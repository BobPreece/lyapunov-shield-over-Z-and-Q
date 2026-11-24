from __future__ import annotations

import dataclasses
import time
from typing import Callable, List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import LPStatus, solve_lp_bareiss
from .lyapunov_verifier import (
    LyapunovVerificationResult,
    SectorPWAPlant,
    check_lyapunov_certificate_bundle,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
)
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
)
from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    Verdict,
    WatchdogTimer,
    check_budget,
    lcm_list,
    q_norm,
    qvec_add,
    qvec_eq,
    qvec_norm,
    qvec_scale,
)


@dataclasses.dataclass(frozen=True)
class AnchorSnapReport:
    """Comparison telemetry between naive lattice quantization and exact post-rounding
    equilibrium anchor subtraction pi_Q(x) - pi_Q(x*) and V_Q(x) - V_Q(x*).
    """

    scale_denom: int
    naive_pi_at_eq: QVec
    naive_v_at_eq: Rat
    anchored_pi_at_eq: QVec
    anchored_v_at_eq: Rat
    naive_result: LyapunovVerificationResult
    anchored_result: LyapunovVerificationResult


@dataclasses.dataclass(frozen=True)
class CEGISSynthesisResult:
    verdict: Verdict
    iterations: int
    witness_count: int
    weights_q: Optional[Tuple[Rat, ...]]
    synthesized_v: Optional[ReLUNetwork]
    verification: Optional[LyapunovVerificationResult]
    elapsed_sec: float
    history_max_violation: Tuple[Rat, ...] = ()
    message: str = ""


def snap_float_matrix_to_lattice(
    W_float: Sequence[Sequence[float]],
    b_float: Sequence[float],
    scale_denom: int = 32,
) -> AffineMapQ:
    """Snap floating-point weights & biases to exact lattice Z / scale_denom."""
    if scale_denom <= 0:
        raise ValueError("scale_denom must be positive")
    out_dim = len(W_float)
    in_dim = len(W_float[0]) if out_dim > 0 else 0
    W_int = [
        [int(round(float(W_float[i][j]) * scale_denom)) for j in range(in_dim)]
        for i in range(out_dim)
    ]
    b_int = [int(round(float(b_float[i]) * scale_denom)) for i in range(out_dim)]
    return AffineMapQ.create(in_dim, out_dim, W_int, b_int, scale_denom)


def snap_relu_network_to_lattice(
    layer_specs: Sequence[Tuple[Sequence[Sequence[float]], Sequence[float], bool]],
    scale_denom: int = 32,
    anchor_equilibrium: Optional[QVec] = None,
    residual_spec: Optional[Tuple[Sequence[Sequence[float]], Sequence[float]]] = None,
    name: str = "snapped_net",
) -> ReLUNetwork:
    """Quantize a floating-point ReLU MLP onto lattice 1/scale_denom, with optional
    mandatory post-rounding equilibrium anchor subtraction y_Q(x) - y_Q(x*).
    """
    layers = tuple(
        ReLULayer(
            affine=snap_float_matrix_to_lattice(Wf, bf, scale_denom=scale_denom),
            use_relu=bool(use_r),
        )
        for Wf, bf, use_r in layer_specs
    )
    in_dim = layers[0].affine.in_dim
    out_dim = layers[-1].affine.out_dim
    res_map = (
        snap_float_matrix_to_lattice(
            residual_spec[0], residual_spec[1], scale_denom=scale_denom
        )
        if residual_spec is not None
        else None
    )
    raw_net = ReLUNetwork(
        in_dim=in_dim,
        out_dim=out_dim,
        layers=layers,
        residual=res_map,
        post_offset=None,
        name=name,
    )
    if anchor_equilibrium is not None:
        return raw_net.with_anchor_zero(anchor_equilibrium)
    return raw_net


def compose_linear_plant_with_relu_controller(
    A_plant: AffineMapQ,
    B_num: Sequence[Sequence[int]],
    B_denom: int,
    controller: ReLUNetwork,
    domain: HPolytope,
    max_cells: int = 2048,
    deadline: Optional[float] = None,
) -> PWAFunction:
    """Build exact closed-loop PWA dynamics F(x) = A_plant(x) + B * controller(x) over domain."""
    n = A_plant.in_dim
    m = controller.out_dim
    pi_pwa = controller.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline)
    b_map = AffineMapQ.create(m, n, B_num, [0] * n, B_denom)
    from .polyhedral_complex import PWACell, bft_merge_cells

    cl_cells = []
    for c in pi_pwa.cells:
        cl_aff = A_plant.add(b_map.compose(c.affine_map))
        cl_cells.append(
            PWACell(
                cell_id=c.cell_id,
                polytope=c.polytope,
                affine_map=cl_aff,
                tag=c.tag,
            )
        )
    merged = bft_merge_cells(cl_cells, deadline=deadline)
    return PWAFunction(
        in_dim=n,
        out_dim=n,
        domain=domain,
        cells=tuple(
            PWACell(cell_id=i, polytope=mc.polytope, affine_map=mc.affine_map, tag=mc.tag)
            for i, mc in enumerate(merged)
        ),
        name=f"closed_loop_{controller.name}",
    )


def build_anchored_template_lyapunov(
    ray_normals: Sequence[Sequence[int]],
    weights_q: Sequence[Rat],
    x_eq: Optional[QVec] = None,
    name: str = "cegis_v",
) -> ReLUNetwork:
    """Construct V(x) = sum_{k=1}^K w_k ReLU(n_k^T (x - x_eq)) anchored at V(x_eq) == 0."""
    K = len(ray_normals)
    if K == 0 or len(weights_q) != K:
        raise ValueError("Mismatched ray_normals and weights_q")
    n = len(ray_normals[0])
    if x_eq is None:
        x_eq = (tuple(0 for _ in range(n)), 1)
    eq_n, eq_d = x_eq

    W_in = [[eq_d * int(ray_normals[k][j]) for j in range(n)] for k in range(K)]
    b_in = [
        -sum(int(ray_normals[k][j]) * eq_n[j] for j in range(n))
        for k in range(K)
    ]
    cd_w = lcm_list([w[1] for w in weights_q])
    w_row = [[weights_q[k][0] * (cd_w // weights_q[k][1]) for k in range(K)]]

    raw = ReLUNetwork(
        in_dim=n,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(n, K, W_in, b_in, eq_d), use_relu=True),
            ReLULayer(AffineMapQ.create(K, 1, w_row, [0], cd_w), use_relu=False),
        ),
        name=name,
    )
    return raw.with_anchor_zero(x_eq)


def synthesize_lyapunov_cegis(
    plant: Union[PWAFunction, ReLUNetwork, SectorPWAPlant],
    domain: HPolytope,
    is_discrete: bool = True,
    ray_normals: Optional[Sequence[Sequence[int]]] = None,
    x_eq: Optional[QVec] = None,
    rate_target: Rat = (1, 20),
    lattice_denom: int = 32,
    max_iters: int = 20,
    timeout_sec: float = 2.5,
) -> CEGISSynthesisResult:
    """Bounded-iteration (< 3s watchdog) exact-Q Counterexample-Guided Inductive Synthesis (CEGIS)
    for polyhedral/ReLU Neural Lyapunov Functions V_w(x) = sum_k w_k ReLU(n_k^T (x - x_eq))
    with mandatory post-rounding equilibrium anchor subtraction!
    """
    t0 = time.monotonic()
    n = domain.dim
    if x_eq is None:
        x_eq = (tuple(0 for _ in range(n)), 1)

    if ray_normals is None:
        # Default symmetric cross-polytope + diagonal basis rays
        rays: List[List[int]] = []
        for j in range(n):
            e_pos = [1 if q == j else 0 for q in range(n)]
            e_neg = [-1 if q == j else 0 for q in range(n)]
            rays.append(e_pos)
            rays.append(e_neg)
        if n <= 4:
            for i in range(n):
                for j in range(i + 1, n):
                    for si in (1, -1):
                        for sj in (1, -1):
                            r = [0] * n
                            r[i] = si
                            r[j] = sj
                            rays.append(r)
        ray_normals = rays

    K = len(ray_normals)
    # Seed initial witness set with unit coordinate corners around x_eq
    witnesses: List[QVec] = []
    for r in ray_normals[: min(K, 4 * n)]:
        cand = qvec_add(x_eq, qvec_norm(r, 4))
        if domain.contains(cand) and not qvec_eq(cand, x_eq):
            witnesses.append(cand)

    # Helper to evaluate feature vector phi(x) = [ReLU(n_k^T (x - x_eq))] in Q^K
    eq_n, eq_d = x_eq

    def phi_eval(x_pt: QVec) -> Tuple[List[int], int]:
        xn, xd = x_pt
        cd = eq_d * xd
        vals = []
        for k in range(K):
            num = sum(
                int(ray_normals[k][j]) * (xn[j] * eq_d - eq_n[j] * xd)
                for j in range(n)
            )
            vals.append(num if num > 0 else 0)
        return (vals, cd)

    def dphi_lie_eval(x_pt: QVec, f_pt: QVec) -> Tuple[List[int], int]:
        # Upper subgradient directional derivative of ReLU(n_k^T (x - x_eq)) along f_pt
        xn, xd = x_pt
        fn, fd = f_pt
        vals = []
        for k in range(K):
            pre = sum(
                int(ray_normals[k][j]) * (xn[j] * eq_d - eq_n[j] * xd)
                for j in range(n)
            )
            df = sum(int(ray_normals[k][j]) * fn[j] for j in range(n))
            if pre > 0 or (pre == 0 and df > 0):
                vals.append(df)
            else:
                vals.append(0)
        return (vals, fd)

    # Pre-extract plant evaluation helper for finite witness constraints
    def plant_next_or_derivs(x_pt: QVec) -> List[QVec]:
        if isinstance(plant, ReLUNetwork):
            return [plant.eval_qvec(x_pt)]
        if isinstance(plant, PWAFunction):
            return [plant.eval_qvec(x_pt)]
        outs: List[QVec] = []
        for sc in plant.cells:
            if sc.polytope.contains(x_pt):
                for g in sc.generators:
                    outs.append(g.eval_qvec(x_pt))
        return outs if outs else [x_eq]

    weights_cur: List[Rat] = [(1, 1) for _ in range(K)]
    hist_viol: List[Rat] = []

    try:
        with WatchdogTimer(timeout_sec, use_sigalrm=False) as wd:
            for it in range(1, max_iters + 1):
                wd.check()
                v_cand = build_anchored_template_lyapunov(
                    ray_normals, weights_cur, x_eq=x_eq, name=f"cegis_v_it{it}"
                )
                if is_discrete:
                    ver_res = verify_discrete_lyapunov(
                        v_cand,
                        plant,
                        domain,
                        x_eq=x_eq,
                        epsilon=rate_target,
                        deadline=wd.deadline,
                    )
                else:
                    ver_res = verify_continuous_lyapunov(
                        v_cand,
                        plant,
                        domain,
                        x_eq=x_eq,
                        decay_lambda=rate_target,
                        deadline=wd.deadline,
                    )

                if ver_res.verdict == Verdict.OPEN_BUDGET_EXCEEDED:
                    return CEGISSynthesisResult(
                        verdict=Verdict.OPEN_BUDGET_EXCEEDED,
                        iterations=it,
                        witness_count=len(witnesses),
                        weights_q=None,
                        synthesized_v=None,
                        verification=ver_res,
                        elapsed_sec=time.monotonic() - t0,
                        history_max_violation=tuple(hist_viol),
                        message=ver_res.message,
                    )

                if ver_res.verdict == Verdict.CERTIFIED_EXACT:
                    assert check_lyapunov_certificate_bundle(ver_res, x_eq=x_eq)
                    return CEGISSynthesisResult(
                        verdict=Verdict.CERTIFIED_EXACT,
                        iterations=it,
                        witness_count=len(witnesses),
                        weights_q=tuple(weights_cur),
                        synthesized_v=v_cand,
                        verification=ver_res,
                        elapsed_sec=time.monotonic() - t0,
                        history_max_violation=tuple(hist_viol),
                        message=f"CEGIS converged to CERTIFIED_EXACT in {it} iteration(s)",
                    )

                # Extract counterexample witness and symmetric reflections around x_eq
                cex = ver_res.counterexample
                if cex is None:
                    break
                hist_viol.append(cex.residual_violation)
                x_w = cex.x_witness
                if qvec_eq(x_w, x_eq):
                    # Equilibrium itself broken; no V template can fix a broken plant equilibrium
                    return CEGISSynthesisResult(
                        verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                        iterations=it,
                        witness_count=len(witnesses),
                        weights_q=tuple(weights_cur),
                        synthesized_v=v_cand,
                        verification=ver_res,
                        elapsed_sec=time.monotonic() - t0,
                        history_max_violation=tuple(hist_viol),
                        message=ver_res.message,
                    )

                # Add x_w, its antipode 2*x_eq - x_w, and scaled midpoint to witnesses
                diff_w = qvec_add(x_w, qvec_scale((-1, 1), x_eq))
                antipode = qvec_add(x_eq, qvec_scale((-1, 1), diff_w))
                half_pt = qvec_add(x_eq, qvec_scale((1, 2), diff_w))
                for cand_pt in (x_w, antipode, half_pt):
                    if domain.contains(cand_pt) and not qvec_eq(cand_pt, x_eq):
                        if not any(qvec_eq(cand_pt, ew) for ew in witnesses):
                            witnesses.append(cand_pt)

                # Solve exact Bareiss LP for weights w in [1/lattice_denom, 8] minimizing slack s:
                # Variables: [w_0 .. w_{K-1}, s] (K + 1 vars), maximize -s!
                lp_A: List[List[int]] = []
                lp_b: List[int] = []
                for k in range(K):
                    # w_k >= 1 / lattice_denom  <=>  -lattice_denom * w_k <= -1
                    r_lo = [0] * (K + 1)
                    r_lo[k] = -lattice_denom
                    lp_A.append(r_lo)
                    lp_b.append(-1)
                    # w_k <= 8
                    r_hi = [0] * (K + 1)
                    r_hi[k] = 1
                    lp_A.append(r_hi)
                    lp_b.append(8)

                rn_t, rd_t = q_norm(rate_target[0], rate_target[1])
                # Require margin +2*rn_t/rd_t on finite witnesses so lattice snapping stays interior!
                margin_num = min(rd_t - 1, 2 * rn_t) if is_discrete else 2 * rn_t
                margin_den = rd_t

                for wit_pt in witnesses:
                    phi_x, d_px = phi_eval(wit_pt)
                    for f_out in plant_next_or_derivs(wit_pt):
                        if is_discrete:
                            phi_fx, d_pfx = phi_eval(f_out)
                            # w^T (phi(F(x))/d_pfx - (1 - margin) phi(x)/d_px) - s <= 0
                            one_m_n = margin_den - margin_num
                            cd_w = lcm_list([d_pfx, d_px * margin_den])
                            sf = cd_w // d_pfx
                            sx = cd_w // (d_px * margin_den)
                            row_w = [
                                sf * phi_fx[k] - sx * one_m_n * phi_x[k]
                                for k in range(K)
                            ] + [-cd_w]
                            lp_A.append(row_w)
                            lp_b.append(0)
                        else:
                            dphi_f, d_df = dphi_lie_eval(wit_pt, f_out)
                            # w^T (dphi/d_df + margin * phi(x)/d_px) - s <= 0
                            cd_w = lcm_list([d_df, d_px * margin_den])
                            sf = cd_w // d_df
                            sx = cd_w // (d_px * margin_den)
                            row_w = [
                                sf * dphi_f[k] + sx * margin_num * phi_x[k]
                                for k in range(K)
                            ] + [-cd_w]
                            lp_A.append(row_w)
                            lp_b.append(0)

                res_w = solve_lp_bareiss(
                    lp_A, lp_b, c=[0] * K + [-1], deadline=wd.deadline
                )
                if res_w.status != LPStatus.OPTIMAL or res_w.x_opt is None:
                    break

                wn, wd_lp = res_w.x_opt
                # Snap weights onto 1 / lattice_denom (enforcing >= 1 / lattice_denom)
                snapped_w: List[Rat] = []
                for k in range(K):
                    # Round (wn[k] * lattice_denom) / wd_lp to nearest integer >= 1
                    num_scaled = (2 * wn[k] * lattice_denom + wd_lp) // (2 * wd_lp)
                    if num_scaled < 1:
                        num_scaled = 1
                    snapped_w.append(q_norm(num_scaled, lattice_denom))
                weights_cur = snapped_w

        return CEGISSynthesisResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            iterations=max_iters,
            witness_count=len(witnesses),
            weights_q=tuple(weights_cur),
            synthesized_v=build_anchored_template_lyapunov(
                ray_normals, weights_cur, x_eq=x_eq
            ),
            verification=None,
            elapsed_sec=time.monotonic() - t0,
            history_max_violation=tuple(hist_viol),
            message=f"CEGIS reached iteration cap ({max_iters}) without global certification",
        )
    except BudgetExceededError as exc:
        return CEGISSynthesisResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            iterations=len(hist_viol),
            witness_count=len(witnesses),
            weights_q=None,
            synthesized_v=None,
            verification=None,
            elapsed_sec=time.monotonic() - t0,
            history_max_violation=tuple(hist_viol),
            message=str(exc),
        )
