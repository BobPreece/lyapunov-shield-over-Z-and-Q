from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .bareiss_simplex import (
    FarkasBoundCert,
    FarkasInfeasibilityCert,
    LPStatus,
    certify_linear_upper_bound,
    check_farkas_certificate,
    check_positivity_farkas,
    check_primal_witness,
    solve_lp_bareiss,
)
from .filippov_crease import _subspace_chebyshev_center
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLUNetwork,
)
from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    Verdict,
    canonical_hyperplane,
    check_budget,
    gcd_list,
    lcm_list,
    lcm_pair,
    q_add,
    q_le,
    q_mul,
    q_norm,
    qvec_add,
    qvec_eq,
    qvec_norm,
    qvec_scale,
)


@dataclasses.dataclass(frozen=True)
class MultiFacetBarrier:
    """Conjunctive barrier C = {x in domain : h_r(x) >= 0 for all r}, i.e., h(x) = min_r h_r(x)."""

    dim: int
    domain: HPolytope
    facets: Tuple[AffineMapQ, ...]
    name: str = "multi_facet_barrier"

    def eval_scalar(self, x_q: QVec) -> Rat:
        vals = [f.eval_scalar(x_q) for f in self.facets]
        best = vals[0]
        for v in vals[1:]:
            if not q_le(best, v):
                best = v
        return best


@dataclasses.dataclass(frozen=True)
class ControlAffineCell:
    """Control-affine dynamics x_dot = f(x) + (B_num / B_denom) u on polytope."""

    cell_id: int
    polytope: HPolytope
    f_map: AffineMapQ
    B_num: Tuple[Tuple[int, ...], ...] = ()
    B_denom: int = 1
    tag: str = ""


@dataclasses.dataclass(frozen=True)
class ControlAffinePWAPlant:
    dim: int
    control_dim: int
    domain: HPolytope
    cells: Tuple[ControlAffineCell, ...]
    u_poly: Optional[HPolytope] = None
    name: str = ""

    @staticmethod
    def autonomous_from_pwa(pwa: PWAFunction) -> ControlAffinePWAPlant:
        return ControlAffinePWAPlant(
            dim=pwa.in_dim,
            control_dim=0,
            domain=pwa.domain,
            cells=tuple(
                ControlAffineCell(
                    cell_id=c.cell_id,
                    polytope=c.polytope,
                    f_map=c.affine_map,
                    B_num=tuple(() for _ in range(pwa.in_dim)),
                    B_denom=1,
                    tag=c.tag,
                )
                for c in pwa.cells
            ),
            u_poly=None,
            name=pwa.name,
        )

    @staticmethod
    def single_cell(
        dim: int,
        control_dim: int,
        domain: HPolytope,
        f_map: AffineMapQ,
        B_num: Sequence[Sequence[int]],
        B_denom: int = 1,
        u_poly: Optional[HPolytope] = None,
        name: str = "",
    ) -> ControlAffinePWAPlant:
        return ControlAffinePWAPlant(
            dim=dim,
            control_dim=control_dim,
            domain=domain,
            cells=(
                ControlAffineCell(
                    cell_id=0,
                    polytope=domain,
                    f_map=f_map,
                    B_num=tuple(tuple(int(v) for v in r) for r in B_num),
                    B_denom=int(B_denom),
                    tag="global",
                ),
            ),
            u_poly=u_poly,
            name=name,
        )


@dataclasses.dataclass(frozen=True)
class BarrierViabilityFaceCert:
    """Constructive integer Farkas certificate proving Bouligand viability on boundary face F
    via exact rational affine safe feedback u(x) = (K_num x + u0_num) / u_scale:
      - Every row of tangent_certs certifies M_tan_row * (f(x) + B u(x)) <= 0 on F.
      - Every row of sat_certs certifies A_u_row * u(x) <= c_u_row on F.
    """

    face_id: int
    valid_cell_id: int
    face_A: Tuple[Tuple[int, ...], ...]
    face_b: Tuple[int, ...]
    K_num: Tuple[Tuple[int, ...], ...]
    u0_num: Tuple[int, ...]
    u_scale: int
    tangent_certs: Tuple[FarkasBoundCert, ...]
    sat_certs: Tuple[FarkasBoundCert, ...]


@dataclasses.dataclass(frozen=True)
class CornerDeadlockSubCert:
    """Exact non-negative integer Farkas dual (lambda_tan, eta_sat) >= 0 proving that
    at corner x_star, no u in {A_u u <= c_u} can enter valid cell's Bouligand tangent cone
    {v : M_act v <= 0}:
      lambda_tan >= 0, eta_sat >= 0,
      lambda_tan^T (M_act * B_scaled) + eta_sat^T (A_u * B_denom) == 0,
      lambda_tan^T (-M_act * f(x*)) + eta_sat^T c_u < 0 (in exact scaled integers!).
    """

    valid_cell_id: int
    M_act: Tuple[Tuple[int, ...], ...]
    control_lp_A: Tuple[Tuple[int, ...], ...]
    control_lp_b: Tuple[int, ...]
    lambda_tan: Tuple[int, ...]
    eta_sat: Tuple[int, ...]
    rhs_dot: int


@dataclasses.dataclass(frozen=True)
class BarrierCounterexample:
    reason: str
    x_star: QVec
    h_at_x: Rat
    deadlock_subcerts: Tuple[CornerDeadlockSubCert, ...] = ()


@dataclasses.dataclass(frozen=True)
class BarrierVerificationResult:
    verdict: Verdict
    faces_certified: int = 0
    corners_checked: int = 0
    init_certs: Tuple[FarkasBoundCert, ...] = ()
    init_polys: Tuple[Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]], ...] = ()
    unsafe_certs: Tuple[FarkasInfeasibilityCert, ...] = ()
    unsafe_polys: Tuple[Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]], ...] = ()
    viability_certs: Tuple[BarrierViabilityFaceCert, ...] = ()
    counterexample: Optional[BarrierCounterexample] = None
    message: str = ""


def check_corner_deadlock_cert(sub: CornerDeadlockSubCert) -> bool:
    """Standalone zero-trust pure-integer checker for Farkas corner-deadlock dual (lambda, eta) >= 0."""
    y_comb = list(sub.lambda_tan) + list(sub.eta_sat)
    if len(sub.lambda_tan) == 0:
        return False
    if any(not isinstance(v, int) or v < 0 for v in y_comb):
        return False
    if len(sub.control_lp_A) == 0:
        # Autonomous (0 control vars): control_lp_A has 0 columns, check sum(y_i * b_i) < 0
        if len(sub.control_lp_b) != len(y_comb):
            return False
        return sum(y_comb[i] * sub.control_lp_b[i] for i in range(len(y_comb))) < 0
    if len(sub.control_lp_A[0]) == 0:
        if len(sub.control_lp_b) != len(y_comb):
            return False
        return sum(y_comb[i] * sub.control_lp_b[i] for i in range(len(y_comb))) < 0
    return check_farkas_certificate(sub.control_lp_A, sub.control_lp_b, y_comb)


def check_barrier_certificate_bundle(result: BarrierVerificationResult) -> bool:
    """Standalone zero-trust pure-integer checker for CERTIFIED_EXACT BarrierVerificationResult."""
    if result.verdict != Verdict.CERTIFIED_EXACT:
        return False
    if len(result.init_certs) != len(result.init_polys):
        return False
    for bc, (pA, pb) in zip(result.init_certs, result.init_polys):
        if not check_positivity_farkas(pA, pb, bc):
            return False
    if len(result.unsafe_certs) != len(result.unsafe_polys):
        return False
    for ic, (pA, pb) in zip(result.unsafe_certs, result.unsafe_polys):
        if not check_farkas_certificate(pA, pb, ic.y):
            return False
    for vc in result.viability_certs:
        if vc.u_scale <= 0:
            return False
        for tc in vc.tangent_certs:
            if not check_positivity_farkas(vc.face_A, vc.face_b, tc):
                return False
        for sc in vc.sat_certs:
            if not check_positivity_farkas(vc.face_A, vc.face_b, sc):
                return False
    return True


def eval_heaviside_formal_lie_derivative(
    h_net: ReLUNetwork,
    x_q: QVec,
    v_q: QVec,
    heaviside_at_zero: Rat = (0, 1),
) -> Rat:
    """Evaluate naive scalar Heaviside step-derivative chain rule (Zhao et al. 2022) at x_q along v_q
    using ReLU'(0) = H in [0, 1]. Demonstrates Trap 2 unsoundness where H cancels (-H + H = 0)!
    """
    if h_net.out_dim != 1:
        raise ValueError("h_net must be scalar")
    hn, hd = q_norm(heaviside_at_zero[0], heaviside_at_zero[1])
    cur = x_q
    # Propagate exact rational directional sensitivity J * v_q layer by layer
    cur_dv = v_q
    for layer in h_net.layers:
        pre = layer.affine.eval_qvec(cur)
        # d(pre) = (W * cur_dv) / denom
        dvn, dvd = cur_dv
        w_dv = [
            sum(layer.affine.W[i][j] * dvn[j] for j in range(layer.affine.in_dim))
            for i in range(layer.affine.out_dim)
        ]
        cur_dv = qvec_norm(w_dv, layer.affine.denom * dvd)
        if not layer.use_relu:
            cur = pre
            continue
        pn, pd = pre
        d_n, d_d = cur_dv
        out_p = [v if v > 0 else 0 for v in pn]
        # Scale derivative component i by 1 if pn[i] > 0, 0 if pn[i] < 0, hn/hd if pn[i] == 0
        out_dv = []
        for i in range(len(pn)):
            if pn[i] > 0:
                out_dv.append(d_n[i] * hd)
            elif pn[i] < 0:
                out_dv.append(0)
            else:
                out_dv.append(d_n[i] * hn)
        cur = qvec_norm(out_p, pd)
        cur_dv = qvec_norm(out_dv, d_d * hd)

    res_val = q_norm(cur_dv[0][0], cur_dv[1])
    if h_net.residual is not None:
        vn, vd = v_q
        rw = sum(h_net.residual.W[0][j] * vn[j] for j in range(h_net.in_dim))
        res_val = q_add(res_val, q_norm(rw, h_net.residual.denom * vd))
    return res_val


@dataclasses.dataclass(frozen=True)
class _ValidSubCell:
    vid: int
    poly: HPolytope
    f_map: AffineMapQ
    B_num: Tuple[Tuple[int, ...], ...]
    B_denom: int
    # Indices of rows in poly.A that represent h_r(x) >= 0 barrier constraints (vs domain box)
    barrier_normals: Set[Tuple[Tuple[int, ...], int]]


def _pointwise_bouligand_check(
    x_pt: QVec,
    valid_cells: Sequence[_ValidSubCell],
    u_poly: Optional[HPolytope],
    control_dim: int,
    deadline: Optional[float] = None,
) -> Tuple[bool, List[CornerDeadlockSubCert]]:
    """Test exact Nagumo-Bouligand tangent-cone viability at point x_pt:
    returns (True, []) if SOME valid cell V_q containing x_pt admits u in U with f_q(x_pt) + B_q u in T_{V_q}(x_pt),
    or (False, subcerts) with an exact integer Farkas certificate on EVERY valid cell V_q containing x_pt!
    """
    xn, xd = x_pt
    n = len(xn)
    containing = [vc for vc in valid_cells if vc.poly.contains(x_pt)]
    if not containing:
        return (True, [])

    # Autonomous discontinuous Filippov boundary invariance check when f jumps across an active crease
    if control_dim == 0 and len(containing) >= 2:
        f_vals = [vc.f_map.eval_qvec(x_pt) for vc in containing]
        if not all(qvec_eq(fv, f_vals[0]) for fv in f_vals[1:]):
            act_per_vc: List[List[Tuple[int, ...]]] = []
            for vc in containing:
                act_per_vc.append(
                    [
                        vc.poly.A[i]
                        for i in range(len(vc.poly.A))
                        if sum(vc.poly.A[i][j] * xn[j] for j in range(n))
                        == vc.poly.b[i] * xd
                    ]
                )

            def _in_bouligand_cone(vq: QVec) -> bool:
                return any(
                    all(
                        sum(ar[j] * vq[0][j] for j in range(n)) <= 0
                        for ar in act_rows
                    )
                    for act_rows in act_per_vc
                )

            found_sliding = False
            sliding_all_safe = True
            for i1 in range(len(containing)):
                set1 = set(act_per_vc[i1])
                for i2 in range(i1 + 1, len(containing)):
                    set2 = set(act_per_vc[i2])
                    for n_sw in set1:
                        neg_sw = tuple(-v for v in n_sw)
                        if neg_sw not in set2:
                            continue
                        f1 = f_vals[i1]
                        f2 = f_vals[i2]
                        nu1 = q_norm(
                            sum(n_sw[j] * f1[0][j] for j in range(n)), f1[1]
                        )
                        nu2 = q_norm(
                            sum(n_sw[j] * f2[0][j] for j in range(n)), f2[1]
                        )
                        if nu1[0] >= 0 and nu2[0] <= 0:
                            found_sliding = True
                            if nu1[0] == 0 and nu2[0] == 0:
                                if not (
                                    _in_bouligand_cone(f1)
                                    and _in_bouligand_cone(f2)
                                ):
                                    sliding_all_safe = False
                            else:
                                diff_nu = q_add(nu1, (-nu2[0], nu2[1]))
                                lam = q_norm(
                                    nu1[0] * diff_nu[1], nu1[1] * diff_nu[0]
                                )
                                one_m_lam = q_add((1, 1), (-lam[0], lam[1]))
                                f_s = qvec_add(
                                    qvec_scale(one_m_lam, f1),
                                    qvec_scale(lam, f2),
                                )
                                if not _in_bouligand_cone(f_s):
                                    sliding_all_safe = False
            if found_sliding:
                if sliding_all_safe:
                    return (True, [])
                # Otherwise fall through to emit deadlock subcerts if individual branches also fail,
                # or reject directly if a non-sliding branch happened to enter while sliding escaped!
                # Check if individual loop would accept falsely when an attractive sliding mode escapes:
                indiv_would_accept = any(
                    all(
                        sum(ar[j] * f_vals[idx_v][0][j] for j in range(n)) <= 0
                        for ar in act_per_vc[idx_v]
                    )
                    for idx_v in range(len(containing))
                )
                if indiv_would_accept:
                    return (False, [])

    subcerts: List[CornerDeadlockSubCert] = []
    for vc in containing:
        check_budget(deadline=deadline)
        # Collect ALL active constraints a_p^T x_pt == b_p of V_q (except outer domain constraints if desired?
        # Wait: at interior points of domain D, domain constraints are strictly inactive! Keep all active rows of vc.poly!)
        M_act: List[Tuple[int, ...]] = []
        for i in range(len(vc.poly.A)):
            ar = vc.poly.A[i]
            br = vc.poly.b[i]
            if sum(ar[j] * xn[j] for j in range(n)) == br * xd:
                M_act.append(ar)
        if not M_act:
            return (True, [])

        f_val = vc.f_map.eval_qvec(x_pt)  # (fn, fd)
        fn, fd = f_val

        if control_dim == 0:
            # Autonomous: check M_act * fn <= 0
            viol_idx = -1
            b_aut: List[int] = []
            for idx_m, m_row in enumerate(M_act):
                # m_row^T (fn / fd) <= 0  <=>  0 * u <= -(m_row^T fn)
                rhs_m = -sum(m_row[j] * fn[j] for j in range(n))
                b_aut.append(rhs_m)
                if rhs_m < 0 and viol_idx == -1:
                    viol_idx = idx_m
            if viol_idx == -1:
                return (True, [])
            lam = tuple(1 if k == viol_idx else 0 for k in range(len(M_act)))
            sub = CornerDeadlockSubCert(
                valid_cell_id=vc.vid,
                M_act=tuple(M_act),
                control_lp_A=tuple(() for _ in range(len(M_act))),
                control_lp_b=tuple(b_aut),
                lambda_tan=lam,
                eta_sat=(),
                rhs_dot=b_aut[viol_idx],
            )
            assert check_corner_deadlock_cert(sub)
            subcerts.append(sub)
            continue

        # Control-affine: M_act * (fn / fd + (B_num / B_denom) u) <= 0
        # Multiply by cd = lcm(fd, B_denom) > 0:
        # (cd // B_denom) * (M_act * B_num) u <= -(cd // fd) * (M_act * fn)
        cd = lcm_pair(fd, vc.B_denom)
        sb = cd // vc.B_denom
        sf = cd // fd
        lp_A: List[List[int]] = []
        lp_b: List[int] = []
        for m_row in M_act:
            row_u = [
                sb * sum(m_row[i] * vc.B_num[i][col] for i in range(n))
                for col in range(control_dim)
            ]
            rhs_u = -sf * sum(m_row[i] * fn[i] for i in range(n))
            lp_A.append(row_u)
            lp_b.append(rhs_u)

        k_tan = len(M_act)
        if u_poly is not None:
            for r_u in range(len(u_poly.A)):
                lp_A.append(list(u_poly.A[r_u]))
                lp_b.append(u_poly.b[r_u])

        res_u = solve_lp_bareiss(lp_A, lp_b, c=None, deadline=deadline)
        if res_u.status == LPStatus.BUDGET_EXCEEDED:
            raise BudgetExceededError("LP budget exceeded in pointwise Bouligand check")
        if res_u.status != LPStatus.INFEASIBLE:
            return (True, [])

        assert res_u.farkas_infeas is not None
        y_all = res_u.farkas_infeas.y
        lam_tan = tuple(y_all[:k_tan])
        eta_sat = tuple(y_all[k_tan:])
        sub = CornerDeadlockSubCert(
            valid_cell_id=vc.vid,
            M_act=tuple(M_act),
            control_lp_A=tuple(tuple(r) for r in lp_A),
            control_lp_b=tuple(lp_b),
            lambda_tan=lam_tan,
            eta_sat=eta_sat,
            rhs_dot=res_u.farkas_infeas.rhs_dot,
        )
        assert check_corner_deadlock_cert(sub)
        subcerts.append(sub)

    return (False, subcerts)


def _try_certify_face_affine_u(
    face_id: int,
    face_poly: HPolytope,
    vc: _ValidSubCell,
    M_tan: Sequence[Tuple[int, ...]],
    u_poly: Optional[HPolytope],
    control_dim: int,
    deadline: Optional[float] = None,
) -> Optional[BarrierViabilityFaceCert]:
    """Synthesize an exact rational affine feedback u(x) = (K x + u0)/scale on face_poly
    and emit standalone integer FarkasBoundCerts proving M_tan (f(x) + B u(x)) <= 0 and A_u u(x) <= c_u!
    """
    n = face_poly.dim
    A_F = face_poly.A
    b_F = face_poly.b
    m_F = len(A_F)
    k_tan = len(M_tan)
    fm = vc.f_map

    if control_dim == 0:
        # Autonomous: directly certify M_tan * f(x) <= 0 on face_poly!
        t_certs: List[FarkasBoundCert] = []
        for m_row in M_tan:
            # m_row^T (W_f x + c_f) / d_f <= 0
            c_num = [sum(m_row[i] * fm.W[i][j] for i in range(n)) for j in range(n)]
            rhs_num = -sum(m_row[i] * fm.c[i] for i in range(n))
            v_str, b_cert, _, _ = certify_linear_upper_bound(
                A_F, b_F, c_num, fm.denom, rhs_num, fm.denom, deadline=deadline
            )
            if v_str == "OPEN_BUDGET_EXCEEDED":
                raise BudgetExceededError("LP budget exceeded in autonomous barrier face")
            if v_str != "CERTIFIED_EXACT" or b_cert is None:
                return None
            t_certs.append(b_cert)
        return BarrierViabilityFaceCert(
            face_id=face_id,
            valid_cell_id=vc.vid,
            face_A=A_F,
            face_b=b_F,
            K_num=(),
            u0_num=(),
            u_scale=1,
            tangent_certs=tuple(t_certs),
            sat_certs=(),
        )

    # Control-affine joint LP in:
    # - K in R^{control_dim x n} (control_dim * n vars)
    # - u0 in R^{control_dim}    (control_dim vars)
    # - Y_tan[r] >= 0 in R^{m_F} (k_tan * m_F vars)
    # - Y_sat[s] >= 0 in R^{m_F} (m_sat * m_F vars)
    m_sat = len(u_poly.A) if u_poly is not None else 0
    k_vars = control_dim * n
    u0_off = k_vars
    ytan_off = u0_off + control_dim
    ysat_off = ytan_off + k_tan * m_F
    total_dec = ysat_off + m_sat * m_F

    cd = lcm_pair(fm.denom, vc.B_denom)
    sf = cd // fm.denom
    sb = cd // vc.B_denom

    joint_A: List[List[int]] = []
    joint_b: List[int] = []

    # 1) Tangent inequalities: for r = 0..k_tan-1:
    # (sf * m_r^T W_f + sb * m_r^T B K) x + (sf * m_r^T c_f + sb * m_r^T B u0) <= 0  forall x in face_poly
    # Farkas dual y_r >= 0:
    #   y_r^T A_F - sb * (m_r^T B) K == sf * (m_r^T W_f)
    #   y_r^T b_F + sb * (m_r^T B) u0 <= -sf * (m_r^T c_f)
    for r_idx, m_row in enumerate(M_tan):
        y_base = ytan_off + r_idx * m_F
        for i in range(m_F):
            rnn = [0] * total_dec
            rnn[y_base + i] = -1
            joint_A.append(rnn)
            joint_b.append(0)
        mb = [
            sb * sum(m_row[i] * vc.B_num[i][u_col] for i in range(n))
            for u_col in range(control_dim)
        ]
        mw = [
            sf * sum(m_row[i] * fm.W[i][j] for i in range(n))
            for j in range(n)
        ]
        mc = sf * sum(m_row[i] * fm.c[i] for i in range(n))

        for j in range(n):
            req = [0] * total_dec
            for u_col in range(control_dim):
                req[u_col * n + j] = -mb[u_col]
            for i in range(m_F):
                req[y_base + i] = A_F[i][j]
            joint_A.append(req)
            joint_b.append(mw[j])
            joint_A.append([-v for v in req])
            joint_b.append(-mw[j])

        rrhs = [0] * total_dec
        for u_col in range(control_dim):
            rrhs[u0_off + u_col] = mb[u_col]
        for i in range(m_F):
            rrhs[y_base + i] = b_F[i]
        joint_A.append(rrhs)
        joint_b.append(-mc)

    # 2) Actuator saturation inequalities: for s = 0..m_sat-1:
    # (a_{u,s}^T K) x + a_{u,s}^T u0 <= c_{u,s}  forall x in face_poly
    # Farkas dual z_s >= 0:
    #   z_s^T A_F - a_{u,s}^T K == 0
    #   z_s^T b_F + a_{u,s}^T u0 <= c_{u,s}
    if u_poly is not None:
        for s_idx in range(m_sat):
            z_base = ysat_off + s_idx * m_F
            au = u_poly.A[s_idx]
            cu = u_poly.b[s_idx]
            for i in range(m_F):
                rnn = [0] * total_dec
                rnn[z_base + i] = -1
                joint_A.append(rnn)
                joint_b.append(0)
            for j in range(n):
                req = [0] * total_dec
                for u_col in range(control_dim):
                    req[u_col * n + j] = -au[u_col]
                for i in range(m_F):
                    req[z_base + i] = A_F[i][j]
                joint_A.append(req)
                joint_b.append(0)
                joint_A.append([-v for v in req])
                joint_b.append(0)
            rrhs = [0] * total_dec
            for u_col in range(control_dim):
                rrhs[u0_off + u_col] = au[u_col]
            for i in range(m_F):
                rrhs[z_base + i] = b_F[i]
            joint_A.append(rrhs)
            joint_b.append(cu)

    res_j = solve_lp_bareiss(joint_A, joint_b, c=None, deadline=deadline)
    if res_j.status == LPStatus.BUDGET_EXCEEDED:
        raise BudgetExceededError("LP budget exceeded in joint barrier feedback LP")
    if res_j.status != LPStatus.OPTIMAL or res_j.x_opt is None:
        return None

    dn, dd = res_j.x_opt
    K_mat = tuple(
        tuple(dn[u_col * n + j] for j in range(n))
        for u_col in range(control_dim)
    )
    u0_vec = tuple(dn[u0_off + u_col] for u_col in range(control_dim))
    u_map = AffineMapQ.create(n, control_dim, K_mat, u0_vec, dd)
    b_aff = AffineMapQ.create(control_dim, n, vc.B_num, [0] * n, vc.B_denom)
    closed_f = fm.add(b_aff.compose(u_map))

    # Emit clean standalone FarkasBoundCerts directly on face_poly!
    t_certs = []
    for m_row in M_tan:
        c_num = [sum(m_row[i] * closed_f.W[i][j] for i in range(n)) for j in range(n)]
        rhs_num = -sum(m_row[i] * closed_f.c[i] for i in range(n))
        v_str, bc, _, _ = certify_linear_upper_bound(
            A_F, b_F, c_num, closed_f.denom, rhs_num, closed_f.denom, deadline=deadline
        )
        if v_str != "CERTIFIED_EXACT" or bc is None:
            return None
        t_certs.append(bc)

    s_certs: List[FarkasBoundCert] = []
    if u_poly is not None:
        for s_idx in range(m_sat):
            au = u_poly.A[s_idx]
            cu = u_poly.b[s_idx]
            c_num = [
                sum(au[u_col] * u_map.W[u_col][j] for u_col in range(control_dim))
                for j in range(n)
            ]
            rhs_num = cu * u_map.denom - sum(
                au[u_col] * u_map.c[u_col] for u_col in range(control_dim)
            )
            v_str, sc, _, _ = certify_linear_upper_bound(
                A_F, b_F, c_num, u_map.denom, rhs_num, u_map.denom, deadline=deadline
            )
            if v_str != "CERTIFIED_EXACT" or sc is None:
                return None
            s_certs.append(sc)

    return BarrierViabilityFaceCert(
        face_id=face_id,
        valid_cell_id=vc.vid,
        face_A=A_F,
        face_b=b_F,
        K_num=K_mat,
        u0_num=u0_vec,
        u_scale=dd,
        tangent_certs=tuple(t_certs),
        sat_certs=tuple(s_certs),
    )


def verify_barrier(
    barrier: Union[PWAFunction, ReLUNetwork, MultiFacetBarrier],
    plant: ControlAffinePWAPlant,
    domain: Optional[HPolytope] = None,
    initial_set: Sequence[HPolytope] = (),
    unsafe_set: Sequence[HPolytope] = (),
    enforce_domain_invariance: bool = False,
    max_cells: int = 2048,
    max_face_splits: int = 16,
    deadline: Optional[float] = None,
) -> BarrierVerificationResult:
    """Spec 2c: Exact Nagumo-Bouligand tangent-cone NCBF & multi-facet corner verifier over Z/Q.

    Verifies:
      1) S_I subset C = {x : h(x) >= 0},
      2) S_U intersect C == empty,
      3) Exact Bouligand tangent-cone viability across all valid boundary facets and crease corners
         under polyhedral actuator saturation A_u u <= c_u (emitting exact integer Farkas
         corner-deadlock duals (lambda, eta) >= 0 when deadlocked).
    """
    try:
        n = plant.dim
        if domain is None:
            domain = plant.domain

        # Step 1: Decompose C = {x in domain : h(x) >= 0} into valid sub-cells V_q
        valid_cells: List[_ValidSubCell] = []
        spurious_pts: List[QVec] = []
        init_certs: List[FarkasBoundCert] = []
        init_polys: List[Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]] = []
        unsafe_certs: List[FarkasInfeasibilityCert] = []
        unsafe_polys: List[Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]] = []

        dom_constraint_set = {
            (domain.A[i], domain.b[i]) for i in range(len(domain.A))
        }

        if isinstance(barrier, MultiFacetBarrier):
            # Initial set S_I subset C
            for p_init in initial_set:
                for fh in barrier.facets:
                    check_budget(deadline=deadline)
                    # -h_r(x) <= 0  <=>  (-W_0 / d) x <= c_0 / d
                    v_str, bc, wit_i, _ = certify_linear_upper_bound(
                        p_init.A,
                        p_init.b,
                        [-v for v in fh.W[0]],
                        fh.denom,
                        fh.c[0],
                        fh.denom,
                        deadline=deadline,
                    )
                    if v_str == "OPEN_BUDGET_EXCEEDED":
                        raise BudgetExceededError("LP budget exceeded in S_I check")
                    if v_str != "CERTIFIED_EXACT" or bc is None:
                        assert wit_i is not None
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            counterexample=BarrierCounterexample(
                                reason="Initial set S_I violates h(x) >= 0",
                                x_star=wit_i,
                                h_at_x=barrier.eval_scalar(wit_i),
                            ),
                            message=f"S_I violated barrier at {wit_i}",
                        )
                    init_certs.append(bc)
                    init_polys.append((p_init.A, p_init.b))

            # Unsafe set S_U intersect C == empty
            c_base = domain
            b_norms: Set[Tuple[Tuple[int, ...], int]] = set()
            for fh in barrier.facets:
                # h_r(x) >= 0  <=>  -W_0 x <= c_0
                nr, br_val, _ = (
                    tuple(-v for v in fh.W[0]),
                    fh.c[0],
                    1,
                )
                c_base = c_base.with_halfspace(nr, br_val)
                # Record primitive inequality row in c_base corresponding to this barrier facet
                from .z_linalg import primitive_ineq_row
                pa, pb, _ = primitive_ineq_row(nr, br_val)
                b_norms.add((pa, pb))

            for p_uns in unsafe_set:
                check_budget(deadline=deadline)
                inter_u = c_base.intersect(p_uns)
                res_u = solve_lp_bareiss(inter_u.A, inter_u.b, c=None, deadline=deadline)
                if res_u.status == LPStatus.BUDGET_EXCEEDED:
                    raise BudgetExceededError("LP budget exceeded in S_U check")
                if res_u.status != LPStatus.INFEASIBLE:
                    assert res_u.x_opt is not None
                    return BarrierVerificationResult(
                        verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                        counterexample=BarrierCounterexample(
                            reason="Unsafe set S_U intersects C = {h(x) >= 0}",
                            x_star=res_u.x_opt,
                            h_at_x=barrier.eval_scalar(res_u.x_opt),
                        ),
                        message=f"S_U intersects safe set at {res_u.x_opt}",
                    )
                assert res_u.farkas_infeas is not None
                unsafe_certs.append(res_u.farkas_infeas)
                unsafe_polys.append((inter_u.A, inter_u.b))

            for pc in plant.cells:
                vp = c_base.intersect(pc.polytope)
                ok_fd, _, _ = vp.is_full_dimensional(deadline=deadline)
                if ok_fd:
                    valid_cells.append(
                        _ValidSubCell(
                            vid=len(valid_cells),
                            poly=vp,
                            f_map=pc.f_map,
                            B_num=pc.B_num,
                            B_denom=pc.B_denom,
                            barrier_normals=set(b_norms),
                        )
                    )
                else:
                    feas_sp, x_sp = vp.is_feasible(deadline=deadline)
                    if feas_sp and x_sp is not None:
                        spurious_pts.append(x_sp)
        else:
            h_pwa = (
                barrier.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline)
                if isinstance(barrier, ReLUNetwork)
                else barrier
            )
            # Initial set S_I subset C
            for p_init in initial_set:
                for hc in h_pwa.cells:
                    check_budget(deadline=deadline)
                    inter_i = hc.polytope.intersect(p_init)
                    hm = hc.affine_map
                    v_str, bc, wit_i, _ = certify_linear_upper_bound(
                        inter_i.A,
                        inter_i.b,
                        [-v for v in hm.W[0]],
                        hm.denom,
                        hm.c[0],
                        hm.denom,
                        deadline=deadline,
                    )
                    if v_str == "OPEN_BUDGET_EXCEEDED":
                        raise BudgetExceededError("LP budget exceeded in S_I PWA check")
                    if v_str != "CERTIFIED_EXACT" or bc is None:
                        assert wit_i is not None
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            counterexample=BarrierCounterexample(
                                reason="Initial set S_I violates h(x) >= 0",
                                x_star=wit_i,
                                h_at_x=hm.eval_scalar(wit_i),
                            ),
                            message=f"S_I violated barrier at {wit_i}",
                        )
                    init_certs.append(bc)
                    init_polys.append((inter_i.A, inter_i.b))

            # Unsafe set S_U intersect C == empty
            for p_uns in unsafe_set:
                for hc in h_pwa.cells:
                    check_budget(deadline=deadline)
                    hm = hc.affine_map
                    inter_u = hc.polytope.intersect(p_uns).with_halfspace(
                        [-v for v in hm.W[0]], hm.c[0]
                    )
                    res_u = solve_lp_bareiss(
                        inter_u.A, inter_u.b, c=None, deadline=deadline
                    )
                    if res_u.status == LPStatus.BUDGET_EXCEEDED:
                        raise BudgetExceededError("LP budget exceeded in S_U PWA check")
                    if res_u.status != LPStatus.INFEASIBLE:
                        assert res_u.x_opt is not None
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            counterexample=BarrierCounterexample(
                                reason="Unsafe set S_U intersects C = {h(x) >= 0}",
                                x_star=res_u.x_opt,
                                h_at_x=hm.eval_scalar(res_u.x_opt),
                            ),
                            message=f"S_U intersects safe set at {res_u.x_opt}",
                        )
                    assert res_u.farkas_infeas is not None
                    unsafe_certs.append(res_u.farkas_infeas)
                    unsafe_polys.append((inter_u.A, inter_u.b))

            # Build valid sub-cells V_q = (hc cap pc) cap {h_c(x) >= 0}
            from .z_linalg import primitive_ineq_row
            for hc in h_pwa.cells:
                hm = hc.affine_map
                h_neg_w = [-v for v in hm.W[0]]
                h_rhs = hm.c[0]
                pa_h, pb_h, _ = primitive_ineq_row(h_neg_w, h_rhs)
                for pc in plant.cells:
                    check_budget(deadline=deadline)
                    vp = hc.polytope.intersect(pc.polytope).with_halfspace(
                        h_neg_w, h_rhs
                    )
                    ok_fd, _, _ = vp.is_full_dimensional(deadline=deadline)
                    if ok_fd:
                        b_set: Set[Tuple[Tuple[int, ...], int]] = set()
                        if any(v != 0 for v in pa_h):
                            b_set.add((pa_h, pb_h))
                        valid_cells.append(
                            _ValidSubCell(
                                vid=len(valid_cells),
                                poly=vp.prune_redundant(deadline=deadline),
                                f_map=pc.f_map,
                                B_num=pc.B_num,
                                B_denom=pc.B_denom,
                                barrier_normals=b_set,
                            )
                        )
                    else:
                        feas_sp, x_sp = vp.is_feasible(deadline=deadline)
                        if feas_sp and x_sp is not None:
                            spurious_pts.append(x_sp)

        # Helper to evaluate h(x) for counterexample reporting
        def eval_h_any(x_pt: QVec) -> Rat:
            if isinstance(barrier, MultiFacetBarrier):
                return barrier.eval_scalar(x_pt)
            if isinstance(barrier, ReLUNetwork):
                return barrier.eval_scalar(x_pt)
            return barrier.eval_scalar(x_pt)

        # Regular closedness verification: Int(C) != empty and C == cl(Int(C))
        if not valid_cells:
            feas_d, x_dom = domain.is_feasible(deadline=deadline)
            wit_rc = (
                spurious_pts[0]
                if spurious_pts
                else (x_dom if (feas_d and x_dom is not None) else (tuple(0 for _ in range(n)), 1))
            )
            return BarrierVerificationResult(
                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                init_certs=tuple(init_certs),
                init_polys=tuple(init_polys),
                unsafe_certs=tuple(unsafe_certs),
                unsafe_polys=tuple(unsafe_polys),
                counterexample=BarrierCounterexample(
                    reason="Safe set C has empty interior (violates regular closedness)",
                    x_star=wit_rc,
                    h_at_x=eval_h_any(wit_rc),
                ),
                message="Regular closedness failed: Int(C) is empty",
            )

        for x_sp in spurious_pts:
            if not any(vc.poly.contains(x_sp) for vc in valid_cells):
                return BarrierVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    init_certs=tuple(init_certs),
                    init_polys=tuple(init_polys),
                    unsafe_certs=tuple(unsafe_certs),
                    unsafe_polys=tuple(unsafe_polys),
                    counterexample=BarrierCounterexample(
                        reason="Safe set C violates regular closedness (isolated lower-dimensional component)",
                        x_star=x_sp,
                        h_at_x=eval_h_any(x_sp),
                    ),
                    message=f"Regular closedness failed at isolated lower-dim point {x_sp}",
                )

        # Helper to check if point x_pt lies strictly in interior of domain (unless enforce_domain_invariance=True)
        def is_interior_to_domain(x_pt: QVec) -> bool:
            if enforce_domain_invariance:
                return True
            xn, xd = x_pt
            for i_d in range(len(domain.A)):
                if sum(domain.A[i_d][j] * xn[j] for j in range(n)) >= domain.b[i_d] * xd:
                    return False
            return True

        # Step 2: Collect all boundary facets F_{q, r} (codim 1) and corner intersections!
        viability_certs: List[BarrierViabilityFaceCert] = []
        raw_facets: List[Tuple[_ValidSubCell, Tuple[int, ...], int, HPolytope]] = []

        for vc in valid_cells:
            target_pairs = (
                {(vc.poly.A[i], vc.poly.b[i]) for i in range(len(vc.poly.A))}
                if enforce_domain_invariance
                else (
                    vc.barrier_normals
                    - dom_constraint_set
                )
            )
            poly_pairs = {(vc.poly.A[i], vc.poly.b[i]) for i in range(len(vc.poly.A))}
            for a_row, b_val in sorted(target_pairs):
                if (a_row, b_val) not in poly_pairs:
                    continue
                check_budget(deadline=deadline)
                f_poly = vc.poly.with_halfspace([-v for v in a_row], -b_val)
                ok_f, x_rel = _subspace_chebyshev_center(
                    f_poly, [a_row], [b_val], deadline=deadline
                )
                if not ok_f or x_rel is None:
                    continue
                # Check that this facet is genuinely on partial C (not shared identically in opposite sign by another valid cell with h > 0)
                raw_facets.append((vc, a_row, b_val, f_poly))

        # Step 2a: Explicit Bouligand Corner & Crease Intersection Audit!
        # Check pairwise intersections of boundary facets AND intersections of boundary facets with cell creases!
        corners_checked = 0
        tested_points: List[QVec] = []

        def already_tested(pt: QVec) -> bool:
            return any(qvec_eq(pt, ex) for ex in tested_points)

        # Gather corner candidate faces:
        # (i) Pairwise intersections of boundary facets (both within same valid cell like Trap 4, and across adjacent cells like Trap 2!)
        for idx1 in range(len(raw_facets)):
            vc1, a1, b1, fp1 = raw_facets[idx1]
            # Also intersect fp1 with every other non-domain constraint of vc1.poly (catches ReLU crease corners like Trap 2!)
            for i_other in range(len(vc1.poly.A)):
                ao = vc1.poly.A[i_other]
                bo = vc1.poly.b[i_other]
                if (ao, bo) == (a1, b1) or ((ao, bo) in dom_constraint_set and not enforce_domain_invariance):
                    continue
                crease_poly = fp1.with_halfspace([-v for v in ao], -bo)
                ok_c, xc_pt = _subspace_chebyshev_center(
                    crease_poly, [a1, ao], [b1, bo], deadline=deadline
                )
                if ok_c and xc_pt is not None and is_interior_to_domain(xc_pt) and not already_tested(xc_pt):
                    tested_points.append(xc_pt)
                    corners_checked += 1
                    viable, d_certs = _pointwise_bouligand_check(
                        xc_pt, valid_cells, plant.u_poly, plant.control_dim, deadline=deadline
                    )
                    if not viable:
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            corners_checked=corners_checked,
                            init_certs=tuple(init_certs),
                            init_polys=tuple(init_polys),
                            unsafe_certs=tuple(unsafe_certs),
                            unsafe_polys=tuple(unsafe_polys),
                            counterexample=BarrierCounterexample(
                                reason="Bouligand crease/corner escape or actuator deadlock",
                                x_star=xc_pt,
                                h_at_x=eval_h_any(xc_pt),
                                deadlock_subcerts=tuple(d_certs),
                            ),
                            message=f"Corner deadlock / escape at x*={xc_pt}",
                        )

            for idx2 in range(idx1 + 1, len(raw_facets)):
                check_budget(deadline=deadline)
                vc2, a2, b2, fp2 = raw_facets[idx2]
                c1_a, c1_b, _ = canonical_hyperplane(a1, b1)
                c2_a, c2_b, _ = canonical_hyperplane(a2, b2)
                if c1_a == c2_a and c1_b == c2_b:
                    continue
                inter_f = fp1.intersect(fp2)
                ok_inter, x_inter = _subspace_chebyshev_center(
                    inter_f, [c1_a, c2_a], [c1_b, c2_b], deadline=deadline
                )
                if not ok_inter or x_inter is None:
                    continue
                if is_interior_to_domain(x_inter) and not already_tested(x_inter):
                    tested_points.append(x_inter)
                    corners_checked += 1
                    viable, d_certs = _pointwise_bouligand_check(
                        x_inter, valid_cells, plant.u_poly, plant.control_dim, deadline=deadline
                    )
                    if not viable:
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            corners_checked=corners_checked,
                            init_certs=tuple(init_certs),
                            init_polys=tuple(init_polys),
                            unsafe_certs=tuple(unsafe_certs),
                            unsafe_polys=tuple(unsafe_polys),
                            counterexample=BarrierCounterexample(
                                reason="Multi-facet corner deadlock under actuator saturation",
                                x_star=x_inter,
                                h_at_x=eval_h_any(x_inter),
                                deadlock_subcerts=tuple(d_certs),
                            ),
                            message=f"Multi-facet corner deadlock at x*={x_inter}",
                        )

        # Step 2b: Certify full (n-1)-dimensional boundary facets constructively (with adaptive splitting if needed)
        for vc, a_row, b_val, fp_init in raw_facets:
            check_budget(deadline=deadline)
            # Check if fp_init is actually an interior glue face where an adjacent valid cell lies on the -a_row side!
            # On true partial C, h(x) == 0 and on the outside h(x) < 0, so no valid cell lies on {-a_row^T x < -b_val}.
            ok_rel, x_mid = _subspace_chebyshev_center(
                fp_init, [a_row], [b_val], deadline=deadline
            )
            if not ok_rel or x_mid is None:
                continue

            # Queue of sub-facets for constructive affine-feedback Farkas certification
            queue: List[HPolytope] = [fp_init]
            splits_done = 0
            while queue:
                check_budget(deadline=deadline)
                cur_f = queue.pop(0)
                ok_cf, x_cf = _subspace_chebyshev_center(
                    cur_f, [a_row], [b_val], deadline=deadline
                )
                if not ok_cf or x_cf is None:
                    continue

                # Pointwise viability test at center first
                if is_interior_to_domain(x_cf):
                    viable_mid, d_certs_mid = _pointwise_bouligand_check(
                        x_cf, valid_cells, plant.u_poly, plant.control_dim, deadline=deadline
                    )
                    if not viable_mid:
                        return BarrierVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            faces_certified=len(viability_certs),
                            corners_checked=corners_checked,
                            init_certs=tuple(init_certs),
                            init_polys=tuple(init_polys),
                            unsafe_certs=tuple(unsafe_certs),
                            unsafe_polys=tuple(unsafe_polys),
                            viability_certs=tuple(viability_certs),
                            counterexample=BarrierCounterexample(
                                reason="Boundary facet escape / actuator saturation violation",
                                x_star=x_cf,
                                h_at_x=eval_h_any(x_cf),
                                deadlock_subcerts=tuple(d_certs_mid),
                            ),
                            message=f"Boundary facet escape at x*={x_cf}",
                        )

                # Try synthesizing affine feedback + Farkas certificate on cur_f
                f_cert = _try_certify_face_affine_u(
                    face_id=len(viability_certs),
                    face_poly=cur_f,
                    vc=vc,
                    M_tan=[a_row],
                    u_poly=plant.u_poly,
                    control_dim=plant.control_dim,
                    deadline=deadline,
                )
                if f_cert is not None:
                    viability_certs.append(f_cert)
                    continue

                # Probe extreme coordinate points of cur_f for potential counterexample before splitting!
                for j_dir in range(n):
                    for sgn in (1, -1):
                        rp = solve_lp_bareiss(
                            cur_f.A,
                            cur_f.b,
                            c=[sgn if q == j_dir else 0 for q in range(n)],
                            deadline=deadline,
                        )
                        if rp.status == LPStatus.OPTIMAL and rp.x_opt is not None:
                            if is_interior_to_domain(rp.x_opt):
                                ok_v, d_v = _pointwise_bouligand_check(
                                    rp.x_opt,
                                    valid_cells,
                                    plant.u_poly,
                                    plant.control_dim,
                                    deadline=deadline,
                                )
                                if not ok_v:
                                    return BarrierVerificationResult(
                                        verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                                        faces_certified=len(viability_certs),
                                        corners_checked=corners_checked + 1,
                                        init_certs=tuple(init_certs),
                                        init_polys=tuple(init_polys),
                                        unsafe_certs=tuple(unsafe_certs),
                                        unsafe_polys=tuple(unsafe_polys),
                                        viability_certs=tuple(viability_certs),
                                        counterexample=BarrierCounterexample(
                                            reason="Boundary extreme escape / deadlock",
                                            x_star=rp.x_opt,
                                            h_at_x=eval_h_any(rp.x_opt),
                                            deadlock_subcerts=tuple(d_v),
                                        ),
                                        message=f"Boundary escape at extreme {rp.x_opt}",
                                    )

                if splits_done >= max_face_splits:
                    raise BudgetExceededError(
                        f"Boundary facet split budget {max_face_splits} exceeded"
                    )

                # Bisect cur_f along its widest coordinate axis through x_cf
                bbox = cur_f.bounding_box(deadline=deadline)
                if bbox is None:
                    raise BudgetExceededError("Unbounded boundary facet in bisection")
                best_axis = 0
                best_span = (0, 1)
                for j_ax in range(n):
                    lo_j, hi_j = bbox[j_ax]
                    span_j = q_add(hi_j, (-lo_j[0], lo_j[1]))
                    if not q_le(span_j, best_span):
                        best_span = span_j
                        best_axis = j_ax
                if best_span[0] <= 0:
                    raise BudgetExceededError("Zero-span facet failed certification")

                # Split at x_{best_axis} <= x_cf[best_axis] vs >= x_cf[best_axis]
                mid_n = x_cf[0][best_axis]
                mid_d = x_cf[1]
                h_le = [mid_d if q == best_axis else 0 for q in range(n)]
                h_ge = [-mid_d if q == best_axis else 0 for q in range(n)]
                queue.append(cur_f.with_halfspace(h_le, mid_n))
                queue.append(cur_f.with_halfspace(h_ge, -mid_n))
                splits_done += 1

        res_final = BarrierVerificationResult(
            verdict=Verdict.CERTIFIED_EXACT,
            faces_certified=len(viability_certs),
            corners_checked=corners_checked,
            init_certs=tuple(init_certs),
            init_polys=tuple(init_polys),
            unsafe_certs=tuple(unsafe_certs),
            unsafe_polys=tuple(unsafe_polys),
            viability_certs=tuple(viability_certs),
            message=(
                f"Certified NCBF across {len(viability_certs)} boundary faces and "
                f"{corners_checked} corner/crease intersections"
            ),
        )
        assert check_barrier_certificate_bundle(res_final)
        return res_final

    except BudgetExceededError as exc:
        return BarrierVerificationResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            message=str(exc),
        )
