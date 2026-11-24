from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import (
    FarkasBoundCert,
    LPStatus,
    certify_linear_upper_bound,
    check_positivity_farkas,
    check_primal_witness,
    solve_lp_bareiss,
)
from .filippov_crease import (
    FilippovAuditResult,
    FilippovSlidingWitness,
    audit_filippov_creases,
    check_filippov_crease_cert,
)
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
    compose_pullback_pwa,
    ct_bab_planet_upper_bound,
)
from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    Verdict,
    check_budget,
    lcm_list,
    lcm_pair,
    q_add,
    q_cmp,
    q_div,
    q_le,
    q_lt,
    q_min,
    q_mul,
    q_norm,
    q_sub,
    qvec_eq,
    qvec_l1,
    qvec_norm,
    qvec_sub,
)


@dataclasses.dataclass(frozen=True)
class SectorPWACell:
    """Polyhedral cell carrying one or more affine plant generators (convex differential/difference inclusion)."""

    cell_id: int
    polytope: HPolytope
    generators: Tuple[AffineMapQ, ...]
    tag: str = ""


@dataclasses.dataclass(frozen=True)
class SectorPWAPlant:
    dim: int
    domain: HPolytope
    cells: Tuple[SectorPWACell, ...]
    is_discrete: bool = True
    name: str = ""

    @staticmethod
    def from_pwa(pwa: PWAFunction, is_discrete: bool = True) -> SectorPWAPlant:
        return SectorPWAPlant(
            dim=pwa.in_dim,
            domain=pwa.domain,
            cells=tuple(
                SectorPWACell(
                    cell_id=c.cell_id,
                    polytope=c.polytope,
                    generators=(c.affine_map,),
                    tag=c.tag,
                )
                for c in pwa.cells
            ),
            is_discrete=is_discrete,
            name=pwa.name,
        )


@dataclasses.dataclass(frozen=True)
class CoercivityCellCert:
    """Zero-hole homogeneous coercivity certificate V_i(x) >= alpha_1 * ||x - x_eq||_1 on cell C_i
    backed by 2n integer Farkas duals (each proving sigma * (x_j - x_eq_j) <= mu_j * V_i(x)).
    """

    cell_id: int
    cell_A: Tuple[Tuple[int, ...], ...]
    cell_b: Tuple[int, ...]
    v_map: AffineMapQ
    mu_bounds: Tuple[Rat, ...]
    alpha_cell: Rat
    bound_certs: Tuple[FarkasBoundCert, ...]
    strict_pos_cert: Optional[FarkasBoundCert] = None


@dataclasses.dataclass(frozen=True)
class ShardDecayCert:
    """Constructive integer Farkas certificate for Lyapunov decay on a shard (or CT-BaB lifted cell)."""

    shard_id: int
    poly_A: Tuple[Tuple[int, ...], ...]
    poly_b: Tuple[int, ...]
    bound_cert: FarkasBoundCert
    tag: str = ""
    is_ct_bab_lifted: bool = False


@dataclasses.dataclass(frozen=True)
class BoundaryLevelCert:
    """Certificate proving V_i(x) >= rho_star on domain boundary facet C_i cap {a_D^T x == b_D}."""

    cell_id: int
    boundary_idx: int
    facet_A: Tuple[Tuple[int, ...], ...]
    facet_b: Tuple[int, ...]
    bound_cert: FarkasBoundCert


@dataclasses.dataclass(frozen=True)
class LyapunovCounterexample:
    reason: str
    x_witness: QVec
    v_at_x: Rat
    v_next_or_deriv: Rat
    residual_violation: Rat


@dataclasses.dataclass(frozen=True)
class LyapunovVerificationResult:
    verdict: Verdict
    is_discrete: bool
    alpha_1: Optional[Rat] = None
    epsilon: Optional[Rat] = None
    rho_star: Optional[Rat] = None
    shards_certified: int = 0
    ct_bab_pruned_cells: int = 0
    coercivity_certs: Tuple[CoercivityCellCert, ...] = ()
    decay_certs: Tuple[ShardDecayCert, ...] = ()
    boundary_certs: Tuple[BoundaryLevelCert, ...] = ()
    filippov_audit: Optional[FilippovAuditResult] = None
    counterexample: Optional[LyapunovCounterexample] = None
    filippov_witness: Optional[FilippovSlidingWitness] = None
    message: str = ""


def check_coercivity_cell_cert(
    cert: CoercivityCellCert, x_eq: QVec, alpha_target: Rat
) -> bool:
    """Zero-trust O(n * m * n) pure-integer checker for 2n-Farkas coercivity certificate."""
    n = cert.v_map.in_dim
    if len(cert.mu_bounds) != n or len(cert.bound_certs) != 2 * n:
        return False
    if alpha_target[0] <= 0 or alpha_target[1] <= 0:
        return False

    mu_sum = (0, 1)
    for mu in cert.mu_bounds:
        if mu[1] <= 0 or mu[0] < 0:
            return False
        mu_sum = q_add(mu_sum, mu)

    # Verify alpha_target * sum(mu_j) <= 1 (or mu_sum == 0 when cell == {x_eq})
    prod = q_mul(alpha_target, mu_sum)
    if not q_le(prod, (1, 1)):
        return False

    eq_n, eq_d = x_eq
    g_V = cert.v_map.W[0]
    c_V = cert.v_map.c[0]
    d_V = cert.v_map.denom

    idx = 0
    for j in range(n):
        mu_n, mu_d = cert.mu_bounds[j]
        # Expected inequality: sigma * (x_j - eq_n[j]/eq_d) - (mu_n/mu_d) * ((g_V^T x + c_V)/d_V) <= 0
        # Let common_c_den = mu_d * d_V:
        # c_vec = [ (sigma if q == j else 0) * mu_d * d_V - mu_n * g_V[q] ] / (mu_d * d_V)
        # rhs = sigma * eq_n[j] / eq_d + (mu_n * c_V) / (mu_d * d_V)
        c_den_exp = mu_d * d_V
        for sgn in (1, -1):
            bc = cert.bound_certs[idx]
            idx += 1
            c_num_exp = [
                (sgn * c_den_exp if q == j else 0) - mu_n * g_V[q]
                for q in range(n)
            ]
            rhs_q = q_add(
                q_norm(sgn * eq_n[j], eq_d),
                q_norm(mu_n * c_V, c_den_exp),
            )
            # Check proportional equality of linear form (bc.c_num / bc.c_den) == (c_num_exp / c_den_exp)
            for q in range(n):
                if bc.c_num[q] * c_den_exp != c_num_exp[q] * bc.c_den:
                    return False
            if bc.rhs_num * rhs_q[1] != rhs_q[0] * bc.rhs_den:
                return False
            if not check_positivity_farkas(cert.cell_A, cert.cell_b, bc):
                return False
    if cert.strict_pos_cert is not None:
        spc = cert.strict_pos_cert
        for q in range(n):
            if spc.c_num[q] * d_V != (-g_V[q]) * spc.c_den:
                return False
        if spc.rhs_num * d_V != c_V * spc.rhs_den:
            return False
        if not check_positivity_farkas(cert.cell_A, cert.cell_b, spc, strict=True):
            return False
    return True


def check_lyapunov_certificate_bundle(
    result: LyapunovVerificationResult, x_eq: Optional[QVec] = None
) -> bool:
    """Standalone zero-trust pure-integer verifier for a CERTIFIED_EXACT LyapunovVerificationResult."""
    if result.verdict != Verdict.CERTIFIED_EXACT:
        return False
    if result.alpha_1 is None or result.alpha_1[0] <= 0:
        return False
    if result.epsilon is None or result.epsilon[0] <= 0:
        return False
    if not result.coercivity_certs or not result.decay_certs:
        return False

    n = result.coercivity_certs[0].v_map.in_dim
    if x_eq is None:
        x_eq = (tuple(0 for _ in range(n)), 1)

    for cc in result.coercivity_certs:
        if not check_coercivity_cell_cert(cc, x_eq, result.alpha_1):
            return False

    for dc in result.decay_certs:
        if not check_positivity_farkas(dc.poly_A, dc.poly_b, dc.bound_cert):
            return False

    for bc in result.boundary_certs:
        if not check_positivity_farkas(bc.facet_A, bc.facet_b, bc.bound_cert):
            return False

    if result.filippov_audit is not None:
        if result.filippov_audit.verdict != Verdict.CERTIFIED_EXACT:
            return False
        for fc in result.filippov_audit.certificates:
            if not check_filippov_crease_cert(fc, x_eq=x_eq):
                return False

    return True


def _certify_cell_coercivity(
    cell: PWACell,
    x_eq: QVec,
    deadline: Optional[float] = None,
) -> Tuple[Optional[CoercivityCellCert], Optional[QVec]]:
    """Compute exact tightest 2n coordinate coercivity ratios mu_j via Charnes-Cooper LP
    and emit 2n integer Farkas duals on cell.polytope.
    """
    poly = cell.polytope
    n = poly.dim
    v_map = cell.affine_map
    g_V = v_map.W[0]
    c_V = v_map.c[0]
    d_V = v_map.denom
    eq_n, eq_d = x_eq

    # First check V_i(x) >= 0 on cell
    neg_g = [-v for v in g_V]
    res_min_v = solve_lp_bareiss(poly.A, poly.b, c=neg_g, deadline=deadline)
    if res_min_v.status == LPStatus.BUDGET_EXCEEDED:
        raise BudgetExceededError("LP budget exceeded in coercivity positivity check")
    if res_min_v.status == LPStatus.OPTIMAL and res_min_v.obj_val is not None:
        # min V_i = (-obj_val / d_V) + (c_V / d_V)
        min_v_q = q_add(
            q_norm(-res_min_v.obj_val[0], res_min_v.obj_val[1] * d_V),
            q_norm(c_V, d_V),
        )
        if min_v_q[0] < 0 and res_min_v.x_opt is not None:
            return (None, res_min_v.x_opt)

    # Charnes-Cooper LP in (y_0 .. y_{n-1}, t >= 0) where y = t * (x - x_eq):
    # 1) eq_d * (A_i y) - (eq_d * b_i - A_i eq_n) * t <= 0
    # 2) eq_d * (g_V^T y) + (g_V^T eq_n + c_V * eq_d) * t <= d_V * eq_d
    # 3) -t <= 0
    v_eq_num = sum(g_V[q] * eq_n[q] for q in range(n)) + c_V * eq_d
    cc_A: List[List[int]] = []
    cc_b: List[int] = []
    for i in range(len(poly.A)):
        ar = poly.A[i]
        rhs_shift = eq_d * poly.b[i] - sum(ar[q] * eq_n[q] for q in range(n))
        cc_A.append([eq_d * ar[q] for q in range(n)] + [-rhs_shift])
        cc_b.append(0)
    cc_A.append([eq_d * g_V[q] for q in range(n)] + [v_eq_num])
    cc_b.append(d_V * eq_d)
    cc_A.append([0] * n + [-1])
    cc_b.append(0)

    mu_bounds: List[Rat] = []
    for j in range(n):
        mu_j = (0, 1)
        for sgn in (1, -1):
            c_obj = [sgn if q == j else 0 for q in range(n)] + [0]
            res_cc = solve_lp_bareiss(cc_A, cc_b, c=c_obj, deadline=deadline)
            if res_cc.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in Charnes-Cooper coercivity")
            if res_cc.status == LPStatus.UNBOUNDED:
                # Non-coercive ray where V_i(x) == 0 away from x_eq!
                # Extract witness in cell maximizing sgn * (x_j - x_eq_j) s.t. V_i(x) <= 0
                zero_v_poly = poly.with_halfspace(g_V, -c_V)
                rw = solve_lp_bareiss(
                    zero_v_poly.A,
                    zero_v_poly.b,
                    c=[sgn if q == j else 0 for q in range(n)],
                    deadline=deadline,
                )
                if rw.status == LPStatus.OPTIMAL and rw.x_opt is not None and not qvec_eq(rw.x_opt, x_eq):
                    return (None, rw.x_opt)
                return (None, res_min_v.x_opt)
            if res_cc.status == LPStatus.OPTIMAL and res_cc.obj_val is not None:
                if q_lt(mu_j, res_cc.obj_val):
                    mu_j = res_cc.obj_val
        mu_bounds.append(mu_j)

    # Emit exact 2n FarkasBoundCerts directly on (poly.A, poly.b)
    bound_certs: List[FarkasBoundCert] = []
    for j in range(n):
        mu_n, mu_d = mu_bounds[j]
        c_den_exp = mu_d * d_V
        for sgn in (1, -1):
            c_num_exp = [
                (sgn * c_den_exp if q == j else 0) - mu_n * g_V[q]
                for q in range(n)
            ]
            rhs_q = q_add(
                q_norm(sgn * eq_n[j], eq_d),
                q_norm(mu_n * c_V, c_den_exp),
            )
            v_str, b_cert, wit_cex, _ = certify_linear_upper_bound(
                poly.A,
                poly.b,
                c_num_exp,
                c_den_exp,
                rhs_q[0],
                rhs_q[1],
                deadline=deadline,
            )
            if v_str == "OPEN_BUDGET_EXCEEDED":
                raise BudgetExceededError("LP budget exceeded in coercivity Farkas emission")
            if v_str != "CERTIFIED_EXACT" or b_cert is None:
                return (None, wit_cex)
            bound_certs.append(b_cert)

    mu_sum = (0, 1)
    for mu in mu_bounds:
        mu_sum = q_add(mu_sum, mu)
    alpha_cell = (1, 1) if mu_sum[0] == 0 else q_div((1, 1), mu_sum)

    strict_pos_cert: Optional[FarkasBoundCert] = None
    if not poly.contains(x_eq):
        v_sp, sp_cert, wit_sp, _ = certify_linear_upper_bound(
            poly.A,
            poly.b,
            [-v for v in g_V],
            d_V,
            c_V,
            d_V,
            strict=True,
            deadline=deadline,
        )
        if v_sp == "OPEN_BUDGET_EXCEEDED":
            raise BudgetExceededError("LP budget exceeded in strict off-origin positivity")
        if v_sp != "CERTIFIED_EXACT" or sp_cert is None:
            return (None, wit_sp)
        strict_pos_cert = sp_cert

    cert = CoercivityCellCert(
        cell_id=cell.cell_id,
        cell_A=poly.A,
        cell_b=poly.b,
        v_map=v_map,
        mu_bounds=tuple(mu_bounds),
        alpha_cell=alpha_cell,
        bound_certs=tuple(bound_certs),
        strict_pos_cert=strict_pos_cert,
    )
    return (cert, None)


def build_dai_l1_lyapunov(
    phi_net: ReLUNetwork,
    R_map: Optional[AffineMapQ] = None,
    x_eq: Optional[QVec] = None,
    alpha_reg: Rat = (1, 1),
    name: str = "dai_l1_v",
) -> ReLUNetwork:
    """Construct Dai et al. l_1-regularized anchored ReLU Lyapunov candidate:
    V(x) = phi(x) - phi(x*) + alpha_reg * ||R(x - x*)||_1.
    """
    if phi_net.out_dim != 1:
        raise ValueError("phi_net must be scalar (out_dim == 1)")
    n = phi_net.in_dim
    if x_eq is None:
        x_eq = (tuple(0 for _ in range(n)), 1)
    an_reg, ad_reg = q_norm(alpha_reg[0], alpha_reg[1])
    if an_reg <= 0:
        return phi_net.with_anchor_zero(x_eq)
    if R_map is None:
        R_map = AffineMapQ.identity(n)
    if R_map.in_dim != n:
        raise ValueError("R_map.in_dim must match phi_net.in_dim")

    p = R_map.out_dim
    eq_n, eq_d = x_eq
    # Build 2p rows for +- R_map * (x - x_eq):
    # r_k^T (x - eq_n/eq_d) / R_denom = ((eq_d * r_k)^T x - (r_k^T eq_n)) / (eq_d * R_denom)
    reg_W: List[List[int]] = []
    reg_c: List[int] = []
    reg_den = eq_d * R_map.denom
    for k in range(p):
        rk = R_map.W[k]
        dot_eq = sum(rk[j] * eq_n[j] for j in range(n))
        reg_W.append([eq_d * rk[j] for j in range(n)])
        reg_c.append(-dot_eq)
        reg_W.append([-eq_d * rk[j] for j in range(n)])
        reg_c.append(dot_eq)

    base_layers = list(phi_net.layers)
    if not base_layers:
        # Pure residual phi_net -> 1-hidden-layer ReLU for ||R(x - x*)||_1
        l1_layer = ReLULayer(AffineMapQ.create(n, 2 * p, reg_W, reg_c, reg_den), use_relu=True)
        out_layer = ReLULayer(
            AffineMapQ.create(2 * p, 1, [[an_reg] * (2 * p)], [0], ad_reg),
            use_relu=False,
        )
        comb = ReLUNetwork(
            in_dim=n,
            out_dim=1,
            layers=(l1_layer, out_layer),
            residual=phi_net.residual,
            name=name,
        )
        return comb.with_anchor_zero(x_eq)

    # Ensure first layer is use_relu=True or prepend/augment cleanly; if last layer is use_relu=True, append readout
    if base_layers[-1].use_relu:
        base_layers.append(
            ReLULayer(AffineMapQ.identity(base_layers[-1].affine.out_dim), use_relu=False)
        )

    if len(base_layers) == 1:
        # Single linear layer -> convert to 1-hidden-layer ReLU for reg + residual for linear phi
        comb_res = base_layers[0].affine
        if phi_net.residual is not None:
            comb_res = comb_res.add(phi_net.residual)
        l1_layer = ReLULayer(AffineMapQ.create(n, 2 * p, reg_W, reg_c, reg_den), use_relu=True)
        out_layer = ReLULayer(
            AffineMapQ.create(2 * p, 1, [[an_reg] * (2 * p)], [0], ad_reg),
            use_relu=False,
        )
        return ReLUNetwork(
            in_dim=n,
            out_dim=1,
            layers=(l1_layer, out_layer),
            residual=comb_res,
            name=name,
        ).with_anchor_zero(x_eq)

    # Multi-layer case (len >= 2): augment layer 0 with 2p ReLU neurons, pass them through intermediate layers,
    # and sum with weight alpha_reg at the final linear readout layer!
    new_layers: List[ReLULayer] = []
    l0 = base_layers[0]
    cd0 = lcm_pair(l0.affine.denom, reg_den)
    s0 = cd0 // l0.affine.denom
    sr = cd0 // reg_den
    if l0.use_relu:
        w0 = [[s0 * v for v in row] for row in l0.affine.W] + [
            [sr * v for v in row] for row in reg_W
        ]
        c0 = [s0 * v for v in l0.affine.c] + [sr * v for v in reg_c]
        new_layers.append(
            ReLULayer(
                AffineMapQ.create(n, l0.affine.out_dim + 2 * p, w0, c0, cd0),
                use_relu=True,
            )
        )
        extra_nonneg = True
    else:
        # Pass raw x through or pass 2p unactivated values and activate at first ReLU layer
        w0 = [[s0 * v for v in row] for row in l0.affine.W] + [
            [sr * v for v in row] for row in reg_W
        ]
        c0 = [s0 * v for v in l0.affine.c] + [sr * v for v in reg_c]
        new_layers.append(
            ReLULayer(
                AffineMapQ.create(n, l0.affine.out_dim + 2 * p, w0, c0, cd0),
                use_relu=False,
            )
        )
        extra_nonneg = False

    for mid_layer in base_layers[1:-1]:
        m_in = mid_layer.affine.in_dim
        m_out = mid_layer.affine.out_dim
        d_m = mid_layer.affine.denom
        w_mid = [list(row) + [0] * (2 * p) for row in mid_layer.affine.W]
        for k_idx in range(2 * p):
            pass_row = [0] * (m_in + 2 * p)
            pass_row[m_in + k_idx] = d_m
            w_mid.append(pass_row)
        c_mid = list(mid_layer.affine.c) + [0] * (2 * p)
        new_layers.append(
            ReLULayer(
                AffineMapQ.create(m_in + 2 * p, m_out + 2 * p, w_mid, c_mid, d_m),
                use_relu=mid_layer.use_relu,
            )
        )
        if mid_layer.use_relu:
            extra_nonneg = True

    if not extra_nonneg:
        # Insert a ReLU activation on the 2p channels if no hidden layer had use_relu=True
        prev_out = new_layers[-1].affine.out_dim
        base_out = prev_out - 2 * p
        # Compose linear phi into residual instead
        return build_dai_l1_lyapunov(
            ReLUNetwork(in_dim=n, out_dim=1, layers=(), residual=phi_net.enumerate_pwa(HPolytope.symmetric_box(n, (1, 1))).cells[0].affine_map),
            R_map=R_map,
            x_eq=x_eq,
            alpha_reg=alpha_reg,
            name=name,
        )

    last_l = base_layers[-1]
    cd_last = lcm_pair(last_l.affine.denom, ad_reg)
    sl = cd_last // last_l.affine.denom
    sa = cd_last // ad_reg
    w_last = [
        [sl * v for v in last_l.affine.W[0]] + [sa * an_reg] * (2 * p)
    ]
    c_last = [sl * last_l.affine.c[0]]
    new_layers.append(
        ReLULayer(
            AffineMapQ.create(last_l.affine.in_dim + 2 * p, 1, w_last, c_last, cd_last),
            use_relu=False,
        )
    )
    return ReLUNetwork(
        in_dim=n,
        out_dim=1,
        layers=tuple(new_layers),
        residual=phi_net.residual,
        name=name,
    ).with_anchor_zero(x_eq)


def _compute_domain_boundary_rho(
    v_pwa: PWAFunction,
    domain: HPolytope,
    deadline: Optional[float] = None,
) -> Optional[Rat]:
    """Compute exact rational rho_dom = min_{x in partial D} V(x)."""
    rho_dom: Optional[Rat] = None
    for b_idx in range(len(domain.A)):
        a_b = domain.A[b_idx]
        rhs_b = domain.b[b_idx]
        neg_a = [-v for v in a_b]
        for cell in v_pwa.cells:
            check_budget(deadline=deadline)
            facet = cell.polytope.with_halfspace(a_b, rhs_b).with_halfspace(
                neg_a, -rhs_b
            )
            vm = cell.affine_map
            res = solve_lp_bareiss(
                facet.A, facet.b, c=[-v for v in vm.W[0]], deadline=deadline
            )
            if res.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in boundary rho")
            if res.status == LPStatus.OPTIMAL and res.obj_val is not None:
                min_v = q_add(
                    q_norm(-res.obj_val[0], res.obj_val[1] * vm.denom),
                    q_norm(vm.c[0], vm.denom),
                )
                if rho_dom is None or q_lt(min_v, rho_dom):
                    rho_dom = min_v
    return rho_dom


def _certify_domain_boundary_rho(
    v_pwa: PWAFunction,
    domain: HPolytope,
    rho_star: Rat,
    deadline: Optional[float] = None,
) -> List[BoundaryLevelCert]:
    """Emit Farkas certificates proving -V_i(x) <= -rho_star on every boundary facet of domain."""
    certs: List[BoundaryLevelCert] = []
    rn, rd = q_norm(rho_star[0], rho_star[1])
    for b_idx in range(len(domain.A)):
        a_b = domain.A[b_idx]
        rhs_b = domain.b[b_idx]
        neg_a = [-v for v in a_b]
        for cell in v_pwa.cells:
            facet = cell.polytope.with_halfspace(a_b, rhs_b).with_halfspace(
                neg_a, -rhs_b
            )
            vm = cell.affine_map
            # Prove (-W_0 / d_V) x <= c_0 / d_V - rn / rd
            rhs_q = q_sub(q_norm(vm.c[0], vm.denom), (rn, rd))
            v_str, b_cert, _, _ = certify_linear_upper_bound(
                facet.A,
                facet.b,
                [-v for v in vm.W[0]],
                vm.denom,
                rhs_q[0],
                rhs_q[1],
                deadline=deadline,
            )
            if v_str == "OPEN_BUDGET_EXCEEDED":
                raise BudgetExceededError("LP budget exceeded in boundary cert")
            if v_str != "CERTIFIED_EXACT" or b_cert is None:
                raise RuntimeError("Boundary level certification failed unexpectedly")
            certs.append(
                BoundaryLevelCert(
                    cell_id=cell.cell_id,
                    boundary_idx=b_idx,
                    facet_A=facet.A,
                    facet_b=facet.b,
                    bound_cert=b_cert,
                )
            )
    return certs


def verify_discrete_lyapunov(
    v_func: Union[PWAFunction, ReLUNetwork],
    plant: Union[PWAFunction, ReLUNetwork, SectorPWAPlant],
    domain: HPolytope,
    x_eq: Optional[QVec] = None,
    epsilon: Rat = (1, 20),
    compute_maximal_roa: bool = False,
    use_ct_bab: bool = False,
    max_cells: int = 2048,
    deadline: Optional[float] = None,
) -> LyapunovVerificationResult:
    """Spec 2a: Exact zero-hole discrete-time Lyapunov & maximal sublevel ROA verifier over Z/Q.

    Certifies V(x) >= alpha_1 ||x - x*||_1 (via 2n Farkas duals) AND
    V(F(x)) - (1 - epsilon) V(x) <= 0 across all pullback/sector shards.
    """
    try:
        n = domain.dim
        if x_eq is None:
            x_eq = (tuple(0 for _ in range(n)), 1)
        eps_q = q_norm(epsilon[0], epsilon[1])
        one_minus_eps = q_sub((1, 1), eps_q)

        # Enumerate V as PWA on domain
        v_pwa = (
            v_func.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline)
            if isinstance(v_func, ReLUNetwork)
            else v_func
        )
        v_eq = v_pwa.eval_scalar(x_eq)
        if v_eq != (0, 1):
            return LyapunovVerificationResult(
                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                is_discrete=True,
                counterexample=LyapunovCounterexample(
                    reason="V(x_eq) != 0 (missing equilibrium anchor)",
                    x_witness=x_eq,
                    v_at_x=v_eq,
                    v_next_or_deriv=v_eq,
                    residual_violation=q_norm(abs(v_eq[0]), v_eq[1]),
                ),
                message=f"V(x_eq) = {v_eq} != 0",
            )

        # Normalize plant into SectorPWAPlant
        if isinstance(plant, ReLUNetwork):
            f_eq = plant.eval_qvec(x_eq)
            if not qvec_eq(f_eq, x_eq):
                v_next = v_func.eval_scalar(f_eq) if domain.contains(f_eq) else (1, 1)
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=True,
                    counterexample=LyapunovCounterexample(
                        reason="F(x_eq) != x_eq (post-rounding equilibrium shift)",
                        x_witness=x_eq,
                        v_at_x=(0, 1),
                        v_next_or_deriv=v_next,
                        residual_violation=qvec_l1(qvec_sub(f_eq, x_eq)),
                    ),
                    message=f"Equilibrium shift: F(x_eq) = {f_eq} != {x_eq}",
                )
            sec_plant = SectorPWAPlant.from_pwa(
                plant.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline),
                is_discrete=True,
            )
        elif isinstance(plant, PWAFunction):
            f_eq = plant.eval_qvec(x_eq)
            if not qvec_eq(f_eq, x_eq):
                v_next = v_pwa.eval_scalar(f_eq) if domain.contains(f_eq) else (1, 1)
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=True,
                    counterexample=LyapunovCounterexample(
                        reason="F(x_eq) != x_eq (equilibrium mismatch)",
                        x_witness=x_eq,
                        v_at_x=(0, 1),
                        v_next_or_deriv=v_next,
                        residual_violation=qvec_l1(qvec_sub(f_eq, x_eq)),
                    ),
                    message=f"Equilibrium mismatch: F(x_eq) = {f_eq} != {x_eq}",
                )
            sec_plant = SectorPWAPlant.from_pwa(plant, is_discrete=True)
        else:
            sec_plant = plant
            for sc in sec_plant.cells:
                if sc.polytope.contains(x_eq):
                    for gen in sc.generators:
                        f_eq = gen.eval_qvec(x_eq)
                        if not qvec_eq(f_eq, x_eq):
                            return LyapunovVerificationResult(
                                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                                is_discrete=True,
                                counterexample=LyapunovCounterexample(
                                    reason="Sector generator F(x_eq) != x_eq",
                                    x_witness=x_eq,
                                    v_at_x=(0, 1),
                                    v_next_or_deriv=(0, 1),
                                    residual_violation=qvec_l1(qvec_sub(f_eq, x_eq)),
                                ),
                                message=f"Sector generator F(x_eq) = {f_eq} != {x_eq}",
                            )

        # Step 1: Certify zero-hole homogeneous coercivity V(x) >= alpha_1 ||x - x_eq||_1 on V cells
        coercivity_certs: List[CoercivityCellCert] = []
        alpha_min: Optional[Rat] = None
        for vc in v_pwa.cells:
            check_budget(deadline=deadline)
            cc, cex_coerc = _certify_cell_coercivity(vc, x_eq, deadline=deadline)
            if cc is None:
                wit = cex_coerc if cex_coerc is not None else x_eq
                vw = vc.affine_map.eval_scalar(wit)
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=True,
                    counterexample=LyapunovCounterexample(
                        reason="Lyapunov coercivity/positivity failed",
                        x_witness=wit,
                        v_at_x=vw,
                        v_next_or_deriv=vw,
                        residual_violation=vw,
                    ),
                    message=f"Coercivity failed on cell {vc.cell_id} at {wit}",
                )
            coercivity_certs.append(cc)
            if alpha_min is None or q_lt(cc.alpha_cell, alpha_min):
                alpha_min = cc.alpha_cell
        assert alpha_min is not None

        # Step 2: Build joint base cells (V_cell cap Plant_cell) with r* > 0
        joint_base: List[Tuple[HPolytope, AffineMapQ, Tuple[AffineMapQ, ...], str]] = []
        for vc in v_pwa.cells:
            for pc in sec_plant.cells:
                check_budget(deadline=deadline)
                inter = vc.polytope.intersect(pc.polytope)
                ok_fd, _, _ = inter.is_full_dimensional(deadline=deadline)
                if ok_fd:
                    joint_base.append(
                        (
                            inter.prune_redundant(deadline=deadline),
                            vc.affine_map,
                            pc.generators,
                            f"v{vc.cell_id}_p{pc.cell_id}",
                        )
                    )
                    if len(joint_base) > max_cells:
                        raise BudgetExceededError("Joint (V, F) cells exceeded max_cells")

        # Step 3: Pull back V(F(x)) across each generator (using CT-BaB Planet pruning on off-origin cells if enabled)
        decay_certs: List[ShardDecayCert] = []
        ct_bab_pruned = 0
        # Store raw pullback shards when compute_maximal_roa=True in case we need to determine rho_star
        pullback_shards: List[Tuple[HPolytope, AffineMapQ, AffineMapQ, str]] = []

        rho_dom = (
            _compute_domain_boundary_rho(v_pwa, domain, deadline=deadline)
            if compute_maximal_roa
            else None
        )
        rho_cap: Optional[Rat] = rho_dom

        for j_poly, v_in_map, gens, j_tag in joint_base:
            sub_map = v_in_map.scale(one_minus_eps)
            contains_origin = j_poly.contains(x_eq)

            for g_idx, f_gen in enumerate(gens):
                check_budget(deadline=deadline)
                # Optional CT-BaB Planet relaxation fast-path on off-origin cells for ReLUNetwork V
                if (
                    use_ct_bab
                    and isinstance(v_func, ReLUNetwork)
                    and not contains_origin
                    and not compute_maximal_roa
                ):
                    # Try proving V(f_gen(x)) - (1 - eps) V_in(x) <= 0 directly via Planet LP!
                    # Note: build lifted LP and extract FarkasBoundCert on lifted polytope
                    planet_res = ct_bab_planet_upper_bound(
                        v_func, f_gen, j_poly, sub_map, deadline=deadline
                    )
                    if planet_res is not None and planet_res[0][0] <= 0:
                        ct_bab_pruned += 1
                        # Also certify on base or record lifted count; for full zero-trust shard certs:
                        # ct_bab_planet_upper_bound returns exact lifted duals; let's enumerate only if Planet > 0!
                        # To keep check_lyapunov_certificate_bundle uniform on ShardDecayCert, reconstruct lifted A,b
                        # OR when Planet <= 0 on small networks, let's build the exact lifted ShardDecayCert!
                        lifted_cert = _build_planet_shard_cert(
                            shard_id=len(decay_certs),
                            v_net=v_func,
                            f_map=f_gen,
                            cell_poly=j_poly,
                            sub_map=sub_map,
                            tag=f"{j_tag}_g{g_idx}_planet",
                            deadline=deadline,
                        )
                        if lifted_cert is not None:
                            decay_certs.append(lifted_cert)
                            continue

                # Exact pullback enumeration on j_poly
                # First ensure f_gen(x) stays inside domain when compute_maximal_roa=False, or pull back v_pwa!
                single_plant_pwa = PWAFunction(
                    in_dim=n,
                    out_dim=n,
                    domain=j_poly,
                    cells=(PWACell(cell_id=0, polytope=j_poly, affine_map=f_gen, tag=j_tag),),
                )
                pb_rep = compose_pullback_pwa(
                    v_func if isinstance(v_func, ReLUNetwork) else v_pwa,
                    single_plant_pwa,
                    merge_bft=True,
                    max_cells=max_cells,
                    deadline=deadline,
                )
                # Note: if compute_maximal_roa=False, also check that f_gen(j_poly) stays in domain
                # if V is only defined on domain (for PWAFunction V, points leaving domain wouldn't hit v_pwa.cells!)
                if not isinstance(v_func, ReLUNetwork) and not compute_maximal_roa:
                    for b_r in range(len(domain.A)):
                        ar = domain.A[b_r]
                        br = domain.b[b_r]
                        # Check ar^T f_gen(x) <= br on j_poly
                        row_f = [
                            sum(ar[k] * f_gen.W[k][col] for k in range(n))
                            for col in range(n)
                        ]
                        rhs_f = f_gen.denom * br - sum(ar[k] * f_gen.c[k] for k in range(n))
                        v_dom, _, wit_out, _ = certify_linear_upper_bound(
                            j_poly.A, j_poly.b, row_f, 1, rhs_f, 1, deadline=deadline
                        )
                        if v_dom == "FALSIFIED_COUNTEREXAMPLE" and wit_out is not None:
                            vw = v_in_map.eval_scalar(wit_out)
                            return LyapunovVerificationResult(
                                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                                is_discrete=True,
                                alpha_1=alpha_min,
                                epsilon=eps_q,
                                counterexample=LyapunovCounterexample(
                                    reason="One-step state exits domain D",
                                    x_witness=wit_out,
                                    v_at_x=vw,
                                    v_next_or_deriv=(1, 1),
                                    residual_violation=(1, 1),
                                ),
                                message=f"Trajectory exits domain from {wit_out}",
                            )

                for sc in pb_rep.pwa.cells:
                    pullback_shards.append(
                        (sc.polytope, v_in_map, sc.affine_map, f"{j_tag}_g{g_idx}_{sc.tag}")
                    )
                    if len(pullback_shards) > max_cells:
                        raise BudgetExceededError("Total pullback shards exceeded max_cells")

        if compute_maximal_roa:
            # Find tightest sublevel rho_star <= rho_dom where every shard has decay <= 0!
            for s_poly, v_in, v_out, s_tag in pullback_shards:
                check_budget(deadline=deadline)
                diff_map = v_out.sub(v_in.scale(one_minus_eps))
                v_str, _, wit_cex, _ = certify_linear_upper_bound(
                    s_poly.A,
                    s_poly.b,
                    diff_map.W[0],
                    diff_map.denom,
                    -diff_map.c[0],
                    diff_map.denom,
                    deadline=deadline,
                )
                if v_str == "OPEN_BUDGET_EXCEEDED":
                    raise BudgetExceededError("LP budget exceeded in ROA sweep")
                if v_str == "CERTIFIED_EXACT":
                    continue
                # Decay violated somewhere on s_poly!
                if s_poly.contains(x_eq):
                    # Origin-incident shard violates decay -> local stability fails!
                    assert wit_cex is not None
                    vx = v_in.eval_scalar(wit_cex)
                    vn = v_out.eval_scalar(wit_cex)
                    return LyapunovVerificationResult(
                        verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                        is_discrete=True,
                        alpha_1=alpha_min,
                        epsilon=eps_q,
                        counterexample=LyapunovCounterexample(
                            reason="Origin-cone discrete Lyapunov decay failed",
                            x_witness=wit_cex,
                            v_at_x=vx,
                            v_next_or_deriv=vn,
                            residual_violation=diff_map.eval_scalar(wit_cex),
                        ),
                        message=f"Origin shard {s_tag} failed decay at {wit_cex}",
                    )
                # Non-origin shard: minimize V_in(x) over {x in s_poly : diff_map(x) >= 0}
                viol_poly = s_poly.with_halfspace(
                    [-v for v in diff_map.W[0]], diff_map.c[0]
                )
                res_min = solve_lp_bareiss(
                    viol_poly.A,
                    viol_poly.b,
                    c=[-v for v in v_in.W[0]],
                    deadline=deadline,
                )
                if res_min.status == LPStatus.BUDGET_EXCEEDED:
                    raise BudgetExceededError("LP budget exceeded in ROA level solve")
                if res_min.status == LPStatus.OPTIMAL and res_min.obj_val is not None:
                    rho_shard = q_add(
                        q_norm(-res_min.obj_val[0], res_min.obj_val[1] * v_in.denom),
                        q_norm(v_in.c[0], v_in.denom),
                    )
                    if rho_cap is None or q_lt(rho_shard, rho_cap):
                        rho_cap = rho_shard

            if rho_cap is None or rho_cap[0] <= 0:
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=True,
                    alpha_1=alpha_min,
                    epsilon=eps_q,
                    message="Maximal certified sublevel ROA radius is <= 0",
                )

            # Re-certify every pullback shard intersected with {V_in(x) <= rho_cap} + boundary certs!
            rn, rd = rho_cap
            for idx_s, (s_poly, v_in, v_out, s_tag) in enumerate(pullback_shards):
                check_budget(deadline=deadline)
                g_row = [rd * v for v in v_in.W[0]]
                rhs_v = v_in.denom * rn - rd * v_in.c[0]
                sub_poly = s_poly.with_halfspace(g_row, rhs_v)
                diff_map = v_out.sub(v_in.scale(one_minus_eps))
                v_str, b_cert, _, _ = certify_linear_upper_bound(
                    sub_poly.A,
                    sub_poly.b,
                    diff_map.W[0],
                    diff_map.denom,
                    -diff_map.c[0],
                    diff_map.denom,
                    deadline=deadline,
                )
                if v_str != "CERTIFIED_EXACT" or b_cert is None:
                    raise BudgetExceededError("Failed re-certifying sublevel shard")
                decay_certs.append(
                    ShardDecayCert(
                        shard_id=idx_s,
                        poly_A=sub_poly.A,
                        poly_b=sub_poly.b,
                        bound_cert=b_cert,
                        tag=s_tag,
                    )
                )
            b_certs = _certify_domain_boundary_rho(
                v_pwa, domain, rho_cap, deadline=deadline
            )
            res_out = LyapunovVerificationResult(
                verdict=Verdict.CERTIFIED_EXACT,
                is_discrete=True,
                alpha_1=alpha_min,
                epsilon=eps_q,
                rho_star=rho_cap,
                shards_certified=len(decay_certs),
                ct_bab_pruned_cells=ct_bab_pruned,
                coercivity_certs=tuple(coercivity_certs),
                decay_certs=tuple(decay_certs),
                boundary_certs=tuple(b_certs),
                message=f"Certified discrete Lyapunov ROA {{V(x) <= {rho_cap[0]}/{rho_cap[1]}}} across {len(decay_certs)} shards",
            )
            assert check_lyapunov_certificate_bundle(res_out, x_eq=x_eq)
            return res_out

        # Global domain verification (compute_maximal_roa=False)
        for s_poly, v_in, v_out, s_tag in pullback_shards:
            check_budget(deadline=deadline)
            diff_map = v_out.sub(v_in.scale(one_minus_eps))
            v_str, b_cert, wit_cex, _ = certify_linear_upper_bound(
                s_poly.A,
                s_poly.b,
                diff_map.W[0],
                diff_map.denom,
                -diff_map.c[0],
                diff_map.denom,
                deadline=deadline,
            )
            if v_str == "OPEN_BUDGET_EXCEEDED":
                raise BudgetExceededError("LP budget exceeded in shard decay check")
            if v_str != "CERTIFIED_EXACT" or b_cert is None:
                assert wit_cex is not None
                vx = v_in.eval_scalar(wit_cex)
                vn = v_out.eval_scalar(wit_cex)
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=True,
                    alpha_1=alpha_min,
                    epsilon=eps_q,
                    counterexample=LyapunovCounterexample(
                        reason="Discrete Lyapunov pullback decay violated",
                        x_witness=wit_cex,
                        v_at_x=vx,
                        v_next_or_deriv=vn,
                        residual_violation=diff_map.eval_scalar(wit_cex),
                    ),
                    message=f"Shard {s_tag} violated V(F(x)) <= (1 - eps) V(x) at x*={wit_cex}",
                )
            decay_certs.append(
                ShardDecayCert(
                    shard_id=len(decay_certs),
                    poly_A=s_poly.A,
                    poly_b=s_poly.b,
                    bound_cert=b_cert,
                    tag=s_tag,
                )
            )

        res_out = LyapunovVerificationResult(
            verdict=Verdict.CERTIFIED_EXACT,
            is_discrete=True,
            alpha_1=alpha_min,
            epsilon=eps_q,
            rho_star=None,
            shards_certified=len(decay_certs),
            ct_bab_pruned_cells=ct_bab_pruned,
            coercivity_certs=tuple(coercivity_certs),
            decay_certs=tuple(decay_certs),
            message=f"Certified zero-hole discrete Lyapunov stability across {len(decay_certs)} shards",
        )
        assert check_lyapunov_certificate_bundle(res_out, x_eq=x_eq)
        return res_out

    except BudgetExceededError as exc:
        return LyapunovVerificationResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            is_discrete=True,
            message=str(exc),
        )


def _build_planet_shard_cert(
    shard_id: int,
    v_net: ReLUNetwork,
    f_map: AffineMapQ,
    cell_poly: HPolytope,
    sub_map: AffineMapQ,
    tag: str,
    deadline: Optional[float] = None,
) -> Optional[ShardDecayCert]:
    """Build explicit lifted Planet polytope + FarkasBoundCert proving V(f(x)) - sub_map(x) <= 0."""
    dim_x = cell_poly.dim
    A_lift: List[List[int]] = [list(r) for r in cell_poly.A]
    b_lift: List[int] = list(cell_poly.b)
    total_vars = dim_x

    def pad_existing(extra: int) -> None:
        nonlocal total_vars
        for r in A_lift:
            r.extend([0] * extra)
        total_vars += extra

    cur_expr = f_map
    for layer in v_net.layers:
        check_budget(deadline=deadline)
        if cur_expr.in_dim < total_vars:
            ext_W = [
                list(row) + [0] * (total_vars - cur_expr.in_dim)
                for row in cur_expr.W
            ]
            cur_expr = AffineMapQ.create(
                total_vars, cur_expr.out_dim, ext_W, cur_expr.c, cur_expr.denom
            )
        pre = layer.affine.compose(cur_expr)
        if not layer.use_relu:
            cur_expr = pre
            continue
        m_neu = pre.out_dim
        bounds_lu: List[Tuple[Rat, Rat]] = []
        for j in range(m_neu):
            wj = list(pre.W[j])
            cj = pre.c[j]
            dj = pre.denom
            r_hi = solve_lp_bareiss(A_lift, b_lift, c=wj, deadline=deadline)
            r_lo = solve_lp_bareiss(A_lift, b_lift, c=[-v for v in wj], deadline=deadline)
            if (
                r_hi.status != LPStatus.OPTIMAL
                or r_lo.status != LPStatus.OPTIMAL
                or r_hi.obj_val is None
                or r_lo.obj_val is None
            ):
                return None
            hi_q = q_add(q_norm(r_hi.obj_val[0], r_hi.obj_val[1] * dj), q_norm(cj, dj))
            lo_q = q_add(q_norm(-r_lo.obj_val[0], r_lo.obj_val[1] * dj), q_norm(cj, dj))
            bounds_lu.append((lo_q, hi_q))

        ambig: Dict[int, int] = {}
        for j, (lo_q, hi_q) in enumerate(bounds_lu):
            if lo_q[0] < 0 and hi_q[0] > 0:
                ambig[j] = total_vars + len(ambig)
        num_ambig = len(ambig)
        prev_vars = total_vars
        if num_ambig > 0:
            pad_existing(num_ambig)
        next_W: List[List[int]] = []
        next_c: List[int] = []
        for j, (lo_q, hi_q) in enumerate(bounds_lu):
            if hi_q[0] <= 0:
                next_W.append([0] * total_vars)
                next_c.append(0)
            elif lo_q[0] >= 0:
                next_W.append(list(pre.W[j]) + [0] * num_ambig)
                next_c.append(pre.c[j])
            else:
                z_col = ambig[j]
                r1 = [0] * total_vars
                r1[z_col] = -1
                A_lift.append(r1)
                b_lift.append(0)
                r2 = list(pre.W[j]) + [0] * num_ambig
                r2[z_col] = -pre.denom
                A_lift.append(r2)
                b_lift.append(-pre.c[j])
                diff_q = q_sub(hi_q, lo_q)
                sn, sd = q_norm(hi_q[0] * diff_q[1], hi_q[1] * diff_q[0])
                ld = lo_q[1]
                ln = lo_q[0]
                dj = pre.denom
                r3 = [-(sn * ld) * pre.W[j][k] for k in range(prev_vars)] + [0] * num_ambig
                r3[z_col] = sd * dj * ld
                rhs3 = sn * (ld * pre.c[j] - dj * ln)
                A_lift.append(r3)
                b_lift.append(rhs3)
                rz = [0] * total_vars
                rz[z_col] = pre.denom
                next_W.append(rz)
                next_c.append(0)
        cur_expr = AffineMapQ.create(total_vars, m_neu, next_W, next_c, pre.denom)

    if cur_expr.in_dim < total_vars:
        ext_W = [list(row) + [0] * (total_vars - cur_expr.in_dim) for row in cur_expr.W]
        cur_expr = AffineMapQ.create(total_vars, 1, ext_W, cur_expr.c, cur_expr.denom)
    if v_net.residual is not None:
        res_comp = v_net.residual.compose(f_map)
        ext_res = AffineMapQ.create(
            total_vars, 1, [list(res_comp.W[0]) + [0] * (total_vars - dim_x)], res_comp.c, res_comp.denom
        )
        cur_expr = cur_expr.add(ext_res)
    if v_net.post_offset is not None:
        cur_expr = cur_expr.shift_output(v_net.post_offset)
    ext_sub = AffineMapQ.create(
        total_vars, 1, [list(sub_map.W[0]) + [0] * (total_vars - dim_x)], sub_map.c, sub_map.denom
    )
    cur_expr = cur_expr.sub(ext_sub)

    v_str, b_cert, _, _ = certify_linear_upper_bound(
        A_lift,
        b_lift,
        cur_expr.W[0],
        cur_expr.denom,
        -cur_expr.c[0],
        cur_expr.denom,
        deadline=deadline,
    )
    if v_str == "CERTIFIED_EXACT" and b_cert is not None:
        return ShardDecayCert(
            shard_id=shard_id,
            poly_A=tuple(tuple(r) for r in A_lift),
            poly_b=tuple(b_lift),
            bound_cert=b_cert,
            tag=tag,
            is_ct_bab_lifted=True,
        )
    return None


def verify_continuous_lyapunov(
    v_func: Union[PWAFunction, ReLUNetwork],
    plant: Union[PWAFunction, ReLUNetwork, SectorPWAPlant],
    domain: HPolytope,
    x_eq: Optional[QVec] = None,
    decay_lambda: Rat = (1, 10),
    compute_maximal_roa: bool = False,
    check_filippov: bool = True,
    max_cells: int = 2048,
    deadline: Optional[float] = None,
) -> LyapunovVerificationResult:
    """Spec 2b: Exact zero-hole continuous-time Lyapunov & Filippov sliding-mode verifier over Z/Q.

    Certifies:
      1) Coercivity V(x) >= alpha_1 ||x - x*||_1 via 2n Farkas duals on all cells,
      2) Interior upper Dini derivative grad V_i . f_{i,nu}(x) <= -lambda * V_i(x) <= -(lambda * alpha_1) ||x - x*||_1,
      3) Joint Filippov crease tangent-multiplier LP across all codimension-1 & codimension-k creases!
    """
    try:
        n = domain.dim
        if x_eq is None:
            x_eq = (tuple(0 for _ in range(n)), 1)
        lam_q = q_norm(decay_lambda[0], decay_lambda[1])

        v_pwa = (
            v_func.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline)
            if isinstance(v_func, ReLUNetwork)
            else v_func
        )
        v_eq = v_pwa.eval_scalar(x_eq)
        if v_eq != (0, 1):
            return LyapunovVerificationResult(
                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                is_discrete=False,
                counterexample=LyapunovCounterexample(
                    reason="V(x_eq) != 0",
                    x_witness=x_eq,
                    v_at_x=v_eq,
                    v_next_or_deriv=(0, 1),
                    residual_violation=q_norm(abs(v_eq[0]), v_eq[1]),
                ),
                message=f"V(x_eq) = {v_eq} != 0",
            )

        if isinstance(plant, ReLUNetwork):
            sec_plant = SectorPWAPlant.from_pwa(
                plant.enumerate_pwa(domain, max_cells=max_cells, deadline=deadline),
                is_discrete=False,
            )
        elif isinstance(plant, PWAFunction):
            sec_plant = SectorPWAPlant.from_pwa(plant, is_discrete=False)
        else:
            sec_plant = plant

        # Step 1: Build joint (V, F) full-dimensional partition so each cell has single affine V_i and generators f_{i,nu}
        joint_v_cells: List[PWACell] = []
        f_gens_map: Dict[int, Sequence[AffineMapQ]] = {}
        for vc in v_pwa.cells:
            for pc in sec_plant.cells:
                check_budget(deadline=deadline)
                inter = vc.polytope.intersect(pc.polytope)
                ok_fd, _, _ = inter.is_full_dimensional(deadline=deadline)
                if ok_fd:
                    cid = len(joint_v_cells)
                    joint_v_cells.append(
                        PWACell(
                            cell_id=cid,
                            polytope=inter.prune_redundant(deadline=deadline),
                            affine_map=vc.affine_map,
                            tag=f"v{vc.cell_id}_p{pc.cell_id}",
                        )
                    )
                    f_gens_map[cid] = pc.generators
                    if len(joint_v_cells) > max_cells:
                        raise BudgetExceededError("Joint continuous cells exceeded max_cells")

        joint_v_pwa = PWAFunction(
            in_dim=n,
            out_dim=1,
            domain=domain,
            cells=tuple(joint_v_cells),
            name=v_pwa.name + "_joint",
        )

        # Step 2: Certify coercivity V(x) >= alpha_1 ||x - x_eq||_1 on all joint cells
        coercivity_certs: List[CoercivityCellCert] = []
        alpha_min: Optional[Rat] = None
        for jc in joint_v_pwa.cells:
            check_budget(deadline=deadline)
            cc, cex_c = _certify_cell_coercivity(jc, x_eq, deadline=deadline)
            if cc is None:
                wit = cex_c if cex_c is not None else x_eq
                vw = jc.affine_map.eval_scalar(wit)
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=False,
                    counterexample=LyapunovCounterexample(
                        reason="Continuous Lyapunov coercivity failed",
                        x_witness=wit,
                        v_at_x=vw,
                        v_next_or_deriv=(0, 1),
                        residual_violation=vw,
                    ),
                    message=f"Coercivity failed on cell {jc.cell_id}",
                )
            coercivity_certs.append(cc)
            if alpha_min is None or q_lt(cc.alpha_cell, alpha_min):
                alpha_min = cc.alpha_cell
        assert alpha_min is not None
        eps_l1 = q_mul(lam_q, alpha_min)

        # Helper to build affine expression for grad V_i . f(x) + lam * V_i(x)
        def lie_plus_lam_v(v_map: AffineMapQ, f_map: AffineMapQ) -> AffineMapQ:
            # grad V_i = v_map.W[0] / v_map.denom (constant vector!)
            # grad V_i . f(x) = (g_V^T W_f x + g_V^T c_f) / (d_V * d_f)
            g_V = v_map.W[0]
            lie_W = [[sum(g_V[i] * f_map.W[i][j] for i in range(n)) for j in range(n)]]
            lie_c = [sum(g_V[i] * f_map.c[i] for i in range(n))]
            lie_aff = AffineMapQ.create(n, 1, lie_W, lie_c, v_map.denom * f_map.denom)
            return lie_aff.add(v_map.scale(lam_q))

        decay_certs: List[ShardDecayCert] = []
        rho_cap: Optional[Rat] = (
            _compute_domain_boundary_rho(joint_v_pwa, domain, deadline=deadline)
            if compute_maximal_roa
            else None
        )

        if compute_maximal_roa:
            for jc in joint_v_pwa.cells:
                for g_idx, fg in enumerate(f_gens_map[jc.cell_id]):
                    check_budget(deadline=deadline)
                    expr = lie_plus_lam_v(jc.affine_map, fg)
                    v_str, _, wit_cex, _ = certify_linear_upper_bound(
                        jc.polytope.A,
                        jc.polytope.b,
                        expr.W[0],
                        expr.denom,
                        -expr.c[0],
                        expr.denom,
                        deadline=deadline,
                    )
                    if v_str == "CERTIFIED_EXACT":
                        continue
                    if jc.polytope.contains(x_eq):
                        assert wit_cex is not None
                        vx = jc.affine_map.eval_scalar(wit_cex)
                        lie_only = expr.sub(jc.affine_map.scale(lam_q)).eval_scalar(wit_cex)
                        return LyapunovVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            is_discrete=False,
                            alpha_1=alpha_min,
                            epsilon=eps_l1,
                            counterexample=LyapunovCounterexample(
                                reason="Origin-cone continuous Lyapunov decay failed",
                                x_witness=wit_cex,
                                v_at_x=vx,
                                v_next_or_deriv=lie_only,
                                residual_violation=expr.eval_scalar(wit_cex),
                            ),
                            message=f"Origin cell {jc.tag} failed continuous decay at {wit_cex}",
                        )
                    viol_poly = jc.polytope.with_halfspace(
                        [-v for v in expr.W[0]], expr.c[0]
                    )
                    res_min = solve_lp_bareiss(
                        viol_poly.A,
                        viol_poly.b,
                        c=[-v for v in jc.affine_map.W[0]],
                        deadline=deadline,
                    )
                    if res_min.status == LPStatus.OPTIMAL and res_min.obj_val is not None:
                        rho_c = q_add(
                            q_norm(-res_min.obj_val[0], res_min.obj_val[1] * jc.affine_map.denom),
                            q_norm(jc.affine_map.c[0], jc.affine_map.denom),
                        )
                        if rho_cap is None or q_lt(rho_c, rho_cap):
                            rho_cap = rho_c

            if rho_cap is None or rho_cap[0] <= 0:
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=False,
                    alpha_1=alpha_min,
                    epsilon=eps_l1,
                    message="Maximal continuous ROA level <= 0",
                )

            rn, rd = rho_cap
            for jc in joint_v_pwa.cells:
                g_row = [rd * v for v in jc.affine_map.W[0]]
                rhs_v = jc.affine_map.denom * rn - rd * jc.affine_map.c[0]
                sub_poly = jc.polytope.with_halfspace(g_row, rhs_v)
                for g_idx, fg in enumerate(f_gens_map[jc.cell_id]):
                    expr = lie_plus_lam_v(jc.affine_map, fg)
                    v_str, b_cert, _, _ = certify_linear_upper_bound(
                        sub_poly.A,
                        sub_poly.b,
                        expr.W[0],
                        expr.denom,
                        -expr.c[0],
                        expr.denom,
                        deadline=deadline,
                    )
                    if v_str != "CERTIFIED_EXACT" or b_cert is None:
                        raise BudgetExceededError("Failed re-certifying continuous ROA cell")
                    decay_certs.append(
                        ShardDecayCert(
                            shard_id=len(decay_certs),
                            poly_A=sub_poly.A,
                            poly_b=sub_poly.b,
                            bound_cert=b_cert,
                            tag=f"{jc.tag}_g{g_idx}",
                        )
                    )
            b_certs = _certify_domain_boundary_rho(
                joint_v_pwa, domain, rho_cap, deadline=deadline
            )
        else:
            b_certs = []
            for jc in joint_v_pwa.cells:
                for g_idx, fg in enumerate(f_gens_map[jc.cell_id]):
                    check_budget(deadline=deadline)
                    expr = lie_plus_lam_v(jc.affine_map, fg)
                    v_str, b_cert, wit_cex, _ = certify_linear_upper_bound(
                        jc.polytope.A,
                        jc.polytope.b,
                        expr.W[0],
                        expr.denom,
                        -expr.c[0],
                        expr.denom,
                        deadline=deadline,
                    )
                    if v_str == "OPEN_BUDGET_EXCEEDED":
                        raise BudgetExceededError("LP budget exceeded in continuous decay")
                    if v_str != "CERTIFIED_EXACT" or b_cert is None:
                        assert wit_cex is not None
                        vx = jc.affine_map.eval_scalar(wit_cex)
                        lie_only = expr.sub(jc.affine_map.scale(lam_q)).eval_scalar(wit_cex)
                        return LyapunovVerificationResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            is_discrete=False,
                            alpha_1=alpha_min,
                            epsilon=eps_l1,
                            counterexample=LyapunovCounterexample(
                                reason="Interior cell continuous Lyapunov decay violated",
                                x_witness=wit_cex,
                                v_at_x=vx,
                                v_next_or_deriv=lie_only,
                                residual_violation=expr.eval_scalar(wit_cex),
                            ),
                            message=f"Cell {jc.tag} violated interior decay at x*={wit_cex}",
                        )
                    decay_certs.append(
                        ShardDecayCert(
                            shard_id=len(decay_certs),
                            poly_A=jc.polytope.A,
                            poly_b=jc.polytope.b,
                            bound_cert=b_cert,
                            tag=f"{jc.tag}_g{g_idx}",
                        )
                    )

        # Step 4: Mandatory Filippov Crease Audit across switching surfaces!
        filippov_res: Optional[FilippovAuditResult] = None
        if check_filippov:
            filippov_res = audit_filippov_creases(
                joint_v_pwa,
                f_gens_map,
                x_eq=x_eq,
                eps_slide=lam_q,
                sublevel_rho=rho_cap,
                deadline=deadline,
            )
            if filippov_res.verdict == Verdict.OPEN_BUDGET_EXCEEDED:
                return LyapunovVerificationResult(
                    verdict=Verdict.OPEN_BUDGET_EXCEEDED,
                    is_discrete=False,
                    filippov_audit=filippov_res,
                    message=filippov_res.message,
                )
            if filippov_res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE:
                wit_f = filippov_res.counterexample
                assert wit_f is not None
                return LyapunovVerificationResult(
                    verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                    is_discrete=False,
                    alpha_1=alpha_min,
                    epsilon=eps_l1,
                    rho_star=rho_cap,
                    filippov_audit=filippov_res,
                    filippov_witness=wit_f,
                    counterexample=LyapunovCounterexample(
                        reason="Uphill/non-decaying Filippov sliding on switching crease",
                        x_witness=wit_f.x_star,
                        v_at_x=wit_f.v_at_x,
                        v_next_or_deriv=wit_f.dini_deriv,
                        residual_violation=q_add(
                            wit_f.dini_deriv, q_mul(lam_q, wit_f.v_at_x)
                        ),
                    ),
                    message=filippov_res.message,
                )

        res_out = LyapunovVerificationResult(
            verdict=Verdict.CERTIFIED_EXACT,
            is_discrete=False,
            alpha_1=alpha_min,
            epsilon=eps_l1,
            rho_star=rho_cap,
            shards_certified=len(decay_certs),
            coercivity_certs=tuple(coercivity_certs),
            decay_certs=tuple(decay_certs),
            boundary_certs=tuple(b_certs),
            filippov_audit=filippov_res,
            message=(
                f"Certified zero-hole continuous Lyapunov + Filippov stability across "
                f"{len(decay_certs)} interior generators and "
                f"{filippov_res.creases_audited if filippov_res else 0} creases"
            ),
        )
        assert check_lyapunov_certificate_bundle(res_out, x_eq=x_eq)
        return res_out

    except BudgetExceededError as exc:
        return LyapunovVerificationResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            is_discrete=False,
            message=str(exc),
        )
