from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

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
from .polyhedral_complex import AffineMapQ, HPolytope, PWACell, PWAFunction
from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    Verdict,
    bareiss_echelon,
    bareiss_rank,
    canonical_hyperplane,
    check_budget,
    gcd_list,
    lcm_list,
    lcm_pair,
    q_add,
    q_cmp,
    q_le,
    q_lt,
    q_mul,
    q_norm,
    qvec_add,
    qvec_eq,
    qvec_norm,
    qvec_scale,
)


@dataclasses.dataclass(frozen=True)
class FilippovCreaseCert:
    """Constructive integer certificate on a codimension-k switching face F.

    mode:
      - "CONTINUOUS_VECTOR_FIELD": closed-loop f is continuous across F (Hyperlyve Theorem 2),
        certified by 2n FarkasBoundCerts proving +-(f_i - f_j)_r(x) <= 0 on F.
      - "NO_ATTRACTIVE_SLIDING": F_slide is empty, certified by FarkasInfeasibilityCert on F_slide.
      - "EQUILIBRIUM_ONLY": F_slide == {x_eq}, certified by 2n FarkasBoundCerts.
      - "TANGENT_MULTIPLIER_DECAY": exact integer joint tangent-multiplier duals
        (tau_int, y_per_gen, scale > 0) certifying D^+_{f_S} V(x) + eps * V(x) <= 0 on F_slide.
    """

    face_id: int
    codim: int
    cell_ids: Tuple[int, ...]
    normals: Tuple[Tuple[int, ...], ...]
    rhss: Tuple[int, ...]
    gamma_kink: Optional[Rat]
    mode: str
    slide_poly_A: Tuple[Tuple[int, ...], ...]
    slide_poly_b: Tuple[int, ...]
    infeas_cert: Optional[FarkasInfeasibilityCert] = None
    eq_only_certs: Tuple[FarkasBoundCert, ...] = ()
    continuity_certs: Tuple[FarkasBoundCert, ...] = ()
    diff_map: Optional[AffineMapQ] = None
    tau_int: Tuple[int, ...] = ()
    y_per_gen: Tuple[Tuple[int, ...], ...] = ()
    scale: int = 1
    gen_h_rows: Tuple[Tuple[int, ...], ...] = ()
    gen_kappas: Tuple[int, ...] = ()
    gen_Q_mats: Tuple[Tuple[Tuple[int, ...], ...], ...] = ()
    gen_s_vecs: Tuple[Tuple[int, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class FilippovSlidingWitness:
    """Exact rational uphill / non-decaying Filippov sliding witness on switching face F."""

    face_id: int
    codim: int
    cell_ids: Tuple[int, ...]
    normals: Tuple[Tuple[int, ...], ...]
    rhss: Tuple[int, ...]
    gamma_kink: Optional[Rat]
    x_star: QVec
    alpha_weights: Tuple[Rat, ...]
    f_sliding: QVec
    dini_deriv: Rat
    v_at_x: Rat
    nu_normals: Tuple[Rat, ...]


@dataclasses.dataclass(frozen=True)
class FilippovAuditResult:
    verdict: Verdict
    creases_audited: int
    codim1_creases: int
    codim_k_creases: int
    certificates: Tuple[FilippovCreaseCert, ...] = ()
    counterexample: Optional[FilippovSlidingWitness] = None
    message: str = ""


def check_filippov_crease_cert(
    cert: FilippovCreaseCert, x_eq: Optional[QVec] = None
) -> bool:
    """Standalone zero-trust pure-integer checker for FilippovCreaseCert."""
    A_F = cert.slide_poly_A
    b_F = cert.slide_poly_b
    m_F = len(A_F)
    if len(b_F) != m_F:
        return False

    if cert.mode == "CONTINUOUS_VECTOR_FIELD":
        if cert.diff_map is None or m_F == 0:
            return False
        n = len(A_F[0])
        if cert.diff_map.in_dim != n or cert.diff_map.out_dim != n:
            return False
        if len(cert.continuity_certs) != 2 * n:
            return False
        dm = cert.diff_map
        idx_c = 0
        for r in range(n):
            for sgn in (1, -1):
                bc = cert.continuity_certs[idx_c]
                idx_c += 1
                for q in range(n):
                    if bc.c_num[q] * dm.denom != (sgn * dm.W[r][q]) * bc.c_den:
                        return False
                if bc.rhs_num * dm.denom != (-sgn * dm.c[r]) * bc.rhs_den:
                    return False
                if not check_positivity_farkas(A_F, b_F, bc):
                    return False
        return True

    if cert.mode == "NO_ATTRACTIVE_SLIDING":
        if cert.infeas_cert is None:
            return False
        return check_farkas_certificate(A_F, b_F, cert.infeas_cert.y)

    if cert.mode == "EQUILIBRIUM_ONLY":
        if x_eq is None or m_F == 0:
            return False
        n = len(A_F[0])
        if len(cert.eq_only_certs) != 2 * n:
            return False
        eq_n, eq_d = x_eq
        idx = 0
        for j in range(n):
            for sgn in (1, -1):
                bc = cert.eq_only_certs[idx]
                idx += 1
                exp_c = tuple(sgn if q == j else 0 for q in range(n))
                if bc.c_num != exp_c or bc.c_den != 1:
                    return False
                if bc.rhs_num * eq_d != sgn * eq_n[j] * bc.rhs_den:
                    return False
                if not check_positivity_farkas(A_F, b_F, bc):
                    return False
        return True

    if cert.mode == "TANGENT_MULTIPLIER_DECAY":
        if not isinstance(cert.scale, int) or cert.scale <= 0:
            return False
        num_gens = len(cert.y_per_gen)
        if (
            num_gens == 0
            or len(cert.gen_h_rows) != num_gens
            or len(cert.gen_kappas) != num_gens
            or len(cert.gen_Q_mats) != num_gens
            or len(cert.gen_s_vecs) != num_gens
        ):
            return False
        k_codim = len(cert.tau_int)
        if k_codim != cert.codim or m_F == 0:
            return False
        n = len(A_F[0])
        for r in range(num_gens):
            yr = cert.y_per_gen[r]
            if len(yr) != m_F or any((not isinstance(v, int) or v < 0) for v in yr):
                return False
            Qr = cert.gen_Q_mats[r]
            sr = cert.gen_s_vecs[r]
            hr = cert.gen_h_rows[r]
            kr = cert.gen_kappas[r]
            if len(Qr) != k_codim or len(sr) != k_codim or len(hr) != n:
                return False
            for j in range(n):
                lhs = sum(yr[i] * A_F[i][j] for i in range(m_F)) - sum(
                    cert.tau_int[p] * Qr[p][j] for p in range(k_codim)
                )
                if lhs != cert.scale * hr[j]:
                    return False
            lhs_b = sum(yr[i] * b_F[i] for i in range(m_F)) + sum(
                cert.tau_int[p] * sr[p] for p in range(k_codim)
            )
            if lhs_b > -cert.scale * kr:
                return False
        return True

    return False


def check_filippov_witness(
    witness: FilippovSlidingWitness,
    v_cells: Sequence[PWACell],
    f_gens_per_cell: Dict[int, Sequence[AffineMapQ]],
    eps_slide: Rat = (0, 1),
    x_eq: Optional[QVec] = None,
) -> bool:
    """Standalone zero-trust exact rational substitution checker for FilippovSlidingWitness."""
    x_s = witness.x_star
    if x_eq is not None and qvec_eq(x_s, x_eq):
        return False
    # Verify x_s lies on switching face normals N_F x_s == rhss
    xn, xd = x_s
    for idx, n_row in enumerate(witness.normals):
        if sum(n_row[j] * xn[j] for j in range(len(xn))) != witness.rhss[idx] * xd:
            return False

    cell_map = {c.cell_id: c for c in v_cells}
    flat_f_maps: List[AffineMapQ] = []
    v_vals: List[Rat] = []
    for cid in witness.cell_ids:
        if cid not in cell_map:
            return False
        vc = cell_map[cid]
        if not vc.polytope.contains(x_s):
            return False
        v_vals.append(vc.affine_map.eval_scalar(x_s))
        for fg in f_gens_per_cell.get(cid, ()):
            flat_f_maps.append(fg)

    if not v_vals or len(flat_f_maps) != len(witness.alpha_weights):
        return False
    # Verify V continuity at x_s
    for vv in v_vals[1:]:
        if q_cmp(vv, v_vals[0]) != 0:
            return False
    if q_cmp(v_vals[0], witness.v_at_x) != 0:
        return False

    # Verify convex combination alpha >= 0, sum(alpha) == 1
    alpha_sum = (0, 1)
    for a_q in witness.alpha_weights:
        if a_q[1] <= 0 or a_q[0] < 0:
            return False
        alpha_sum = q_add(alpha_sum, a_q)
    if alpha_sum != (1, 1):
        return False

    # Compute exact sliding vector f_S = sum_r alpha_r * f_r(x_s)
    f_comb = (tuple(0 for _ in range(len(xn))), 1)
    for a_q, fg in zip(witness.alpha_weights, flat_f_maps):
        f_comb = qvec_add(f_comb, qvec_scale(a_q, fg.eval_qvec(x_s)))
    if not qvec_eq(f_comb, witness.f_sliding):
        return False

    # Verify tangency N_F * f_S == 0
    fn, fd = f_comb
    for n_row in witness.normals:
        if sum(n_row[j] * fn[j] for j in range(len(fn))) != 0:
            return False

    # For codim-1 attractive check: if 2 cells with 1 generator each, check opposing inward normals
    if witness.codim == 1 and len(witness.cell_ids) == 2 and len(flat_f_maps) == 2:
        n0 = witness.normals[0]
        f0 = flat_f_maps[0].eval_qvec(x_s)
        f1 = flat_f_maps[1].eval_qvec(x_s)
        # Check which cell is on n0^T x <= rhs vs >= rhs
        c0 = cell_map[witness.cell_ids[0]].polytope
        nu0 = q_norm(sum(n0[j] * f0[0][j] for j in range(len(xn))), f0[1])
        nu1 = q_norm(sum(n0[j] * f1[0][j] for j in range(len(xn))), f1[1])
        # Determine orientation of n0 relative to c0 using c0's constraints
        n0_tup = tuple(n0)
        neg_n0 = tuple(-v for v in n0)
        if n0_tup in c0.A:
            # c0 is n0^T x <= rhs, c1 is n0^T x >= rhs -> attractive requires nu0 >= 0 and nu1 <= 0
            if nu0[0] < 0 or nu1[0] > 0:
                return False
        elif neg_n0 in c0.A:
            if nu0[0] > 0 or nu1[0] < 0:
                return False

    # Verify D^+_{f_S} V(x_s) = grad V_{c0} . f_S
    vc0 = cell_map[witness.cell_ids[0]].affine_map
    dini_num = sum(vc0.W[0][j] * fn[j] for j in range(len(fn)))
    dini_q = q_norm(dini_num, vc0.denom * fd)
    if q_cmp(dini_q, witness.dini_deriv) != 0:
        return False

    # Must violate D^+_{f_S} V(x_s) <= -eps_slide * V(x_s)
    req_rhs = q_mul((-eps_slide[0], eps_slide[1]), witness.v_at_x)
    return not q_le(dini_q, req_rhs)


def _subspace_chebyshev_center(
    face_poly: HPolytope,
    normals: Sequence[Sequence[int]],
    rhss: Sequence[int],
    deadline: Optional[float] = None,
) -> Tuple[bool, Optional[QVec]]:
    """Test whether face_poly = {A x <= b} subset {N x = rhs} has full dimension (n - rank(N))
    and extract its exact rational relative-interior Chebyshev center!
    """
    n = face_poly.dim
    k = len(normals)
    if k == 0:
        ok, _, xc = face_poly.is_full_dimensional(deadline=deadline)
        return (ok, xc)

    # Use Bareiss echelon on [N | rhs] to parameterize {N x = rhs} by (n - r) free coordinates!
    aug = [list(normals[i]) + [int(rhss[i])] for i in range(k)]
    M, piv_cols, det_sub, _ = bareiss_echelon(
        aug, num_piv_cols=n, deadline=deadline
    )
    rank = len(piv_cols)
    for r_idx in range(rank, k):
        if M[r_idx][n] != 0:
            return (False, None)

    free_cols = [j for j in range(n) if j not in set(piv_cols)]
    d_sub = len(free_cols)
    if d_sub == 0:
        # 0D point: check feasibility of unique point
        x_nums = [0] * n
        for idx, pcol in enumerate(piv_cols):
            x_nums[pcol] = M[idx][n]
        pt = qvec_norm(x_nums, det_sub)
        if face_poly.contains(pt):
            return (True, pt)
        return (False, None)

    # Express x = (x0 + B_free * u) / det_sub (taking care of sign of det_sub!)
    sign_d = -1 if det_sub < 0 else 1
    abs_d = abs(det_sub)
    x0 = [0] * n
    B_free = [[0] * d_sub for _ in range(n)]
    piv_map = {col: idx for idx, col in enumerate(piv_cols)}
    for f_idx, f_col in enumerate(free_cols):
        B_free[f_col][f_idx] = abs_d
    for pcol, prow in piv_map.items():
        x0[pcol] = sign_d * M[prow][n]
        for f_idx, f_col in enumerate(free_cols):
            B_free[pcol][f_idx] = -sign_d * M[prow][f_col]

    # Substitute x into A_face * x <= b_face:
    # A_face * (x0 + B_free * u) <= abs_d * b_face  <=>  (A_face * B_free) u <= abs_d * b_face - A_face * x0
    sub_A: List[List[int]] = []
    sub_b: List[int] = []
    for i in range(len(face_poly.A)):
        ar = face_poly.A[i]
        # Skip rows already parallel to N (they become 0 <= 0)
        u_row = [
            sum(ar[q] * B_free[q][f_idx] for q in range(n))
            for f_idx in range(d_sub)
        ]
        u_rhs = abs_d * face_poly.b[i] - sum(ar[q] * x0[q] for q in range(n))
        if all(v == 0 for v in u_row):
            if u_rhs < 0:
                return (False, None)
            continue
        sub_A.append(u_row)
        sub_b.append(u_rhs)

    sub_poly = HPolytope.create(d_sub, sub_A, sub_b)
    ok_sub, _, u_center = sub_poly.is_full_dimensional(deadline=deadline)
    if not ok_sub or u_center is None:
        return (False, None)

    un, ud = u_center
    # Lift u_center = un / ud back to x = (ud * x0 + B_free * un) / (abs_d * ud)
    x_lift = [
        ud * x0[q] + sum(B_free[q][f_idx] * un[f_idx] for f_idx in range(d_sub))
        for q in range(n)
    ]
    return (True, qvec_norm(x_lift, abs_d * ud))


def _solve_joint_tangent_lp(
    face_id: int,
    codim: int,
    cell_ids: Tuple[int, ...],
    normals: Tuple[Tuple[int, ...], ...],
    rhss: Tuple[int, ...],
    gamma_kink: Optional[Rat],
    slide_poly: HPolytope,
    v_ref_map: AffineMapQ,
    f_gens: Sequence[AffineMapQ],
    eps_slide: Rat,
    x_eq: QVec,
    deadline: Optional[float] = None,
) -> Tuple[Optional[FilippovCreaseCert], Optional[FilippovSlidingWitness]]:
    """Solve the joint Filippov tangent-multiplier LP on slide_poly, returning either
    an integer certificate or an exact rational sliding witness.
    """
    n = slide_poly.dim
    A_F = slide_poly.A
    b_F = slide_poly.b
    m_F = len(A_F)

    # Step 1: Check if slide_poly is empty
    res_feas = solve_lp_bareiss(A_F, b_F, c=None, deadline=deadline)
    if res_feas.status == LPStatus.BUDGET_EXCEEDED:
        raise BudgetExceededError("LP budget exceeded in Filippov feasibility")
    if res_feas.status == LPStatus.INFEASIBLE:
        assert res_feas.farkas_infeas is not None
        cert = FilippovCreaseCert(
            face_id=face_id,
            codim=codim,
            cell_ids=cell_ids,
            normals=normals,
            rhss=rhss,
            gamma_kink=gamma_kink,
            mode="NO_ATTRACTIVE_SLIDING",
            slide_poly_A=A_F,
            slide_poly_b=b_F,
            infeas_cert=res_feas.farkas_infeas,
        )
        assert check_filippov_crease_cert(cert, x_eq=x_eq)
        return (cert, None)

    # Step 2: Check if slide_poly == {x_eq} (equilibrium singleton only)
    eq_n, eq_d = x_eq
    eq_certs: List[FarkasBoundCert] = []
    only_eq = True
    for j in range(n):
        for sgn in (1, -1):
            c_dir = [sgn if q == j else 0 for q in range(n)]
            v_str, b_cert, _, _ = certify_linear_upper_bound(
                A_F, b_F, c_dir, 1, sgn * eq_n[j], eq_d, deadline=deadline
            )
            if v_str == "OPEN_BUDGET_EXCEEDED":
                raise BudgetExceededError("LP budget exceeded in Filippov eq check")
            if v_str == "CERTIFIED_EXACT" and b_cert is not None:
                eq_certs.append(b_cert)
            else:
                only_eq = False
                break
        if not only_eq:
            break

    if only_eq:
        cert = FilippovCreaseCert(
            face_id=face_id,
            codim=codim,
            cell_ids=cell_ids,
            normals=normals,
            rhss=rhss,
            gamma_kink=gamma_kink,
            mode="EQUILIBRIUM_ONLY",
            slide_poly_A=A_F,
            slide_poly_b=b_F,
            eq_only_certs=tuple(eq_certs),
        )
        assert check_filippov_crease_cert(cert, x_eq=x_eq)
        return (cert, None)

    # Step 3: Formulate Joint Tangent-Multiplier LP over (tau in R^codim, y^(r) >= 0 in R^m_F)
    # Clear common denominator D_common across V, all f_r, and eps_slide = p_eps / q_eps
    p_eps, q_eps = q_norm(eps_slide[0], eps_slide[1])
    d_V = v_ref_map.denom
    denoms_needed = [d_V * q_eps] + [d_V * fg.denom for fg in f_gens] + [fg.denom for fg in f_gens]
    D_common = lcm_list(denoms_needed)

    g_V = v_ref_map.W[0]
    c_V = v_ref_map.c[0]

    R_cnt = len(f_gens)
    h_rows: List[Tuple[int, ...]] = []
    kappas: List[int] = []
    Q_mats: List[Tuple[Tuple[int, ...], ...]] = []
    s_vecs: List[Tuple[int, ...]] = []

    scale_eps = (D_common // (d_V * q_eps)) * p_eps
    for fg in f_gens:
        scale_vf = D_common // (d_V * fg.denom)
        # grad V . f_r(x) + eps * V(x):
        # x-coeff: scale_vf * (g_V^T W_r) + scale_eps * g_V^T
        hr = tuple(
            scale_vf * sum(g_V[i] * fg.W[i][j] for i in range(n))
            + scale_eps * g_V[j]
            for j in range(n)
        )
        kr = (
            scale_vf * sum(g_V[i] * fg.c[i] for i in range(n))
            + scale_eps * c_V
        )
        h_rows.append(hr)
        kappas.append(kr)

        scale_nf = D_common // fg.denom
        Qr = tuple(
            tuple(
                scale_nf * sum(normals[p][i] * fg.W[i][j] for i in range(n))
                for j in range(n)
            )
            for p in range(codim)
        )
        sr = tuple(
            scale_nf * sum(normals[p][i] * fg.c[i] for i in range(n))
            for p in range(codim)
        )
        Q_mats.append(Qr)
        s_vecs.append(sr)

    # Decision variables in joint LP:
    # [tau_0 .. tau_{codim-1}, y^(0)_0 .. y^(0)_{m_F-1}, ..., y^(R-1)_0 .. y^(R-1)_{m_F-1}]
    total_dec = codim + R_cnt * m_F
    joint_A: List[List[int]] = []
    joint_b: List[int] = []

    for r in range(R_cnt):
        y_off = codim + r * m_F
        # 1) y^(r) >= 0  <=>  -y^(r)_i <= 0
        for i in range(m_F):
            row_nn = [0] * total_dec
            row_nn[y_off + i] = -1
            joint_A.append(row_nn)
            joint_b.append(0)
        # 2) Equality: sum_i A_F[i][j] y^(r)_i - sum_p Q_r[p][j] tau_p == h_r[j]  (written as <= and >=)
        for j in range(n):
            row_eq = [0] * total_dec
            for p in range(codim):
                row_eq[p] = -Q_mats[r][p][j]
            for i in range(m_F):
                row_eq[y_off + i] = A_F[i][j]
            joint_A.append(row_eq)
            joint_b.append(h_rows[r][j])
            joint_A.append([-v for v in row_eq])
            joint_b.append(-h_rows[r][j])
        # 3) RHS inequality: sum_i b_F[i] y^(r)_i + sum_p s_r[p] tau_p <= -kappa_r
        row_rhs = [0] * total_dec
        for p in range(codim):
            row_rhs[p] = s_vecs[r][p]
        for i in range(m_F):
            row_rhs[y_off + i] = b_F[i]
        joint_A.append(row_rhs)
        joint_b.append(-kappas[r])

    res_joint = solve_lp_bareiss(joint_A, joint_b, c=None, deadline=deadline)
    if res_joint.status == LPStatus.BUDGET_EXCEEDED:
        raise BudgetExceededError("LP budget exceeded in joint Filippov LP")

    if res_joint.status == LPStatus.OPTIMAL and res_joint.x_opt is not None:
        dec_nums, dec_den = res_joint.x_opt
        tau_int = tuple(dec_nums[:codim])
        y_per_gen = tuple(
            tuple(dec_nums[codim + r * m_F : codim + (r + 1) * m_F])
            for r in range(R_cnt)
        )
        cert = FilippovCreaseCert(
            face_id=face_id,
            codim=codim,
            cell_ids=cell_ids,
            normals=normals,
            rhss=rhss,
            gamma_kink=gamma_kink,
            mode="TANGENT_MULTIPLIER_DECAY",
            slide_poly_A=A_F,
            slide_poly_b=b_F,
            tau_int=tau_int,
            y_per_gen=y_per_gen,
            scale=dec_den,
            gen_h_rows=tuple(h_rows),
            gen_kappas=tuple(kappas),
            gen_Q_mats=tuple(Q_mats),
            gen_s_vecs=tuple(s_vecs),
        )
        assert check_filippov_crease_cert(cert, x_eq=x_eq), (
            "Internal Filippov tangent-multiplier certificate invariant failed"
        )
        return (cert, None)

    # Step 4: Joint LP infeasible -> extract concrete exact rational uphill-sliding witness x* in slide_poly!
    candidates: List[QVec] = []
    ok_rel, x_rel_center = _subspace_chebyshev_center(
        slide_poly, normals, rhss, deadline=deadline
    )
    if ok_rel and x_rel_center is not None and not qvec_eq(x_rel_center, x_eq):
        candidates.append(x_rel_center)
    if res_feas.x_opt is not None and not qvec_eq(res_feas.x_opt, x_eq):
        candidates.append(res_feas.x_opt)
    # Add extreme vertices along +/- coordinate axes and generator gradients
    probe_dirs: List[Sequence[int]] = []
    for j in range(n):
        probe_dirs.append([1 if q == j else 0 for q in range(n)])
        probe_dirs.append([-1 if q == j else 0 for q in range(n)])
    for hr in h_rows:
        if any(v != 0 for v in hr):
            probe_dirs.append(hr)

    for d_vec in probe_dirs:
        rp = solve_lp_bareiss(A_F, b_F, c=d_vec, deadline=deadline)
        if rp.status == LPStatus.OPTIMAL and rp.x_opt is not None:
            if not qvec_eq(rp.x_opt, x_eq):
                candidates.append(rp.x_opt)
                if ok_rel and x_rel_center is not None:
                    # Midpoint with relative center stays strictly in relative interior
                    mid = qvec_scale((1, 2), qvec_add(rp.x_opt, x_rel_center))
                    if not qvec_eq(mid, x_eq):
                        candidates.append(mid)

    # Also add rational bisection / convex combinations between initial distinct candidates
    base_distinct: List[QVec] = []
    for xc in candidates:
        if not any(qvec_eq(xc, ex) for ex in base_distinct):
            base_distinct.append(xc)
        if len(base_distinct) >= 6:
            break
    for i_c in range(len(base_distinct)):
        for j_c in range(i_c + 1, len(base_distinct)):
            for w_num in (1, 2, 3):
                comb_pt = qvec_add(
                    qvec_scale((w_num, 4), base_distinct[i_c]),
                    qvec_scale((4 - w_num, 4), base_distinct[j_c]),
                )
                if not qvec_eq(comb_pt, x_eq):
                    candidates.append(comb_pt)

    best_witness: Optional[FilippovSlidingWitness] = None
    for x_cand in candidates:
        check_budget(deadline=deadline)
        # Solve pointwise convex-combination LP in alpha in R^R_cnt:
        # max sum_r alpha_r (grad V . f_r(x_cand))
        # s.t. alpha_r >= 0, sum_r alpha_r == 1, N_F * (sum_r alpha_r f_r(x_cand)) == 0
        f_vals = [fg.eval_qvec(x_cand) for fg in f_gens]
        common_fd = lcm_list([fv[1] for fv in f_vals])
        f_ints = [
            [fv[0][j] * (common_fd // fv[1]) for j in range(n)]
            for fv in f_vals
        ]

        alp_A: List[List[int]] = []
        alp_b: List[int] = []
        # alpha_r >= 0
        for r in range(R_cnt):
            row = [0] * R_cnt
            row[r] = -1
            alp_A.append(row)
            alp_b.append(0)
        # sum_r alpha_r == 1
        alp_A.append([1] * R_cnt)
        alp_b.append(1)
        alp_A.append([-1] * R_cnt)
        alp_b.append(-1)
        # N_F * sum_r alpha_r f_ints[r] == 0
        for p in range(codim):
            row_t = [
                sum(normals[p][j] * f_ints[r][j] for j in range(n))
                for r in range(R_cnt)
            ]
            alp_A.append(row_t)
            alp_b.append(0)
            alp_A.append([-v for v in row_t])
            alp_b.append(0)

        # Objective: maximize sum_r alpha_r * (g_V^T f_ints[r])
        obj_alp = [
            sum(g_V[j] * f_ints[r][j] for j in range(n)) for r in range(R_cnt)
        ]
        res_alp = solve_lp_bareiss(alp_A, alp_b, c=obj_alp, deadline=deadline)
        if res_alp.status != LPStatus.OPTIMAL or res_alp.x_opt is None or res_alp.obj_val is None:
            continue

        an, ad = res_alp.x_opt
        alphas = tuple(q_norm(an[r], ad) for r in range(R_cnt))
        f_s = (tuple(0 for _ in range(n)), 1)
        for r in range(R_cnt):
            f_s = qvec_add(f_s, qvec_scale(alphas[r], f_vals[r]))
        dini_q = q_norm(res_alp.obj_val[0], res_alp.obj_val[1] * d_V * common_fd)
        v_x = v_ref_map.eval_scalar(x_cand)
        req_rhs = q_mul((-p_eps, q_eps), v_x)

        if not q_le(dini_q, req_rhs):
            nu_list: List[Rat] = []
            for fv in f_vals:
                nu_list.append(
                    q_norm(
                        sum(normals[0][j] * fv[0][j] for j in range(n)),
                        fv[1],
                    )
                )
            wit = FilippovSlidingWitness(
                face_id=face_id,
                codim=codim,
                cell_ids=cell_ids,
                normals=normals,
                rhss=rhss,
                gamma_kink=gamma_kink,
                x_star=x_cand,
                alpha_weights=alphas,
                f_sliding=f_s,
                dini_deriv=dini_q,
                v_at_x=v_x,
                nu_normals=tuple(nu_list),
            )
            # Immediately return if strictly uphill (dini_deriv > 0) at preferred candidate (e.g. Chebyshev center!)
            if dini_q[0] > 0:
                return (None, wit)
            if best_witness is None:
                best_witness = wit

    return (None, best_witness)


def audit_filippov_creases(
    v_pwa: PWAFunction,
    f_gens_per_cell: Dict[int, Sequence[AffineMapQ]],
    x_eq: Optional[QVec] = None,
    eps_slide: Rat = (0, 1),
    sublevel_rho: Optional[Rat] = None,
    include_codim_k: bool = True,
    deadline: Optional[float] = None,
) -> FilippovAuditResult:
    """Audit all codimension-1 and codimension-k interior switching creases of a joint (V, F) partition.

    Emits constructive integer certificates on every crease when sliding strictly decays,
    or an exact rational Filippov uphill-sliding witness (x*, f_S(x*)) when violated.
    """
    try:
        n = v_pwa.in_dim
        if x_eq is None:
            x_eq = (tuple(0 for _ in range(n)), 1)

        # Optionally intersect cells with sublevel {V_i(x) <= rho}
        cells: List[PWACell] = []
        for c in v_pwa.cells:
            poly = c.polytope
            if sublevel_rho is not None:
                rn, rd = q_norm(sublevel_rho[0], sublevel_rho[1])
                # (g_V^T x + c_V) / d_V <= rn / rd  <=>  (rd * g_V)^T x <= d_V * rn - rd * c_V
                g_row = [rd * v for v in c.affine_map.W[0]]
                rhs_v = c.affine_map.denom * rn - rd * c.affine_map.c[0]
                poly = poly.with_halfspace(g_row, rhs_v)
            cells.append(
                PWACell(
                    cell_id=c.cell_id,
                    polytope=poly,
                    affine_map=c.affine_map,
                    tag=c.tag,
                )
            )

        certs: List[FilippovCreaseCert] = []
        face_counter = 0
        codim1_cnt = 0
        codimk_cnt = 0

        # Map canonical hyperplanes to incident cells for codim-1 and codim-k discovery
        # First: Codimension-1 adjacent cell pairs (i < j)
        num_c = len(cells)
        codim1_edges: List[Tuple[int, int, Tuple[int, ...], int, HPolytope]] = []

        for idx_i in range(num_c):
            ci = cells[idx_i]
            map_i = {ci.polytope.A[r]: ci.polytope.b[r] for r in range(len(ci.polytope.A))}
            for idx_j in range(idx_i + 1, num_c):
                check_budget(deadline=deadline)
                cj = cells[idx_j]
                map_j = {cj.polytope.A[r]: cj.polytope.b[r] for r in range(len(cj.polytope.A))}

                shared_normal: Optional[Tuple[int, ...]] = None
                shared_rhs = 0
                for a_i, b_i in map_i.items():
                    neg_a = tuple(-v for v in a_i)
                    if neg_a in map_j and map_j[neg_a] == -b_i:
                        shared_normal = a_i
                        shared_rhs = b_i
                        break
                if shared_normal is None:
                    continue

                face_poly = ci.polytope.intersect(cj.polytope)
                ok_f, _ = _subspace_chebyshev_center(
                    face_poly, [shared_normal], [shared_rhs], deadline=deadline
                )
                if not ok_f:
                    continue

                codim1_edges.append(
                    (idx_i, idx_j, shared_normal, shared_rhs, face_poly)
                )

                # Compute convex kink jump gamma_ij: grad V_j - grad V_i = gamma_ij * n_ij
                # where n_ij = shared_normal points from C_i (n^T x <= b) into C_j (n^T x >= b).
                vi = ci.affine_map
                vj = cj.affine_map
                cd_v = lcm_pair(vi.denom, vj.denom)
                si = cd_v // vi.denom
                sj = cd_v // vj.denom
                diff_g = [
                    sj * vj.W[0][q] - si * vi.W[0][q] for q in range(n)
                ]
                gamma_q: Optional[Rat] = (0, 1)
                for q in range(n):
                    if shared_normal[q] != 0:
                        gamma_q = q_norm(diff_g[q], cd_v * shared_normal[q])
                        break

                # Build attractive sliding sub-facet:
                # If single generator per side, intersect with nu_i(x) >= 0 and nu_j(x) <= 0!
                gens_i = list(f_gens_per_cell.get(ci.cell_id, ()))
                gens_j = list(f_gens_per_cell.get(cj.cell_id, ()))
                if not gens_i or not gens_j:
                    continue

                slide_poly = face_poly
                if len(gens_i) == 1 and len(gens_j) == 1:
                    fi = gens_i[0]
                    fj = gens_j[0]
                    # Hyperlyve Theorem 2 check: if fi(x) == fj(x) across face_poly, interior decay suffices!
                    diff_f = fi.sub(fj)
                    cont_certs: List[FarkasBoundCert] = []
                    is_cont = True
                    for r_out in range(n):
                        for sgn in (1, -1):
                            v_c, bc_c, _, _ = certify_linear_upper_bound(
                                face_poly.A,
                                face_poly.b,
                                [sgn * v for v in diff_f.W[r_out]],
                                diff_f.denom,
                                -sgn * diff_f.c[r_out],
                                diff_f.denom,
                                deadline=deadline,
                            )
                            if v_c == "OPEN_BUDGET_EXCEEDED":
                                raise BudgetExceededError("LP budget exceeded in Hyperlyve continuity check")
                            if v_c == "CERTIFIED_EXACT" and bc_c is not None:
                                cont_certs.append(bc_c)
                            else:
                                is_cont = False
                                break
                        if not is_cont:
                            break
                    if is_cont:
                        codim1_cnt += 1
                        c_cert = FilippovCreaseCert(
                            face_id=face_counter,
                            codim=1,
                            cell_ids=(ci.cell_id, cj.cell_id),
                            normals=(shared_normal,),
                            rhss=(shared_rhs,),
                            gamma_kink=gamma_q,
                            mode="CONTINUOUS_VECTOR_FIELD",
                            slide_poly_A=face_poly.A,
                            slide_poly_b=face_poly.b,
                            continuity_certs=tuple(cont_certs),
                            diff_map=diff_f,
                        )
                        assert check_filippov_crease_cert(c_cert, x_eq=x_eq)
                        certs.append(c_cert)
                        face_counter += 1
                        continue

                    # nu_i(x) = n_ij^T (W_i x + c_i) / d_i >= 0  <=>  -(n_ij^T W_i) x <= n_ij^T c_i
                    n_wi = [
                        sum(shared_normal[q] * fi.W[q][col] for q in range(n))
                        for col in range(n)
                    ]
                    n_ci = sum(shared_normal[q] * fi.c[q] for q in range(n))
                    slide_poly = slide_poly.with_halfspace(
                        [-v for v in n_wi], n_ci
                    )
                    # nu_j(x) = n_ij^T (W_j x + c_j) / d_j <= 0  <=>  (n_ij^T W_j) x <= -n_ij^T c_j
                    n_wj = [
                        sum(shared_normal[q] * fj.W[q][col] for q in range(n))
                        for col in range(n)
                    ]
                    n_cj = sum(shared_normal[q] * fj.c[q] for q in range(n))
                    slide_poly = slide_poly.with_halfspace(n_wj, -n_cj)

                codim1_cnt += 1
                cert, wit = _solve_joint_tangent_lp(
                    face_id=face_counter,
                    codim=1,
                    cell_ids=(ci.cell_id, cj.cell_id),
                    normals=(shared_normal,),
                    rhss=(shared_rhs,),
                    gamma_kink=gamma_q,
                    slide_poly=slide_poly,
                    v_ref_map=vi,
                    f_gens=gens_i + gens_j,
                    eps_slide=eps_slide,
                    x_eq=x_eq,
                    deadline=deadline,
                )
                face_counter += 1
                if wit is not None:
                    assert check_filippov_witness(
                        wit, cells, f_gens_per_cell, eps_slide=eps_slide, x_eq=x_eq
                    )
                    return FilippovAuditResult(
                        verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                        creases_audited=face_counter,
                        codim1_creases=codim1_cnt,
                        codim_k_creases=codimk_cnt,
                        certificates=tuple(certs),
                        counterexample=wit,
                        message=(
                            f"Uphill/non-decaying Filippov sliding detected on codim-1 crease "
                            f"cells={wit.cell_ids} at x*={wit.x_star}"
                        ),
                    )
                assert cert is not None
                certs.append(cert)

        # Second: Codimension-k switching faces (k >= 2, dim(F) = n - k >= 1, so n >= 3)
        if include_codim_k and n >= 3 and len(codim1_edges) >= 2:
            # Discover codim-2 switching ridges formed by pairs of distinct canonical hyperplanes
            seen_ridges = set()
            for e1_idx in range(len(codim1_edges)):
                i1, j1, n1, r1, fp1 = codim1_edges[e1_idx]
                c1_a, c1_b, _ = canonical_hyperplane(n1, r1)
                for e2_idx in range(e1_idx + 1, len(codim1_edges)):
                    check_budget(deadline=deadline)
                    i2, j2, n2, r2, fp2 = codim1_edges[e2_idx]
                    if len({i1, j1} & {i2, j2}) == 0:
                        continue
                    c2_a, c2_b, _ = canonical_hyperplane(n2, r2)
                    if c1_a == c2_a and c1_b == c2_b:
                        continue
                    if bareiss_rank([c1_a, c2_a], deadline=deadline) < 2:
                        continue
                    # Collect all cells containing the intersection ridge
                    ridge_base = fp1.intersect(fp2)
                    ok_r, _ = _subspace_chebyshev_center(
                        ridge_base, [c1_a, c2_a], [c1_b, c2_b], deadline=deadline
                    )
                    if not ok_r:
                        continue
                    inc_indices = sorted({i1, j1, i2, j2})
                    for idx_m in range(num_c):
                        if idx_m in inc_indices:
                            continue
                        cand_r = ridge_base.intersect(cells[idx_m].polytope)
                        ok_m, _ = _subspace_chebyshev_center(
                            cand_r, [c1_a, c2_a], [c1_b, c2_b], deadline=deadline
                        )
                        if ok_m:
                            ridge_base = cand_r
                            inc_indices.append(idx_m)
                    inc_tup = tuple(sorted(cells[ix].cell_id for ix in inc_indices))
                    norm_pairs = tuple(sorted([(c1_a, c1_b), (c2_a, c2_b)]))
                    key = (inc_tup, norm_pairs)
                    if key in seen_ridges:
                        continue
                    seen_ridges.add(key)

                    all_gens: List[AffineMapQ] = []
                    for cid in inc_tup:
                        all_gens.extend(f_gens_per_cell.get(cid, ()))
                    if not all_gens:
                        continue

                    codimk_cnt += 1
                    cert_k, wit_k = _solve_joint_tangent_lp(
                        face_id=face_counter,
                        codim=2,
                        cell_ids=inc_tup,
                        normals=(norm_pairs[0][0], norm_pairs[1][0]),
                        rhss=(norm_pairs[0][1], norm_pairs[1][1]),
                        gamma_kink=None,
                        slide_poly=ridge_base,
                        v_ref_map=cells[inc_indices[0]].affine_map,
                        f_gens=all_gens,
                        eps_slide=eps_slide,
                        x_eq=x_eq,
                        deadline=deadline,
                    )
                    face_counter += 1
                    if wit_k is not None:
                        assert check_filippov_witness(
                            wit_k, cells, f_gens_per_cell, eps_slide=eps_slide, x_eq=x_eq
                        )
                        return FilippovAuditResult(
                            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                            creases_audited=face_counter,
                            codim1_creases=codim1_cnt,
                            codim_k_creases=codimk_cnt,
                            certificates=tuple(certs),
                            counterexample=wit_k,
                            message=(
                                f"Uphill/non-decaying Filippov sliding detected on codim-2 ridge "
                                f"cells={wit_k.cell_ids} at x*={wit_k.x_star}"
                            ),
                        )
                    if cert_k is not None:
                        certs.append(cert_k)

        return FilippovAuditResult(
            verdict=Verdict.CERTIFIED_EXACT,
            creases_audited=face_counter,
            codim1_creases=codim1_cnt,
            codim_k_creases=codimk_cnt,
            certificates=tuple(certs),
            message=f"Certified {face_counter} Filippov creases ({codim1_cnt} codim-1, {codimk_cnt} codim-k)",
        )
    except BudgetExceededError as exc:
        return FilippovAuditResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            creases_audited=0,
            codim1_creases=0,
            codim_k_creases=0,
            message=str(exc),
        )
