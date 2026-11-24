from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from .bareiss_simplex import (
    FarkasInfeasibilityCert,
    LPStatus,
    check_farkas_certificate,
    check_primal_witness,
    solve_lp_bareiss,
)
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
    check_budget,
    q_abs,
    q_add,
    q_lt,
    q_norm,
    qvec_add,
    qvec_eq,
    qvec_norm,
    qvec_scale,
    solve_one_minus_m_q,
)


@dataclasses.dataclass(frozen=True)
class FixedPointRecord:
    point: QVec
    cell_ids: Tuple[int, ...]
    is_isolated: bool
    segment_other_end: Optional[QVec] = None
    stability: str = "UNKNOWN"
    jacobian_W: Tuple[Tuple[int, ...], ...] = ()
    jacobian_denom: int = 1


@dataclasses.dataclass(frozen=True)
class NonFixedPointCellCert:
    """Exact integer Farkas certificate proving {x in C_i : F_i(x) == x} is empty."""

    cell_id: int
    aug_A: Tuple[Tuple[int, ...], ...]
    aug_b: Tuple[int, ...]
    infeas_cert: FarkasInfeasibilityCert


@dataclasses.dataclass(frozen=True)
class PeriodicOrbitWitness:
    """Exact rational minimal Period-k orbit (x^(0), ..., x^(k-1)) with F(x^(t)) = x^(t+1 mod k)."""

    period: int
    points: Tuple[QVec, ...]
    cell_itinerary: Tuple[int, ...]
    stability: str
    cycle_jacobian_W: Tuple[Tuple[int, ...], ...]
    cycle_jacobian_denom: int


@dataclasses.dataclass(frozen=True)
class OrbitDetectionResult:
    verdict: Verdict
    num_cells: int
    transition_edges: Tuple[Tuple[int, int], ...]
    fixed_points: Tuple[FixedPointRecord, ...]
    periodic_orbits: Tuple[PeriodicOrbitWitness, ...]
    unique_origin_fixed_point_rn: bool
    origin_locally_exp_stable: bool
    non_fixed_cell_certs: Tuple[NonFixedPointCellCert, ...] = ()
    cycles_checked: int = 0
    message: str = ""


def _classify_discrete_jacobian(amap: AffineMapQ) -> str:
    """Exact rational discrete-time Jacobian stability classification (Schur-Cohn in 1D/2D + power norm)."""
    n = amap.in_dim
    W = amap.W
    d = amap.denom
    if all(W[i][j] == 0 for i in range(n) for j in range(n)):
        return "SUPER_STABLE"

    if n == 1:
        if abs(W[0][0]) < d:
            return "LOCALLY_STABLE"
        return "UNSTABLE_OR_MARGINAL"

    if n == 2:
        # Jury's exact rational Schur-Cohn criterion for 2x2 matrix J = W / d:
        # Spectral radius rho(J) < 1  <=>  |det(J)| < 1  AND  |tr(J)| < 1 + det(J)
        # Write det(W) = w00*w11 - w01*w10, tr(W) = w00 + w11:
        # |det(W)| < d^2  AND  d * |tr(W)| < d^2 + det(W)
        det_w = W[0][0] * W[1][1] - W[0][1] * W[1][0]
        tr_w = W[0][0] + W[1][1]
        d2 = d * d
        if abs(det_w) < d2 and d * abs(tr_w) < d2 + det_w:
            return "LOCALLY_STABLE"
        return "UNSTABLE_OR_MARGINAL"

    # General dimension n >= 3: test exact matrix powers J^r for l_inf operator norm < 1
    cur = AffineMapQ.create(n, n, W, [0] * n, d)
    for _ in range(n + 4):
        max_row_sum = max(sum(abs(v) for v in row) for row in cur.W)
        if max_row_sum < cur.denom:
            return "LOCALLY_STABLE"
        cur = cur.compose(AffineMapQ.create(n, n, W, [0] * n, d))
    return "UNSTABLE_OR_MARGINAL"


def check_periodic_orbit_witness(
    orbit: PeriodicOrbitWitness,
    plant: Union[PWAFunction, ReLUNetwork],
) -> bool:
    """Standalone zero-trust exact rational verifier for a minimal Period-k limit cycle."""
    k = orbit.period
    if k < 2 or len(orbit.points) != k or len(orbit.cell_itinerary) != k:
        return False
    # Verify F(x^(t)) == x^((t + 1) % k) for all t
    for t in range(k):
        xt = orbit.points[t]
        x_next_expected = orbit.points[(t + 1) % k]
        x_next_actual = plant.eval_qvec(xt)
        if not qvec_eq(x_next_actual, x_next_expected):
            return False
        if isinstance(plant, PWAFunction):
            cid = orbit.cell_itinerary[t]
            matching = [c for c in plant.cells if c.cell_id == cid]
            if not matching or not matching[0].polytope.contains(xt):
                return False

    # Reject any proper divisor period d | k (including d == 1 fixed point!)
    for d in range(1, k):
        if k % d == 0:
            if qvec_eq(orbit.points[0], orbit.points[d]):
                return False
    return True


def check_non_fixed_point_certs(certs: Sequence[NonFixedPointCellCert]) -> bool:
    """Standalone zero-trust checker for Farkas certificates proving F_i(x) != x on C_i."""
    for c in certs:
        if not check_farkas_certificate(c.aug_A, c.aug_b, c.infeas_cert.y):
            return False
    return True


def _canonical_orbit_rotation(
    pts: Sequence[QVec], itin: Sequence[int]
) -> Tuple[Tuple[QVec, ...], Tuple[int, ...]]:
    """Rotate periodic orbit so lexicographically smallest rational state comes first."""
    k = len(pts)

    def key_fn(idx: int) -> Tuple[Tuple[float, ...], Tuple[int, ...], int]:
        qv = pts[idx]
        # Exact pairwise rational comparison below; float tuple only coarse pre-key
        return (tuple(v / qv[1] for v in qv[0]), qv[0], qv[1])

    best = 0
    for idx in range(1, k):
        # Exact lexicographic comparison of QVecs pts[idx] vs pts[best]
        p_a = pts[idx]
        p_b = pts[best]
        less = False
        for q in range(len(p_a[0])):
            lhs = p_a[0][q] * p_b[1]
            rhs = p_b[0][q] * p_a[1]
            if lhs != rhs:
                less = lhs < rhs
                break
        if less:
            best = idx
    rot_pts = tuple(pts[(best + t) % k] for t in range(k))
    rot_itin = tuple(itin[(best + t) % k] for t in range(k))
    return (rot_pts, rot_itin)


def _is_primitive_itinerary(seq: Tuple[int, ...]) -> bool:
    k = len(seq)
    for d in range(1, k):
        if k % d == 0:
            if all(seq[i] == seq[i % d] for i in range(k)):
                return False
    return True


def detect_periodic_orbits_and_fixed_points(
    plant: Union[PWAFunction, ReLUNetwork],
    domain: Optional[HPolytope] = None,
    max_period: int = 4,
    check_global_rn_uniqueness: bool = True,
    max_cells: int = 2048,
    max_cycles: int = 4000,
    deadline: Optional[float] = None,
) -> OrbitDetectionResult:
    """Build exact Bareiss Phase-I cell-transition digraph G = (V, E), find all isolated &
    singular-segment fixed points x = F(x), verify global R^n fixed-point uniqueness (when
    plant is a ReLUNetwork), and algebraically extract exact rational Period-k limit cycles.
    """
    try:
        if isinstance(plant, ReLUNetwork):
            n = plant.in_dim
            if domain is None:
                domain = HPolytope.symmetric_box(n, (2, 1))
            pwa = plant.enumerate_pwa(
                domain, merge_bft=True, max_cells=max_cells, deadline=deadline
            )
        else:
            n = plant.in_dim
            domain = plant.domain if domain is None else domain
            pwa = plant

        cells = pwa.cells
        num_c = len(cells)

        # Step 1: Build exact Bareiss Phase-I one-step transition digraph C_i -> C_j
        adj: Dict[int, List[int]] = {c.cell_id: [] for c in cells}
        edges: List[Tuple[int, int]] = []
        cell_by_id: Dict[int, PWACell] = {c.cell_id: c for c in cells}

        for ci in cells:
            fm = ci.affine_map
            for cj in cells:
                check_budget(deadline=deadline)
                # Preimage C_i cap F_i^{-1}(C_j): A_i x <= b_i AND (A_j W_i) x <= d_i b_j - A_j c_i
                pre_A: List[List[int]] = [list(r) for r in ci.polytope.A]
                pre_b: List[int] = list(ci.polytope.b)
                for r_idx in range(len(cj.polytope.A)):
                    aj = cj.polytope.A[r_idx]
                    bj = cj.polytope.b[r_idx]
                    row = [
                        sum(aj[q] * fm.W[q][col] for q in range(n))
                        for col in range(n)
                    ]
                    rhs = fm.denom * bj - sum(aj[q] * fm.c[q] for q in range(n))
                    pre_A.append(row)
                    pre_b.append(rhs)
                res_e = solve_lp_bareiss(pre_A, pre_b, c=None, deadline=deadline)
                if res_e.status == LPStatus.BUDGET_EXCEEDED:
                    raise BudgetExceededError("LP budget exceeded building transition digraph")
                if res_e.status != LPStatus.INFEASIBLE:
                    adj[ci.cell_id].append(cj.cell_id)
                    edges.append((ci.cell_id, cj.cell_id))

        # Step 2: Complete Fixed-Point Finder (isolated + singular fixed-point segments)
        fixed_records: List[FixedPointRecord] = []
        non_fixed_certs: List[NonFixedPointCellCert] = []

        def find_fixed_on_pwa(
            target_pwa: PWAFunction, emit_certs: bool
        ) -> Tuple[List[FixedPointRecord], List[NonFixedPointCellCert]]:
            recs: List[FixedPointRecord] = []
            nf_certs: List[NonFixedPointCellCert] = []
            for ci in target_pwa.cells:
                check_budget(deadline=deadline)
                fm = ci.affine_map
                # Augmented LP for {x in C_i : (d_i I - W_i) x == c_i}
                aug_A: List[List[int]] = [list(r) for r in ci.polytope.A]
                aug_b: List[int] = list(ci.polytope.b)
                for row_i in range(n):
                    eq_row = [
                        (fm.denom if col == row_i else 0) - fm.W[row_i][col]
                        for col in range(n)
                    ]
                    aug_A.append(eq_row)
                    aug_b.append(fm.c[row_i])
                    aug_A.append([-v for v in eq_row])
                    aug_b.append(-fm.c[row_i])

                res_fp = solve_lp_bareiss(aug_A, aug_b, c=None, deadline=deadline)
                if res_fp.status == LPStatus.BUDGET_EXCEEDED:
                    raise BudgetExceededError("LP budget exceeded in fixed-point solver")
                if res_fp.status == LPStatus.INFEASIBLE:
                    if emit_certs and res_fp.farkas_infeas is not None:
                        nfc = NonFixedPointCellCert(
                            cell_id=ci.cell_id,
                            aug_A=tuple(tuple(r) for r in aug_A),
                            aug_b=tuple(aug_b),
                            infeas_cert=res_fp.farkas_infeas,
                        )
                        assert check_farkas_certificate(nfc.aug_A, nfc.aug_b, nfc.infeas_cert.y)
                        nf_certs.append(nfc)
                    continue

                # Solve algebraic structure via Bareiss echelon
                st_lin, x_part, ns_basis = solve_one_minus_m_q(
                    fm.W, fm.denom, fm.c, fm.denom, deadline=deadline
                )
                stab = _classify_discrete_jacobian(fm)
                if st_lin == "UNIQUE" and x_part is not None:
                    if ci.polytope.contains(x_part):
                        # Deduplicate if already discovered on adjacent cell boundary
                        merged = False
                        for idx_r, ex in enumerate(recs):
                            if ex.is_isolated and qvec_eq(ex.point, x_part):
                                recs[idx_r] = FixedPointRecord(
                                    point=ex.point,
                                    cell_ids=tuple(sorted(set(ex.cell_ids + (ci.cell_id,)))),
                                    is_isolated=True,
                                    stability=ex.stability
                                    if ex.stability == stab
                                    else ex.stability,
                                    jacobian_W=ex.jacobian_W,
                                    jacobian_denom=ex.jacobian_denom,
                                )
                                merged = True
                                break
                        if not merged:
                            recs.append(
                                FixedPointRecord(
                                    point=x_part,
                                    cell_ids=(ci.cell_id,),
                                    is_isolated=True,
                                    stability=stab,
                                    jacobian_W=fm.W,
                                    jacobian_denom=fm.denom,
                                )
                            )
                elif st_lin == "AFFINE_SUBSPACE" and res_fp.x_opt is not None:
                    # Check if fixed-point set in C_i is a singleton (boundary corner) or non-trivial segment
                    p_first = res_fp.x_opt
                    p_other: Optional[QVec] = None
                    for v_dir in ns_basis:
                        r_hi = solve_lp_bareiss(aug_A, aug_b, c=v_dir, deadline=deadline)
                        r_lo = solve_lp_bareiss(
                            aug_A, aug_b, c=[-v for v in v_dir], deadline=deadline
                        )
                        if r_hi.status == LPStatus.OPTIMAL and r_hi.x_opt is not None:
                            if not qvec_eq(r_hi.x_opt, p_first):
                                p_other = r_hi.x_opt
                                break
                        if r_lo.status == LPStatus.OPTIMAL and r_lo.x_opt is not None:
                            if not qvec_eq(r_lo.x_opt, p_first):
                                p_other = r_lo.x_opt
                                break
                        if r_hi.status == LPStatus.UNBOUNDED and r_hi.unbounded_ray is not None:
                            p_other = qvec_add(p_first, r_hi.unbounded_ray)
                            break
                    recs.append(
                        FixedPointRecord(
                            point=p_first,
                            cell_ids=(ci.cell_id,),
                            is_isolated=(p_other is None),
                            segment_other_end=p_other,
                            stability="MARGINAL_CONTINUUM" if p_other is not None else stab,
                            jacobian_W=fm.W,
                            jacobian_denom=fm.denom,
                        )
                    )
            return (recs, nf_certs)

        fixed_records, non_fixed_certs = find_fixed_on_pwa(pwa, emit_certs=True)

        # Check global R^n uniqueness of origin fixed point when plant is a ReLUNetwork!
        x_zero = (tuple(0 for _ in range(n)), 1)
        unique_origin_rn = False
        origin_exp_stable = False

        for fp in fixed_records:
            if fp.is_isolated and qvec_eq(fp.point, x_zero):
                if fp.stability in ("SUPER_STABLE", "LOCALLY_STABLE"):
                    # Verify every cell actually containing origin is locally contracting
                    origin_cells_ok = True
                    for ci in cells:
                        if ci.polytope.contains(x_zero):
                            if _classify_discrete_jacobian(ci.affine_map) not in (
                                "SUPER_STABLE",
                                "LOCALLY_STABLE",
                            ):
                                origin_cells_ok = False
                                break
                    origin_exp_stable = origin_cells_ok

        if check_global_rn_uniqueness and isinstance(plant, ReLUNetwork):
            unbounded_rn = HPolytope(dim=n, A=(), b=())
            rn_pwa = plant.enumerate_pwa(
                unbounded_rn, merge_bft=True, max_cells=max_cells, deadline=deadline
            )
            rn_fps, rn_nf_certs = find_fixed_on_pwa(rn_pwa, emit_certs=True)
            if (
                len(rn_fps) == 1
                and rn_fps[0].is_isolated
                and qvec_eq(rn_fps[0].point, x_zero)
            ):
                unique_origin_rn = True
                non_fixed_certs = rn_nf_certs
        else:
            if (
                len(fixed_records) == 1
                and fixed_records[0].is_isolated
                and qvec_eq(fixed_records[0].point, x_zero)
            ):
                unique_origin_rn = True

        # Step 3: Enumerate primitive closed cell walks of length 2 <= k <= max_period in G = (V, E)
        # Prioritize elementary cycles first, then general closed walks up to max_cycles
        candidate_itineraries: List[Tuple[int, ...]] = []
        seen_canon_itin: Set[Tuple[int, ...]] = set()

        sorted_cids = sorted(adj.keys())

        def add_itinerary(seq: Tuple[int, ...]) -> None:
            if not _is_primitive_itinerary(seq):
                return
            k = len(seq)
            canon = min(seq[r:] + seq[:r] for r in range(k))
            if canon not in seen_canon_itin:
                seen_canon_itin.add(canon)
                candidate_itineraries.append(canon)

        # Bounded DFS for closed walks from start_cid (only exploring nodes >= start_cid so canonical min is start_cid!)
        for start_cid in sorted_cids:
            if len(candidate_itineraries) >= max_cycles:
                break
            stack: List[Tuple[int, Tuple[int, ...]]] = [(start_cid, (start_cid,))]
            while stack:
                check_budget(deadline=deadline)
                if len(candidate_itineraries) >= max_cycles:
                    break
                curr, path = stack.pop()
                depth = len(path)
                for nxt in adj[curr]:
                    if nxt < start_cid:
                        continue
                    if nxt == start_cid and depth >= 2:
                        add_itinerary(path)
                    if depth < max_period:
                        # Prefer elementary cycles; allow revisits only when max_period <= 4 and graph small
                        if nxt not in path or num_c <= 16:
                            stack.append((nxt, path + (nxt,)))

        # Step 4: Solve unrolled algebraic linear system (I - M_gamma) x^(0) = c_gamma along each itinerary
        orbits: List[PeriodicOrbitWitness] = []
        cycles_checked = 0

        for itin in candidate_itineraries:
            check_budget(deadline=deadline)
            cycles_checked += 1
            k = len(itin)
            comp = AffineMapQ.identity(n)
            for cid in itin:
                comp = cell_by_id[cid].affine_map.compose(comp)

            st_cyc, x0_part, ns_cyc = solve_one_minus_m_q(
                comp.W, comp.denom, comp.c, comp.denom, deadline=deadline
            )
            cand_x0s: List[QVec] = []
            if st_cyc == "UNIQUE" and x0_part is not None:
                cand_x0s.append(x0_part)
            elif st_cyc == "AFFINE_SUBSPACE":
                # Intersect fixed-point subspace of F^k with full k-step itinerary polytope!
                step_A: List[List[int]] = []
                step_b: List[int] = []
                prefix = AffineMapQ.identity(n)
                for t in range(k):
                    ct = cell_by_id[itin[t]]
                    for r_idx in range(len(ct.polytope.A)):
                        ar = ct.polytope.A[r_idx]
                        br = ct.polytope.b[r_idx]
                        row = [
                            sum(ar[q] * prefix.W[q][col] for q in range(n))
                            for col in range(n)
                        ]
                        rhs = prefix.denom * br - sum(
                            ar[q] * prefix.c[q] for q in range(n)
                        )
                        step_A.append(row)
                        step_b.append(rhs)
                    prefix = ct.affine_map.compose(prefix)
                for row_i in range(n):
                    eq_r = [
                        (comp.denom if col == row_i else 0) - comp.W[row_i][col]
                        for col in range(n)
                    ]
                    step_A.append(eq_r)
                    step_b.append(comp.c[row_i])
                    step_A.append([-v for v in eq_r])
                    step_b.append(-comp.c[row_i])
                res_sub = solve_lp_bareiss(step_A, step_b, c=None, deadline=deadline)
                if res_sub.status == LPStatus.OPTIMAL and res_sub.x_opt is not None:
                    cand_x0s.append(res_sub.x_opt)

            for x0 in cand_x0s:
                # Roll forward along itinerary and verify cell membership + exact closure
                pts: List[QVec] = []
                cur_x = x0
                valid_roll = True
                for t in range(k):
                    ct = cell_by_id[itin[t]]
                    if not ct.polytope.contains(cur_x):
                        valid_roll = False
                        break
                    pts.append(cur_x)
                    cur_x = ct.affine_map.eval_qvec(cur_x)
                if not valid_roll or not qvec_eq(cur_x, x0):
                    continue

                # Reject proper divisor period d | k
                is_divisor = False
                for d in range(1, k):
                    if k % d == 0 and qvec_eq(pts[0], pts[d]):
                        is_divisor = True
                        break
                if is_divisor:
                    continue

                canon_pts, canon_itin = _canonical_orbit_rotation(pts, itin)
                # Recompute cycle Jacobian in canonical order
                canon_comp = AffineMapQ.identity(n)
                for cid in canon_itin:
                    canon_comp = cell_by_id[cid].affine_map.compose(canon_comp)

                # Deduplicate identical periodic orbits across boundary-adjacent itineraries
                dup = False
                for ex_orb in orbits:
                    if ex_orb.period == k and all(
                        qvec_eq(ex_orb.points[t], canon_pts[t]) for t in range(k)
                    ):
                        dup = True
                        break
                if dup:
                    continue

                orb_wit = PeriodicOrbitWitness(
                    period=k,
                    points=canon_pts,
                    cell_itinerary=canon_itin,
                    stability=_classify_discrete_jacobian(canon_comp),
                    cycle_jacobian_W=canon_comp.W,
                    cycle_jacobian_denom=canon_comp.denom,
                )
                assert check_periodic_orbit_witness(orb_wit, plant)
                orbits.append(orb_wit)

        verdict = (
            Verdict.FALSIFIED_COUNTEREXAMPLE
            if len(orbits) > 0
            else Verdict.CERTIFIED_EXACT
        )
        return OrbitDetectionResult(
            verdict=verdict,
            num_cells=num_c,
            transition_edges=tuple(edges),
            fixed_points=tuple(fixed_records),
            periodic_orbits=tuple(orbits),
            unique_origin_fixed_point_rn=unique_origin_rn,
            origin_locally_exp_stable=origin_exp_stable,
            non_fixed_cell_certs=tuple(non_fixed_certs),
            cycles_checked=cycles_checked,
            message=(
                f"Detected {len(orbits)} minimal periodic orbit(s) and {len(fixed_records)} fixed point(s) "
                f"across {num_c} cells ({cycles_checked} itineraries checked)"
            ),
        )

    except BudgetExceededError as exc:
        return OrbitDetectionResult(
            verdict=Verdict.OPEN_BUDGET_EXCEEDED,
            num_cells=0,
            transition_edges=(),
            fixed_points=(),
            periodic_orbits=(),
            unique_origin_fixed_point_rn=False,
            origin_locally_exp_stable=False,
            message=str(exc),
        )
