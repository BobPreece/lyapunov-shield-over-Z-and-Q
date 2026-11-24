from __future__ import annotations

import dataclasses
import time
from typing import List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import (
    FarkasBoundCert,
    FarkasInfeasibilityCert,
    LPStatus,
    check_farkas_certificate,
    check_positivity_farkas,
    check_primal_witness,
    solve_lp_bareiss,
)
from .barrier_verifier import ControlAffinePWAPlant, MultiFacetBarrier
from .polyhedral_complex import AffineMapQ, HPolytope, PWAFunction, ReLUNetwork
from .z_linalg import (
    QVec,
    Rat,
    Verdict,
    lcm_list,
    lcm_pair,
    q_add,
    q_le,
    q_lt,
    q_mul,
    q_norm,
    qvec_add,
    qvec_l1,
    qvec_norm,
    qvec_scale,
    qvec_sub,
)


@dataclasses.dataclass(frozen=True)
class ShieldStepResult:
    """Result of one online exact-Q Runtime CBF-LP Safety Shield step."""

    verdict: Verdict
    u_safe: Optional[QVec]
    u_nominal: QVec
    l1_intervention: Optional[Rat]
    intervened: bool
    active_barrier_indices: Tuple[int, ...]
    lp_A: Tuple[Tuple[int, ...], ...] = ()
    lp_b: Tuple[int, ...] = ()
    opt_cert: Optional[FarkasBoundCert] = None
    deadlock_cert: Optional[FarkasInfeasibilityCert] = None
    pivots: int = 0
    solve_time_us: float = 0.0


def check_shield_step_result(res: ShieldStepResult) -> bool:
    """Standalone zero-trust pure-integer checker for Runtime CBF-LP Shield output."""
    if res.verdict == Verdict.CERTIFIED_EXACT:
        if res.u_safe is None or res.l1_intervention is None:
            return False
        diff_l1 = qvec_l1(qvec_sub(res.u_safe, res.u_nominal))
        if diff_l1 != res.l1_intervention:
            return False
        if not res.intervened:
            return res.l1_intervention == (0, 1)
        if res.opt_cert is None or len(res.lp_A) == 0:
            return False
        # Reconstruct lifted (u, t) primal vector and verify primal + dual optimality!
        un, ud = res.u_safe
        nom_n, nom_d = res.u_nominal
        cd = lcm_pair(ud, nom_d)
        su = cd // ud
        sn = cd // nom_d
        m = len(un)
        u_scaled = [un[j] * su for j in range(m)]
        t_scaled = [abs(un[j] * su - nom_n[j] * sn) for j in range(m)]
        ut_q = qvec_norm(u_scaled + t_scaled, cd)
        if not check_primal_witness(res.lp_A, res.lp_b, ut_q):
            return False
        return check_positivity_farkas(res.lp_A, res.lp_b, res.opt_cert)

    if res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE:
        if res.deadlock_cert is None or len(res.lp_A) == 0:
            return False
        return check_farkas_certificate(res.lp_A, res.lp_b, res.deadlock_cert.y)

    return False


class RuntimeCBFShield:
    """Sub-millisecond exact-Q online CBF-LP safety filter minimizing ||u - u_nom||_1
    subject to stacked active/near-active Bouligand barrier constraints and A_u u <= c_u.
    """

    def __init__(
        self,
        barrier: Union[MultiFacetBarrier, PWAFunction, ReLUNetwork],
        plant: ControlAffinePWAPlant,
        alpha_gain: Rat = (1, 1),
        collar_margin: Optional[Rat] = None,
    ) -> None:
        self.plant = plant
        self.alpha_gain = q_norm(alpha_gain[0], alpha_gain[1])
        self.collar_margin = (
            q_norm(collar_margin[0], collar_margin[1])
            if collar_margin is not None
            else None
        )
        if isinstance(barrier, ReLUNetwork):
            self.barrier: Union[MultiFacetBarrier, PWAFunction] = (
                barrier.enumerate_pwa(plant.domain)
            )
        else:
            self.barrier = barrier

    def eval_barrier(self, x_q: QVec) -> Rat:
        if isinstance(self.barrier, MultiFacetBarrier):
            return self.barrier.eval_scalar(x_q)
        return self.barrier.eval_scalar(x_q)

    def _solve_branch_cbf_lp(
        self,
        x_q: QVec,
        u_nom: QVec,
        facets: Sequence[Tuple[int, AffineMapQ]],
        cell_crease_constraints: Sequence[Tuple[Sequence[int], int]],
        f_val: QVec,
        B_num: Sequence[Sequence[int]],
        B_denom: int,
    ) -> ShieldStepResult:
        t0 = time.perf_counter()
        n = self.plant.dim
        m = self.plant.control_dim
        fn, fd = f_val
        un_nom, ud_nom = u_nom
        an_gain, ad_gain = self.alpha_gain

        # Build linear inequalities on u in R^m: cbf_A * u <= cbf_b
        cbf_A: List[List[int]] = []
        cbf_b: List[int] = []
        active_ids: List[int] = []

        for f_idx, h_aff in facets:
            h_val = h_aff.eval_scalar(x_q)
            if self.collar_margin is not None and not q_le(h_val, self.collar_margin):
                continue
            active_ids.append(f_idx)
            # Need: grad_h . (f(x) + B u) + alpha * h(x) >= 0
            # grad_h = W_h[0] / d_h
            # -(W_h[0]^T B_num) / (d_h * B_denom) * u <= (W_h[0]^T fn)/(d_h * fd) + (an_gain * hn)/(ad_gain * hd)
            d_h = h_aff.denom
            wh = h_aff.W[0]
            hn, hd = h_val
            cd = lcm_list([d_h * B_denom, d_h * fd, ad_gain * hd])
            s_ub = cd // (d_h * B_denom)
            s_lf = cd // (d_h * fd)
            s_ah = cd // (ad_gain * hd)

            row_u = [
                -s_ub * sum(wh[i] * B_num[i][col] for i in range(n))
                for col in range(m)
            ]
            rhs_u = s_lf * sum(wh[i] * fn[i] for i in range(n)) + s_ah * (
                an_gain * hn
            )
            cbf_A.append(row_u)
            cbf_b.append(rhs_u)

        # Include active cell-crease tangent constraints when x_q lies on a cell crease!
        xn, xd = x_q
        cd_cr = lcm_pair(fd, B_denom)
        sb_cr = cd_cr // B_denom
        sf_cr = cd_cr // fd
        for a_cr, b_cr in cell_crease_constraints:
            if sum(a_cr[i] * xn[i] for i in range(n)) == b_cr * xd:
                # a_cr^T (fn / fd + (B_num / B_denom) u) <= 0
                row_cr = [
                    sb_cr * sum(a_cr[i] * B_num[i][col] for i in range(n))
                    for col in range(m)
                ]
                rhs_cr = -sf_cr * sum(a_cr[i] * fn[i] for i in range(n))
                cbf_A.append(row_cr)
                cbf_b.append(rhs_cr)

        if self.plant.u_poly is not None:
            for r_u in range(len(self.plant.u_poly.A)):
                cbf_A.append(list(self.plant.u_poly.A[r_u]))
                cbf_b.append(self.plant.u_poly.b[r_u])

        # Fast-path zero-pivot check: test if u_nom is already safe & within saturation!
        if check_primal_witness(cbf_A, cbf_b, u_nom):
            lp_A_fast: List[Tuple[int, ...]] = []
            lp_b_fast: List[int] = []
            y_fast: List[int] = []
            for j in range(m):
                r_pos = [0] * (2 * m)
                r_pos[j] = ud_nom
                r_pos[m + j] = -ud_nom
                lp_A_fast.append(tuple(r_pos))
                lp_b_fast.append(un_nom[j])
                y_fast.append(1)

                r_neg = [0] * (2 * m)
                r_neg[j] = -ud_nom
                r_neg[m + j] = -ud_nom
                lp_A_fast.append(tuple(r_neg))
                lp_b_fast.append(-un_nom[j])
                y_fast.append(1)
            for idx_c in range(len(cbf_A)):
                lp_A_fast.append(tuple(list(cbf_A[idx_c]) + [0] * m))
                lp_b_fast.append(cbf_b[idx_c])
                y_fast.append(0)
            fast_cert = FarkasBoundCert(
                y=tuple(y_fast),
                scale=2 * ud_nom,
                c_num=tuple([0] * m + [-1] * m),
                c_den=1,
                rhs_num=0,
                rhs_den=1,
                is_vacuous=False,
            )
            dt_us = (time.perf_counter() - t0) * 1e6
            return ShieldStepResult(
                verdict=Verdict.CERTIFIED_EXACT,
                u_safe=u_nom,
                u_nominal=u_nom,
                l1_intervention=(0, 1),
                intervened=False,
                active_barrier_indices=tuple(active_ids),
                lp_A=tuple(lp_A_fast),
                lp_b=tuple(lp_b_fast),
                opt_cert=fast_cert,
                pivots=0,
                solve_time_us=dt_us,
            )

        # Solve L1 Manhattan intervention LP in variables (u_0..u_{m-1}, t_0..t_{m-1}):
        # Minimize sum_j t_j  <=>  Maximize -sum_j t_j
        # Subject to:
        #   ud_nom * u_j - ud_nom * t_j <= un_nom[j]
        #  -ud_nom * u_j - ud_nom * t_j <= -un_nom[j]
        #   cbf_A * u <= cbf_b
        lp_A: List[List[int]] = []
        lp_b: List[int] = []
        for j in range(m):
            r_pos = [0] * (2 * m)
            r_pos[j] = ud_nom
            r_pos[m + j] = -ud_nom
            lp_A.append(r_pos)
            lp_b.append(un_nom[j])

            r_neg = [0] * (2 * m)
            r_neg[j] = -ud_nom
            r_neg[m + j] = -ud_nom
            lp_A.append(r_neg)
            lp_b.append(-un_nom[j])

        for idx_c in range(len(cbf_A)):
            lp_A.append(list(cbf_A[idx_c]) + [0] * m)
            lp_b.append(cbf_b[idx_c])

        c_max = [0] * m + [-1] * m
        res_lp = solve_lp_bareiss(lp_A, lp_b, c=c_max, max_pivots=1000)
        dt_us = (time.perf_counter() - t0) * 1e6

        if res_lp.status == LPStatus.BUDGET_EXCEEDED:
            return ShieldStepResult(
                verdict=Verdict.OPEN_BUDGET_EXCEEDED,
                u_safe=None,
                u_nominal=u_nom,
                l1_intervention=None,
                intervened=False,
                active_barrier_indices=tuple(active_ids),
                pivots=res_lp.pivots,
                solve_time_us=dt_us,
            )

        if res_lp.status == LPStatus.INFEASIBLE:
            assert res_lp.farkas_infeas is not None
            # Also extract pure (u)-space Farkas infeasibility certificate on cbf_A u <= cbf_b!
            res_u_inf = solve_lp_bareiss(cbf_A, cbf_b, c=None, max_pivots=1000)
            assert res_u_inf.status == LPStatus.INFEASIBLE and res_u_inf.farkas_infeas is not None
            return ShieldStepResult(
                verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
                u_safe=None,
                u_nominal=u_nom,
                l1_intervention=None,
                intervened=False,
                active_barrier_indices=tuple(active_ids),
                lp_A=tuple(tuple(r) for r in cbf_A),
                lp_b=tuple(cbf_b),
                deadlock_cert=res_u_inf.farkas_infeas,
                pivots=res_lp.pivots + res_u_inf.pivots,
                solve_time_us=dt_us,
            )

        assert (
            res_lp.status == LPStatus.OPTIMAL
            and res_lp.x_opt is not None
            and res_lp.obj_val is not None
            and res_lp.dual_y is not None
        )
        ut_n, ut_d = res_lp.x_opt
        u_safe = qvec_norm(ut_n[:m], ut_d)
        l1_int = q_norm(-res_lp.obj_val[0], res_lp.obj_val[1])
        opt_cert = FarkasBoundCert(
            y=res_lp.dual_y,
            scale=res_lp.dual_scale,
            c_num=tuple(c_max),
            c_den=1,
            rhs_num=res_lp.obj_val[0],
            rhs_den=res_lp.obj_val[1],
            is_vacuous=False,
        )
        out = ShieldStepResult(
            verdict=Verdict.CERTIFIED_EXACT,
            u_safe=u_safe,
            u_nominal=u_nom,
            l1_intervention=l1_int,
            intervened=(l1_int != (0, 1)),
            active_barrier_indices=tuple(active_ids),
            lp_A=tuple(tuple(r) for r in lp_A),
            lp_b=tuple(lp_b),
            opt_cert=opt_cert,
            pivots=res_lp.pivots,
            solve_time_us=dt_us,
        )
        assert check_shield_step_result(out)
        return out

    def step(self, x_q: QVec, u_nominal: QVec) -> ShieldStepResult:
        """Filter nominal action u_nominal at rational state x_q in exact Q."""
        p_cell = None
        for pc in self.plant.cells:
            if pc.polytope.contains(x_q):
                p_cell = pc
                break
        if p_cell is None:
            p_cell = self.plant.cells[0]
        f_val = p_cell.f_map.eval_qvec(x_q)

        if isinstance(self.barrier, MultiFacetBarrier):
            facets = [
                (idx, fh) for idx, fh in enumerate(self.barrier.facets)
            ]
            return self._solve_branch_cbf_lp(
                x_q,
                u_nominal,
                facets,
                (),
                f_val,
                p_cell.B_num,
                p_cell.B_denom,
            )

        # PWA barrier (supports both convex & non-convex ReLU/obstacle-exterior barriers!):
        # Evaluate all cells containing x_q with h_c(x_q) >= 0 (or max h_c(x_q) if slightly outside),
        # and choose the valid branch minimizing Manhattan intervention ||u - u_nom||_1!
        dom_pairs = {
            (self.plant.domain.A[i], self.plant.domain.b[i])
            for i in range(len(self.plant.domain.A))
        }
        cand_cells = [
            hc for hc in self.barrier.cells if hc.polytope.contains(x_q)
        ]
        if not cand_cells:
            cand_cells = [self.barrier.cells[0]]

        valid_cands = [
            hc for hc in cand_cells if hc.affine_map.eval_scalar(x_q)[0] >= 0
        ]
        if not valid_cands:
            valid_cands = cand_cells

        best_res: Optional[ShieldStepResult] = None
        for hc in valid_cands:
            creases = [
                (hc.polytope.A[i], hc.polytope.b[i])
                for i in range(len(hc.polytope.A))
                if (hc.polytope.A[i], hc.polytope.b[i]) not in dom_pairs
            ]
            res = self._solve_branch_cbf_lp(
                x_q,
                u_nominal,
                [(hc.cell_id, hc.affine_map)],
                creases,
                f_val,
                p_cell.B_num,
                p_cell.B_denom,
            )
            if res.verdict == Verdict.CERTIFIED_EXACT:
                if (
                    best_res is None
                    or best_res.verdict != Verdict.CERTIFIED_EXACT
                    or (
                        res.l1_intervention is not None
                        and best_res.l1_intervention is not None
                        and q_lt(res.l1_intervention, best_res.l1_intervention)
                    )
                ):
                    best_res = res
            elif best_res is None:
                best_res = res

        assert best_res is not None
        return best_res
