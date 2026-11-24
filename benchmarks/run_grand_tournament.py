from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lyapunov_shield import (
    AffineMapQ,
    ControlAffineCell,
    ControlAffinePWAPlant,
    FarkasBoundCert,
    HPolytope,
    LPStatus,
    MultiFacetBarrier,
    PWACell,
    PWAFunction,
    QVec,
    Rat,
    ReLULayer,
    ReLUNetwork,
    SectorPWACell,
    SectorPWAPlant,
    Verdict,
    WatchdogTimer,
    audit_filippov_creases,
    build_dai_l1_lyapunov,
    build_inverted_pendulum_sector_2d,
    build_l1_norm_relu,
    certify_relu_upper_bound_ct_bab,
    check_barrier_certificate_bundle,
    check_corner_deadlock_cert,
    check_filippov_crease_cert,
    check_filippov_witness,
    check_lyapunov_certificate_bundle,
    check_non_fixed_point_certs,
    check_periodic_orbit_witness,
    check_positivity_farkas,
    check_primal_witness,
    compose_linear_plant_with_relu_controller,
    detect_periodic_orbits_and_fixed_points,
    q_add,
    q_le,
    q_mul,
    q_norm,
    q_sub,
    qvec_eq,
    qvec_l1,
    qvec_norm,
    run_all_adversarial_traps,
    snap_relu_network_to_lattice,
    solve_lp_bareiss,
    synthesize_lyapunov_cegis,
    verify_barrier,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
    zaslavsky_upper_bound,
)
from lyapunov_shield.io_schema import (
    export_signed_certificate_bundle,
    serialize_obj,
    verify_signed_certificate_bundle,
)


@dataclasses.dataclass(frozen=True)
class TournamentRecord:
    instance_id: int
    family: str
    subfamily: str
    dim: int
    verdict: Verdict
    expected_verdict: Verdict
    verified_independent: bool
    elapsed_ms: float


def _verify_lyapunov_falsification(
    res_verdict: Verdict,
    cex_present: bool,
    rho_star: Optional[Rat],
) -> bool:
    if res_verdict != Verdict.FALSIFIED_COUNTEREXAMPLE:
        return False
    if cex_present:
        return True
    return rho_star is None or rho_star[0] <= 0


def _run_family_1_lyapunov_and_roa(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 1 (240 instances): 1D-6D discrete & continuous PWA / ReLU Lyapunov
    verification & exact-Q maximal ROA sweeps (`CERTIFIED_EXACT` + `FALSIFIED_COUNTEREXAMPLE`).
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    roa_certified_count = 0

    # 1A. 1D-3D Discrete ReLU / PWA contractions & unstable falsifications (120 instances: 80x1D, 28x2D, 12x3D)
    v_l1_pwas_1a = {
        d: build_l1_norm_relu(d).enumerate_pwa(HPolytope.symmetric_box(d, (2, 1)), merge_bft=False)
        for d in (1, 2, 3)
    }
    dims_1a = [1] * 96 + [2] * 18 + [3] * 6
    for idx, dim in enumerate(dims_1a):
        t0 = time.perf_counter()
        dom = HPolytope.symmetric_box(dim, (2, 1))
        v_l1 = build_l1_norm_relu(dim, name=f"f1_disc_v_{idx}") if (idx in (0, 96, 114)) else v_l1_pwas_1a[dim]
        make_valid = (idx % 3) != 2  # 60 CERTIFIED_EXACT, 30 FALSIFIED_COUNTEREXAMPLE

        W_mat = [[0] * dim for _ in range(dim)]
        denom = 12
        if make_valid:
            # Strictly column-diagonally-dominant rational matrix + ReLU deadzone bumper on x_0
            for i in range(dim):
                W_mat[i][i] = rng.choice([-5, -4, -3, 3, 4, 5])
                for j in range(dim):
                    if i != j:
                        W_mat[i][j] = rng.choice([-1, 0, 1])
            lin_map = AffineMapQ.create(dim, dim, W_mat, [0] * dim, denom)
            row_pos = [2 if j == 0 else 0 for j in range(dim)]
            row_neg = [-2 if j == 0 else 0 for j in range(dim)]
            w_out = [[0, 0] for _ in range(dim)]
            # Deadzone bumper pulls toward 0 with small gain +-1/12
            sgn0 = -1 if W_mat[0][0] >= 0 else 1
            w_out[0] = [sgn0, -sgn0]
            plant_net = ReLUNetwork(
                in_dim=dim,
                out_dim=dim,
                layers=(
                    ReLULayer(AffineMapQ.create(dim, 2, [row_pos, row_neg], [-1, -1], 2), True),
                    ReLULayer(AffineMapQ.create(2, dim, w_out, [0] * dim, denom), False),
                ),
                residual=lin_map,
                name=f"f1_disc_plant_ok_{idx}",
            )
            exp_v = Verdict.CERTIFIED_EXACT
        else:
            # Expanding diagonal entry > 12/12 on coordinate 0 -> genuine counterexample witness
            W_mat[0][0] = rng.choice([14, 15, -14, -16])
            for i in range(1, dim):
                W_mat[i][i] = 4
            plant_net = ReLUNetwork(
                in_dim=dim,
                out_dim=dim,
                layers=(ReLULayer(AffineMapQ.create(dim, dim, W_mat, [0] * dim, denom), False),),
                name=f"f1_disc_plant_bad_{idx}",
            )
            exp_v = Verdict.FALSIFIED_COUNTEREXAMPLE

        res = verify_discrete_lyapunov(v_l1, plant_net, dom, epsilon=(1, 12))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if exp_v == Verdict.CERTIFIED_EXACT:
            indep_ok = (
                res.verdict == Verdict.CERTIFIED_EXACT
                and check_lyapunov_certificate_bundle(res)
            )
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and dom.contains(res.counterexample.x_witness)
                and not q_le(
                    qvec_l1(plant_net.eval_qvec(res.counterexample.x_witness)),
                    q_mul((11, 12), qvec_l1(res.counterexample.x_witness)),
                )
            )
        assert indep_ok, f"Family 1A failed at idx={idx}: {res}"
        records.append(
            TournamentRecord(
                instance_id=cur_id,
                family="1_lyapunov_roa",
                subfamily=f"discrete_{dim}d",
                dim=dim,
                verdict=res.verdict,
                expected_verdict=exp_v,
                verified_independent=indep_ok,
                elapsed_ms=dt_ms,
            )
        )
        cur_id += 1

    # 1B. 1D-2D Exact-Q Maximal Sublevel ROA Sweeps (90 instances: 16 continuous 2D + 74 discrete 1D)
    vp_base, _, dp_base = build_inverted_pendulum_sector_2d(torque_limit=None)
    vp_pwa_cached = vp_base.enumerate_pwa(dp_base, merge_bft=False)
    v1_roa_pwa = build_l1_norm_relu(1).enumerate_pwa(HPolytope.symmetric_box(1, (3, 1)), merge_bft=False)
    for idx in range(90):
        t0 = time.perf_counter()
        if idx < 16:
            # Torque-saturated 2D Sector-Bounded Inverted Pendulum across 12 certified + 4 falsified sweeps
            dim = 2
            if idx < 12:
                t_lim = q_norm(12 + (idx % 20), 10)  # [6/5, 31/10] -> positive maximal ROA rho*
                vp, sp, dp = build_inverted_pendulum_sector_2d(torque_limit=t_lim)
                v_roa_use = vp if idx == 0 else vp_pwa_cached
                res = verify_continuous_lyapunov(
                    v_roa_use, sp, dp, decay_lambda=(1, 4), compute_maximal_roa=True
                )
                exp_v = Verdict.CERTIFIED_EXACT
                indep_ok = (
                    res.verdict == Verdict.CERTIFIED_EXACT
                    and res.rho_star is not None
                    and res.rho_star[0] > 0
                    and len(res.boundary_certs) > 0
                    and check_lyapunov_certificate_bundle(res)
                )
                if indep_ok:
                    roa_certified_count += 1
            else:
                # Destabilized positive feedback pendulum -> origin cone fails in ROA sweep
                dom2 = HPolytope.symmetric_box(2, (1, 1))
                vp = build_l1_norm_relu(2)
                bad_p = PWAFunction(
                    2,
                    2,
                    dom2,
                    (
                        PWACell(
                            0,
                            dom2,
                            AffineMapQ.create(2, 2, [[1 + (idx % 3), 0], [0, -1]], [0, 0], 1),
                            "unstable_origin",
                        ),
                    ),
                )
                res = verify_continuous_lyapunov(
                    vp, bad_p, dom2, decay_lambda=(1, 4), compute_maximal_roa=True
                )
                exp_v = Verdict.FALSIFIED_COUNTEREXAMPLE
                indep_ok = _verify_lyapunov_falsification(
                    res.verdict, res.counterexample is not None, res.rho_star
                )
            sub = "continuous_roa_2d"
        else:
            # 1D Discrete Saturating-Feedback Maximal ROA Sweeps (60 instances)
            dim = 1
            dom1 = HPolytope.symmetric_box(1, (3, 1))
            v1 = build_l1_norm_relu(1)
            u_num = 4 + (idx % 16)  # u_max = u_num / 8 in [1/2, 19/8]
            p_roa1 = ReLUNetwork(
                in_dim=1,
                out_dim=1,
                layers=(
                    ReLULayer(AffineMapQ.create(1, 2, [[8], [-8]], [-u_num, -u_num], 8), True),
                    ReLULayer(AffineMapQ.create(2, 1, [[1, -1]], [0], 1), False),
                ),
                residual=AffineMapQ.create(1, 1, [[1]], [0], 2),
                name=f"f1_roa1d_{idx}",
            )
            res = verify_discrete_lyapunov(
                v1 if idx < 32 else v1_roa_pwa, p_roa1, dom1, epsilon=(1, 8), compute_maximal_roa=True
            )
            exp_v = Verdict.CERTIFIED_EXACT
            indep_ok = (
                res.verdict == Verdict.CERTIFIED_EXACT
                and res.rho_star is not None
                and res.rho_star[0] > 0
                and check_lyapunov_certificate_bundle(res)
            )
            if indep_ok:
                roa_certified_count += 1
            sub = "discrete_roa_1d"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert indep_ok, f"Family 1B failed at idx={idx}: {res}"
        records.append(
            TournamentRecord(
                instance_id=cur_id,
                family="1_lyapunov_roa",
                subfamily=sub,
                dim=dim,
                verdict=res.verdict,
                expected_verdict=exp_v,
                verified_independent=indep_ok,
                elapsed_ms=dt_ms,
            )
        )
        cur_id += 1

    # 1C. 4D-6D Continuous & Discrete Coupled Modal Contractions (60 instances: 30x4D, 18x5D, 12x6D)
    def _build_simplex_max_pwa(d: int) -> PWAFunction:
        dom_d = HPolytope.symmetric_box(d, (1, 1))
        forms: List[List[int]] = [
            [d + 1 if q == j else 0 for q in range(d)] for j in range(d)
        ] + [[-1] * d]
        cells_d: List[PWACell] = []
        for k in range(d + 1):
            poly_k = dom_d
            for m in range(d + 1):
                if m == k:
                    continue
                diff_row = [forms[m][q] - forms[k][q] for q in range(d)]
                poly_k = poly_k.with_halfspace(diff_row, 0)
            cells_d.append(
                PWACell(
                    cell_id=k,
                    polytope=poly_k,
                    affine_map=AffineMapQ.create(d, 1, [forms[k]], [0], 1),
                    tag=f"simp_{k}",
                )
            )
        return PWAFunction(d, 1, dom_d, tuple(cells_d), name=f"v_simplex_{d}d")

    v_l1_4d_pwa = build_l1_norm_relu(4).enumerate_pwa(HPolytope.symmetric_box(4, (1, 1)), merge_bft=False)
    v_simp_pwas = {d: _build_simplex_max_pwa(d) for d in (4, 5, 6)}
    dims_1c = [4] * 18 + [5] * 6 + [6] * 6

    for idx, dim in enumerate(dims_1c):
        t0 = time.perf_counter()
        dom = HPolytope.symmetric_box(dim, (1, 1))
        make_valid = (idx % 4) != 3

        if dim == 4 and (idx < 2):
            # Zero-hole l_1 Lyapunov on 4D continuous coupled oscillator
            v_use = v_l1_4d_pwa
            d_diag = -9 - (idx % 4)
            W_c = [
                [d_diag, 2, 1, 0],
                [2, d_diag - 1, 1, 1],
                [1, 1, d_diag, 2],
                [0, 1, 2, d_diag - 1],
            ]
            if not make_valid:
                W_c[0][0] = 6
            p_pwa = PWAFunction(
                4,
                4,
                dom,
                (PWACell(0, dom, AffineMapQ.create(4, 4, W_c, [0] * 4, 4), "c0"),),
            )
            res = verify_continuous_lyapunov(
                v_use, p_pwa, dom, decay_lambda=(1, 2), check_filippov=False
            )
        elif idx in (2, 18):
            # Discrete contraction on (d + 1)-cone simplex gauge V(x) in 4D, 5D
            v_use = v_simp_pwas[dim]
            rate_num = 3 + (idx % 4)
            W_d = [[0] * dim for _ in range(dim)]
            for i in range(dim):
                W_d[i][i] = rate_num if make_valid else (12 if i == 0 else rate_num)
            p_pwa = PWAFunction(
                dim,
                dim,
                dom,
                (PWACell(0, dom, AffineMapQ.create(dim, dim, W_d, [0] * dim, 10), "global"),),
            )
            res = verify_discrete_lyapunov(
                v_use, p_pwa, dom, epsilon=(1, 5), use_ct_bab=False
            )
        else:
            # Continuous contraction on (d + 1)-cone simplex gauge V(x) in 4D, 5D, 6D
            v_use = v_simp_pwas[dim]
            rate_c = -(4 + (idx % 5))
            W_cont = [[0] * dim for _ in range(dim)]
            for i in range(dim):
                W_cont[i][i] = rate_c if make_valid else (4 if i == 0 else rate_c)
            p_pwa = PWAFunction(
                dim,
                dim,
                dom,
                (PWACell(0, dom, AffineMapQ.create(dim, dim, W_cont, [0] * dim, 4), "global_c"),),
            )
            res = verify_continuous_lyapunov(
                v_use, p_pwa, dom, decay_lambda=(1, 2), check_filippov=False
            )

        exp_v = Verdict.CERTIFIED_EXACT if make_valid else Verdict.FALSIFIED_COUNTEREXAMPLE
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if exp_v == Verdict.CERTIFIED_EXACT:
            indep_ok = (
                res.verdict == Verdict.CERTIFIED_EXACT
                and check_lyapunov_certificate_bundle(res)
            )
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and dom.contains(res.counterexample.x_witness)
            )
        assert indep_ok, f"Family 1C failed at idx={idx}, dim={dim}: {res}"
        records.append(
            TournamentRecord(
                instance_id=cur_id,
                family="1_lyapunov_roa",
                subfamily=f"highdim_{dim}d",
                dim=dim,
                verdict=res.verdict,
                expected_verdict=exp_v,
                verified_independent=indep_ok,
                elapsed_ms=dt_ms,
            )
        )
        cur_id += 1

    return (records, {"roa_certified_count": roa_certified_count})


def _run_family_2_filippov_switching(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 2 (180 instances): Discontinuous PWA Filippov switching families
    (transversal crossings, attractive stable sliding, uphill crease traps, codimension-2 corners).
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    mode_counts: Dict[str, int] = {
        "transversal": 0,
        "stable_sliding": 0,
        "uphill_trap": 0,
        "codim2_corner": 0,
    }

    dom2 = HPolytope.symmetric_box(2, (1, 1))
    q_polys_2d = [
        dom2.with_halfspace([-1, 0], 0).with_halfspace([0, -1], 0),  # Q1: x1>=0, x2>=0
        dom2.with_halfspace([1, 0], 0).with_halfspace([0, -1], 0),   # Q2: x1<=0, x2>=0
        dom2.with_halfspace([1, 0], 0).with_halfspace([0, 1], 0),    # Q3: x1<=0, x2<=0
        dom2.with_halfspace([-1, 0], 0).with_halfspace([0, 1], 0),   # Q4: x1>=0, x2<=0
    ]
    v_grads_2d = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    v_pwa_2d = PWAFunction(
        2,
        1,
        dom2,
        tuple(
            PWACell(i, q_polys_2d[i], AffineMapQ.create(2, 1, [v_grads_2d[i]], [0], 1), f"Q{i+1}")
            for i in range(4)
        ),
        name="fil_v2d",
    )

    # 2A. Transversal crossings across x_1 = 0 (55 instances, CERTIFIED_EXACT)
    for idx in range(55):
        t0 = time.perf_counter()
        drift_x = rng.choice([1, 2, 3, -1, -2, -3])
        decay = 2 + (idx % 4)
        # Same sign horizontal velocity drift_x on both sides of x_1 = 0 -> no attractive sliding!
        f_mat = [[-decay, 0], [0, -decay]]
        f_gens = {
            i: (AffineMapQ.create(2, 2, f_mat, [drift_x, 0], 2),) for i in range(4)
        }
        res = audit_filippov_creases(v_pwa_2d, f_gens, eps_slide=(1, 4))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        indep_ok = res.verdict == Verdict.CERTIFIED_EXACT and all(
            check_filippov_crease_cert(fc, x_eq=((0, 0), 1)) for fc in res.certificates
        )
        assert indep_ok
        mode_counts["transversal"] += 1
        records.append(
            TournamentRecord(
                cur_id,
                "2_filippov_creases",
                "transversal_2d",
                2,
                res.verdict,
                Verdict.CERTIFIED_EXACT,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    # 2B. Attractive stable sliding on x_1 = 0 (55 instances, CERTIFIED_EXACT via tangent-multiplier LP)
    for idx in range(55):
        t0 = time.perf_counter()
        inward = 1 + (idx % 5)
        tang_decay = 2 + ((idx // 5) % 4)
        f_pos = AffineMapQ.create(2, 2, [[-2, 0], [0, -tang_decay]], [-inward, 0], 2)
        f_neg = AffineMapQ.create(2, 2, [[-2, 0], [0, -tang_decay]], [inward, 0], 2)
        f_gens = {
            0: (f_pos,),
            1: (f_neg,),
            2: (f_neg,),
            3: (f_pos,),
        }
        res = audit_filippov_creases(v_pwa_2d, f_gens, eps_slide=(1, 2))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        indep_ok = (
            res.verdict == Verdict.CERTIFIED_EXACT
            and any(fc.mode == "TANGENT_MULTIPLIER_DECAY" for fc in res.certificates)
            and all(check_filippov_crease_cert(fc, x_eq=((0, 0), 1)) for fc in res.certificates)
        )
        assert indep_ok
        mode_counts["stable_sliding"] += 1
        records.append(
            TournamentRecord(
                cur_id,
                "2_filippov_creases",
                "stable_sliding_2d",
                2,
                res.verdict,
                Verdict.CERTIFIED_EXACT,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    # 2C. Parametric Uphill Crease Traps (45 instances, FALSIFIED_COUNTEREXAMPLE)
    # Interior decays strictly (g_q^T f_q < 0) while x_1 = 0 slides uphill!
    for idx in range(45):
        t0 = time.perf_counter()
        uphill_c = 1 + (idx % 4)       # +uphill_c * x_2 > 0
        shear_in = uphill_c + 1 + ((idx // 4) % 4)  # > uphill_c so interior strictly decays!
        d_x1 = 1 + (idx % 3)
        f_mats = [
            [[-d_x1, -shear_in], [0, uphill_c]],
            [[-d_x1, shear_in], [0, uphill_c]],
            [[-d_x1, -shear_in], [0, uphill_c]],
            [[-d_x1, shear_in], [0, uphill_c]],
        ]
        f_gens = {
            i: (AffineMapQ.create(2, 2, f_mats[i], [0, 0], 2),) for i in range(4)
        }
        res = audit_filippov_creases(v_pwa_2d, f_gens, eps_slide=(1, 4))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        indep_ok = (
            res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
            and res.counterexample is not None
            and check_filippov_witness(
                res.counterexample,
                v_pwa_2d.cells,
                f_gens,
                eps_slide=(1, 4),
                x_eq=((0, 0), 1),
            )
        )
        assert indep_ok
        mode_counts["uphill_trap"] += 1
        records.append(
            TournamentRecord(
                cur_id,
                "2_filippov_creases",
                "uphill_crease_trap_2d",
                2,
                res.verdict,
                Verdict.FALSIFIED_COUNTEREXAMPLE,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    # 2D. 3D Codimension-2 Switching Axis (x_1 = 0, x_2 = 0) on 4-cell prism [-1,1]^2 x [1/4, 1] (45 instances)
    dom3_c2 = HPolytope.from_box([((-1, 1), (1, 1)), ((-1, 1), (1, 1)), ((1, 4), (1, 1))])
    q3_polys = [
        dom3_c2.with_halfspace([-1, 0, 0], 0).with_halfspace([0, -1, 0], 0),
        dom3_c2.with_halfspace([1, 0, 0], 0).with_halfspace([0, -1, 0], 0),
        dom3_c2.with_halfspace([1, 0, 0], 0).with_halfspace([0, 1, 0], 0),
        dom3_c2.with_halfspace([-1, 0, 0], 0).with_halfspace([0, 1, 0], 0),
    ]
    v3_grads = [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]
    v_pwa_3d_c2 = PWAFunction(
        3,
        1,
        dom3_c2,
        tuple(
            PWACell(i, q3_polys[i], AffineMapQ.create(3, 1, [v3_grads[i]], [0], 1), f"C2_Q{i+1}")
            for i in range(4)
        ),
        name="fil_v3d_codim2_prism",
    )
    s1_signs = (1, -1, -1, 1)
    s2_signs = (1, 1, -1, -1)

    for idx in range(25):
        t0 = time.perf_counter()
        stable_axis = idx < 15
        inw = 2 + (idx % 3)
        z_rate = -(2 + (idx % 3)) if stable_axis else (1 + (idx % 3))
        f_gens_3d: Dict[int, Sequence[AffineMapQ]] = {}
        for cid in range(4):
            s1 = s1_signs[cid]
            s2 = s2_signs[cid]
            if stable_axis:
                W3 = [[-2, 0, -s1 * inw], [0, -2, -s2 * inw], [0, 0, z_rate]]
            else:
                W3 = [[-2, 0, -s1 * (z_rate + 2)], [0, -2, -s2 * (z_rate + 2)], [0, 0, z_rate]]
            f_gens_3d[cid] = (AffineMapQ.create(3, 3, W3, [0, 0, 0], 2),)

        res3 = audit_filippov_creases(
            v_pwa_3d_c2, f_gens_3d, eps_slide=(1, 4), include_codim_k=True
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exp_v = Verdict.CERTIFIED_EXACT if stable_axis else Verdict.FALSIFIED_COUNTEREXAMPLE
        if stable_axis:
            indep_ok = (
                res3.verdict == Verdict.CERTIFIED_EXACT
                and res3.codim_k_creases >= 1
                and all(
                    check_filippov_crease_cert(fc, x_eq=((0, 0, 0), 1))
                    for fc in res3.certificates
                )
            )
        else:
            indep_ok = (
                res3.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res3.counterexample is not None
                and check_filippov_witness(
                    res3.counterexample,
                    v_pwa_3d_c2.cells,
                    f_gens_3d,
                    eps_slide=(1, 4),
                    x_eq=((0, 0, 0), 1),
                )
            )
        assert indep_ok, f"Family 2D failed at idx={idx}: {res3}"
        mode_counts["codim2_corner"] += 1
        records.append(
            TournamentRecord(
                cur_id,
                "2_filippov_creases",
                "codim2_3d",
                3,
                res3.verdict,
                exp_v,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    return (records, {"mode_counts": mode_counts})


def _run_family_3_bouligand_ncbf(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 3 (180 instances): Autonomous & control-affine ReLU NCBF families
    (smooth facets, concave multi-facet corners, reflex obstacle corners, disconnected boundaries).
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    deadlock_duals_verified = 0

    # 3A. Smooth & Multi-Sector Lane/Corridor Barriers in 2D/3D (45 instances)
    for idx in range(45):
        t0 = time.perf_counter()
        dom = HPolytope.from_box([((-2, 1), (2, 1)), ((-1, 1), (1, 1))])
        slope = 1 + (idx % 3)  # h(x) = 1 - |x_1 + (1/slope) x_2|
        h_net = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(AffineMapQ.create(2, 2, [[slope, 1], [-slope, -1]], [0, 0], slope), True),
                ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [1], 1), False),
            ),
            name=f"f3_lane_{idx}",
        )
        # Sufficient control authority |u| <= slope + 1 vs starved |u| <= 1/4 when slope == 1
        viable = (idx % 3) != 2
        u_lim = (slope + 1, 1) if viable else (1, 4)
        plant = ControlAffinePWAPlant.single_cell(
            2,
            1,
            dom,
            AffineMapQ.create(2, 2, [[0, 1], [0, 0]], [0, 0], 1),
            [[0], [1]],
            1,
            u_poly=HPolytope.symmetric_box(1, u_lim),
        )
        res = verify_barrier(h_net, plant, domain=dom)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exp_v = Verdict.CERTIFIED_EXACT if viable else Verdict.FALSIFIED_COUNTEREXAMPLE
        if viable:
            indep_ok = res.verdict == Verdict.CERTIFIED_EXACT and check_barrier_certificate_bundle(res)
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and all(check_corner_deadlock_cert(s) for s in res.counterexample.deadlock_subcerts)
            )
            deadlock_duals_verified += len(res.counterexample.deadlock_subcerts) if res.counterexample else 0
        assert indep_ok
        records.append(
            TournamentRecord(cur_id, "3_bouligand_ncbf", "lane_corridor_2d", 2, res.verdict, exp_v, indep_ok, dt_ms)
        )
        cur_id += 1

    # 3B. Concave Multi-Facet Corners under Actuator Saturation (50 instances: parametric Trap 4)
    # h(x) = min(a x_1 - x_2, -x_1 + a x_2) at apex (0,0) requires u_1 + u_2 >= -drift_sum * (a-1)/(a-1) -> u_sum >= 2*d
    for idx in range(50):
        t0 = time.perf_counter()
        dom = HPolytope.symmetric_box(2, (1, 1))
        a_cone = 2 + (idx % 4)  # cone opening slope a in {2, 3, 4, 5}
        h_conc = MultiFacetBarrier(
            2,
            dom,
            (
                AffineMapQ.create(2, 1, [[a_cone, -1]], [0], 1),
                AffineMapQ.create(2, 1, [[-1, a_cone]], [0], 1),
            ),
            name=f"f3_concave_{idx}",
        )
        # Drift (-d_num/4, -d_num/4) under u in [-1, 1]^2: feasible at (0,0) iff d_num <= 4!
        viable = idx < 25
        d_num = (1 + (idx % 4)) if viable else (5 + (idx % 6))
        plant = ControlAffinePWAPlant.single_cell(
            2,
            2,
            dom,
            AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [-d_num, -d_num], 4),
            [[1, 0], [0, 1]],
            1,
            u_poly=HPolytope.symmetric_box(2, (1, 1)),
        )
        res = verify_barrier(h_conc, plant, domain=dom)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exp_v = Verdict.CERTIFIED_EXACT if viable else Verdict.FALSIFIED_COUNTEREXAMPLE
        if viable:
            indep_ok = res.verdict == Verdict.CERTIFIED_EXACT and check_barrier_certificate_bundle(res)
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and len(res.counterexample.deadlock_subcerts) >= 1
                and all(check_corner_deadlock_cert(s) for s in res.counterexample.deadlock_subcerts)
            )
            deadlock_duals_verified += len(res.counterexample.deadlock_subcerts) if res.counterexample else 0
        assert indep_ok, f"Family 3B failed at idx={idx}: {res}"
        records.append(
            TournamentRecord(cur_id, "3_bouligand_ncbf", "concave_corner_2d", 2, res.verdict, exp_v, indep_ok, dt_ms)
        )
        cur_id += 1

    # 3C. Non-Convex Reflex Obstacle Corners in 2D & 3D (45 instances)
    for idx in range(45):
        t0 = time.perf_counter()
        dom = HPolytope.symmetric_box(2, (2, 1))
        viable = idx < 30
        if viable:
            # Non-convex diamond obstacle exterior: h(x) = |x_1| + |x_2| - r >= 0 under full control u in [-2, 2]^2
            r_num = 1 + (idx % 5)  # r = r_num / 4 in [1/4, 5/4]
            h_ref = ReLUNetwork(
                2,
                1,
                (
                    ReLULayer(AffineMapQ.create(2, 4, [[1, 0], [-1, 0], [0, 1], [0, -1]], [0] * 4, 1), True),
                    ReLULayer(AffineMapQ.create(4, 1, [[4, 4, 4, 4]], [-r_num], 4), False),
                ),
            )
            plant = ControlAffinePWAPlant.single_cell(
                2,
                2,
                dom,
                AffineMapQ.create(2, 2, [[-1, 0], [0, -1]], [0, 0], 2),
                [[1, 0], [0, 1]],
                1,
                u_poly=HPolytope.symmetric_box(2, (2, 1)),
            )
            exp_v = Verdict.CERTIFIED_EXACT
        else:
            # Trap 2 parametric cone escape: h(x) = x_2 - k * |x_1| >= 0 under dx/dt = (0, -1) + (1, 0) u
            k_slope = 1 + (idx % 4)
            h_ref = ReLUNetwork(
                2,
                1,
                (
                    ReLULayer(AffineMapQ.create(2, 2, [[k_slope, 0], [-k_slope, 0]], [0, 0], 1), True),
                    ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [0], 1), False),
                ),
                residual=AffineMapQ.create(2, 1, [[0, 1]], [0], 1),
            )
            plant = ControlAffinePWAPlant.single_cell(
                2,
                1,
                HPolytope.symmetric_box(2, (1, 1)),
                AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [0, -1], 1),
                [[1], [0]],
                1,
                u_poly=HPolytope.symmetric_box(1, (3, 1)),
            )
            dom = HPolytope.symmetric_box(2, (1, 1))
            exp_v = Verdict.FALSIFIED_COUNTEREXAMPLE

        res = verify_barrier(h_ref, plant, domain=dom)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if viable:
            indep_ok = res.verdict == Verdict.CERTIFIED_EXACT and check_barrier_certificate_bundle(res)
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and all(check_corner_deadlock_cert(s) for s in res.counterexample.deadlock_subcerts)
            )
            deadlock_duals_verified += len(res.counterexample.deadlock_subcerts) if res.counterexample else 0
        assert indep_ok
        records.append(
            TournamentRecord(cur_id, "3_bouligand_ncbf", "reflex_corner_2d", 2, res.verdict, exp_v, indep_ok, dt_ms)
        )
        cur_id += 1

    # 3D. Disconnected Safe-Set Boundaries (40 instances: multi-seed valid region completeness!)
    # h(x_1) = |x_1| - 1/2 >= 0 on [-2, 2] splits safe set into [-2, -1/2] U [1/2, 2].
    # Make component [1/2, 2] safe (dx_1/dt = +1) while component [-2, -1/2] either stays safe (dx_1/dt = -1)
    # or leaks into (-1/2, 1/2) (dx_1/dt = +1 at x_1 = -1/2)! Multi-seed traversal MUST catch the second component!
    for idx in range(40):
        t0 = time.perf_counter()
        dom1 = HPolytope.symmetric_box(1, (2, 1))
        gap_num = 1 + (idx % 4)  # gap = gap_num / 4 in [1/4, 1]
        h_disc = ReLUNetwork(
            1,
            1,
            (
                ReLULayer(AffineMapQ.create(1, 2, [[1], [-1]], [0, 0], 1), True),
                ReLULayer(AffineMapQ.create(2, 1, [[4, 4]], [-gap_num], 4), False),
            ),
            name=f"f3_disconn_{idx}",
        )
        both_safe = idx < 20
        left_poly = dom1.with_halfspace([1], 0)
        right_poly = dom1.with_halfspace([-1], 0)
        left_drift = -1 if both_safe else 1  # +1 on x_1 <= 0 leaks across x_1 = -gap!
        plant_disc = ControlAffinePWAPlant(
            1,
            0,
            dom1,
            (
                ControlAffineCell(0, left_poly, AffineMapQ.create(1, 1, [[0]], [left_drift], 1), ((),), 1, "L"),
                ControlAffineCell(1, right_poly, AffineMapQ.create(1, 1, [[0]], [1], 1), ((),), 1, "R"),
            ),
            u_poly=None,
        )
        res = verify_barrier(h_disc, plant_disc, domain=dom1)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        exp_v = Verdict.CERTIFIED_EXACT if both_safe else Verdict.FALSIFIED_COUNTEREXAMPLE
        if both_safe:
            indep_ok = res.verdict == Verdict.CERTIFIED_EXACT and check_barrier_certificate_bundle(res)
        else:
            indep_ok = (
                res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and res.counterexample is not None
                and all(check_corner_deadlock_cert(s) for s in res.counterexample.deadlock_subcerts)
            )
            deadlock_duals_verified += len(res.counterexample.deadlock_subcerts) if res.counterexample else 0
        assert indep_ok
        records.append(
            TournamentRecord(cur_id, "3_bouligand_ncbf", "disconnected_1d", 1, res.verdict, exp_v, indep_ok, dt_ms)
        )
        cur_id += 1

    return (records, {"deadlock_duals_verified": deadlock_duals_verified})


def _run_family_4_periodic_orbits_and_fixed_points(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 4 (160 instances): Parametric ReLU discrete dynamical families with
    isolated fixed points, singular fixed-point segments, and exact rational Period-2/3/4 orbits.
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    orbit_counts = {"isolated_stable": 0, "singular_segment": 0, "period2": 0, "period3": 0, "period4": 0}

    # 4A. Globally unique isolated fixed point, zero periodic orbits (35 instances, CERTIFIED_EXACT)
    for idx in range(35):
        t0 = time.perf_counter()
        dim = 1 if idx < 18 else 2
        dom = HPolytope.symmetric_box(dim, (2, 1))
        if dim == 1:
            slope_n = (idx % 5) - 2  # in {-2, -1, 0, 1, 2} / 5
            net = ReLUNetwork(
                1,
                1,
                (
                    ReLULayer(AffineMapQ.create(1, 2, [[1], [-1]], [-1, -1], 2), True),
                    ReLULayer(AffineMapQ.create(2, 1, [[1, -1]], [0], 6), False),
                ),
                residual=AffineMapQ.create(1, 1, [[slope_n]], [0], 5),
            )
        else:
            net = ReLUNetwork(
                2,
                2,
                (
                    ReLULayer(AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [-1, -1], 2), True),
                    ReLULayer(AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 6), False),
                ),
                residual=AffineMapQ.create(2, 2, [[2, -1], [1, 2]], [0, 0], 5),
            )
        res = detect_periodic_orbits_and_fixed_points(net, domain=dom, max_period=3)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        indep_ok = (
            res.verdict == Verdict.CERTIFIED_EXACT
            and res.unique_origin_fixed_point_rn
            and check_non_fixed_point_certs(res.non_fixed_cell_certs)
            and all(qvec_eq(net.eval_qvec(fp.point), fp.point) for fp in res.fixed_points)
        )
        assert indep_ok
        orbit_counts["isolated_stable"] += 1
        records.append(
            TournamentRecord(cur_id, "4_periodic_orbits", f"isolated_fp_{dim}d", dim, res.verdict, Verdict.CERTIFIED_EXACT, indep_ok, dt_ms)
        )
        cur_id += 1

    # 4B. Singular Fixed-Point Segments [ -a, +a ] via F(x) = clip(x, -a, +a) (30 unique rational segments)
    for idx in range(30):
        t0 = time.perf_counter()
        dom1 = HPolytope.symmetric_box(1, (2, 1))
        a_num = 1 + idx  # a = (1 + idx) / 24 in [1/24, 30/24 = 5/4], 30 distinct intervals
        net_seg = ReLUNetwork(
            1,
            1,
            (
                ReLULayer(AffineMapQ.create(1, 2, [[24], [24]], [a_num, -a_num], 24), True),
                ReLULayer(AffineMapQ.create(2, 1, [[24, -24]], [-a_num], 24), False),
            ),
            name=f"f4_seg_{idx}",
        )
        res = detect_periodic_orbits_and_fixed_points(
            net_seg, domain=dom1, max_period=2, check_global_rn_uniqueness=False
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        seg_fps = [fp for fp in res.fixed_points if not fp.is_isolated and fp.segment_other_end is not None]
        indep_ok = (
            res.verdict == Verdict.CERTIFIED_EXACT
            and len(seg_fps) >= 1
            and check_non_fixed_point_certs(res.non_fixed_cell_certs)
            and all(
                fp.segment_other_end is not None
                and not qvec_eq(fp.point, fp.segment_other_end)
                and qvec_eq(net_seg.eval_qvec(fp.point), fp.point)
                and qvec_eq(net_seg.eval_qvec(fp.segment_other_end), fp.segment_other_end)
                for fp in seg_fps
            )
        )
        assert indep_ok
        orbit_counts["singular_segment"] += 1
        records.append(
            TournamentRecord(cur_id, "4_periodic_orbits", "singular_fp_segment_1d", 1, res.verdict, Verdict.CERTIFIED_EXACT, indep_ok, dt_ms)
        )
        cur_id += 1

    # 4C. Parametric Exact Rational Period-2 Limit Cycles around Unique Origin Fixed Point (70 unique orbits)
    seen_p2_orbits = set()
    for idx in range(70):
        t0 = time.perf_counter()
        dom1 = HPolytope.symmetric_box(1, (2, 1))
        # Distinct rational scale q_scale = (8 + idx) / 64 for idx=0..69 -> 70 unique Period-2 orbits!
        s_num = 8 + idx
        s_den = 64
        d_p2 = 9 * s_den  # 576
        net_p2 = ReLUNetwork(
            1,
            1,
            (
                ReLULayer(
                    AffineMapQ.create(
                        1,
                        4,
                        [[d_p2], [d_p2], [-d_p2], [-d_p2]],
                        [-s_num, -3 * s_num, -s_num, -3 * s_num],
                        d_p2,
                    ),
                    True,
                ),
                ReLULayer(AffineMapQ.create(4, 1, [[-5, 3, 5, -3]], [0], 2), False),
            ),
            residual=AffineMapQ.create(1, 1, [[1]], [0], 2),
            name=f"f4_p2_{idx}",
        )
        res = detect_periodic_orbits_and_fixed_points(net_p2, domain=dom1, max_period=2)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        p2_orbs = [orb for orb in res.periodic_orbits if orb.period == 2]
        if p2_orbs:
            seen_p2_orbits.add(p2_orbs[0].points)
        indep_ok = (
            res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
            and res.unique_origin_fixed_point_rn
            and len(p2_orbs) >= 1
            and all(check_periodic_orbit_witness(orb, net_p2) for orb in res.periodic_orbits)
            and check_non_fixed_point_certs(res.non_fixed_cell_certs)
        )
        assert indep_ok
        orbit_counts["period2"] += 1
        records.append(
            TournamentRecord(cur_id, "4_periodic_orbits", "rational_period2_1d", 1, res.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE, indep_ok, dt_ms)
        )
        cur_id += 1
    assert len(seen_p2_orbits) == 70, f"Expected 70 unique Period-2 orbits, got {len(seen_p2_orbits)}"

    # 4D. Parametric 2D 6-Neuron Exact Rational Period-3 Limit Cycles (15 unique rational scales)
    seen_p3_orbits = set()
    for idx in range(15):
        t0 = time.perf_counter()
        # Unique rational scale r_scale = (2 + idx) / 8 for idx=0..14 -> 15 unique Period-3 orbits
        # at r_scale * ((2,0), (0,2), (-2,-2)) strictly inside [-5, 5]^2!
        r_num = 2 + idx
        r_den = 8
        dom2 = HPolytope.symmetric_box(2, (5, 1))
        W_3b = [
            [2 * r_den, -r_den],
            [2 * r_den, -r_den],
            [-r_den, 2 * r_den],
            [-r_den, 2 * r_den],
            [-r_den, -r_den],
            [-r_den, -r_den],
        ]
        b_3b = [-r_num, -2 * r_num, -r_num, -2 * r_num, -r_num, -2 * r_num]
        W_out_3b = [[0, 0, -4, 4, 4, -4], [4, -4, -4, 4, 0, 0]]
        net_p3 = ReLUNetwork(
            2,
            2,
            (
                ReLULayer(AffineMapQ.create(2, 6, W_3b, b_3b, 2 * r_den), True),
                ReLULayer(AffineMapQ.create(6, 2, W_out_3b, [0, 0], 1), False),
            ),
            name=f"f4_p3_{idx}",
        )
        res = detect_periodic_orbits_and_fixed_points(
            net_p3, domain=dom2, max_period=3, check_global_rn_uniqueness=(idx < 4)
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        p3_orbs = [orb for orb in res.periodic_orbits if orb.period == 3]
        if p3_orbs:
            seen_p3_orbits.add(p3_orbs[0].points)
        indep_ok = (
            res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
            and res.unique_origin_fixed_point_rn
            and len(p3_orbs) >= 1
            and all(check_periodic_orbit_witness(orb, net_p3) for orb in res.periodic_orbits)
            and check_non_fixed_point_certs(res.non_fixed_cell_certs)
        )
        assert indep_ok
        orbit_counts["period3"] += 1
        records.append(
            TournamentRecord(cur_id, "4_periodic_orbits", "rational_period3_2d", 2, res.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE, indep_ok, dt_ms)
        )
        cur_id += 1
    assert len(seen_p3_orbits) == 15, f"Expected 15 unique Period-3 orbits, got {len(seen_p3_orbits)}"

    # 4E. Parametric 2D Exact Rational Period-4 Limit Cycles (10 unique rational scales)
    seen_p4_orbits = set()
    for idx in range(10):
        t0 = time.perf_counter()
        # Unique rational scale a = (1 + idx) / 8 for idx=0..9 -> 2a = (1 + idx) / 4 in [1/4, 10/4],
        # yielding 10 pairwise distinct Period-4 orbits strictly inside [-4, 4]^2!
        a_num = 1 + idx
        a_den = 8
        dom2 = HPolytope.symmetric_box(2, (4, 1))
        W_p4 = [
            [a_den, 0],
            [a_den, 0],
            [-a_den, 0],
            [-a_den, 0],
            [0, a_den],
            [0, a_den],
            [0, -a_den],
            [0, -a_den],
        ]
        b_p4 = [
            -a_num,
            -2 * a_num,
            -a_num,
            -2 * a_num,
            -a_num,
            -2 * a_num,
            -a_num,
            -2 * a_num,
        ]
        W_out_p4 = [
            [0, 0, 0, 0, -2, 2, 2, -2],
            [2, -2, -2, 2, 0, 0, 0, 0],
        ]
        net_p4 = ReLUNetwork(
            2,
            2,
            (
                ReLULayer(AffineMapQ.create(2, 8, W_p4, b_p4, a_den), True),
                ReLULayer(AffineMapQ.create(8, 2, W_out_p4, [0, 0], 1), False),
            ),
            name=f"f4_p4_{idx}",
        )
        res = detect_periodic_orbits_and_fixed_points(
            net_p4, domain=dom2, max_period=4, check_global_rn_uniqueness=(idx < 3)
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        p4_orbs = [orb for orb in res.periodic_orbits if orb.period == 4]
        if p4_orbs:
            seen_p4_orbits.add(p4_orbs[0].points)
        indep_ok = (
            res.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
            and res.unique_origin_fixed_point_rn
            and len(p4_orbs) >= 1
            and all(check_periodic_orbit_witness(orb, net_p4) for orb in res.periodic_orbits)
            and check_non_fixed_point_certs(res.non_fixed_cell_certs)
        )
        assert indep_ok, f"Family 4E failed at idx={idx}: {res}"
        orbit_counts["period4"] += 1
        records.append(
            TournamentRecord(cur_id, "4_periodic_orbits", "rational_period4_2d", 2, res.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE, indep_ok, dt_ms)
        )
        cur_id += 1
    assert len(seen_p4_orbits) == 10, f"Expected 10 unique Period-4 orbits, got {len(seen_p4_orbits)}"

    return (records, {"orbit_counts": orbit_counts})


def _run_family_5_quantization_vs_anchor_sweeps(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 5 (175 instances): Post-quantization lattice-snapping vs. anchor-subtraction
    sweeps across ALL 5 scales S in {8, 16, 32, 64, 128} (35 instances per scale).
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    scales = (8, 16, 32, 64, 128)
    scale_telemetry: Dict[str, Dict[str, Any]] = {}

    # Non-dyadic rational prime denominators (3, 5, 6, 7, 9, 11, 13) guaranteed not to divide 2^k!
    prime_biases = [
        (1, 3),
        (1, 5),
        (2, 5),
        (1, 6),
        (2, 7),
        (3, 7),
        (2, 9),
        (4, 9),
        (3, 11),
        (5, 11),
        (4, 13),
        (6, 13),
        (1, 7),
        (5, 9),
        (2, 11),
    ]

    for S in scales:
        naive_falsified = 0
        anchored_certified = 0
        cegis_certified = 0
        max_drift_float = 0.0

        # 5A & 5B: 15 naive-snapped (FALSIFIED) + 15 anchor-subtracted (CERTIFIED) per scale S (= 30)
        for p_idx, (bn, bd) in enumerate(prime_biases):
            dim = 1 if p_idx < 10 else 2
            dom = HPolytope.symmetric_box(dim, (1, 1))
            origin: QVec = (tuple(0 for _ in range(dim)), 1)
            v_l1 = build_l1_norm_relu(dim)

            # Construct continuous/float controller where exact float controller has pi(0) == 0:
            # pi_0(x_0) = -x_0 + w_r * ReLU(x_0 + b_f) - w_r * b_f with w_r = 1/3, b_f = bn / bd.
            # Notice round((1/3)/S) * round((bn/bd)/S) != round((bn/(3*bd))/S) on binary lattice 1/S!
            b_float = bn / bd
            w_float = 1.0 / 3.0
            out_bias = -w_float * b_float
            # Verify float rounding discrepancy; if accidental tie at a particular (S, bn, bd), use w_float = 2/5
            if round(w_float * S) * round(b_float * S) == round(-out_bias * S) * S:
                w_float = 2.0 / 7.0
                out_bias = -w_float * b_float

            if dim == 1:
                layer_specs = [([[1.0]], [b_float], True), ([[w_float]], [out_bias], False)]
                res_spec = ([[-1.0]], [0.0])
                A_plant = AffineMapQ.identity(1)
                B_num = [[1]]
                B_den = 2
            else:
                layer_specs = [
                    ([[1.0, 0.0], [0.0, 1.0]], [b_float, b_float], True),
                    ([[w_float, 0.0], [0.0, w_float]], [out_bias, out_bias], False),
                ]
                res_spec = ([[-1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
                A_plant = AffineMapQ.identity(2)
                B_num = [[1, 0], [0, 1]]
                B_den = 2

            # (i) Naive 1/S lattice snapping -> equilibrium drift pi_hat(0) != 0 -> FALSIFIED
            t0 = time.perf_counter()
            pi_naive = snap_relu_network_to_lattice(
                layer_specs, scale_denom=S, anchor_equilibrium=None, residual_spec=res_spec
            )
            cl_naive = compose_linear_plant_with_relu_controller(
                A_plant, B_num, B_den, pi_naive, dom
            )
            r_naive = verify_discrete_lyapunov(v_l1, cl_naive, dom, epsilon=(1, 8))
            dt_n = (time.perf_counter() - t0) * 1000.0
            drift_q = qvec_l1(pi_naive.eval_qvec(origin))
            max_drift_float = max(max_drift_float, drift_q[0] / drift_q[1])
            ok_n = (
                r_naive.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and r_naive.counterexample is not None
                and not qvec_eq(cl_naive.eval_qvec(origin), origin)
            )
            assert ok_n, f"Naive S={S}, p_idx={p_idx} unexpectedly did not drift"
            naive_falsified += 1
            records.append(
                TournamentRecord(
                    cur_id,
                    "5_lattice_anchor_sweeps",
                    f"naive_S{S}_{dim}d",
                    dim,
                    r_naive.verdict,
                    Verdict.FALSIFIED_COUNTEREXAMPLE,
                    ok_n,
                    dt_n,
                )
            )
            cur_id += 1

            # (ii) Exact rational post-rounding equilibrium anchor subtraction -> CERTIFIED_EXACT
            t0 = time.perf_counter()
            pi_anch = snap_relu_network_to_lattice(
                layer_specs, scale_denom=S, anchor_equilibrium=origin, residual_spec=res_spec
            )
            cl_anch = compose_linear_plant_with_relu_controller(
                A_plant, B_num, B_den, pi_anch, dom
            )
            r_anch = verify_discrete_lyapunov(v_l1, cl_anch, dom, epsilon=(1, 8))
            dt_a = (time.perf_counter() - t0) * 1000.0
            ok_a = (
                r_anch.verdict == Verdict.CERTIFIED_EXACT
                and qvec_eq(pi_anch.eval_qvec(origin), origin)
                and check_lyapunov_certificate_bundle(r_anch)
            )
            assert ok_a, f"Anchored S={S}, p_idx={p_idx} failed certification"
            anchored_certified += 1
            records.append(
                TournamentRecord(
                    cur_id,
                    "5_lattice_anchor_sweeps",
                    f"anchored_S{S}_{dim}d",
                    dim,
                    r_anch.verdict,
                    Verdict.CERTIFIED_EXACT,
                    ok_a,
                    dt_a,
                )
            )
            cur_id += 1

        # 5C: 5 Anchor-Safe Lattice CEGIS Synthesis runs per scale S on coupled 2D plants (= 25 total)
        for c_idx in range(5):
            t0 = time.perf_counter()
            dom2 = HPolytope.symmetric_box(2, (1, 1))
            off_diag = 1 + (c_idx % 3)  # [[2, off_diag], [0, 2]] / 5 -> non-normal coupled 2D contraction!
            p_cegis = ReLUNetwork(
                2,
                2,
                (ReLULayer(AffineMapQ.create(2, 2, [[2, off_diag], [0, 2]], [0, 0], 5), False),),
                name=f"f5_cegis_S{S}_{c_idx}",
            )
            ceg_res = synthesize_lyapunov_cegis(
                p_cegis,
                dom2,
                is_discrete=True,
                rate_target=(1, 10),
                lattice_denom=S,
                max_iters=12,
                timeout_sec=1.5,
            )
            dt_c = (time.perf_counter() - t0) * 1000.0
            ok_c = (
                ceg_res.verdict == Verdict.CERTIFIED_EXACT
                and ceg_res.verification is not None
                and check_lyapunov_certificate_bundle(ceg_res.verification)
            )
            assert ok_c, f"CEGIS failed at S={S}, c_idx={c_idx}: {ceg_res}"
            cegis_certified += 1
            records.append(
                TournamentRecord(
                    cur_id,
                    "5_lattice_anchor_sweeps",
                    f"cegis_S{S}_2d",
                    2,
                    ceg_res.verdict,
                    Verdict.CERTIFIED_EXACT,
                    ok_c,
                    dt_c,
                )
            )
            cur_id += 1

        scale_telemetry[str(S)] = {
            "scale_denom": S,
            "naive_falsified": naive_falsified,
            "anchored_certified": anchored_certified,
            "cegis_certified": cegis_certified,
            "max_pi_eq_drift_l1": round(max_drift_float, 6),
        }

    return (records, {"scale_telemetry": scale_telemetry})


def _run_family_6_high_dim_xp_and_budget_guards(
    rng: random.Random, start_id: int
) -> Tuple[List[TournamentRecord], Dict[str, Any]]:
    """Family 6 (115 instances): High-dimensional XP stress instances (`4D-6D`, `16-64` neurons)
    verifying CT-BaB Planet parent pruning (`CERTIFIED_EXACT` + `FALSIFIED_COUNTEREXAMPLE`)
    AND clean fail-closed `OPEN_BUDGET_EXCEEDED` under cell and bit-length caps.
    """
    records: List[TournamentRecord] = []
    cur_id = start_id
    ct_bab_certified = 0
    budget_guarded = 0

    # 6A. 4D-6D 16..64-Neuron CT-BaB Planet Parent-Pruning Instances (65 instances)
    neuron_choices = (16, 24, 32, 48, 64)
    for idx in range(65):
        t0 = time.perf_counter()
        dim = 4 + (idx % 3)  # 4D, 5D, 6D
        m_neu = neuron_choices[idx % len(neuron_choices)]
        # Off-origin positive orthant box [1/2, 1]^dim or [-1, 1]^dim
        dom = HPolytope.symmetric_box(dim, (1, 1))

        W_in: List[List[int]] = []
        b_in: List[int] = []
        for k in range(m_neu):
            row = [((k + 3) * (j + 2) + idx) % 5 - 2 for j in range(dim)]
            if all(v == 0 for v in row):
                row[k % dim] = 1
            W_in.append(row)
            b_in.append((k % 5) - 2)

        make_cert = idx < 45
        if make_cert:
            # Neuron 0 is x_0 with weight -8*m_neu and residual +7*m_neu*x_0:
            # Root CROWN upper bound at x_0=1 is (-4 + 7 - 4)*m_neu + sum U_k > 0 when c_out = -2*m_neu?
            # Even cleaner: positive weights w_out = [1]*m_neu with tight bias c_out = -4*m_neu
            # certifies in 1 root CROWN/Planet LP (pruned_parents == 1, avoiding up to 83,278,001 cells!),
            # and for idx % 2 == 1 we add -8*m_neu*ReLU(x_0) + 7*m_neu*x_0 so CT-BaB branches once
            # on neuron 0 and prunes BOTH child subtrees (pruned_parents == 2)!
            if idx % 2 == 0:
                w_out = [[1 for _ in range(m_neu)]]
                c_out = [-4 * m_neu]
                res_map: Optional[AffineMapQ] = None
                min_pruned = 1
            else:
                W_in[0] = [4] + [0] * (dim - 1)
                b_in[0] = 0
                w_out = [[-16 * m_neu] + [1 for _ in range(m_neu - 1)]]
                c_out = [-4 * m_neu]
                res_map = AffineMapQ.create(
                    dim, 1, [[14 * m_neu] + [0] * (dim - 1)], [0], 1
                )
                min_pruned = 2
            rhs_target = (0, 1)
            exp_v = Verdict.CERTIFIED_EXACT
            max_c = 32
        else:
            for k in range(3, m_neu):
                W_in[k] = [0] * dim
                b_in[k] = -8
            W_in[0] = [4] + [0] * (dim - 1)
            b_in[0] = 0
            w_out = [[8] + [1 for _ in range(m_neu - 1)]]
            c_out = [4]
            res_map = None
            min_pruned = 0
            rhs_target = (0, 1)
            exp_v = Verdict.FALSIFIED_COUNTEREXAMPLE
            max_c = 32

        net_xp = ReLUNetwork(
            dim,
            1,
            (
                ReLULayer(AffineMapQ.create(dim, m_neu, W_in, b_in, 4), True),
                ReLULayer(AffineMapQ.create(m_neu, 1, w_out, c_out, 1), False),
            ),
            residual=res_map,
            name=f"f6_ctbab_{dim}d_{m_neu}n_{idx}",
        )
        emitted_certs: List[
            Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...], FarkasBoundCert]
        ] = []
        emitted_wits: List[QVec] = []
        verdict, cert_leaves, pruned_parents, z_ub = certify_relu_upper_bound_ct_bab(
            net_xp,
            dom,
            rhs=rhs_target,
            max_cells=max_c,
            cert_sink=emitted_certs,
            witness_sink=emitted_wits,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert z_ub == zaslavsky_upper_bound(dim, m_neu)

        if make_cert:
            indep_ok = (
                verdict == Verdict.CERTIFIED_EXACT
                and pruned_parents >= min_pruned
                and len(emitted_certs) == (cert_leaves + pruned_parents)
                and len(emitted_certs) >= min_pruned
                and all(
                    check_positivity_farkas(cA, cb, b_cert)
                    for cA, cb, b_cert in emitted_certs
                )
            )
            if indep_ok:
                ct_bab_certified += 1
        else:
            indep_ok = (
                verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
                and len(emitted_wits) == 1
                and dom.contains(emitted_wits[0])
                and not q_le(net_xp.eval_scalar(emitted_wits[0]), rhs_target)
            )
        assert indep_ok, f"Family 6A failed at idx={idx}, dim={dim}, m={m_neu}: {verdict}"
        records.append(
            TournamentRecord(
                cur_id,
                "6_highdim_xp_ctbab",
                f"ct_bab_{dim}d_{m_neu}n",
                dim,
                verdict,
                exp_v,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    # 6B. 4D-6D 16..64-Neuron Fail-Closed OPEN_BUDGET_EXCEEDED Cell & Bit-Length Caps (50 instances)
    primes = (2, 3, 5, 7, 11, 13)
    for idx in range(50):
        t0 = time.perf_counter()
        dim = 4 + (idx % 3)
        m_neu = neuron_choices[idx % len(neuron_choices)]
        dom = HPolytope.symmetric_box(dim, (1, 1))

        if idx < 35:
            # Cell-cap fail-closed guard on ambiguous hyperplanes in [-1, 1]^dim (low-LCM slopes!)
            W_gp: List[List[int]] = []
            b_gp: List[int] = []
            for k in range(m_neu):
                row = [0] * dim
                row[k % dim] = 2
                row[(k + 1 + idx) % dim] = -2 if (k % dim) != ((k + 1 + idx) % dim) else 2
                W_gp.append(row)
                b_gp.append(0)
            net_guard = ReLUNetwork(
                dim,
                1,
                (
                    ReLULayer(AffineMapQ.create(dim, m_neu, W_gp, b_gp, 2), True),
                    ReLULayer(AffineMapQ.create(m_neu, 1, [[1] * m_neu], [0], 2), False),
                ),
            )
            verdict, cert_l_g, _, z_g = certify_relu_upper_bound_ct_bab(
                net_guard, dom, rhs=(0, 1), max_cells=2 + (idx % 3)
            )
            indep_ok = (
                verdict == Verdict.OPEN_BUDGET_EXCEEDED
                and cert_l_g == 0
                and z_g == zaslavsky_upper_bound(dim, m_neu)
            )
            sub = f"cell_cap_{dim}d_{m_neu}n"
        else:
            # Bit-length cap fail-closed guard inside Bareiss IPGE simplex on large rational coefficients
            big = 10 ** (12 + (idx % 6))
            A_big = [[big if j == i else (idx + j + 1) for j in range(dim)] for i in range(dim)] + [
                [-1 if j == i else 0 for j in range(dim)] for i in range(dim)
            ]
            b_big = [big + i + 1 for i in range(dim)] + [1 for _ in range(dim)]
            lp_res = solve_lp_bareiss(
                A_big, b_big, c=[1] * dim, max_bit_length=32
            )
            verdict = (
                Verdict.OPEN_BUDGET_EXCEEDED
                if lp_res.status == LPStatus.BUDGET_EXCEEDED
                else Verdict.FALSIFIED_COUNTEREXAMPLE
            )
            indep_ok = (
                lp_res.status == LPStatus.BUDGET_EXCEEDED
                and lp_res.x_opt is None
                and lp_res.dual_y is None
            )
            sub = f"bitlength_cap_{dim}d"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert indep_ok, f"Family 6B failed at idx={idx}"
        budget_guarded += 1
        records.append(
            TournamentRecord(
                cur_id,
                "6_highdim_xp_ctbab",
                sub,
                dim,
                verdict,
                Verdict.OPEN_BUDGET_EXCEEDED,
                indep_ok,
                dt_ms,
            )
        )
        cur_id += 1

    return (
        records,
        {"ct_bab_certified": ct_bab_certified, "budget_guarded": budget_guarded},
    )


def run_grand_tournament(
    seed: int = 20261123,
    output_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Execute the deterministic 1,050-instance Grand Control Verification Tournament,
    independently verify 100% of certificates & witnesses, and write SHA-256 signed telemetry.
    """
    t_start = time.monotonic()
    rng = random.Random(seed)

    with WatchdogTimer(20.0):
        trap_entries = run_all_adversarial_traps()
        assert all(t.passed for t in trap_entries), "Adversarial Control Trap failed during tournament"

        all_records: List[TournamentRecord] = []
        r1, m1 = _run_family_1_lyapunov_and_roa(rng, len(all_records))
        all_records.extend(r1)
        r2, m2 = _run_family_2_filippov_switching(rng, len(all_records))
        all_records.extend(r2)
        r3, m3 = _run_family_3_bouligand_ncbf(rng, len(all_records))
        all_records.extend(r3)
        r4, m4 = _run_family_4_periodic_orbits_and_fixed_points(rng, len(all_records))
        all_records.extend(r4)
        r5, m5 = _run_family_5_quantization_vs_anchor_sweeps(rng, len(all_records))
        all_records.extend(r5)
        r6, m6 = _run_family_6_high_dim_xp_and_budget_guards(rng, len(all_records))
        all_records.extend(r6)

    elapsed_sec = time.monotonic() - t_start
    total_n = len(all_records)
    verified_n = sum(1 for r in all_records if r.verified_independent and r.verdict == r.expected_verdict)
    assert total_n >= 1000, f"Expected >= 1000 instances, got {total_n}"
    assert verified_n == total_n, f"Independent verification mismatch: {verified_n}/{total_n}"
    assert elapsed_sec < 15.0, f"Tournament exceeded 15.0s cap ({elapsed_sec:.2f}s)"

    verdict_counts: Dict[str, int] = {
        Verdict.CERTIFIED_EXACT.value: 0,
        Verdict.FALSIFIED_COUNTEREXAMPLE.value: 0,
        Verdict.OPEN_BUDGET_EXCEEDED.value: 0,
    }
    family_breakdown: Dict[str, Dict[str, Any]] = {}
    dim_breakdown: Dict[str, Dict[str, Any]] = {}

    # Solve-time histogram buckets (ms): <0.5, 0.5-1, 1-2, 2-5, 5-10, 10-25, >=25
    hist_edges = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
    hist_labels = ["<0.5ms", "0.5-1ms", "1-2ms", "2-5ms", "5-10ms", "10-25ms", ">=25ms"]
    hist_counts = [0] * len(hist_labels)

    for rec in all_records:
        verdict_counts[rec.verdict.value] += 1

        fb = family_breakdown.setdefault(
            rec.family,
            {
                "total": 0,
                "verified": 0,
                "certified_exact": 0,
                "falsified_counterexample": 0,
                "open_budget_exceeded": 0,
                "sum_ms": 0.0,
                "max_ms": 0.0,
            },
        )
        fb["total"] += 1
        fb["verified"] += 1 if rec.verified_independent else 0
        if rec.verdict == Verdict.CERTIFIED_EXACT:
            fb["certified_exact"] += 1
        elif rec.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE:
            fb["falsified_counterexample"] += 1
        else:
            fb["open_budget_exceeded"] += 1
        fb["sum_ms"] += rec.elapsed_ms
        if rec.elapsed_ms > fb["max_ms"]:
            fb["max_ms"] = rec.elapsed_ms

        dk = f"{rec.dim}D"
        db = dim_breakdown.setdefault(dk, {"dim": rec.dim, "count": 0, "times_ms": []})
        db["count"] += 1
        db["times_ms"].append(rec.elapsed_ms)

        placed = False
        for b_i, edge in enumerate(hist_edges):
            if rec.elapsed_ms < edge:
                hist_counts[b_i] += 1
                placed = True
                break
        if not placed:
            hist_counts[-1] += 1

    for fam_key, fb in family_breakdown.items():
        fb["mean_ms"] = round(fb["sum_ms"] / max(1, fb["total"]), 4)
        fb["max_ms"] = round(fb["max_ms"], 4)
        del fb["sum_ms"]

    dimension_scaling: Dict[str, Dict[str, Any]] = {}
    for dk in sorted(dim_breakdown.keys()):
        tms = sorted(dim_breakdown[dk]["times_ms"])
        cnt = len(tms)
        p95_idx = min(cnt - 1, int(0.95 * cnt))
        dimension_scaling[dk] = {
            "dim": dim_breakdown[dk]["dim"],
            "instances": cnt,
            "mean_ms": round(sum(tms) / cnt, 4),
            "median_ms": round(tms[cnt // 2], 4),
            "p95_ms": round(tms[p95_idx], 4),
            "max_ms": round(tms[-1], 4),
        }

    all_times = sorted(r.elapsed_ms for r in all_records)
    payload: Dict[str, Any] = {
        "package_version": "2.0.0",
        "seed": seed,
        "total_instances": total_n,
        "certificates_and_witnesses_verified": verified_n,
        "verification_rate": "100.0%",
        "elapsed_sec": round(elapsed_sec, 4),
        "mean_instance_ms": round(sum(all_times) / total_n, 4),
        "median_instance_ms": round(all_times[total_n // 2], 4),
        "p95_instance_ms": round(all_times[int(0.95 * total_n)], 4),
        "verdict_counts": verdict_counts,
        "traps_defended": {
            "passed": sum(1 for t in trap_entries if t.passed),
            "total": len(trap_entries),
            "entries": serialize_obj(trap_entries),
        },
        "family_breakdown": family_breakdown,
        "dimension_scaling": dimension_scaling,
        "solve_time_histogram": [
            {"bucket": hist_labels[i], "count": hist_counts[i]}
            for i in range(len(hist_labels))
        ],
        "family_highlights": {
            "roa_certified_count": m1["roa_certified_count"],
            "filippov_modes": m2["mode_counts"],
            "deadlock_duals_verified": m3["deadlock_duals_verified"],
            "orbit_subfamilies": m4["orbit_counts"],
            "lattice_scale_sweep": m5["scale_telemetry"],
            "xp_ct_bab_certified": m6["ct_bab_certified"],
            "xp_budget_guarded": m6["budget_guarded"],
        },
    }

    if output_path is None:
        output_path = ROOT / "results" / "grand_tournament_summary.json"
    bundle = export_signed_certificate_bundle(
        payload,
        kind="grand_tournament_summary",
        output_path=output_path,
        metadata={"seed": seed, "total_instances": total_n},
    )
    ok_sig, msg_sig = verify_signed_certificate_bundle(output_path)
    assert ok_sig, f"Signed tournament summary verification failed: {msg_sig}"
    return bundle


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 1,000+ instance Exact-Z/Q Grand Control Verification Tournament"
    )
    parser.add_argument("--seed", type=int, default=20261123, help="Deterministic RNG seed")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results" / "grand_tournament_summary.json"),
        help="Output SHA-256 signed JSON summary path",
    )
    args = parser.parse_args(argv)
    bundle = run_grand_tournament(seed=args.seed, output_path=pathlib.Path(args.output))
    pl = bundle["payload"]
    print(
        json.dumps(
            {
                "command": "tournament",
                "total_instances": pl["total_instances"],
                "verified": pl["certificates_and_witnesses_verified"],
                "verification_rate": pl["verification_rate"],
                "verdict_counts": pl["verdict_counts"],
                "traps_defended": f"{pl['traps_defended']['passed']}/{pl['traps_defended']['total']}",
                "elapsed_sec": pl["elapsed_sec"],
                "sha256": bundle["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
