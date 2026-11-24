from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Dict, List, Sequence, Tuple

from lyapunov_shield import (
    AffineMapQ,
    BarrierVerificationResult,
    BarrierViabilityFaceCert,
    BoundaryLevelCert,
    CoercivityCellCert,
    CornerDeadlockSubCert,
    FarkasBoundCert,
    FarkasInfeasibilityCert,
    FilippovCreaseCert,
    FilippovSlidingWitness,
    HPolytope,
    LPStatus,
    MultiFacetBarrier,
    NonFixedPointCellCert,
    PWACell,
    PWAFunction,
    PeriodicOrbitWitness,
    QVec,
    ReLULayer,
    ReLUNetwork,
    RuntimeCBFShield,
    ShardDecayCert,
    Verdict,
    WatchdogTimer,
    audit_filippov_creases,
    build_dubins_3sector_obstacle_3d,
    build_inverted_pendulum_sector_2d,
    build_l1_norm_relu,
    build_trap1_filippov_uphill_sliding,
    build_trap3_hidden_periodic_orbits,
    build_trap4_deadlocked_concave_corner,
    build_trap7_composition_shard_blowup,
    certify_linear_upper_bound,
    check_affine_neg_farkas,
    check_affine_nonpos_farkas,
    check_barrier_certificate_bundle,
    check_coercivity_cell_cert,
    check_corner_deadlock_cert,
    check_farkas_infeasible,
    check_filippov_crease_cert,
    check_filippov_witness,
    check_lyapunov_certificate_bundle,
    check_non_fixed_point_certs,
    check_periodic_orbit_witness,
    check_positivity_farkas,
    check_shield_step_result,
    detect_periodic_orbits_and_fixed_points,
    q_add,
    q_norm,
    qvec_add,
    qvec_norm,
    solve_lp_bareiss,
    verify_barrier,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
)
from lyapunov_shield.io_schema import (
    compute_bundle_sha256,
    export_signed_certificate_bundle,
    verify_signed_certificate_bundle,
)


class TestCertificateBitFlipRedTeam(unittest.TestCase):
    """Cycle 5.1: Deterministic 1-unit / 1-sign / 1-coordinate mutation across EVERY
    certificate and witness class in `lyapunov_shield`, asserting 100% rejection.
    """

    def test_01_inhomogeneous_and_homogeneous_farkas_positivity_coercivity_tamper(self) -> None:
        """Mutate inhomogeneous & homogeneous Farkas positivity/coercivity duals:
        - y_i sign flip,
        - A^T y mismatch (+1/-1 on y_i and c_num),
        - q = 0 (scale = 0),
        - d^T y slack breach (rhs tightening & strictness breach).
        """
        with WatchdogTimer(15.0):
            # 1A. Inhomogeneous off-origin cell and homogeneous origin-cone coercivity from Trap 7
            v_oct, p_quad, dom = build_trap7_composition_shard_blowup()
            # Shift domain split to create both origin-incident cones and strictly off-origin cells
            v_shifted_layers = (
                ReLULayer(
                    AffineMapQ.create(
                        2,
                        6,
                        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]],
                        [0, 0, 0, 0, -1, -1],
                        2,
                    ),
                    use_relu=True,
                ),
                ReLULayer(AffineMapQ.create(6, 1, [[2, 2, 2, 2, 1, 1]], [0], 2), use_relu=False),
            )
            v_net = ReLUNetwork(2, 1, v_shifted_layers, name="v_inhom_hom").with_anchor_zero(
                ((0, 0), 1)
            )
            res_disc = verify_discrete_lyapunov(v_net, p_quad, dom, epsilon=(1, 4))
            self.assertEqual(res_disc.verdict, Verdict.CERTIFIED_EXACT)
            self.assertTrue(check_lyapunov_certificate_bundle(res_disc))

            x_eq: QVec = ((0, 0), 1)
            alpha_1 = res_disc.alpha_1
            assert alpha_1 is not None

            checked_origin_cone = False
            checked_off_origin = False

            for cc in res_disc.coercivity_certs:
                self.assertTrue(check_coercivity_cell_cert(cc, x_eq, alpha_1))
                if cc.strict_pos_cert is None:
                    checked_origin_cone = True
                else:
                    checked_off_origin = True

                # Mutate each of the 2n homogeneous FarkasBoundCerts inside CoercivityCellCert
                for b_idx, bc in enumerate(cc.bound_certs):
                    self.assertTrue(check_positivity_farkas(cc.cell_A, cc.cell_b, bc))
                    # (a) y_i sign flip on every positive multiplier
                    for i, yi in enumerate(bc.y):
                        if yi > 0:
                            bad_y = list(bc.y)
                            bad_y[i] = -yi
                            bad_bc = dataclasses.replace(bc, y=tuple(bad_y))
                            self.assertFalse(check_positivity_farkas(cc.cell_A, cc.cell_b, bad_bc))
                            bad_cc = dataclasses.replace(
                                cc,
                                bound_certs=cc.bound_certs[:b_idx]
                                + (bad_bc,)
                                + cc.bound_certs[b_idx + 1 :],
                            )
                            self.assertFalse(check_coercivity_cell_cert(bad_cc, x_eq, alpha_1))

                    # (b) A^T y mismatch: +1 and -1 on every entry of y, and +1/-1 on c_num
                    for i in range(len(bc.y)):
                        for delta in (1, -1):
                            bad_y = list(bc.y)
                            bad_y[i] += delta
                            bad_bc = dataclasses.replace(bc, y=tuple(bad_y))
                            self.assertFalse(check_positivity_farkas(cc.cell_A, cc.cell_b, bad_bc))
                    for j in range(len(bc.c_num)):
                        for delta in (1, -1):
                            bad_c = list(bc.c_num)
                            bad_c[j] += delta
                            bad_bc = dataclasses.replace(bc, c_num=tuple(bad_c))
                            self.assertFalse(check_positivity_farkas(cc.cell_A, cc.cell_b, bad_bc))

                    # (c) q = 0 (scale = 0 and negative scale)
                    for bad_q in (0, -1, -bc.scale):
                        bad_bc = dataclasses.replace(bc, scale=bad_q)
                        self.assertFalse(check_positivity_farkas(cc.cell_A, cc.cell_b, bad_bc))

                    # (d) d^T y slack breach: decrease rhs_num below exact dual dot product
                    y_dot_b = sum(bc.y[i] * cc.cell_b[i] for i in range(len(bc.y)))
                    # Require rhs_den * y_dot_b <= scale * rhs_num -> set rhs_num so scale * rhs_num = rhs_den * y_dot_b - 1
                    breach_rhs_num = (bc.rhs_den * y_dot_b) - 1
                    bad_bc_slack = dataclasses.replace(
                        bc, rhs_num=breach_rhs_num, rhs_den=bc.rhs_den * bc.scale
                    )
                    self.assertFalse(check_positivity_farkas(cc.cell_A, cc.cell_b, bad_bc_slack))

                # Also test alpha_target inflation (violating sum_j mu_j * alpha_1 <= 1 by 1 unit)
                bad_alpha = (alpha_1[0] + alpha_1[1] * 10, alpha_1[1])
                self.assertFalse(check_coercivity_cell_cert(cc, x_eq, bad_alpha))

                # Test inhomogeneous strict positivity certificate (-V_i(x) < 0) on off-origin cells
                if cc.strict_pos_cert is not None:
                    spc = cc.strict_pos_cert
                    self.assertTrue(check_positivity_farkas(cc.cell_A, cc.cell_b, spc, strict=True))
                    for i, yi in enumerate(spc.y):
                        if yi > 0:
                            bad_y = list(spc.y)
                            bad_y[i] = -yi
                            bad_spc = dataclasses.replace(spc, y=tuple(bad_y))
                            self.assertFalse(
                                check_positivity_farkas(cc.cell_A, cc.cell_b, bad_spc, strict=True)
                            )
                            self.assertFalse(
                                check_coercivity_cell_cert(
                                    dataclasses.replace(cc, strict_pos_cert=bad_spc), x_eq, alpha_1
                                )
                            )
                    # q = 0 on strict_pos_cert
                    self.assertFalse(
                        check_coercivity_cell_cert(
                            dataclasses.replace(cc, strict_pos_cert=dataclasses.replace(spc, scale=0)),
                            x_eq,
                            alpha_1,
                        )
                    )
                    # Slack breach on strict inequality (lhs_b == rhs_b must FAIL strict=True!)
                    y_dot_b = sum(spc.y[i] * cc.cell_b[i] for i in range(len(spc.y)))
                    eq_spc = dataclasses.replace(spc, rhs_num=y_dot_b, rhs_den=spc.scale)
                    self.assertFalse(
                        check_positivity_farkas(cc.cell_A, cc.cell_b, eq_spc, strict=True)
                    )

            self.assertTrue(checked_origin_cone)
            self.assertTrue(checked_off_origin)

            # 1B. Direct standalone check_affine_nonpos_farkas & check_affine_neg_farkas vector API tamper
            A_box = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            d_box = [2, -1, 3, 0]  # [1, 2] x [0, 3]
            # Prove -x_1 + 1 <= 0 (non-strict, tight at x_1=1) and -x_1 < 0 (strict, since -x_1 <= -1)
            self.assertTrue(check_affine_nonpos_farkas(A_box, d_box, [-1, 0], c0=1, y=[0, 1, 0, 0], q=1))
            self.assertFalse(check_affine_nonpos_farkas(A_box, d_box, [-1, 0], c0=1, y=[0, -1, 0, 0], q=1))
            self.assertFalse(check_affine_nonpos_farkas(A_box, d_box, [-1, 0], c0=1, y=[0, 2, 0, 0], q=1))
            self.assertFalse(check_affine_nonpos_farkas(A_box, d_box, [-1, 0], c0=1, y=[0, 1, 0, 0], q=0))
            self.assertFalse(check_affine_nonpos_farkas(A_box, d_box, [-1, 0], c0=2, y=[0, 1, 0, 0], q=1))

            self.assertTrue(check_affine_neg_farkas(A_box, d_box, [-1, 0], c0=0, y=[0, 1, 0, 0], q=1))
            # Slack == 0 fails strict negativity!
            self.assertFalse(check_affine_neg_farkas(A_box, d_box, [-1, 0], c0=1, y=[0, 1, 0, 0], q=1))
            self.assertFalse(check_affine_neg_farkas(A_box, d_box, [-1, 0], c0=0, y=[0, -1, 0, 0], q=1))
            self.assertFalse(check_affine_neg_farkas(A_box, d_box, [-1, 0], c0=0, y=[0, 1, 0, 0], q=0))

    def test_02_discrete_pullback_and_continuous_dini_decrease_tamper(self) -> None:
        """Mutate discrete pullback decay duals, CT-BaB Planet lifted decay duals,
        continuous Dini decrease duals, and maximal ROA boundary certificates.
        """
        with WatchdogTimer(15.0):
            # Discrete pullback + CT-BaB lifted certificates
            v7, p7, dom7 = build_trap7_composition_shard_blowup()
            r_disc = verify_discrete_lyapunov(v7, p7, dom7, epsilon=(1, 4))
            self.assertEqual(r_disc.verdict, Verdict.CERTIFIED_EXACT)
            self.assertTrue(check_lyapunov_certificate_bundle(r_disc))

            # Continuous Dini + maximal ROA boundary certificates on saturated Inverted Pendulum
            vp, sp, dp = build_inverted_pendulum_sector_2d(torque_limit=(3, 2))
            r_cont = verify_continuous_lyapunov(
                vp, sp, dp, decay_lambda=(1, 2), compute_maximal_roa=True
            )
            self.assertEqual(r_cont.verdict, Verdict.CERTIFIED_EXACT)
            self.assertTrue(check_lyapunov_certificate_bundle(r_cont))
            self.assertGreater(len(r_cont.boundary_certs), 0)

            for bundle in (r_disc, r_cont):
                for s_idx, dc in enumerate(bundle.decay_certs):
                    bc = dc.bound_cert
                    # 1) y_i sign flip
                    for i, yi in enumerate(bc.y):
                        if yi > 0:
                            bad_y = list(bc.y)
                            bad_y[i] = -yi
                            bad_dc = dataclasses.replace(
                                dc, bound_cert=dataclasses.replace(bc, y=tuple(bad_y))
                            )
                            bad_b = dataclasses.replace(
                                bundle,
                                decay_certs=bundle.decay_certs[:s_idx]
                                + (bad_dc,)
                                + bundle.decay_certs[s_idx + 1 :],
                            )
                            self.assertFalse(check_lyapunov_certificate_bundle(bad_b))
                    # 2) A^T y mismatch (+1 / -1 on y_0)
                    for delta in (1, -1):
                        bad_y = list(bc.y)
                        bad_y[0] += delta
                        bad_dc = dataclasses.replace(
                            dc, bound_cert=dataclasses.replace(bc, y=tuple(bad_y))
                        )
                        bad_b = dataclasses.replace(
                            bundle,
                            decay_certs=bundle.decay_certs[:s_idx]
                            + (bad_dc,)
                            + bundle.decay_certs[s_idx + 1 :],
                        )
                        self.assertFalse(check_lyapunov_certificate_bundle(bad_b))
                    # 3) q = 0 (scale = 0 on non-vacuous, scale = 1 or is_vacuous=False on vacuous)
                    bad_bc_q = (
                        dataclasses.replace(bc, is_vacuous=False, scale=0)
                        if bc.is_vacuous
                        else dataclasses.replace(bc, scale=0)
                    )
                    bad_dc_q0 = dataclasses.replace(dc, bound_cert=bad_bc_q)
                    bad_b_q0 = dataclasses.replace(
                        bundle,
                        decay_certs=bundle.decay_certs[:s_idx]
                        + (bad_dc_q0,)
                        + bundle.decay_certs[s_idx + 1 :],
                    )
                    self.assertFalse(check_lyapunov_certificate_bundle(bad_b_q0))
                    # 4) d^T y slack breach
                    if not bc.is_vacuous:
                        y_dot_b = sum(bc.y[i] * dc.poly_b[i] for i in range(len(bc.y)))
                        bad_dc_sl = dataclasses.replace(
                            dc,
                            bound_cert=dataclasses.replace(
                                bc, rhs_num=bc.rhs_den * y_dot_b - 1, rhs_den=bc.rhs_den * bc.scale
                            ),
                        )
                        bad_b_sl = dataclasses.replace(
                            bundle,
                            decay_certs=bundle.decay_certs[:s_idx]
                            + (bad_dc_sl,)
                            + bundle.decay_certs[s_idx + 1 :],
                        )
                        self.assertFalse(check_lyapunov_certificate_bundle(bad_b_sl))
                    else:
                        # Vacuous infeasibility requires y^T b < 0; shift poly_b so y^T b >= 0
                        act_i = [i for i, yi in enumerate(bc.y) if yi > 0][0]
                        y_dot_b = sum(bc.y[i] * dc.poly_b[i] for i in range(len(bc.y)))
                        bump = (-y_dot_b) // bc.y[act_i] + 1
                        bad_pb = list(dc.poly_b)
                        bad_pb[act_i] += bump
                        bad_dc_sl = dataclasses.replace(dc, poly_b=tuple(bad_pb))
                        bad_b_sl = dataclasses.replace(
                            bundle,
                            decay_certs=bundle.decay_certs[:s_idx]
                            + (bad_dc_sl,)
                            + bundle.decay_certs[s_idx + 1 :],
                        )
                        self.assertFalse(check_lyapunov_certificate_bundle(bad_b_sl))

            # Also tamper BoundaryLevelCert on r_cont
            b0 = r_cont.boundary_certs[0]
            y_dot_f = sum(b0.bound_cert.y[i] * b0.facet_b[i] for i in range(len(b0.bound_cert.y)))
            bad_b0 = dataclasses.replace(
                b0,
                bound_cert=dataclasses.replace(
                    b0.bound_cert,
                    rhs_num=b0.bound_cert.rhs_den * y_dot_f - 1,
                    rhs_den=b0.bound_cert.rhs_den * b0.bound_cert.scale,
                ),
            )
            self.assertFalse(
                check_lyapunov_certificate_bundle(
                    dataclasses.replace(r_cont, boundary_certs=(bad_b0,) + r_cont.boundary_certs[1:])
                )
            )

    def test_03_filippov_crease_certs_and_sliding_witness_tamper(self) -> None:
        """Mutate Filippov tangent-multiplier certificates (tau_int, y_per_gen, scale, kappas)
        AND uphill sliding-mode counterexample witnesses (x*, lambda* = alpha_weights, f_sliding).
        """
        with WatchdogTimer(15.0):
            # 3A. Construct a genuine attractive stable-sliding crease yielding TANGENT_MULTIPLIER_DECAY:
            # On [-1, 1]^2, V(x) = |x_1| + |x_2|.
            # For x_1 >= 0: f_pos(x) = (-1 - x_1, -x_2) (pushes left into x_1 = 0!)
            # For x_1 <= 0: f_neg(x) = (+1 - x_1, -x_2) (pushes right into x_1 = 0!)
            # Along x_1 = 0, alpha = (1/2, 1/2) gives f_S = (0, -x_2), so D^+_{f_S} V = -|x_2| = -V(x)!
            dom = HPolytope.symmetric_box(2, (1, 1))
            v_l1 = build_l1_norm_relu(2).enumerate_pwa(dom, merge_bft=False)
            f_pos = AffineMapQ.create(2, 2, [[-1, 0], [0, -1]], [-1, 0], 1)
            f_neg = AffineMapQ.create(2, 2, [[-1, 0], [0, -1]], [1, 0], 1)
            f_gens_map: Dict[int, Sequence[AffineMapQ]] = {}
            for c in v_l1.cells:
                xc = c.polytope.chebyshev_ball()[1]
                assert xc is not None
                f_gens_map[c.cell_id] = (f_pos,) if xc[0][0] > 0 else (f_neg,)

            f_res = audit_filippov_creases(v_l1, f_gens_map, eps_slide=(1, 2))
            self.assertEqual(f_res.verdict, Verdict.CERTIFIED_EXACT)
            tm_certs = [c for c in f_res.certificates if c.mode == "TANGENT_MULTIPLIER_DECAY"]
            self.assertGreaterEqual(len(tm_certs), 2)

            for fc in f_res.certificates:
                self.assertTrue(check_filippov_crease_cert(fc, x_eq=((0, 0), 1)))

            for fc in tm_certs:
                # 1) scale = 0 or negative
                self.assertFalse(
                    check_filippov_crease_cert(dataclasses.replace(fc, scale=0), x_eq=((0, 0), 1))
                )
                self.assertFalse(
                    check_filippov_crease_cert(
                        dataclasses.replace(fc, scale=-fc.scale), x_eq=((0, 0), 1)
                    )
                )
                # 2) y_per_gen sign flip and +-1 mutation
                for r_idx, yr in enumerate(fc.y_per_gen):
                    for i, yi in enumerate(yr):
                        if yi > 0:
                            bad_yr = list(yr)
                            bad_yr[i] = -yi
                            bad_y_all = (
                                fc.y_per_gen[:r_idx] + (tuple(bad_yr),) + fc.y_per_gen[r_idx + 1 :]
                            )
                            self.assertFalse(
                                check_filippov_crease_cert(
                                    dataclasses.replace(fc, y_per_gen=bad_y_all), x_eq=((0, 0), 1)
                                )
                            )
                        for delta in (1, -1):
                            bad_yr = list(yr)
                            bad_yr[i] += delta
                            bad_y_all = (
                                fc.y_per_gen[:r_idx] + (tuple(bad_yr),) + fc.y_per_gen[r_idx + 1 :]
                            )
                            self.assertFalse(
                                check_filippov_crease_cert(
                                    dataclasses.replace(fc, y_per_gen=bad_y_all), x_eq=((0, 0), 1)
                                )
                            )
                # 3) tau_int +-1 mutation
                for p_idx in range(len(fc.tau_int)):
                    for delta in (1, -1):
                        bad_tau = list(fc.tau_int)
                        bad_tau[p_idx] += delta
                        self.assertFalse(
                            check_filippov_crease_cert(
                                dataclasses.replace(fc, tau_int=tuple(bad_tau)), x_eq=((0, 0), 1)
                            )
                        )
                # 4) kappa slack breach (+1000 on gen_kappas[0])
                bad_kappas = (fc.gen_kappas[0] + 1000,) + fc.gen_kappas[1:]
                self.assertFalse(
                    check_filippov_crease_cert(
                        dataclasses.replace(fc, gen_kappas=bad_kappas), x_eq=((0, 0), 1)
                    )
                )

            # 3B. Trap 1 uphill Filippov sliding witness tamper (x*, lambda* = alpha_weights, f_sliding, dini_deriv)
            v1, p1, d1 = build_trap1_filippov_uphill_sliding()
            r1 = verify_continuous_lyapunov(v1, p1, d1, decay_lambda=(1, 2), check_filippov=True)
            self.assertEqual(r1.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
            wit = r1.filippov_witness
            assert wit is not None
            p1_gens = {c.cell_id: (c.affine_map,) for c in p1.cells}
            self.assertTrue(check_filippov_witness(wit, v1.cells, p1_gens, eps_slide=(1, 2), x_eq=((0, 0), 1)))

            # Perturb x* off crease (x_1 = 1/10^6) and along crease (x_2 + 1/10^6 without updating v_at_x/f_sliding)
            x_off_crease = qvec_add(wit.x_star, ((1, 0), 1_000_000))
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, x_star=x_off_crease),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )
            x_along_crease = qvec_add(wit.x_star, ((0, 1), 1_000_000))
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, x_star=x_along_crease),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )
            # Disguise equilibrium x_eq = (0,0) as sliding witness
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(
                        wit,
                        x_star=((0, 0), 1),
                        f_sliding=((0, 0), 1),
                        dini_deriv=(0, 1),
                        v_at_x=(0, 1),
                    ),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )

            # Perturb convex-combination multiplier lambda* (alpha_weights):
            # Case 1: sum != 1
            a0, a1 = wit.alpha_weights
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, alpha_weights=(q_add(a0, (1, 1_000_000)), a1)),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )
            # Case 2: sum == 1, non-negative, but breaks tangency N_F * f_S == 0!
            a0_shift = q_add(a0, (1, 8))
            a1_shift = q_add(a1, (-1, 8))
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, alpha_weights=(a0_shift, a1_shift)),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )
            # Case 3: negative alpha weight (-1/2, 3/2)
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, alpha_weights=((-1, 2), (3, 2))),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )
            # Case 4: flip sign of dini_deriv
            self.assertFalse(
                check_filippov_witness(
                    dataclasses.replace(wit, dini_deriv=(-wit.dini_deriv[0], wit.dini_deriv[1])),
                    v1.cells,
                    p1_gens,
                    eps_slide=(1, 2),
                    x_eq=((0, 0), 1),
                )
            )

    def test_04_bouligand_ncbf_duals_and_corner_deadlock_tamper(self) -> None:
        """Mutate Bouligand NCBF validity duals (init, unsafe, tangent, saturation)
        AND actuator-saturated corner-deadlock duals (lambda_tan, eta_sat).
        """
        with WatchdogTimer(15.0):
            # 4A. Certified 3D Dubins NCBF bundle tamper
            hd, pd, dd, init_d, uns_d = build_dubins_3sector_obstacle_3d()
            r_bar = verify_barrier(hd, pd, domain=dd, initial_set=init_d, unsafe_set=uns_d)
            self.assertEqual(r_bar.verdict, Verdict.CERTIFIED_EXACT)
            self.assertTrue(check_barrier_certificate_bundle(r_bar))

            for ic_idx, ic in enumerate(r_bar.init_certs):
                for i, yi in enumerate(ic.y):
                    if yi > 0:
                        bad_y = list(ic.y)
                        bad_y[i] = -yi
                        bad_ic = dataclasses.replace(ic, y=tuple(bad_y))
                        self.assertFalse(
                            check_barrier_certificate_bundle(
                                dataclasses.replace(
                                    r_bar,
                                    init_certs=r_bar.init_certs[:ic_idx]
                                    + (bad_ic,)
                                    + r_bar.init_certs[ic_idx + 1 :],
                                )
                            )
                        )
                bad_q_ic = dataclasses.replace(ic, scale=1 if ic.is_vacuous else 0)
                self.assertFalse(
                    check_barrier_certificate_bundle(
                        dataclasses.replace(
                            r_bar,
                            init_certs=r_bar.init_certs[:ic_idx]
                            + (bad_q_ic,)
                            + r_bar.init_certs[ic_idx + 1 :],
                        )
                    )
                )

            # Tamper unsafe_certs[0]
            uc0 = r_bar.unsafe_certs[0]
            for i, yi in enumerate(uc0.y):
                if yi > 0:
                    bad_y = list(uc0.y)
                    bad_y[i] = -yi
                    bad_uc = dataclasses.replace(uc0, y=tuple(bad_y))
                    self.assertFalse(
                        check_barrier_certificate_bundle(
                            dataclasses.replace(
                                r_bar, unsafe_certs=(bad_uc,) + r_bar.unsafe_certs[1:]
                            )
                        )
                    )
                bad_y_p1 = list(uc0.y)
                bad_y_p1[i] += 1
                self.assertFalse(
                    check_barrier_certificate_bundle(
                        dataclasses.replace(
                            r_bar,
                            unsafe_certs=(dataclasses.replace(uc0, y=tuple(bad_y_p1)),)
                            + r_bar.unsafe_certs[1:],
                        )
                    )
                )

            # Tamper viability_certs (u_scale, tangent_certs, sat_certs)
            vc0 = r_bar.viability_certs[0]
            self.assertFalse(
                check_barrier_certificate_bundle(
                    dataclasses.replace(
                        r_bar,
                        viability_certs=(dataclasses.replace(vc0, u_scale=0),)
                        + r_bar.viability_certs[1:],
                    )
                )
            )
            tc0 = vc0.tangent_certs[0]
            bad_tc0 = dataclasses.replace(tc0, y=(tc0.y[0] + 1,) + tc0.y[1:])
            self.assertFalse(
                check_barrier_certificate_bundle(
                    dataclasses.replace(
                        r_bar,
                        viability_certs=(
                            dataclasses.replace(vc0, tangent_certs=(bad_tc0,) + vc0.tangent_certs[1:]),
                        )
                        + r_bar.viability_certs[1:],
                    )
                )
            )
            sc0 = vc0.sat_certs[0]
            bad_sc0 = dataclasses.replace(sc0, scale=0)
            self.assertFalse(
                check_barrier_certificate_bundle(
                    dataclasses.replace(
                        r_bar,
                        viability_certs=(
                            dataclasses.replace(vc0, sat_certs=(bad_sc0,) + vc0.sat_certs[1:]),
                        )
                        + r_bar.viability_certs[1:],
                    )
                )
            )

            # 4B. Trap 4 actuator-saturated corner-deadlock dual (lambda_tan, eta_sat) tamper
            b4, p4_sat, _, _ = build_trap4_deadlocked_concave_corner()
            r4 = verify_barrier(b4, p4_sat)
            self.assertEqual(r4.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
            assert r4.counterexample is not None
            sub = r4.counterexample.deadlock_subcerts[0]
            self.assertTrue(check_corner_deadlock_cert(sub))

            # Mutate every coordinate of lambda_tan (sign flip, +1, -1)
            for i, lv in enumerate(sub.lambda_tan):
                if lv > 0:
                    bad_l = list(sub.lambda_tan)
                    bad_l[i] = -lv
                    self.assertFalse(
                        check_corner_deadlock_cert(dataclasses.replace(sub, lambda_tan=tuple(bad_l)))
                    )
                for delta in (1, -1):
                    bad_l = list(sub.lambda_tan)
                    bad_l[i] += delta
                    self.assertFalse(
                        check_corner_deadlock_cert(dataclasses.replace(sub, lambda_tan=tuple(bad_l)))
                    )

            # Mutate every coordinate of eta_sat (sign flip, +1, -1)
            for j, ev in enumerate(sub.eta_sat):
                if ev > 0:
                    bad_e = list(sub.eta_sat)
                    bad_e[j] = -ev
                    self.assertFalse(
                        check_corner_deadlock_cert(dataclasses.replace(sub, eta_sat=tuple(bad_e)))
                    )
                for delta in (1, -1):
                    bad_e = list(sub.eta_sat)
                    bad_e[j] += delta
                    self.assertFalse(
                        check_corner_deadlock_cert(dataclasses.replace(sub, eta_sat=tuple(bad_e)))
                    )

            # Zero dual vector (0, 0) must be rejected
            self.assertFalse(
                check_corner_deadlock_cert(
                    dataclasses.replace(
                        sub,
                        lambda_tan=tuple(0 for _ in sub.lambda_tan),
                        eta_sat=tuple(0 for _ in sub.eta_sat),
                    )
                )
            )

    def test_05_periodic_orbit_witness_perturbation_and_disguised_fixed_point(self) -> None:
        """Perturb Period-k limit-cycle coordinates by 1/10^6 AND feed period-1 fixed points
        disguised as Period-2 / Period-3 (and Period-2 disguised as Period-4).
        """
        with WatchdogTimer(15.0):
            net3a, dom3a, net3b, dom3b = build_trap3_hidden_periodic_orbits()
            r3a = detect_periodic_orbits_and_fixed_points(net3a, domain=dom3a, max_period=3)
            r3b = detect_periodic_orbits_and_fixed_points(net3b, domain=dom3b, max_period=3)

            orb2 = [o for o in r3a.periodic_orbits if o.period == 2][0]
            orb3 = [o for o in r3b.periodic_orbits if o.period == 3][0]
            self.assertTrue(check_periodic_orbit_witness(orb2, net3a))
            self.assertTrue(check_periodic_orbit_witness(orb3, net3b))

            # 1) Perturb each point & coordinate of Period-2 by +- 1/10^6
            for t in range(2):
                for sgn in (1, -1):
                    pert_pt = qvec_add(orb2.points[t], ((sgn,), 1_000_000))
                    bad_pts = orb2.points[:t] + (pert_pt,) + orb2.points[t + 1 :]
                    self.assertFalse(
                        check_periodic_orbit_witness(
                            dataclasses.replace(orb2, points=bad_pts), net3a
                        )
                    )

            # 2) Perturb each point & coordinate of 2D Period-3 by +- 1/10^6
            for t in range(3):
                for coord in range(2):
                    for sgn in (1, -1):
                        delta_vec = ((sgn, 0), 1_000_000) if coord == 0 else ((0, sgn), 1_000_000)
                        pert_pt = qvec_add(orb3.points[t], delta_vec)
                        bad_pts = orb3.points[:t] + (pert_pt,) + orb3.points[t + 1 :]
                        self.assertFalse(
                            check_periodic_orbit_witness(
                                dataclasses.replace(orb3, points=bad_pts), net3b
                            )
                        )

            # 3) Feed genuine period-1 fixed point 0 disguised as Period-2 and Period-3!
            fp_1d: QVec = ((0,), 1)
            for fake_k in (2, 3):
                disguised_1d = PeriodicOrbitWitness(
                    period=fake_k,
                    points=tuple(fp_1d for _ in range(fake_k)),
                    cell_itinerary=tuple(0 for _ in range(fake_k)),
                    stability="LOCALLY_STABLE",
                    cycle_jacobian_W=((1,),),
                    cycle_jacobian_denom=2**fake_k,
                )
                self.assertFalse(check_periodic_orbit_witness(disguised_1d, net3a))

            fp_2d: QVec = ((0, 0), 1)
            for fake_k in (2, 3):
                disguised_2d = PeriodicOrbitWitness(
                    period=fake_k,
                    points=tuple(fp_2d for _ in range(fake_k)),
                    cell_itinerary=tuple(0 for _ in range(fake_k)),
                    stability="SUPER_STABLE",
                    cycle_jacobian_W=((0, 0), (0, 0)),
                    cycle_jacobian_denom=1,
                )
                self.assertFalse(check_periodic_orbit_witness(disguised_2d, net3b))

            # 4) Feed genuine Period-2 orbit {-4/9, 4/9} disguised as Period-4!
            disguised_p4 = PeriodicOrbitWitness(
                period=4,
                points=orb2.points + orb2.points,
                cell_itinerary=orb2.cell_itinerary + orb2.cell_itinerary,
                stability=orb2.stability,
                cycle_jacobian_W=orb2.cycle_jacobian_W,
                cycle_jacobian_denom=orb2.cycle_jacobian_denom,
            )
            self.assertFalse(check_periodic_orbit_witness(disguised_p4, net3a))

            # 5) Wrong cell itinerary on PWAFunction must also be rejected
            pwa_3b = net3b.enumerate_pwa(dom3b)
            # Re-detect on pwa_3b to get exact PWA cell IDs, then corrupt 1 cell ID
            r3b_pwa = detect_periodic_orbits_and_fixed_points(pwa_3b, domain=dom3b, max_period=3)
            orb3_pwa = [o for o in r3b_pwa.periodic_orbits if o.period == 3][0]
            self.assertTrue(check_periodic_orbit_witness(orb3_pwa, pwa_3b))
            wrong_cid = (orb3_pwa.cell_itinerary[0] + 1) % len(pwa_3b.cells)
            bad_itin = (wrong_cid,) + orb3_pwa.cell_itinerary[1:]
            self.assertFalse(
                check_periodic_orbit_witness(
                    dataclasses.replace(orb3_pwa, cell_itinerary=bad_itin), pwa_3b
                )
            )

            # 6) Tamper NonFixedPointCellCerts
            self.assertTrue(check_non_fixed_point_certs(r3b.non_fixed_cell_certs))
            nfc0 = r3b.non_fixed_cell_certs[0]
            bad_nfc0 = dataclasses.replace(
                nfc0,
                infeas_cert=FarkasInfeasibilityCert(
                    y=(nfc0.infeas_cert.y[0] + 1,) + nfc0.infeas_cert.y[1:],
                    rhs_dot=nfc0.infeas_cert.rhs_dot,
                ),
            )
            self.assertFalse(check_non_fixed_point_certs((bad_nfc0,) + r3b.non_fixed_cell_certs[1:]))

    def test_06_sha256_signed_json_proof_bundles_and_semantic_rehash_tamper(self) -> None:
        """Tamper SHA-256 signed JSON proof bundles across all bundle kinds in `io_schema.py`:
        both raw byte/field mutations AND recomputed-SHA-256 semantic certificate forgeries.
        """
        with WatchdogTimer(15.0):
            vp, sp, dp = build_inverted_pendulum_sector_2d(torque_limit=None)
            lyap_res = verify_continuous_lyapunov(vp, sp, dp, decay_lambda=(1, 2))
            lyap_bundle = export_signed_certificate_bundle(lyap_res, "lyapunov_continuous")

            hd, pd, dd, id_, ud = build_dubins_3sector_obstacle_3d()
            bar_res = verify_barrier(hd, pd, domain=dd, initial_set=id_, unsafe_set=ud)
            bar_bundle = export_signed_certificate_bundle(bar_res, "barrier")

            b4, p4_sat, _, _ = build_trap4_deadlocked_concave_corner()
            dead_res = verify_barrier(b4, p4_sat)
            dead_bundle = export_signed_certificate_bundle(dead_res, "barrier")

            sh = RuntimeCBFShield(hd, pd, alpha_gain=(1, 1))
            sh_res = sh.step(((0, -1, 0), 2), ((-3,), 1))
            sh_bundle = export_signed_certificate_bundle(sh_res, "shield")

            net3a, dom3a, _, _ = build_trap3_hidden_periodic_orbits()
            orb_res = detect_periodic_orbits_and_fixed_points(net3a, domain=dom3a, max_period=3)
            orb_bundle = export_signed_certificate_bundle(orb_res, "orbits")

            for bnd in (lyap_bundle, bar_bundle, dead_bundle, sh_bundle, orb_bundle):
                ok, msg = verify_signed_certificate_bundle(bnd)
                self.assertTrue(ok, msg)

                # 1) Single-nibble flip in sha256 digest
                bad_sig = json.loads(json.dumps(bnd))
                h0 = bad_sig["sha256"]
                flipped_char = "0" if h0[0] != "0" else "1"
                bad_sig["sha256"] = flipped_char + h0[1:]
                ok_s, msg_s = verify_signed_certificate_bundle(bad_sig)
                self.assertFalse(ok_s)
                self.assertIn("SHA-256", msg_s)

                # 2) Metadata tamper without rehashing
                bad_meta = json.loads(json.dumps(bnd))
                bad_meta["metadata"]["injected"] = 1
                ok_m, _ = verify_signed_certificate_bundle(bad_meta)
                self.assertFalse(ok_m)

            # 3) Semantic forgery WITH recomputed valid SHA-256 across every certificate kind!
            # (a) Lyapunov decay dual +1 with re-signed SHA-256
            forg_lyap = json.loads(json.dumps(lyap_bundle))
            forg_lyap["payload"]["decay_certs"][0]["bound_cert"]["y"][0] += 1
            forg_lyap["sha256"] = compute_bundle_sha256(forg_lyap)
            self.assertFalse(verify_signed_certificate_bundle(forg_lyap)[0])

            # (b) Certified Barrier tangent dual scale=0 with re-signed SHA-256
            forg_bar = json.loads(json.dumps(bar_bundle))
            forg_bar["payload"]["viability_certs"][0]["tangent_certs"][0]["scale"] = 0
            forg_bar["sha256"] = compute_bundle_sha256(forg_bar)
            self.assertFalse(verify_signed_certificate_bundle(forg_bar)[0])

            # (c) Deadlock Barrier lambda_tan +1 with re-signed SHA-256
            forg_dead = json.loads(json.dumps(dead_bundle))
            forg_dead["payload"]["counterexample"]["deadlock_subcerts"][0]["lambda_tan"][0] += 1
            forg_dead["sha256"] = compute_bundle_sha256(forg_dead)
            self.assertFalse(verify_signed_certificate_bundle(forg_dead)[0])

            # (d) Runtime Shield u_safe coordinate +1 with re-signed SHA-256
            forg_sh = json.loads(json.dumps(sh_bundle))
            forg_sh["payload"]["u_safe"][0][0] += 1
            forg_sh["sha256"] = compute_bundle_sha256(forg_sh)
            self.assertFalse(verify_signed_certificate_bundle(forg_sh)[0])

            # (e) Orbit non_fixed_cell_certs y[0] +1 with re-signed SHA-256
            forg_orb = json.loads(json.dumps(orb_bundle))
            forg_orb["payload"]["non_fixed_cell_certs"][0]["infeas_cert"]["y"][0] += 1
            forg_orb["sha256"] = compute_bundle_sha256(forg_orb)
            self.assertFalse(verify_signed_certificate_bundle(forg_orb)[0])


if __name__ == "__main__":
    unittest.main()
