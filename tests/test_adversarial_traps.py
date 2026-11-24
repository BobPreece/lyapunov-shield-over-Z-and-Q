from __future__ import annotations

import time
import unittest

from lyapunov_shield import (
    LPStatus,
    MultiFacetBarrier,
    RuntimeCBFShield,
    Verdict,
    build_trap1_filippov_uphill_sliding,
    build_trap2_bouligand_corner_escape,
    build_trap3_hidden_periodic_orbits,
    build_trap4_deadlocked_concave_corner,
    build_trap5_sub_tolerance_and_sliver,
    build_trap6_rounding_drift_vs_anchor,
    build_trap7_composition_shard_blowup,
    build_trap8_degenerate_origin_cone_3d,
    build_trap9_6d_64neuron_budget_guard,
    certify_linear_upper_bound,
    certify_relu_upper_bound_ct_bab,
    chebyshev_ball_linf,
    check_barrier_certificate_bundle,
    check_corner_deadlock_cert,
    check_filippov_witness,
    check_lyapunov_certificate_bundle,
    check_non_fixed_point_certs,
    check_periodic_orbit_witness,
    check_positivity_farkas,
    check_primal_witness,
    compose_pullback_pwa,
    detect_periodic_orbits_and_fixed_points,
    eval_heaviside_formal_lie_derivative,
    qvec_eq,
    solve_lp_bareiss,
    verify_barrier,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
    zaslavsky_upper_bound,
)


class TestAdversarialTraps(unittest.TestCase):
    """Comprehensive 9/9 Adversarial Control Trap verification suite."""

    def test_trap1_filippov_uphill_sliding_crease(self) -> None:
        v1, p1, dom1 = build_trap1_filippov_uphill_sliding()
        # Interior-only check certifies (all 4 open quadrants satisfy g_q^T f_q = -||x||_1 < 0)
        res_int = verify_continuous_lyapunov(v1, p1, dom1, decay_lambda=(1, 2), check_filippov=False)
        self.assertEqual(res_int.verdict, Verdict.CERTIFIED_EXACT)

        # Full Filippov crease audit catches uphill sliding on x_1 = 0, x_2 > 0
        res_full = verify_continuous_lyapunov(v1, p1, dom1, decay_lambda=(1, 2), check_filippov=True)
        self.assertEqual(res_full.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        wit = res_full.filippov_witness
        self.assertIsNotNone(wit)
        assert wit is not None
        self.assertEqual(wit.x_star, ((0, 1), 2))
        self.assertEqual(wit.f_sliding, ((0, 1), 2))
        self.assertEqual(wit.dini_deriv, (1, 2))
        assert wit.gamma_kink is not None
        self.assertEqual(abs(wit.gamma_kink[0]), 2 * wit.gamma_kink[1])
        self.assertTrue(
            check_filippov_witness(wit, v1.cells, {c.cell_id: (c.affine_map,) for c in p1.cells})
        )

    def test_trap2_bouligand_corner_escape_defeating_heaviside(self) -> None:
        h2, p2a, p2b, dom2 = build_trap2_bouligand_corner_escape()
        origin = ((0, 0), 1)

        # (2a) Every scalar Heaviside H(0) in [0, 1] cancels (-H + H = 0) and gives 0 >= 0
        for h_val in ((0, 1), (1, 4), (1, 2), (3, 4), (1, 1)):
            self.assertEqual(
                eval_heaviside_formal_lie_derivative(h2, origin, ((1, 0), 1), h_val),
                (0, 1),
            )
        r2a = verify_barrier(h2, p2a, domain=dom2)
        self.assertEqual(r2a.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        assert r2a.counterexample is not None
        self.assertEqual(r2a.counterexample.x_star, origin)
        for sub in r2a.counterexample.deadlock_subcerts:
            self.assertTrue(check_corner_deadlock_cert(sub))

        # (2b) Controlled dx/dt = (0,-1) + (1,0) u, u in [-2,2]: viable on x_1 != 0, impossible at (0,0)
        shield_2b = RuntimeCBFShield(h2, p2b, alpha_gain=(1, 1))
        self.assertEqual(shield_2b.step(((1, 1), 2), ((0,), 1)).verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(shield_2b.step(((-1, 1), 2), ((0,), 1)).verdict, Verdict.CERTIFIED_EXACT)
        r2b = verify_barrier(h2, p2b, domain=dom2)
        self.assertEqual(r2b.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        assert r2b.counterexample is not None
        self.assertEqual(r2b.counterexample.x_star, origin)
        for sub in r2b.counterexample.deadlock_subcerts:
            self.assertTrue(check_corner_deadlock_cert(sub))

    def test_trap3_hidden_rational_period2_and_period3_orbits(self) -> None:
        net3a, dom3a, net3b, dom3b = build_trap3_hidden_periodic_orbits()

        # (3a) 1D odd clamped-ReLU map -> unique locally stable fixed point 0 + Period-2 {-4/9, +4/9}
        r3a = detect_periodic_orbits_and_fixed_points(net3a, domain=dom3a, max_period=3)
        self.assertEqual(r3a.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        self.assertTrue(r3a.unique_origin_fixed_point_rn)
        self.assertTrue(r3a.origin_locally_exp_stable)
        self.assertTrue(check_non_fixed_point_certs(r3a.non_fixed_cell_certs))
        p2_pts = [orb.points for orb in r3a.periodic_orbits if orb.period == 2]
        self.assertIn((((-4,), 9), ((4,), 9)), p2_pts)
        for orb in r3a.periodic_orbits:
            self.assertTrue(check_periodic_orbit_witness(orb, net3a))

        # (3b) 2D 6-neuron plateau map -> 19-27 regions, sole fixed point (0,0), super-stable Period-3 ((2,0),(0,2),(-2,-2))
        unmerged_3b = net3b.enumerate_pwa(dom3b, merge_bft=False)
        self.assertGreaterEqual(len(unmerged_3b.cells), 19)
        r3b = detect_periodic_orbits_and_fixed_points(net3b, domain=dom3b, max_period=3)
        self.assertEqual(r3b.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        self.assertTrue(r3b.unique_origin_fixed_point_rn)
        self.assertTrue(r3b.origin_locally_exp_stable)
        self.assertTrue(check_non_fixed_point_certs(r3b.non_fixed_cell_certs))
        super_p3 = [
            orb for orb in r3b.periodic_orbits
            if orb.period == 3 and orb.stability == "SUPER_STABLE"
        ]
        self.assertGreaterEqual(len(super_p3), 1)
        self.assertIn(
            (((-2, -2), 1), ((2, 0), 1), ((0, 2), 1)),
            [orb.points for orb in super_p3],
        )
        for orb in r3b.periodic_orbits:
            self.assertTrue(check_periodic_orbit_witness(orb, net3b))

    def test_trap4_actuator_saturated_deadlocked_concave_corner(self) -> None:
        b4, p4_sat, p4_unsat, dom4 = build_trap4_deadlocked_concave_corner()
        r_f1 = verify_barrier(MultiFacetBarrier(2, dom4, (b4.facets[0],)), p4_sat)
        r_f2 = verify_barrier(MultiFacetBarrier(2, dom4, (b4.facets[1],)), p4_sat)
        r_un = verify_barrier(b4, p4_unsat)
        self.assertEqual(r_f1.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_f2.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_un.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(r_f1))
        self.assertTrue(check_barrier_certificate_bundle(r_f2))
        self.assertTrue(check_barrier_certificate_bundle(r_un))

        r_sat = verify_barrier(b4, p4_sat)
        self.assertEqual(r_sat.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        assert r_sat.counterexample is not None
        self.assertEqual(r_sat.counterexample.x_star, ((0, 0), 1))
        sub = r_sat.counterexample.deadlock_subcerts[0]
        self.assertTrue(check_corner_deadlock_cert(sub))
        self.assertTrue(all(v > 0 for v in sub.lambda_tan))
        self.assertTrue(any(v > 0 for v in sub.eta_sat))

    def test_trap5_sub_tolerance_leak_and_sliver(self) -> None:
        p_large, p_cube, p_sliver = build_trap5_sub_tolerance_and_sliver()
        v_l, _, wit_l, _ = certify_linear_upper_bound(p_large.A, p_large.b, [1, 0, 0], 20_000_000, 1, 4)
        self.assertEqual(v_l, "FALSIFIED_COUNTEREXAMPLE")
        assert wit_l is not None
        self.assertTrue(check_primal_witness(p_large.A, p_large.b, wit_l))
        self.assertEqual(wit_l[0][0], 10_000_000 * wit_l[1])

        v_c, _, wit_c, _ = certify_linear_upper_bound(p_cube.A, p_cube.b, [0, 0, 1], 1, 0, 1)
        self.assertEqual(v_c, "FALSIFIED_COUNTEREXAMPLE")
        assert wit_c is not None
        self.assertTrue(check_primal_witness(p_cube.A, p_cube.b, wit_c))

        s14 = 10**14
        r_q, xc_q, lp_res = chebyshev_ball_linf(p_sliver.A, p_sliver.b, r_cap=10)
        self.assertEqual(lp_res.status, LPStatus.OPTIMAL)
        self.assertEqual(r_q, (1, 4 * s14))
        assert xc_q is not None
        self.assertTrue(check_primal_witness(p_sliver.A, p_sliver.b, xc_q))
        v_s, cert_s, _, _ = certify_linear_upper_bound(p_sliver.A, p_sliver.b, [0, 1], 1, s14 + 1, s14)
        self.assertEqual(v_s, "CERTIFIED_EXACT")
        assert cert_s is not None
        self.assertTrue(check_positivity_farkas(p_sliver.A, p_sliver.b, cert_s))

    def test_trap6_post_rounding_equilibrium_drift_vs_anchor(self) -> None:
        pi_n, pi_a, cl_n, cl_a, v6, dom6 = build_trap6_rounding_drift_vs_anchor()
        self.assertEqual(pi_n.eval_qvec(((0,), 1)), ((-1,), 32))
        fp_n = detect_periodic_orbits_and_fixed_points(
            cl_n, domain=dom6, max_period=2, check_global_rn_uniqueness=False
        )
        self.assertTrue(any(qvec_eq(fp.point, ((-1,), 22)) for fp in fp_n.fixed_points))
        self.assertEqual(verify_discrete_lyapunov(v6, cl_n, dom6, epsilon=(1, 4)).verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)

        self.assertEqual(pi_a.eval_qvec(((0,), 1)), ((0,), 1))
        r_anc = verify_discrete_lyapunov(v6, cl_a, dom6, epsilon=(1, 4))
        self.assertEqual(r_anc.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(r_anc))

    def test_trap7_two_stage_composition_shard_blowup(self) -> None:
        v7, p7, dom7 = build_trap7_composition_shard_blowup()
        rep = compose_pullback_pwa(v7, p7, merge_bft=False, count_point_feasible=True)
        self.assertEqual(rep.combinatorial_candidates, 1024)
        self.assertEqual(rep.point_feasible_shards, 1024)
        self.assertEqual(rep.pruned_degenerate_shards, 1016)
        self.assertEqual(rep.pre_bft_full_dim_shards, 8)
        r7 = verify_discrete_lyapunov(v7, p7, dom7, epsilon=(1, 4))
        self.assertEqual(r7.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(r7))

    def test_trap8_12_hyperplane_degenerate_origin_cone_3d(self) -> None:
        v8, p8, dom8, (A_cone, b_cone) = build_trap8_degenerate_origin_cone_3d()
        for obj in ([0, 0, 1], [1, 1, 3], [-2, 1, 4], [3, -2, 6]):
            res_lp = solve_lp_bareiss(A_cone, b_cone, c=obj, max_pivots=200)
            self.assertEqual(res_lp.status, LPStatus.OPTIMAL)
            self.assertEqual(res_lp.obj_val, (0, 1))
            self.assertLess(res_lp.pivots, 50)
        r8 = verify_discrete_lyapunov(v8, p8, dom8, epsilon=(1, 4), max_cells=512)
        self.assertEqual(r8.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(r8))

    def test_trap9_6d_64neuron_arrangement_budget_guard(self) -> None:
        v9, _, dom9 = build_trap9_6d_64neuron_budget_guard()
        self.assertEqual(zaslavsky_upper_bound(6, 64), 83_278_001)
        t0 = time.perf_counter()
        verdict, _, _, z_rep = certify_relu_upper_bound_ct_bab(v9, dom9, rhs=(0, 1), max_cells=24)
        dt = time.perf_counter() - t0
        self.assertEqual(z_rep, 83_278_001)
        self.assertEqual(verdict, Verdict.OPEN_BUDGET_EXCEEDED)
        self.assertLess(dt, 0.5)


if __name__ == "__main__":
    unittest.main()
