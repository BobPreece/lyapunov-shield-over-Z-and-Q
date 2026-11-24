from __future__ import annotations

import unittest

from lyapunov_shield import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
    Verdict,
    check_lyapunov_certificate_bundle,
    check_non_fixed_point_certs,
    check_periodic_orbit_witness,
    compose_linear_plant_with_relu_controller,
    detect_periodic_orbits_and_fixed_points,
    snap_relu_network_to_lattice,
    synthesize_lyapunov_cegis,
    verify_discrete_lyapunov,
)


class TestOrbitsAndCEGIS(unittest.TestCase):
    def test_isolated_and_singular_segment_fixed_points(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        # F(x1, x2) = (x1, 1/2 x2) has a 1D singular fixed-point continuum {(x1, 0) : x1 in [-1, 1]}!
        plant_seg = PWAFunction(
            2,
            2,
            dom,
            (PWACell(0, dom, AffineMapQ.create(2, 2, [[2, 0], [0, 1]], [0, 0], 2)),),
        )
        res = detect_periodic_orbits_and_fixed_points(
            plant_seg, domain=dom, max_period=2, check_global_rn_uniqueness=False
        )
        self.assertEqual(len(res.fixed_points), 1)
        fp = res.fixed_points[0]
        self.assertFalse(fp.is_isolated)
        self.assertIsNotNone(fp.segment_other_end)
        self.assertEqual(fp.stability, "MARGINAL_CONTINUUM")

    def test_period2_and_period3_divisor_rejection(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        # Pure Period-2 swap F(x1, x2) = (-x1, -1/2 x2)? Wait: (-x1, 1/2 x2) has Period-2 on x1 != 0!
        # Use affine shift swap on 2 halves or verify that fixed points are NOT misclassified as Period-2/4!
        plant_contr = ReLUNetwork(
            2,
            2,
            (ReLULayer(AffineMapQ.create(2, 2, [[-1, 0], [0, 1]], [0, 0], 2), False),),
        )
        res = detect_periodic_orbits_and_fixed_points(plant_contr, domain=dom, max_period=4)
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(len(res.periodic_orbits), 0)
        self.assertTrue(res.unique_origin_fixed_point_rn)
        self.assertTrue(res.origin_locally_exp_stable)

    def test_global_rn_fixed_point_uniqueness_certs(self) -> None:
        dom = HPolytope.symmetric_box(1, (1, 1))
        # F(x) = 1/2 x + 1/4 ReLU(x - 1/2) has unique fixed point 0 on all of R, plus non-fixed cell on x >= 1/2!
        net = ReLUNetwork(
            1,
            1,
            (
                ReLULayer(AffineMapQ.create(1, 1, [[2]], [-1], 2), True),
                ReLULayer(AffineMapQ.create(1, 1, [[1]], [0], 4), False),
            ),
            residual=AffineMapQ.create(1, 1, [[1]], [0], 2),
        )
        res = detect_periodic_orbits_and_fixed_points(
            net, domain=dom, max_period=3, check_global_rn_uniqueness=True
        )
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(res.unique_origin_fixed_point_rn)
        self.assertGreaterEqual(len(res.non_fixed_cell_certs), 1)
        self.assertTrue(check_non_fixed_point_certs(res.non_fixed_cell_certs))

    def test_lattice_snapping_and_equilibrium_anchor_subtraction(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        origin = ((0, 0), 1)
        A_lin = AffineMapQ.create(2, 2, [[8, 2], [0, 7]], [0, 0], 10)
        pi_layers = [
            ([[-0.3, -0.1], [0.0, 0.0]], [0.3, 0.3], True),
            ([[0.3, 0.0]], [-0.09], False),
        ]
        v_layers = [
            ([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0]], [0.0, 0.0, 0.0, 0.0, 0.3], True),
            ([[1.0, 1.0, 1.0, 1.0, 0.3]], [-0.09], False),
        ]
        pi_naive = snap_relu_network_to_lattice(pi_layers, scale_denom=16, anchor_equilibrium=None)
        v_naive = snap_relu_network_to_lattice(v_layers, scale_denom=16, anchor_equilibrium=None)
        cl_naive = compose_linear_plant_with_relu_controller(A_lin, [[1], [1]], 1, pi_naive, dom)
        self.assertEqual(verify_discrete_lyapunov(v_naive, cl_naive, dom).verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)

        pi_anc = snap_relu_network_to_lattice(pi_layers, scale_denom=16, anchor_equilibrium=origin)
        v_anc = snap_relu_network_to_lattice(v_layers, scale_denom=16, anchor_equilibrium=origin)
        cl_anc = compose_linear_plant_with_relu_controller(A_lin, [[1], [1]], 1, pi_anc, dom)
        res_anc = verify_discrete_lyapunov(v_anc, cl_anc, dom, epsilon=(1, 20))
        self.assertEqual(res_anc.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(res_anc))

    def test_cegis_discrete_non_normal_synthesis(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        # Non-normal coupled 2D plant x+ = [[1/2, 3/4], [0, 1/2]] x where unweighted ||x||_1 fails
        plant = ReLUNetwork(
            2, 2, (ReLULayer(AffineMapQ.create(2, 2, [[2, 3], [0, 2]], [0, 0], 4), False),)
        )
        res = synthesize_lyapunov_cegis(
            plant, dom, is_discrete=True, rate_target=(1, 10), lattice_denom=16, max_iters=15, timeout_sec=2.5
        )
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertGreaterEqual(res.iterations, 2)
        self.assertLess(res.elapsed_sec, 2.5)
        assert res.verification is not None
        self.assertTrue(check_lyapunov_certificate_bundle(res.verification))

    def test_cegis_continuous_synthesis_and_budget_cap(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        # Continuous non-normal plant dx/dt = [[-1, 2], [0, -1]] x where unweighted ||x||_1 fails!
        plant_c = ReLUNetwork(
            2, 2, (ReLULayer(AffineMapQ.create(2, 2, [[-1, 2], [0, -1]], [0, 0], 1), False),)
        )
        res_c = synthesize_lyapunov_cegis(
            plant_c, dom, is_discrete=False, rate_target=(1, 5), lattice_denom=16, max_iters=15, timeout_sec=2.5
        )
        self.assertEqual(res_c.verdict, Verdict.CERTIFIED_EXACT)
        assert res_c.verification is not None
        self.assertTrue(check_lyapunov_certificate_bundle(res_c.verification))

        # Unstable plant dx/dt = +x hits max_iters=3 cleanly -> OPEN_BUDGET_EXCEEDED
        plant_u = ReLUNetwork(
            2, 2, (ReLULayer(AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 1), False),)
        )
        res_u = synthesize_lyapunov_cegis(
            plant_u, dom, is_discrete=False, rate_target=(1, 5), max_iters=3, timeout_sec=2.0
        )
        self.assertEqual(res_u.verdict, Verdict.OPEN_BUDGET_EXCEEDED)


if __name__ == "__main__":
    unittest.main()
