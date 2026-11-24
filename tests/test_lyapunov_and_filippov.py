from __future__ import annotations

import dataclasses
import unittest

from lyapunov_shield import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
    Verdict,
    audit_filippov_creases,
    build_dai_l1_lyapunov,
    build_l1_norm_relu,
    check_affine_neg_farkas,
    check_coercivity_cell_cert,
    check_filippov_crease_cert,
    check_lyapunov_certificate_bundle,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
)


class TestLyapunovAndFilippov(unittest.TestCase):
    def test_zero_hole_coercivity_and_strict_off_origin_farkas(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        # V(x) = |x1| + |x2| + 1/2 ReLU(x1 - 1) has off-origin cells {x1 >= 1, x2 >= 0} etc.
        v_net = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(
                    AffineMapQ.create(
                        2,
                        5,
                        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 0]],
                        [0, 0, 0, 0, -1],
                        1,
                    ),
                    True,
                ),
                ReLULayer(AffineMapQ.create(5, 1, [[2, 2, 2, 2, 1]], [0], 2), False),
            ),
        )
        plant = PWAFunction(
            2, 2, dom, (PWACell(0, dom, AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2)),)
        )
        res = verify_discrete_lyapunov(v_net, plant, dom, epsilon=(1, 4))
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(res))

        # Check that off-origin cells emit strict_pos_cert proving -V_i(x) < 0!
        strict_certs = [c.strict_pos_cert for c in res.coercivity_certs if c.strict_pos_cert is not None]
        self.assertGreaterEqual(len(strict_certs), 2)
        for cc in res.coercivity_certs:
            self.assertTrue(check_coercivity_cell_cert(cc, ((0, 0), 1), res.alpha_1))  # type: ignore[arg-type]
            if cc.strict_pos_cert is not None:
                self.assertTrue(check_affine_neg_farkas(cc.cell_A, cc.cell_b, cc.strict_pos_cert))

    def test_dai_l1_regularized_anchored_lyapunov(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        x_eq = ((1, -1), 4)
        # Base phi(x) with nonzero value at x_eq and degenerate rank-1 hidden neuron
        phi = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(AffineMapQ.create(2, 1, [[1, 2]], [3], 2), True),
                ReLULayer(AffineMapQ.create(1, 1, [[1]], [5], 4), False),
            ),
        )
        R_map = AffineMapQ.create(2, 2, [[2, 0], [0, 3]], [0, 0], 1)
        v_dai = build_dai_l1_lyapunov(phi, R_map=R_map, x_eq=x_eq, alpha_reg=(1, 2))
        self.assertEqual(v_dai.eval_scalar(x_eq), (0, 1))

        # Contracting affine plant around x_eq: F(x) = x_eq + 1/2 (x - x_eq) = 1/2 x + 1/2 x_eq
        plant_eq = PWAFunction(
            2,
            2,
            dom,
            (PWACell(0, dom, AffineMapQ.create(2, 2, [[4, 0], [0, 4]], [1, -1], 8)),),
        )
        res = verify_discrete_lyapunov(v_dai, plant_eq, dom, x_eq=x_eq, epsilon=(1, 8))
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(res, x_eq=x_eq))

    def test_discrete_lyapunov_and_exact_maximal_sublevel_roa(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        v_l1 = build_l1_norm_relu(2)
        plant_roa = ReLUNetwork(
            2,
            2,
            (
                ReLULayer(AffineMapQ.create(2, 1, [[1, 0]], [-1], 1), True),
                ReLULayer(AffineMapQ.create(1, 2, [[2], [0]], [0, 0], 1), False),
            ),
            residual=AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2),
        )
        r_glob = verify_discrete_lyapunov(v_l1, plant_roa, dom, epsilon=(1, 4), compute_maximal_roa=False)
        self.assertEqual(r_glob.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)

        r_roa = verify_discrete_lyapunov(v_l1, plant_roa, dom, epsilon=(1, 4), compute_maximal_roa=True)
        self.assertEqual(r_roa.verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(r_roa.rho_star, (8, 7))
        self.assertTrue(check_lyapunov_certificate_bundle(r_roa))

    def test_continuous_lyapunov_and_hyperlyve_continuity_shortcut(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        v_l1 = build_l1_norm_relu(2)
        # Continuous closed-loop f(x) = (-x1 - 1/2 ReLU(x1), -x2)
        plant_cont = ReLUNetwork(
            2,
            2,
            (
                ReLULayer(AffineMapQ.create(2, 1, [[1, 0]], [0], 1), True),
                ReLULayer(AffineMapQ.create(1, 2, [[-1], [0]], [0, 0], 2), False),
            ),
            residual=AffineMapQ.create(2, 2, [[-1, 0], [0, -1]], [0, 0], 1),
        )
        res = verify_continuous_lyapunov(v_l1, plant_cont, dom, decay_lambda=(1, 2))
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_lyapunov_certificate_bundle(res))
        assert res.filippov_audit is not None
        modes = {fc.mode for fc in res.filippov_audit.certificates}
        self.assertIn("CONTINUOUS_VECTOR_FIELD", modes)

    def test_filippov_downhill_sliding_tangent_multiplier_cert(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        v_pwa = build_l1_norm_relu(2).enumerate_pwa(dom)
        # Discontinuous relay pushing inward to x1 = 0 AND downhill toward (0,0)
        q_polys = [c.polytope for c in v_pwa.cells]
        # Build discontinuous relay: dx1 = -sgn(x1) - x1, dx2 = -x2
        f_gens = {}
        for c in v_pwa.cells:
            xc = c.polytope.chebyshev_ball()[1]
            assert xc is not None
            sx = -1 if xc[0][0] > 0 else 1
            f_gens[c.cell_id] = (AffineMapQ.create(2, 2, [[-1, 0], [0, -1]], [sx, 0], 1),)
        aud = audit_filippov_creases(v_pwa, f_gens, eps_slide=(1, 2))
        self.assertEqual(aud.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(any(fc.mode == "TANGENT_MULTIPLIER_DECAY" for fc in aud.certificates))
        for fc in aud.certificates:
            self.assertTrue(check_filippov_crease_cert(fc, x_eq=((0, 0), 1)))

    def test_filippov_codim2_ridge_3d_audit(self) -> None:
        dom = HPolytope.symmetric_box(3, (1, 1))
        v_3d = build_l1_norm_relu(3)
        plant_3d = PWAFunction(
            3,
            3,
            dom,
            (PWACell(0, dom, AffineMapQ.create(3, 3, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [0, 0, 0], 1)),),
        )
        res = verify_continuous_lyapunov(v_3d, plant_3d, dom, decay_lambda=(1, 2))
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        assert res.filippov_audit is not None
        self.assertGreater(res.filippov_audit.codim_k_creases, 0)
        self.assertTrue(check_lyapunov_certificate_bundle(res))

    def test_lyapunov_bundle_tamper_rejection(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        v_l1 = build_l1_norm_relu(2)
        plant = PWAFunction(
            2, 2, dom, (PWACell(0, dom, AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2)),)
        )
        res = verify_discrete_lyapunov(v_l1, plant, dom, epsilon=(1, 4))
        self.assertTrue(check_lyapunov_certificate_bundle(res))

        # Corrupt one dual multiplier inside decay_certs[0] -> must fail check!
        dc0 = res.decay_certs[0]
        bad_bc = dataclasses.replace(
            dc0.bound_cert,
            y=tuple(dc0.bound_cert.y[i] + (1 if i == 0 else 0) for i in range(len(dc0.bound_cert.y))),
        )
        bad_dc = dataclasses.replace(dc0, bound_cert=bad_bc)
        bad_res = dataclasses.replace(res, decay_certs=(bad_dc,) + res.decay_certs[1:])
        self.assertFalse(check_lyapunov_certificate_bundle(bad_res))


if __name__ == "__main__":
    unittest.main()
