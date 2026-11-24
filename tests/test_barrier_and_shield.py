from __future__ import annotations

import unittest

from lyapunov_shield import (
    AffineMapQ,
    ControlAffineCell,
    ControlAffinePWAPlant,
    HPolytope,
    MultiFacetBarrier,
    ReLULayer,
    ReLUNetwork,
    RuntimeCBFShield,
    Verdict,
    check_barrier_certificate_bundle,
    check_shield_step_result,
    verify_barrier,
)


class TestBarrierAndShield(unittest.TestCase):
    def test_barrier_regular_closedness_rejection(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        # h(x) = -|x1| has C = {x1 == 0} with empty interior Int(C) == empty!
        h_line = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(AffineMapQ.create(2, 2, [[1, 0], [-1, 0]], [0, 0], 1), True),
                ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [0], 1), False),
            ),
        )
        plant = ControlAffinePWAPlant.single_cell(
            2, 0, dom, AffineMapQ.zero(2, 2), [[], []], 1
        )
        res = verify_barrier(h_line, plant, domain=dom)
        self.assertEqual(res.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        self.assertIn("regular closedness", res.message.lower())

    def test_autonomous_locally_lipschitz_and_filippov_boundary_invariance(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        # Safe half-plane C = {x2 >= 0} with discontinuous relay across x1 = 0:
        # Left (x1 <= 0): f_1 = (+1, +1) [pushes into x1 = 0 AND upward into Int(C)]
        # Right (x1 >= 0): f_2 = (-1, +1) [pushes into x1 = 0 AND upward into Int(C)]
        # Filippov sliding at (0, 0) is f_S = (0, +1) -> enters Int(C)!
        h_half = MultiFacetBarrier(2, dom, (AffineMapQ.create(2, 1, [[0, 1]], [0], 1),))
        p_left = dom.with_halfspace([1, 0], 0)
        p_right = dom.with_halfspace([-1, 0], 0)
        plant_inward = ControlAffinePWAPlant(
            2,
            0,
            dom,
            (
                ControlAffineCell(0, p_left, AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [1, 1], 1), ((), ()), 1),
                ControlAffineCell(1, p_right, AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [-1, 1], 1), ((), ()), 1),
            ),
        )
        res_ok = verify_barrier(h_half, plant_inward, domain=dom)
        self.assertEqual(res_ok.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(res_ok))

        # Now flip vertical component on right to f_2 = (-1, -3) so f_S = (0, -1) escapes C at (0, 0)!
        plant_escape = ControlAffinePWAPlant(
            2,
            0,
            dom,
            (
                ControlAffineCell(0, p_left, AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [1, 1], 1), ((), ()), 1),
                ControlAffineCell(1, p_right, AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [-1, -3], 1), ((), ()), 1),
            ),
        )
        res_bad = verify_barrier(h_half, plant_escape, domain=dom)
        self.assertEqual(res_bad.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)

    def test_concave_vs_reflex_corner_bouligand_viability(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        # Non-convex reflex obstacle exterior: h(x) = |x1| + |x2| - 1/2 >= 0
        # Under dx/dt = u with u in [-1, 1]^2, every smooth facet AND reflex apex is viable!
        h_reflex = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(AffineMapQ.create(2, 4, [[1, 0], [-1, 0], [0, 1], [0, -1]], [0] * 4, 1), True),
                ReLULayer(AffineMapQ.create(4, 1, [[2, 2, 2, 2]], [-1], 2), False),
            ),
        )
        plant = ControlAffinePWAPlant.single_cell(
            2,
            2,
            dom,
            AffineMapQ.zero(2, 2),
            [[1, 0], [0, 1]],
            1,
            u_poly=HPolytope.symmetric_box(2, (1, 1)),
        )
        res = verify_barrier(h_reflex, plant, domain=dom)
        self.assertEqual(res.verdict, Verdict.CERTIFIED_EXACT)
        self.assertGreaterEqual(res.corners_checked, 4)
        self.assertTrue(check_barrier_certificate_bundle(res))

    def test_initial_and_unsafe_set_farkas_separation(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        h = MultiFacetBarrier(2, dom, (AffineMapQ.create(2, 1, [[1, 0]], [0], 1),))
        plant = ControlAffinePWAPlant.single_cell(
            2, 1, dom, AffineMapQ.zero(2, 2), [[1], [0]], 1, u_poly=HPolytope.symmetric_box(1, (1, 1))
        )
        s_init_ok = (HPolytope.from_box([((1, 2), (1, 1)), ((-1, 2), (1, 2))]),)
        s_uns_ok = (HPolytope.from_box([((-1, 1), (-1, 2)), ((-1, 2), (1, 2))]),)
        res_ok = verify_barrier(h, plant, domain=dom, initial_set=s_init_ok, unsafe_set=s_uns_ok)
        self.assertEqual(res_ok.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(check_barrier_certificate_bundle(res_ok))

        # Violating unsafe set overlapping x1 >= 0
        s_uns_bad = (HPolytope.from_box([((-1, 4), (1, 4)), ((-1, 4), (1, 4))]),)
        res_bad = verify_barrier(h, plant, domain=dom, initial_set=s_init_ok, unsafe_set=s_uns_bad)
        self.assertEqual(res_bad.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)

    def test_runtime_cbf_lp_shield_fast_path_and_intervention(self) -> None:
        dom = HPolytope.symmetric_box(2, (4, 1))
        barrier = MultiFacetBarrier(
            2,
            dom,
            (
                AffineMapQ.create(2, 1, [[1, 0]], [1], 1),
                AffineMapQ.create(2, 1, [[0, 1]], [1], 1),
            ),
        )
        plant = ControlAffinePWAPlant.single_cell(
            2, 2, dom, AffineMapQ.zero(2, 2), [[1, 0], [0, 1]], 1, HPolytope.symmetric_box(2, (2, 1))
        )
        shield = RuntimeCBFShield(barrier, plant, alpha_gain=(2, 1))

        # Fast-path safe nominal control (0 pivots + valid Farkas opt_cert!)
        s_fast = shield.step(((0, 0), 1), ((1, 1), 1))
        self.assertEqual(s_fast.verdict, Verdict.CERTIFIED_EXACT)
        self.assertFalse(s_fast.intervened)
        self.assertEqual(s_fast.pivots, 0)
        self.assertIsNotNone(s_fast.opt_cert)
        self.assertTrue(check_shield_step_result(s_fast))

        # Intervening nominal control
        s_int = shield.step(((-3, -3), 4), ((-3, 1), 1))
        self.assertEqual(s_int.verdict, Verdict.CERTIFIED_EXACT)
        self.assertTrue(s_int.intervened)
        self.assertEqual(s_int.u_safe, ((-1, 2), 2))
        self.assertTrue(check_shield_step_result(s_int))

    def test_runtime_cbf_lp_shield_deadlock_farkas_cert(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        barrier = MultiFacetBarrier(
            2,
            dom,
            (
                AffineMapQ.create(2, 1, [[3, -1]], [0], 1),
                AffineMapQ.create(2, 1, [[-1, 3]], [0], 1),
            ),
        )
        plant = ControlAffinePWAPlant.single_cell(
            2,
            2,
            dom,
            AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [-7, -7], 4),
            [[1, 0], [0, 1]],
            1,
            HPolytope.symmetric_box(2, (1, 1)),
        )
        shield = RuntimeCBFShield(barrier, plant, alpha_gain=(1, 1))
        res_dead = shield.step(((0, 0), 1), ((0, 0), 1))
        self.assertEqual(res_dead.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
        self.assertIsNotNone(res_dead.deadlock_cert)
        self.assertTrue(check_shield_step_result(res_dead))

    def test_submillisecond_shield_latency(self) -> None:
        dom = HPolytope.symmetric_box(2, (2, 1))
        barrier = MultiFacetBarrier(
            2,
            dom,
            (
                AffineMapQ.create(2, 1, [[1, 0]], [1], 1),
                AffineMapQ.create(2, 1, [[0, 1]], [1], 1),
            ),
        )
        plant = ControlAffinePWAPlant.single_cell(
            2, 2, dom, AffineMapQ.zero(2, 2), [[1, 0], [0, 1]], 1, HPolytope.symmetric_box(2, (2, 1))
        )
        shield = RuntimeCBFShield(barrier, plant, alpha_gain=(1, 1))
        # Warm-up + average solve_time_us check (< 1000 us = 1ms)
        shield.step(((-3, -3), 4), ((-2, -2), 1))
        times = [shield.step(((-3, -3), 4), ((-2, -2), 1)).solve_time_us for _ in range(10)]
        avg_us = sum(times) / len(times)
        self.assertLess(avg_us, 1000.0)


if __name__ == "__main__":
    unittest.main()
