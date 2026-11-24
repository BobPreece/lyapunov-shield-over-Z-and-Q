from __future__ import annotations

import unittest
from typing import List, Optional, Tuple

from lyapunov_shield import (
    AffineMapQ,
    MultiFacetBarrier,
    QVec,
    Rat,
    RuntimeCBFShield,
    ShieldStepResult,
    Verdict,
    WatchdogTimer,
    build_double_integrator_2d_corridor_diamond,
    build_double_integrator_4d_corridor_diamond,
    build_dubins_3sector_obstacle_3d,
    check_farkas_certificate,
    check_shield_step_result,
    q_add,
    q_le,
    q_lt,
    q_mul,
    q_norm,
    qvec_add,
    qvec_norm,
    qvec_scale,
)


class TestOnlineShieldStressSuite(unittest.TestCase):
    """Cycle 5.2: Stress-test `runtime_shield.py` across >= 500 deterministic adversarial
    nominal actions near multi-facet corridor and reflex diamond-obstacle corners on
    2D/4D Double Integrator and 3D Dubins benchmarks.
    Asserts:
      - 0 barrier breaches on viable states (both continuous-time CBF condition >= 0 and safe discrete step),
      - Exact Farkas deadlock certificates on intentionally trapped states,
      - < 1.0 ms mean Bareiss LP solve latency.
    """

    def test_500_plus_adversarial_shield_interventions_and_deadlocks(self) -> None:
        with WatchdogTimer(20.0):
            solve_times_us: List[float] = []
            viable_count = 0
            intervened_count = 0
            breaches = 0

            # ------------------------------------------------------------------
            # 1. 2D Double Integrator: Braking Corridor Corners & Reflex Diamond
            #    (200 deterministic adversarial queries)
            # ------------------------------------------------------------------
            corr_2d, diam_2d, plant_2d, _ = build_double_integrator_2d_corridor_diamond()
            sh_corr_2d = RuntimeCBFShield(corr_2d, plant_2d, alpha_gain=(1, 1))
            sh_diam_2d = RuntimeCBFShield(diam_2d, plant_2d, alpha_gain=(1, 1))
            f_2d = plant_2d.cells[0].f_map
            b_2d = AffineMapQ.create(1, 2, plant_2d.cells[0].B_num, [0, 0], 1)

            # 100 states near concave braking corners (0, 1), (0, -1), (2, 0), (-2, 0) of corridor_2d:
            # 2 - p - 2v >= 0, 1 - v >= 0, 2 + p + 2v >= 0, 1 + v >= 0
            for k in range(100):
                # Deterministic rational state strictly/boundary-viable inside corridor
                if k % 4 == 0:
                    # Near upper-right concave corner (p=0, v=1): v = 1 - eps_v, p = -eps_p
                    v_q = q_norm(20 - (k % 5), 20)  # in [16/20, 1]
                    # Ensure p + 2v <= 2: set p = 2 - 2v - delta
                    delta = q_norm(k % 7, 40)
                    p_q = q_add(q_add((2, 1), q_mul((-2, 1), v_q)), (-delta[0], delta[1]))
                elif k % 4 == 1:
                    # Near lower-left concave corner (p=0, v=-1)
                    v_q = q_norm(-(20 - (k % 5)), 20)
                    delta = q_norm(k % 7, 40)
                    p_q = q_add(q_add((-2, 1), q_mul((-2, 1), v_q)), delta)
                elif k % 4 == 2:
                    # Near right stopping boundary p + 2v = 2 with v in [-1/2, 1/2]
                    v_q = q_norm((k % 9) - 4, 8)
                    delta = q_norm(k % 6, 32)
                    p_q = q_add(q_add((2, 1), q_mul((-2, 1), v_q)), (-delta[0], delta[1]))
                else:
                    # Exact concave corner (0, 1) or (0, -1)
                    p_q = (0, 1)
                    v_q = (1, 1) if (k // 4) % 2 == 0 else (-1, 1)

                cd = p_q[1] * v_q[1]
                x_2d: QVec = qvec_norm([p_q[0] * v_q[1], v_q[0] * p_q[1]], cd)
                self.assertGreaterEqual(corr_2d.eval_scalar(x_2d)[0], 0)

                # Adversarial nominal control slamming outward (+5, +2, +1, -5, -2, -1)
                u_adv_num = 5 - (k % 3) * 2 if (k % 2 == 0) else -(5 - (k % 3) * 2)
                u_nom_2d: QVec = ((u_adv_num,), 1)

                res = sh_corr_2d.step(x_2d, u_nom_2d)
                solve_times_us.append(res.solve_time_us)
                viable_count += 1
                if res.verdict != Verdict.CERTIFIED_EXACT or res.u_safe is None:
                    breaches += 1
                    continue
                self.assertTrue(check_shield_step_result(res))
                if res.intervened:
                    intervened_count += 1
                # Verify continuous-time CBF condition grad_h . (f(x) + B u_safe) + h(x) >= 0 on ALL facets
                dx = qvec_add(f_2d.eval_qvec(x_2d), b_2d.eval_qvec(res.u_safe))
                for fh in corr_2d.facets:
                    hx = fh.eval_scalar(x_2d)
                    lh = q_norm(
                        sum(fh.W[0][j] * dx[0][j] for j in range(2)),
                        fh.denom * dx[1],
                    )
                    if q_add(lh, hx)[0] < 0:
                        breaches += 1
                # Also verify Euler step dt = 1/20 stays strictly h(x_next) >= 0
                x_next = qvec_add(x_2d, qvec_scale((1, 20), dx))
                if corr_2d.eval_scalar(x_next)[0] < 0:
                    breaches += 1

            # 100 states near 2D reflex diamond-obstacle boundary |p + 2v| >= 1/2, |v| <= 1
            for k in range(100):
                sgn_q = 1 if k % 2 == 0 else -1
                v_q = q_norm(((k * 3) % 15) - 7, 8)  # in [-7/8, 7/8], strictly inside |v| <= 1
                q_mag = q_add((1, 2), q_norm(k % 6, 32))  # >= 1/2 (includes exact boundary 1/2!)
                q_val = (sgn_q * q_mag[0], q_mag[1])
                # p = q_val - 2 * v_q
                p_q = q_add(q_val, q_mul((-2, 1), v_q))
                cd = p_q[1] * v_q[1]
                x_2d = qvec_norm([p_q[0] * v_q[1], v_q[0] * p_q[1]], cd)
                self.assertGreaterEqual(diam_2d.eval_scalar(x_2d)[0], 0)

                # Adversarial nominal control trying to push q = p + 2v straight into (-1/2, 1/2) obstacle
                # dq/dt = v + 2u -> to push sgn_q * q downward, command u = -sgn_q * 3
                u_nom_2d = ((-3 * sgn_q,), 1)
                res = sh_diam_2d.step(x_2d, u_nom_2d)
                solve_times_us.append(res.solve_time_us)
                viable_count += 1
                if res.verdict != Verdict.CERTIFIED_EXACT or res.u_safe is None:
                    breaches += 1
                    continue
                self.assertTrue(check_shield_step_result(res))
                if res.intervened:
                    intervened_count += 1
                dx = qvec_add(f_2d.eval_qvec(x_2d), b_2d.eval_qvec(res.u_safe))
                x_next = qvec_add(x_2d, qvec_scale((1, 20), dx))
                if diam_2d.eval_scalar(x_next)[0] < 0:
                    breaches += 1

            # ------------------------------------------------------------------
            # 2. 4D Planar Double Integrator: Multi-Facet Corridor & Reflex Diamond
            #    (220 deterministic adversarial queries)
            # ------------------------------------------------------------------
            (
                corr_4d,
                diam_4d,
                plant_c4,
                plant_d4,
                _,
                _,
                _,
            ) = build_double_integrator_4d_corridor_diamond()
            sh_corr_4d = RuntimeCBFShield(corr_4d, plant_c4, alpha_gain=(1, 1))
            diam_4d_pwa = diam_4d.enumerate_pwa(plant_d4.domain)
            f_4d = plant_c4.cells[0].f_map
            b_4d = AffineMapQ.create(2, 4, plant_c4.cells[0].B_num, [0, 0, 0, 0], 1)

            # 110 queries near 4D multi-facet corridor corners (simultaneously active x and y braking facets)
            for k in range(110):
                vx_n = 16 - (k % 5) if k % 2 == 0 else -(16 - (k % 5))
                vy_n = 16 - ((k // 2) % 5) if (k // 3) % 2 == 0 else -(16 - ((k // 2) % 5))
                # Set px = sgn * (2 - 2*|vx|) * 16/16 - slack so |px + 2 vx| <= 2
                slack_x = k % 4
                slack_y = (k * 3) % 4
                if vx_n >= 0:
                    px_n = 32 - 2 * vx_n - slack_x
                else:
                    px_n = -32 - 2 * vx_n + slack_x
                if vy_n >= 0:
                    py_n = 32 - 2 * vy_n - slack_y
                else:
                    py_n = -32 - 2 * vy_n + slack_y
                x_4d: QVec = qvec_norm([px_n, py_n, vx_n, vy_n], 16)
                self.assertGreaterEqual(corr_4d.eval_scalar(x_4d)[0], 0)

                # Adversarial 2D control trying to breach both x and y braking walls simultaneously
                ux_adv = 4 if vx_n >= 0 else -4
                uy_adv = 4 if vy_n >= 0 else -4
                u_nom_4d: QVec = ((ux_adv, uy_adv), 1)

                res4 = sh_corr_4d.step(x_4d, u_nom_4d)
                solve_times_us.append(res4.solve_time_us)
                viable_count += 1
                if res4.verdict != Verdict.CERTIFIED_EXACT or res4.u_safe is None:
                    breaches += 1
                    continue
                self.assertTrue(check_shield_step_result(res4))
                if res4.intervened:
                    intervened_count += 1
                dx4 = qvec_add(f_4d.eval_qvec(x_4d), b_4d.eval_qvec(res4.u_safe))
                x4_next = qvec_add(x_4d, qvec_scale((1, 20), dx4))
                if corr_4d.eval_scalar(x4_next)[0] < 0:
                    breaches += 1

            # 110 queries near 4D reflex diamond-obstacle corners & creases:
            # |p_x + 2 v_x| + |p_y + 2 v_y| - 1/2 >= 0 combined with velocity box |v_x|<=1, |v_y|<=1
            vel_facets = (
                AffineMapQ.create(4, 1, [[0, 0, -1, 0]], [1], 1),
                AffineMapQ.create(4, 1, [[0, 0, 1, 0]], [1], 1),
                AffineMapQ.create(4, 1, [[0, 0, 0, -1]], [1], 1),
                AffineMapQ.create(4, 1, [[0, 0, 0, 1]], [1], 1),
            )
            for k in range(110):
                # Place (qx, qy) = (px + 2 vx, py + 2 vy) right on or near the 4 reflex vertices
                # (+-1/2, 0), (0, +-1/2), and diagonal diamond facets |qx| + |qy| >= 1/2!
                mode = k % 5
                extra = k % 5  # in 0..4 / 32
                if mode == 0:
                    qx_n, qy_n = 16 + extra, 0  # reflex corner (+1/2, 0) on crease qy == 0!
                elif mode == 1:
                    qx_n, qy_n = -(16 + extra), 0  # reflex corner (-1/2, 0) on crease qy == 0!
                elif mode == 2:
                    qx_n, qy_n = 0, 16 + extra  # reflex corner (0, +1/2) on crease qx == 0!
                elif mode == 3:
                    qx_n, qy_n = 0, -(16 + extra)  # reflex corner (0, -1/2) on crease qx == 0!
                else:
                    sx = 1 if (k // 5) % 2 == 0 else -1
                    sy = 1 if (k // 10) % 2 == 0 else -1
                    qx_n, qy_n = sx * (9 + extra), sy * 8  # |qx| + |qy| = (17+extra)/32 > 1/2

                vx_n = ((k * 5) % 17) - 8  # in [-8/32, 8/32] = [-1/4, 1/4]
                vy_n = ((k * 7) % 17) - 8
                px_n = qx_n - 2 * vx_n
                py_n = qy_n - 2 * vy_n
                x_4d = qvec_norm([px_n, py_n, vx_n, vy_n], 32)
                self.assertGreaterEqual(diam_4d.eval_scalar(x_4d)[0], 0)

                # Adversarial nominal control aiming straight into the origin of stopping space (qx, qy) -> (0, 0)
                ux_adv = -2 if qx_n >= 0 else 2
                uy_adv = -2 if qy_n >= 0 else 2
                u_nom_4d = ((ux_adv, uy_adv), 1)

                # Evaluate disjunctive active diamond branches + velocity limits (exact reflex NCBF filter)
                valid_branches = [
                    hc
                    for hc in diam_4d_pwa.cells
                    if hc.polytope.contains(x_4d) and hc.affine_map.eval_scalar(x_4d)[0] >= 0
                ]
                best_d_res: Optional[ShieldStepResult] = None
                for hc in valid_branches:
                    jb = MultiFacetBarrier(4, plant_d4.domain, vel_facets + (hc.affine_map,))
                    sh_jb = RuntimeCBFShield(jb, plant_d4, alpha_gain=(2, 1))
                    cand_r = sh_jb.step(x_4d, u_nom_4d)
                    solve_times_us.append(cand_r.solve_time_us)
                    if cand_r.verdict == Verdict.CERTIFIED_EXACT:
                        if (
                            best_d_res is None
                            or best_d_res.l1_intervention is None
                            or (
                                cand_r.l1_intervention is not None
                                and q_lt(cand_r.l1_intervention, best_d_res.l1_intervention)
                            )
                        ):
                            best_d_res = cand_r

                viable_count += 1
                if best_d_res is None or best_d_res.u_safe is None:
                    breaches += 1
                    continue
                self.assertTrue(check_shield_step_result(best_d_res))
                if best_d_res.intervened:
                    intervened_count += 1
                dx4 = qvec_add(f_4d.eval_qvec(x_4d), b_4d.eval_qvec(best_d_res.u_safe))
                x4_next = qvec_add(x_4d, qvec_scale((1, 20), dx4))
                if diam_4d.eval_scalar(x4_next)[0] < 0:
                    breaches += 1

            # ------------------------------------------------------------------
            # 3. 3D 3-Sector Dubins Obstacle Avoidance: Reflex Creases p_x = +-1
            #    & Sector Transitions theta = +-1/3 (120 deterministic queries)
            # ------------------------------------------------------------------
            h_dub, p_dub, _, _, _ = build_dubins_3sector_obstacle_3d()
            sh_dub = RuntimeCBFShield(h_dub, p_dub, alpha_gain=(1, 1))
            B_dub = AffineMapQ.create(1, 3, [[0], [0], [1]], [0, 0, 0], 1)

            for k in range(120):
                # Focus near ReLU creases p_x = -1, 0, +1 and heading sector boundaries theta = -1/3, +1/3
                px_choices = [-24, -16, -15, 0, 15, 16, 24]  # / 16 -> includes exact -1 and +1!
                th_choices = [-12, -6, -5, 0, 5, 6, 12]      # / 18 -> includes exact -1/3 and +1/3!
                px_q = q_norm(px_choices[k % len(px_choices)], 16)
                th_q = q_norm(th_choices[(k // 3) % len(th_choices)], 18)
                # Compute minimal safe p_y so h_dub(px, py, theta) >= margin >= 0:
                # h(x) = p_y - (1/4)ReLU(1 - px) - (1/4)ReLU(1 + px) + (1/2)theta + 1 >= 0
                base_h_at_py0 = h_dub.eval_scalar(
                    qvec_norm([px_q[0] * th_q[1], 0, th_q[0] * px_q[1]], px_q[1] * th_q[1])
                )
                margin = q_norm(k % 5, 16)  # includes exact boundary h(x) == 0 when k % 5 == 0!
                py_q = q_add(margin, (-base_h_at_py0[0], base_h_at_py0[1]))
                cd3 = px_q[1] * py_q[1] * th_q[1]
                x_3d: QVec = qvec_norm(
                    [
                        px_q[0] * py_q[1] * th_q[1],
                        py_q[0] * px_q[1] * th_q[1],
                        th_q[0] * px_q[1] * py_q[1],
                    ],
                    cd3,
                )
                self.assertGreaterEqual(h_dub.eval_scalar(x_3d)[0], 0)

                # Adversarial steering command diving downward into obstacle (u_nom in {-6, -4, -3})
                u_nom_3d: QVec = ((-6 + (k % 3),), 1)
                res_dub = sh_dub.step(x_3d, u_nom_3d)
                solve_times_us.append(res_dub.solve_time_us)
                viable_count += 1
                if res_dub.verdict != Verdict.CERTIFIED_EXACT or res_dub.u_safe is None:
                    breaches += 1
                    continue
                self.assertTrue(check_shield_step_result(res_dub))
                if res_dub.intervened:
                    intervened_count += 1

                # Check Euler step dt = 1/30 stays h >= 0
                p_cell = [pc for pc in p_dub.cells if pc.polytope.contains(x_3d)][0]
                dx3 = qvec_add(p_cell.f_map.eval_qvec(x_3d), B_dub.eval_qvec(res_dub.u_safe))
                x3_next = qvec_add(x_3d, qvec_scale((1, 30), dx3))
                if h_dub.eval_scalar(x3_next)[0] < 0:
                    breaches += 1

            # ------------------------------------------------------------------
            # 4. Intentionally Trapped States: Assert Exact Farkas Deadlock Duals
            #    across 2D DI, 4D DI, and 3D Dubins (30 deterministic trapped states)
            # ------------------------------------------------------------------
            deadlock_certified = 0
            # (a) 10 trapped 2D Double Integrator states: naive box |p| <= 1, |v| <= 2 at p = 1, v = 3/2..2
            # where dp/dt = v > 1 and |u| <= 1 cannot arrest forward motion in 1st-order box!
            naive_box_2d = MultiFacetBarrier(
                2,
                plant_2d.domain,
                (
                    AffineMapQ.create(2, 1, [[-1, 0]], [1], 1),  # 1 - p >= 0
                    AffineMapQ.create(2, 1, [[1, 0]], [1], 1),   # 1 + p >= 0
                ),
                name="naive_pos_box_2d",
            )
            sh_trapped_2d = RuntimeCBFShield(naive_box_2d, plant_2d, alpha_gain=(1, 1))
            for k in range(10):
                # At p = 1, v = 1 + (k+1)/10 > 0: h_0(x) = 0, grad h_0 . (v, u) + h_0 = -v < 0 for ALL u!
                x_trap_2d: QVec = qvec_norm([10, 11 + k], 10)
                r_t2 = sh_trapped_2d.step(x_trap_2d, ((-1,), 1))
                solve_times_us.append(r_t2.solve_time_us)
                self.assertEqual(r_t2.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
                self.assertIsNotNone(r_t2.deadlock_cert)
                self.assertTrue(check_shield_step_result(r_t2))
                assert r_t2.deadlock_cert is not None
                self.assertTrue(check_farkas_certificate(r_t2.lp_A, r_t2.lp_b, r_t2.deadlock_cert.y))
                deadlock_certified += 1

            # (b) 10 trapped 4D Double Integrator states: high speed at Braking Corridor violation
            # e.g., at p_x = 2, v_x = 2 (where 2 - p_x - 2 v_x = -4, so even max braking u_x = -1 gives
            # -v_x - 2 u_x + h = -2 + 2 - 4 = -4 < 0!)
            for k in range(10):
                x_trap_4d: QVec = qvec_norm([20 + k, 0, 20, 0], 10)
                r_t4 = sh_corr_4d.step(x_trap_4d, ((-1, 0), 1))
                solve_times_us.append(r_t4.solve_time_us)
                self.assertEqual(r_t4.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
                self.assertIsNotNone(r_t4.deadlock_cert)
                self.assertTrue(check_shield_step_result(r_t4))
                assert r_t4.deadlock_cert is not None
                self.assertTrue(check_farkas_certificate(r_t4.lp_A, r_t4.lp_b, r_t4.deadlock_cert.y))
                deadlock_certified += 1

            # (c) 10 trapped 3D Dubins states: deep inside obstacle with downward heading theta = -1
            # where max upward turn u = +3 gives grad_h . (f + B u) + alpha * h < 0
            for k in range(10):
                # p_x = 0, p_y = -2 - k/10, theta = -1 -> h = p_y - 1/2 - 1/2 + 1 = p_y <= -2
                # L_f h + L_g h u + h = theta + (1/2)*u + p_y <= -1 + 3/2 - 2 = -3/2 < 0 for all |u| <= 3!
                x_trap_3d: QVec = qvec_norm([0, -(20 + k), -10], 10)
                r_t3 = sh_dub.step(x_trap_3d, ((3,), 1))
                solve_times_us.append(r_t3.solve_time_us)
                self.assertEqual(r_t3.verdict, Verdict.FALSIFIED_COUNTEREXAMPLE)
                self.assertIsNotNone(r_t3.deadlock_cert)
                self.assertTrue(check_shield_step_result(r_t3))
                assert r_t3.deadlock_cert is not None
                self.assertTrue(check_farkas_certificate(r_t3.lp_A, r_t3.lp_b, r_t3.deadlock_cert.y))
                deadlock_certified += 1

            # Global assertions across all 570 deterministic shield solves
            self.assertGreaterEqual(viable_count, 540)
            self.assertGreaterEqual(intervened_count, 400)
            self.assertEqual(breaches, 0, f"Expected 0 barrier breaches on viable states, got {breaches}")
            self.assertEqual(deadlock_certified, 30)

            mean_latency_ms = (sum(solve_times_us) / len(solve_times_us)) / 1000.0
            self.assertLess(
                mean_latency_ms,
                1.0,
                f"Mean Bareiss CBF-LP latency {mean_latency_ms:.3f} ms exceeded 1.0 ms target",
            )


if __name__ == "__main__":
    unittest.main()
