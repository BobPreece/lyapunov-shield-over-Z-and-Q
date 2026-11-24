from __future__ import annotations

import unittest
from typing import List, Sequence, Tuple

from lyapunov_shield import (
    AffineMapQ,
    BudgetExceededError,
    FarkasBoundCert,
    FarkasInfeasibilityCert,
    HPolytope,
    LPStatus,
    PWACell,
    PWAFunction,
    QVec,
    ReLULayer,
    ReLUNetwork,
    Verdict,
    WatchdogTimer,
    bareiss_det,
    bareiss_echelon,
    bareiss_nullspace,
    bareiss_rank,
    bft_merge_cells,
    certify_linear_upper_bound,
    chebyshev_ball_linf,
    check_farkas_certificate,
    check_positivity_farkas,
    check_primal_witness,
    compose_pullback_pwa,
    ct_bab_planet_upper_bound,
    enumerate_hyperplane_arrangement,
    q_le,
    q_lt,
    q_norm,
    qvec_eq,
    qvec_norm,
    solve_linear_system_int,
    solve_lp_bareiss,
    solve_one_minus_m_q,
)


class TestGate1InquisitorRedTeam(unittest.TestCase):
    """Adversarial red-team test suite for Gate 1 exact Z/Q kernel modules."""

    def test_01_degenerate_negative_rhs_tied_min_phase1(self) -> None:
        """1. Degenerate negative-RHS tied-minimum Phase-I polytopes (singletons, lines, infeasible)."""
        with WatchdogTimer(15.0):
            # Case 1A: Feasible 2D singleton x* = (5, 5) with 4-way tied min_b == -5
            # plus duplicate tied negative-RHS rows (6 tight constraints at 0D vertex).
            # Phase I must drive z0 to 0 AND pivot z0 out of basis_p1 cleanly.
            A_sing = [
                [-1, 0],   # -x1 <= -5 (b_red = -5)
                [0, -1],   # -x2 <= -5 (b_red = -5)
                [1, -2],   # x1 - 2*x2 <= -5 (b_red = -5)
                [-2, 1],   # -2*x1 + x2 <= -5 (b_red = -5)
                [-1, 0],   # exact duplicate tied min_b = -5 row
                [1, 0],    # x1 <= 5
                [0, 1],    # x2 <= 5
            ]
            b_sing = [-5, -5, -5, -5, -5, 5, 5]
            for c_obj, expected_opt in (
                ([1, 1], (10, 1)),
                ([-3, 2], (-5, 1)),
                ([0, 0], (0, 1)),
            ):
                res = solve_lp_bareiss(A_sing, b_sing, c=c_obj)
                self.assertEqual(res.status, LPStatus.OPTIMAL)
                self.assertEqual(res.x_opt, ((5, 5), 1))
                self.assertEqual(res.obj_val, expected_opt)
                self.assertGreaterEqual(res.phase1_pivots, 1)
                assert res.x_opt is not None and res.dual_y is not None
                self.assertTrue(check_primal_witness(A_sing, b_sing, res.x_opt))
                bcert = FarkasBoundCert(
                    y=res.dual_y,
                    scale=res.dual_scale,
                    c_num=tuple(c_obj),
                    c_den=1,
                    rhs_num=expected_opt[0],
                    rhs_den=expected_opt[1],
                )
                self.assertTrue(check_positivity_farkas(A_sing, b_sing, bcert))

            # Case 1B: Feasible 1D affine line L = {(3+t, 6+t, 6+t) : t in R} in R^3
            # with 4-way tied negative RHS min_b == -3 and all 6 rows tight on all of L.
            A_line = [
                [1, -1, 0],   # x1 - x2 <= -3 (b_red = -3)
                [1, 0, -1],   # x1 - x3 <= -3 (b_red = -3)
                [1, 1, -2],   # x1 + x2 - 2*x3 <= -3 (b_red = -3)
                [1, -2, 1],   # x1 - 2*x2 + x3 <= -3 (b_red = -3)
                [-1, 1, 0],   # -x1 + x2 <= 3
                [-1, 0, 1],   # -x1 + x3 <= 3
            ]
            b_line = [-3, -3, -3, -3, 3, 3]
            # Orthogonal objective [-2, 1, 1] is constant 6 on L -> OPTIMAL
            res_line_opt = solve_lp_bareiss(A_line, b_line, c=[-2, 1, 1])
            self.assertEqual(res_line_opt.status, LPStatus.OPTIMAL)
            self.assertEqual(res_line_opt.obj_val, (6, 1))
            assert res_line_opt.x_opt is not None
            self.assertTrue(check_primal_witness(A_line, b_line, res_line_opt.x_opt))

            # Along-line objective [1, 1, 1] -> Phase I exits with z0 out of basis, Phase II UNBOUNDED
            res_line_unb = solve_lp_bareiss(A_line, b_line, c=[1, 1, 1])
            self.assertEqual(res_line_unb.status, LPStatus.UNBOUNDED)
            assert res_line_unb.x_opt is not None and res_line_unb.unbounded_ray is not None
            self.assertTrue(check_primal_witness(A_line, b_line, res_line_unb.x_opt))

            # Case 1C: Infeasible 3D irreducible inconsistent subsystem (IIS) with 4-way tied min_b == -4
            # (every 3-row subset is feasible; positive sum 1+1+1+1 yields 0 <= -16).
            A_iis = [
                [2, -1, 0],
                [0, 2, -1],
                [-1, 0, 2],
                [-1, -1, -1],
            ]
            b_iis = [-4, -4, -4, -4]
            res_inf = solve_lp_bareiss(A_iis, b_iis, c=[1, -2, 3])
            self.assertEqual(res_inf.status, LPStatus.INFEASIBLE)
            assert res_inf.farkas_infeas is not None
            self.assertTrue(
                check_farkas_certificate(A_iis, b_iis, res_inf.farkas_infeas.y)
            )
            self.assertTrue(all(yi > 0 for yi in res_inf.farkas_infeas.y))

            # Case 1D: Infeasible perturbed line (5-way tied min_b == -3)
            A_line_inf = A_line + [[-1, -1, 2]]  # on L, -x1 - x2 + 2*x3 == +3 <= -3 impossible
            b_line_inf = b_line + [-3]
            res_line_inf = solve_lp_bareiss(A_line_inf, b_line_inf, c=[0, 0, 1])
            self.assertEqual(res_line_inf.status, LPStatus.INFEASIBLE)
            assert res_line_inf.farkas_infeas is not None
            self.assertTrue(
                check_farkas_certificate(
                    A_line_inf, b_line_inf, res_line_inf.farkas_infeas.y
                )
            )

    def test_02_unbounded_cones_counterexample_extraction(self) -> None:
        """2. Unbounded polyhedral cones extracting finite rational counterexamples along rays."""
        with WatchdogTimer(15.0):
            # Shifted pointed cone in R^2 requiring Phase I (negative RHS) + unbounded ray in R_+^2:
            # x1 - x2 <= -3, -2*x1 + x2 <= -5  =>  x1 >= 8, x2 >= 11, ray (t, 2t) for t >= 0
            A_cone = [
                [1, -1],
                [-2, 1],
            ]
            b_cone = [-3, -5]

            # Test huge positive threshold (10^12 / 7) with non-unit c_den = 13, both non-strict and strict
            c_num = [3, 5]
            c_den = 13
            rhs_num = 10**12
            rhs_den = 7
            for strict_flag in (False, True):
                verdict, cert, wit, pivs = certify_linear_upper_bound(
                    A_cone,
                    b_cone,
                    c_num=c_num,
                    c_den=c_den,
                    rhs_num=rhs_num,
                    rhs_den=rhs_den,
                    strict=strict_flag,
                )
                self.assertEqual(verdict, Verdict.FALSIFIED_COUNTEREXAMPLE.value)
                self.assertIsNone(cert)
                self.assertIsNotNone(wit)
                self.assertGreaterEqual(pivs, 1)
                assert wit is not None
                self.assertTrue(check_primal_witness(A_cone, b_cone, wit))
                wn, wd = wit
                self.assertGreater(wd, 0)
                # Verify exact rational inequality (c_num^T wit) / c_den > rhs_num / rhs_den
                lhs_cross = rhs_den * (c_num[0] * wn[0] + c_num[1] * wn[1])
                rhs_cross = c_den * wd * rhs_num
                self.assertGreater(lhs_cross, rhs_cross)

            # Dual-cone bounded direction on the SAME unbounded cone:
            # c = [-3, -1] = 1 * [1, -1] + 2 * [-2, 1] <= (-3) + 2*(-5) = -13!
            verdict_ok, cert_ok, wit_ok, _ = certify_linear_upper_bound(
                A_cone, b_cone, [-3, -1], 5, -13, 5, strict=False
            )
            self.assertEqual(verdict_ok, Verdict.CERTIFIED_EXACT.value)
            self.assertIsNone(wit_ok)
            assert cert_ok is not None
            self.assertTrue(check_positivity_farkas(A_cone, b_cone, cert_ok))

            # Unbounded non-pointed wedge in R^3 + unconstrained R^3 (m == 0)
            A_wedge = [[1, 1, 0], [1, -1, 0]]  # x1 <= -|x2|, x3 completely free
            b_wedge = [0, 0]
            v_w, c_w, wit_w, _ = certify_linear_upper_bound(
                A_wedge, b_wedge, [2, 0, 7], 3, 500_000, 11, strict=True
            )
            self.assertEqual(v_w, Verdict.FALSIFIED_COUNTEREXAMPLE.value)
            self.assertIsNone(c_w)
            assert wit_w is not None
            self.assertTrue(check_primal_witness(A_wedge, b_wedge, wit_w))
            self.assertGreater(
                11 * (2 * wit_w[0][0] + 7 * wit_w[0][2]),
                3 * wit_w[1] * 500_000,
            )

            v_rn, c_rn, wit_rn, _ = certify_linear_upper_bound(
                [], [], [-4, 9, -2], 7, 99_999, 3, strict=False
            )
            self.assertEqual(v_rn, Verdict.FALSIFIED_COUNTEREXAMPLE.value)
            self.assertIsNone(c_rn)
            assert wit_rn is not None
            self.assertTrue(check_primal_witness([], [], wit_rn))
            self.assertGreater(
                3 * (-4 * wit_rn[0][0] + 9 * wit_rn[0][1] - 2 * wit_rn[0][2]),
                7 * wit_rn[1] * 99_999,
            )

            # Strictly negative-orthant cone where v0 > 0 enters unbounded ray (-t, -t, -t)
            A_neg_cone = [[-1, 1, 0], [0, -1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            b_neg_cone = [0, 0, -2, -2, -2]
            v_neg, c_neg, wit_neg, _ = certify_linear_upper_bound(
                A_neg_cone, b_neg_cone, [-1, -2, -3], 5, 10**9, 3, strict=True
            )
            self.assertEqual(v_neg, Verdict.FALSIFIED_COUNTEREXAMPLE.value)
            self.assertIsNone(c_neg)
            assert wit_neg is not None
            self.assertTrue(check_primal_witness(A_neg_cone, b_neg_cone, wit_neg))
            self.assertGreater(
                3 * (-wit_neg[0][0] - 2 * wit_neg[0][1] - 3 * wit_neg[0][2]),
                5 * wit_neg[1] * (10**9),
            )

    def test_03_one_bit_and_pm1_certificate_and_witness_tampering(self) -> None:
        """3. Exhaustive bit-flip / +1 / -1 tampering on dual certificates and primal witnesses."""
        with WatchdogTimer(15.0):
            # 3A: FarkasInfeasibilityCert.y tampering on 3D infeasible system (no zero rows)
            A_inf = [
                [3, -2, 1],
                [-1, 4, -3],
                [-2, -2, 2],
                [1, 1, 1],
            ]
            b_inf = [-2, -5, -1, 10]
            res_inf = solve_lp_bareiss(A_inf, b_inf, c=[1, 0, 0])
            self.assertEqual(res_inf.status, LPStatus.INFEASIBLE)
            assert res_inf.farkas_infeas is not None
            y_inf = res_inf.farkas_infeas.y
            self.assertTrue(check_farkas_certificate(A_inf, b_inf, y_inf))

            for idx in range(len(y_inf)):
                for delta in (1, -1):
                    tampered_y = list(y_inf)
                    tampered_y[idx] += delta
                    self.assertFalse(
                        check_farkas_certificate(A_inf, b_inf, tampered_y)
                    )
                for bit in range(16):
                    tampered_y = list(y_inf)
                    tampered_y[idx] ^= 1 << bit
                    self.assertFalse(
                        check_farkas_certificate(A_inf, b_inf, tampered_y)
                    )
            self.assertFalse(check_farkas_certificate(A_inf, b_inf, [0] * len(y_inf)))
            self.assertFalse(check_farkas_certificate(A_inf, b_inf, y_inf[:-1]))

            # 3B & 3C: FarkasBoundCert.y and FarkasBoundCert.scale tampering (non-vacuous & vacuous)
            A_bnd = [
                [2, 3, -1],
                [-3, 1, 4],
                [1, -4, 2],
                [-1, 0, -3],
                [0, 2, 1],
                [4, -1, -2],
            ]
            b_bnd = [7, 11, 5, 6, 8, 9]
            res_opt = solve_lp_bareiss(A_bnd, b_bnd, c=[5, 3, 2])
            self.assertEqual(res_opt.status, LPStatus.OPTIMAL)
            assert res_opt.obj_val is not None
            opt_n, opt_d = res_opt.obj_val

            verdict_b, cert_b, _, _ = certify_linear_upper_bound(
                A_bnd, b_bnd, [5, 3, 2], 7, opt_n, 7 * opt_d, strict=False
            )
            self.assertEqual(verdict_b, Verdict.CERTIFIED_EXACT.value)
            assert cert_b is not None
            self.assertTrue(check_positivity_farkas(A_bnd, b_bnd, cert_b, strict=False))
            # Tight upper bound must fail strict=True and off-by-one rhs_num - 1!
            self.assertFalse(check_positivity_farkas(A_bnd, b_bnd, cert_b, strict=True))
            tighter_rhs_cert = FarkasBoundCert(
                y=cert_b.y,
                scale=cert_b.scale,
                c_num=cert_b.c_num,
                c_den=cert_b.c_den,
                rhs_num=cert_b.rhs_num - 1,
                rhs_den=cert_b.rhs_den,
                is_vacuous=False,
            )
            self.assertFalse(
                check_positivity_farkas(A_bnd, b_bnd, tighter_rhs_cert, strict=False)
            )

            # Tamper every entry of FarkasBoundCert.y (+1, -1, 16 bit flips)
            for idx in range(len(cert_b.y)):
                for delta in (1, -1):
                    ty = list(cert_b.y)
                    ty[idx] += delta
                    tc = FarkasBoundCert(
                        y=tuple(ty),
                        scale=cert_b.scale,
                        c_num=cert_b.c_num,
                        c_den=cert_b.c_den,
                        rhs_num=cert_b.rhs_num,
                        rhs_den=cert_b.rhs_den,
                        is_vacuous=False,
                    )
                    self.assertFalse(check_positivity_farkas(A_bnd, b_bnd, tc))
                for bit in range(16):
                    ty = list(cert_b.y)
                    ty[idx] ^= 1 << bit
                    tc = FarkasBoundCert(
                        y=tuple(ty),
                        scale=cert_b.scale,
                        c_num=cert_b.c_num,
                        c_den=cert_b.c_den,
                        rhs_num=cert_b.rhs_num,
                        rhs_den=cert_b.rhs_den,
                        is_vacuous=False,
                    )
                    self.assertFalse(check_positivity_farkas(A_bnd, b_bnd, tc))

            # Tamper FarkasBoundCert.scale (+1, -1, 0, -scale, 16 bit flips)
            for bad_scale in [
                cert_b.scale + 1,
                cert_b.scale - 1,
                0,
                -cert_b.scale,
            ] + [cert_b.scale ^ (1 << bit) for bit in range(16)]:
                tc_s = FarkasBoundCert(
                    y=cert_b.y,
                    scale=bad_scale,
                    c_num=cert_b.c_num,
                    c_den=cert_b.c_den,
                    rhs_num=cert_b.rhs_num,
                    rhs_den=cert_b.rhs_den,
                    is_vacuous=False,
                )
                self.assertFalse(check_positivity_farkas(A_bnd, b_bnd, tc_s))

            # Also tamper vacuous FarkasBoundCert (scale == 0 required)
            v_vac, cert_vac, _, _ = certify_linear_upper_bound(
                A_inf, b_inf, [5, -3, 1], 3, -999, 1, strict=True
            )
            self.assertEqual(v_vac, Verdict.CERTIFIED_EXACT.value)
            assert cert_vac is not None and cert_vac.is_vacuous
            self.assertTrue(check_positivity_farkas(A_inf, b_inf, cert_vac, strict=True))
            for bad_vac_scale in [1, -1] + [1 << bit for bit in range(8)]:
                tc_vac = FarkasBoundCert(
                    y=cert_vac.y,
                    scale=bad_vac_scale,
                    c_num=cert_vac.c_num,
                    c_den=cert_vac.c_den,
                    rhs_num=cert_vac.rhs_num,
                    rhs_den=cert_vac.rhs_den,
                    is_vacuous=True,
                )
                self.assertFalse(
                    check_positivity_farkas(A_inf, b_inf, tc_vac, strict=True)
                )

            # 3D: Primal witness x_q tampering on non-integer rational singleton ((11, -17, 23), 13)
            # Invertible unimodular matrix U (det == -1) with U * (11, -17, 23)^T = (17, 5, 52)^T
            U_rows = [
                [1, 1, 1],    # 11 - 17 + 23 = 17
                [2, 1, 0],    # 22 - 17 = 5
                [0, 1, 3],    # -17 + 69 = 52
            ]
            rhs_u = [17, 5, 52]
            A_wit: List[List[int]] = []
            b_wit: List[int] = []
            for r_idx in range(3):
                scaled_r = [13 * v for v in U_rows[r_idx]]
                A_wit.append(scaled_r)
                b_wit.append(rhs_u[r_idx])
                A_wit.append([-v for v in scaled_r])
                b_wit.append(-rhs_u[r_idx])

            res_wit = solve_lp_bareiss(A_wit, b_wit, c=[1, -1, 1])
            self.assertEqual(res_wit.status, LPStatus.OPTIMAL)
            assert res_wit.x_opt is not None
            x_q: QVec = res_wit.x_opt
            self.assertEqual(x_q, ((11, -17, 23), 13))
            self.assertTrue(check_primal_witness(A_wit, b_wit, x_q))

            nums, den = x_q
            for j in range(len(nums)):
                for delta in (1, -1):
                    t_nums = list(nums)
                    t_nums[j] += delta
                    self.assertFalse(
                        check_primal_witness(A_wit, b_wit, (tuple(t_nums), den))
                    )
                t_neg = list(nums)
                t_neg[j] = -t_neg[j]
                self.assertFalse(
                    check_primal_witness(A_wit, b_wit, (tuple(t_neg), den))
                )
                for bit in range(16):
                    t_nums = list(nums)
                    t_nums[j] ^= 1 << bit
                    self.assertFalse(
                        check_primal_witness(A_wit, b_wit, (tuple(t_nums), den))
                    )

            for bad_den in [den + 1, den - 1, 0, -den] + [
                den ^ (1 << bit) for bit in range(16)
            ]:
                self.assertFalse(
                    check_primal_witness(A_wit, b_wit, (nums, bad_den))
                )

    def test_04_non_primitive_scaled_rows_dual_lifting(self) -> None:
        """4. Non-primitive scaled rows (12, 18, 1000000) + tautological zero rows dual lifting."""
        with WatchdogTimer(15.0):
            # Base primitive 3D polytope
            A_prim = [
                [2, 1, -1],
                [-1, 3, 2],
                [1, -2, 4],
                [-3, -1, -2],
                [1, 1, 1],
            ]
            b_prim = [5, 8, 7, 6, 4]
            scales = [12, 18, 1_000_000, 720_720, 36]

            # Interleave scaled rows with tautological 0^T x <= b_i >= 0 rows so indices shift!
            A_scaled: List[List[int]] = [[0, 0, 0]]
            b_scaled: List[int] = [500]
            for i in range(len(A_prim)):
                sc = scales[i]
                A_scaled.append([sc * v for v in A_prim[i]])
                b_scaled.append(sc * b_prim[i])
                if i == 2:
                    A_scaled.append([0, 0, 0])
                    b_scaled.append(0)

            c_obj = [7, 4, 5]
            res_prim = solve_lp_bareiss(A_prim, b_prim, c=c_obj)
            res_scaled = solve_lp_bareiss(A_scaled, b_scaled, c=c_obj)
            self.assertEqual(res_prim.status, LPStatus.OPTIMAL)
            self.assertEqual(res_scaled.status, LPStatus.OPTIMAL)
            self.assertEqual(res_scaled.obj_val, res_prim.obj_val)
            self.assertEqual(res_scaled.x_opt, res_prim.x_opt)

            assert res_scaled.obj_val is not None and res_scaled.dual_y is not None
            # Tautological zero rows at indices 0 and 4 must carry dual weight 0
            self.assertEqual(res_scaled.dual_y[0], 0)
            self.assertEqual(res_scaled.dual_y[4], 0)

            v_sc, cert_sc, _, _ = certify_linear_upper_bound(
                A_scaled,
                b_scaled,
                c_obj,
                3,
                res_scaled.obj_val[0],
                3 * res_scaled.obj_val[1],
                strict=False,
            )
            self.assertEqual(v_sc, Verdict.CERTIFIED_EXACT.value)
            assert cert_sc is not None
            # Lifted Farkas dual must validate on the EXACT original unscaled (A_scaled, b_scaled)
            self.assertTrue(
                check_positivity_farkas(A_scaled, b_scaled, cert_sc, strict=False)
            )
            # Verify directly via integer identity c_den * (y^T A_scaled) == scale * c_obj
            for j in range(3):
                col_dot = sum(
                    cert_sc.y[i] * A_scaled[i][j] for i in range(len(A_scaled))
                )
                self.assertEqual(cert_sc.c_den * col_dot, cert_sc.scale * c_obj[j])

            # Also verify non-primitive scaling on an INFEASIBLE system (12, 18, 1000000)
            A_inf_scaled = [
                [0, 0],
                [12 * 1, 12 * 2],
                [18 * (-3), 18 * (-1)],
                [1_000_000 * 2, 1_000_000 * (-1)],
            ]
            b_inf_scaled = [25, 12 * (-3), 18 * (-2), 1_000_000 * (-4)]
            # Sum of primitive rows: (1 - 3 + 2, 2 - 1 - 1) = (0, 0) <= -3 - 2 - 4 = -9 < 0
            res_inf_sc = solve_lp_bareiss(A_inf_scaled, b_inf_scaled, c=[1, 1])
            self.assertEqual(res_inf_sc.status, LPStatus.INFEASIBLE)
            assert res_inf_sc.farkas_infeas is not None
            self.assertEqual(res_inf_sc.farkas_infeas.y[0], 0)
            self.assertTrue(
                check_farkas_certificate(
                    A_inf_scaled, b_inf_scaled, res_inf_sc.farkas_infeas.y
                )
            )

    def test_05_sliver_1e_minus_15_vs_zero_width_equality_facets(self) -> None:
        """5. High-aspect-ratio 10^-15 slivers vs explicit and implicit 0-width equality facets."""
        with WatchdogTimer(15.0):
            s15 = 10**15

            # 5A: Off-origin [1, 2] 2D sliver of width 10^-15 (where float64 1.0 + 2.5e-16 collapses)
            # 1 <= x1 <= 2, x1 <= x2 <= x1 + 10^-15
            sliver_2d = HPolytope.create(
                2,
                [
                    [-1, 0],
                    [1, 0],
                    [1, -1],
                    [-s15, s15],
                ],
                [-1, 2, 0, 1],
            )
            ok_2d, r_2d, xc_2d = sliver_2d.is_full_dimensional()
            self.assertTrue(ok_2d)
            self.assertEqual(r_2d, (1, 4 * s15))
            assert r_2d is not None and xc_2d is not None
            self.assertTrue(sliver_2d.contains(xc_2d))
            # Verify all 4 corners of B_inf(xc_2d, r_2d) lie inside sliver_2d
            cxn, cxd = xc_2d
            rn, rd = r_2d
            common_d = 4 * s15 * cxd
            for sgn0 in (-1, 1):
                for sgn1 in (-1, 1):
                    corner = qvec_norm(
                        [
                            cxn[0] * (common_d // cxd) + sgn0 * rn * (common_d // rd),
                            cxn[1] * (common_d // cxd) + sgn1 * rn * (common_d // rd),
                        ],
                        common_d,
                    )
                    self.assertTrue(sliver_2d.contains(corner))

            # 5B: 3D skewed 10^-15 sliver: 0 <= x1, x2 <= 1, x1 + 2*x2 <= x3 <= x1 + 2*x2 + 10^-15
            # ||(1, 2, -1)||_1 == 4 => 8 * r <= 10^-15 => r* == 1 / (8 * 10^15)
            sliver_3d = HPolytope.create(
                3,
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [1, 2, -1],
                    [-s15, -2 * s15, s15],
                ],
                [0, 1, 0, 1, 0, 1],
            )
            ok_3d, r_3d, xc_3d = sliver_3d.is_full_dimensional()
            self.assertTrue(ok_3d)
            self.assertEqual(r_3d, (1, 8 * s15))
            assert xc_3d is not None
            self.assertTrue(sliver_3d.contains(xc_3d))

            # 5C: True 0-width explicit equality facet: a^T x <= b and -a^T x <= -b (with non-primitive scales)
            eq_facet_2d = HPolytope.create(
                2,
                [
                    [-1, 0],
                    [1, 0],
                    [0, -1],
                    [0, 1],
                    [12 * 3, 12 * (-5)],      # 3*x1 - 5*x2 <= 1
                    [18 * (-3), 18 * 5],      # -3*x1 + 5*x2 <= -1
                ],
                [0, 2, 0, 2, 12 * 1, 18 * (-1)],
            )
            feas_eq, wit_eq = eq_facet_2d.is_feasible()
            self.assertTrue(feas_eq)
            assert wit_eq is not None
            self.assertTrue(eq_facet_2d.contains(wit_eq))
            ok_eq, r_eq, _ = eq_facet_2d.is_full_dimensional()
            self.assertFalse(ok_eq)
            self.assertEqual(r_eq, (0, 1))
            # Also confirm raw chebyshev_ball_linf LP directly returns (0, 1)
            r_raw, _, lp_raw = chebyshev_ball_linf(eq_facet_2d.A, eq_facet_2d.b)
            self.assertEqual(lp_raw.status, LPStatus.OPTIMAL)
            self.assertEqual(r_raw, (0, 1))

            # 5D: Implicit 0-width 1D segment in 3D with NO opposing (a, -a) row pair
            # x1 + x2 <= 1, x2 + x3 <= 1, -x1 - 2*x2 - x3 <= -2 inside [0, 1]^3
            # Feasible along {(t, 1-t, t) : t in [0, 1]}, but Chebyshev radius r* == 0!
            implicit_eq_3d = HPolytope.create(
                3,
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1],
                    [-1, -2, -1],
                ],
                [0, 1, 0, 1, 0, 1, 1, 1, -2],
            )
            feas_imp, wit_imp = implicit_eq_3d.is_feasible()
            self.assertTrue(feas_imp)
            assert wit_imp is not None
            self.assertTrue(implicit_eq_3d.contains(wit_imp))
            ok_imp, r_imp, _ = implicit_eq_3d.is_full_dimensional()
            self.assertFalse(ok_imp)
            self.assertEqual(r_imp, (0, 1))

    def test_06_two_hidden_layer_relu_enumerate_pwa_exact_grid(self) -> None:
        """6. 2-hidden-layer ReLU network enumerate_pwa exact pointwise agreement on 49 rational points."""
        with WatchdogTimer(15.0):
            domain = HPolytope.from_box([((-2, 1), (2, 1)), ((-2, 1), (2, 1))])

            # 2-hidden-layer ReLU MLP (2 -> 3 -> 3 -> 2) with rational denominators 3, 5, 2,
            # residual skip map, and equilibrium anchor zeroing at (1/3, -1/2).
            layer1 = ReLULayer(
                AffineMapQ.create(
                    2,
                    3,
                    [
                        [3, -2],
                        [-1, 4],
                        [2, 2],
                    ],
                    [1, -2, -1],
                    denom=3,
                ),
                use_relu=True,
            )
            layer2 = ReLULayer(
                AffineMapQ.create(
                    3,
                    3,
                    [
                        [4, -3, 2],
                        [-2, 5, -1],
                        [1, 2, -4],
                    ],
                    [-1, 2, 3],
                    denom=5,
                ),
                use_relu=True,
            )
            layer3_out = ReLULayer(
                AffineMapQ.create(
                    3,
                    2,
                    [
                        [3, -2, 5],
                        [-4, 1, 3],
                    ],
                    [2, -3],
                    denom=2,
                ),
                use_relu=False,
            )
            skip_map = AffineMapQ.create(
                2,
                2,
                [
                    [1, -3],
                    [2, 1],
                ],
                [0, 1],
                denom=7,
            )
            base_net = ReLUNetwork(
                in_dim=2,
                out_dim=2,
                layers=(layer1, layer2, layer3_out),
                residual=skip_map,
                name="two_hidden_mlp",
            )
            anchor_pt: QVec = qvec_norm([2, -3], 6)  # (1/3, -1/2)
            net = base_net.with_anchor_zero(anchor_pt)
            self.assertEqual(net.eval_qvec(anchor_pt), ((0, 0), 1))

            pwa_unmerged = net.enumerate_pwa(domain, merge_bft=False)
            pwa_merged = net.enumerate_pwa(domain, merge_bft=True)
            self.assertGreater(len(pwa_unmerged.cells), 4)
            self.assertGreaterEqual(len(pwa_unmerged.cells), len(pwa_merged.cells))

            # Every cell must be strictly full-dimensional, and distinct cells must have disjoint interiors
            for cell in pwa_merged.cells:
                ok_fd, r_fd, _ = cell.polytope.is_full_dimensional()
                self.assertTrue(ok_fd)
                assert r_fd is not None and r_fd[0] > 0

            for i in range(len(pwa_merged.cells)):
                for j in range(i + 1, len(pwa_merged.cells)):
                    inter = pwa_merged.cells[i].polytope.intersect(
                        pwa_merged.cells[j].polytope
                    )
                    ok_inter, _, _ = inter.is_full_dimensional()
                    self.assertFalse(ok_inter)

            # 7 x 7 = 49 >= 40 deterministic rational grid points spanning [-2, 2]^2
            # including boundary vertices (-2, 2), origin (0, 0), anchor (1/3, -1/2), and 11ths/7ths
            coords_1d: Sequence[Tuple[int, int]] = (
                (-2, 1),
                (-13, 11),
                (-1, 2),
                (0, 1),
                (1, 3),
                (9, 7),
                (2, 1),
            )
            grid_points: List[QVec] = []
            for x1_n, x1_d in coords_1d:
                for x2_n, x2_d in coords_1d:
                    cd = x1_d * x2_d
                    grid_points.append(
                        qvec_norm([x1_n * x2_d, x2_n * x1_d], cd)
                    )
            self.assertEqual(len(grid_points), 49)

            for pt in grid_points:
                expected = net.eval_qvec(pt)
                got_unmerged = pwa_unmerged.eval_qvec(pt)
                got_merged = pwa_merged.eval_qvec(pt)
                self.assertTrue(qvec_eq(got_unmerged, expected))
                self.assertTrue(qvec_eq(got_merged, expected))
                self.assertEqual(got_unmerged, expected)
                self.assertEqual(got_merged, expected)

    def test_07_bft_orthogonal_fake_creases_vs_genuine_and_nonconvex_l_shape(self) -> None:
        """7. BFT fake-crease merging across 2 orthogonal creases (4 -> 1) vs genuine (4 -> 4) & L-shape."""
        with WatchdogTimer(15.0):
            domain = HPolytope.symmetric_box(2, (2, 1))

            # 7A: 2 rotated orthogonal fake creases: n1 = (1, -1) and n2 = (1, 1) (n1 . n2 == 0)
            # 3*(ReLU(x1-x2) - ReLU(x2-x1)) - 2*(ReLU(x1+x2) - ReLU(-x1-x2)) + 1 == x1 - 5*x2 + 1
            fake_2d_net = ReLUNetwork(
                in_dim=2,
                out_dim=1,
                layers=(
                    ReLULayer(
                        AffineMapQ.create(
                            2,
                            4,
                            [
                                [1, -1],
                                [-1, 1],
                                [1, 1],
                                [-1, -1],
                            ],
                            [0, 0, 0, 0],
                            1,
                        ),
                        use_relu=True,
                    ),
                    ReLULayer(
                        AffineMapQ.create(4, 1, [[3, -3, -2, 2]], [1], 1),
                        use_relu=False,
                    ),
                ),
                name="two_ortho_fake_creases",
            )
            pwa_fake_raw = fake_2d_net.enumerate_pwa(domain, merge_bft=False)
            pwa_fake_bft = fake_2d_net.enumerate_pwa(domain, merge_bft=True)
            self.assertEqual(len(pwa_fake_raw.cells), 4)
            self.assertEqual(len(pwa_fake_bft.cells), 1)
            expected_aff = AffineMapQ.create(2, 1, [[1, -5]], [1], 1)
            self.assertEqual(pwa_fake_bft.cells[0].affine_map, expected_aff)

            # 7B: 2 rotated orthogonal genuine creases: 3*|x1 - x2| + 2*|x1 + x2| + 1 (4 -> 4 preserved)
            genuine_2d_net = ReLUNetwork(
                in_dim=2,
                out_dim=1,
                layers=(
                    ReLULayer(
                        AffineMapQ.create(
                            2,
                            4,
                            [
                                [1, -1],
                                [-1, 1],
                                [1, 1],
                                [-1, -1],
                            ],
                            [0, 0, 0, 0],
                            1,
                        ),
                        use_relu=True,
                    ),
                    ReLULayer(
                        AffineMapQ.create(4, 1, [[3, 3, 2, 2]], [1], 1),
                        use_relu=False,
                    ),
                ),
                name="two_ortho_genuine_creases",
            )
            pwa_gen_raw = genuine_2d_net.enumerate_pwa(domain, merge_bft=False)
            pwa_gen_bft = genuine_2d_net.enumerate_pwa(domain, merge_bft=True)
            self.assertEqual(len(pwa_gen_raw.cells), 4)
            self.assertEqual(len(pwa_gen_bft.cells), 4)

            # 7C: Adversarial non-convex L-shape & checkerboard trap for bft_merge_cells!
            # Q1, Q2, Q3 share affine map 0 (L-shaped union is NOT convex!), while Q4 has x1 - x2.
            # BFT may merge 2 adjacent quadrants into a half-box, leaving 2 convex pieces for map 0 + Q4 = 3 cells,
            # and MUST NOT merge into the full box (which would overwrite Q4!).
            q1 = domain.with_halfspace([-1, 0], 0).with_halfspace([0, -1], 0)
            q2 = domain.with_halfspace([1, 0], 0).with_halfspace([0, -1], 0)
            q3 = domain.with_halfspace([1, 0], 0).with_halfspace([0, 1], 0)
            q4 = domain.with_halfspace([-1, 0], 0).with_halfspace([0, 1], 0)
            zero_map = AffineMapQ.zero(2, 1)
            q4_map = AffineMapQ.create(2, 1, [[1, -1]], [0], 1)

            l_cells = [
                PWACell(0, q1, zero_map, "Q1"),
                PWACell(1, q2, zero_map, "Q2"),
                PWACell(2, q3, zero_map, "Q3"),
                PWACell(3, q4, q4_map, "Q4"),
            ]
            merged_l = bft_merge_cells(l_cells)
            self.assertEqual(len(merged_l), 3)
            pwa_l = PWAFunction(2, 1, domain, tuple(merged_l), "l_shape_pwa")
            # Interior point (1, -1) in Q4 must still evaluate to 1 - (-1) = 2, NOT 0!
            self.assertEqual(pwa_l.eval_qvec(((1, -1), 1)), ((2,), 1))
            self.assertEqual(pwa_l.eval_qvec(((-1, -1), 1)), ((0,), 1))

            # Checkerboard: Q1 and Q3 both have zero_map, Q2 and Q4 have q4_map -> 0 merges (4 -> 4)
            checker_cells = [
                PWACell(0, q1, zero_map, "Q1"),
                PWACell(1, q3, zero_map, "Q3"),
                PWACell(2, q2, q4_map, "Q2"),
                PWACell(3, q4, q4_map, "Q4"),
            ]
            self.assertEqual(len(bft_merge_cells(checker_cells)), 4)

    def test_08_tight_budget_exceeded_fail_closed(self) -> None:
        """8. Tight max_pivots=2 and max_bit_length=16 fail-closed behavior with zero silent truncation."""
        with WatchdogTimer(15.0):
            # 8A: 3D integer Klee-Minty deformed cube requiring 7 Bland pivots (> max_pivots=2)
            # 0 <= x1 <= 5
            # 4*x1 <= x2 <= 100 - 4*x1
            # 8*x1 + 4*x2 <= x3 <= 2000 - 8*x1 - 4*x2
            # max 4*x1 + 2*x2 + x3  (visits all 8 vertices = 7 pivots under Bland's rule)
            A_km = [
                [-1, 0, 0],
                [1, 0, 0],
                [4, -1, 0],
                [4, 1, 0],
                [8, 4, -1],
                [8, 4, 1],
            ]
            b_km = [0, 5, 0, 100, 0, 2000]
            c_km = [4, 2, 1]

            # Confirm unrestricted solve succeeds in > 2 pivots with exact optimum 2000 at (0, 0, 2000)
            res_full = solve_lp_bareiss(A_km, b_km, c=c_km, max_pivots=50)
            self.assertEqual(res_full.status, LPStatus.OPTIMAL)
            self.assertGreater(res_full.pivots, 2)
            self.assertEqual(res_full.obj_val, (2000, 1))

            # Under tight max_pivots=2: must fail closed with BUDGET_EXCEEDED and zero leaked partial fields
            res_piv2 = solve_lp_bareiss(A_km, b_km, c=c_km, max_pivots=2)
            self.assertEqual(res_piv2.status, LPStatus.BUDGET_EXCEEDED)
            self.assertIsNone(res_piv2.x_opt)
            self.assertIsNone(res_piv2.obj_val)
            self.assertIsNone(res_piv2.dual_y)
            self.assertIsNone(res_piv2.farkas_infeas)

            v_piv2, cert_piv2, wit_piv2, _ = certify_linear_upper_bound(
                A_km, b_km, c_km, 1, 2000, 1, max_pivots=2
            )
            self.assertEqual(v_piv2, Verdict.OPEN_BUDGET_EXCEEDED.value)
            self.assertIsNone(cert_piv2)
            self.assertIsNone(wit_piv2)

            # 8B: Tight max_bit_length=16 on prime-coefficient LP whose initial entries are <= 9 bits (< 16),
            # but whose 2-step/3-step Bareiss subdeterminants exceed 65,535 (> 16 bits).
            A_bits = [
                [251, 241, 239],
                [233, 257, 229],
                [227, 223, 263],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
            b_bits = [500, 500, 500, 0, 0, 0]
            c_bits = [269, 271, 277]

            # Confirm default max_bit_length=4096 solves cleanly and exact determinant exceeds 16 bits
            det_3 = bareiss_det([row[:] for row in A_bits[:3]], max_bit_length=4096)
            self.assertGreater(abs(det_3).bit_length(), 16)
            res_bits_ok = solve_lp_bareiss(
                A_bits, b_bits, c=c_bits, max_bit_length=4096
            )
            self.assertEqual(res_bits_ok.status, LPStatus.OPTIMAL)

            # Under tight max_bit_length=16: LP, upper-bound certifier, and Bareiss echelon fail closed
            res_bits_16 = solve_lp_bareiss(
                A_bits, b_bits, c=c_bits, max_bit_length=16
            )
            self.assertEqual(res_bits_16.status, LPStatus.BUDGET_EXCEEDED)
            self.assertIsNone(res_bits_16.x_opt)
            self.assertIsNone(res_bits_16.obj_val)
            self.assertIsNone(res_bits_16.dual_y)

            v_b16, cert_b16, wit_b16, _ = certify_linear_upper_bound(
                A_bits, b_bits, c_bits, 1, 10_000, 1, max_bit_length=16
            )
            self.assertEqual(v_b16, Verdict.OPEN_BUDGET_EXCEEDED.value)
            self.assertIsNone(cert_b16)
            self.assertIsNone(wit_b16)

            with self.assertRaises(BudgetExceededError):
                bareiss_det([row[:] for row in A_bits[:3]], max_bit_length=16)

            with self.assertRaises(BudgetExceededError):
                solve_linear_system_int(
                    [row[:] for row in A_bits[:3]], [1, 2, 3], max_bit_length=16
                )

    def test_09_exact_z_linalg_fixed_point_and_affine_subspaces(self) -> None:
        """9. Adversarial Bareiss linear solver & (I - M)x = c fixed-point subspace/inconsistency checks."""
        with WatchdogTimer(15.0):
            # Rank-deficient 3x4 integer system with 2D nullspace
            A_sub = [
                [2, -4, 6, -2],
                [1, -2, 3, -1],   # dependent row
                [3, -1, 4, 2],
            ]
            b_sub = [10, 5, 15]
            self.assertEqual(bareiss_rank(A_sub), 2)
            status, x_part, basis = solve_linear_system_int(A_sub, b_sub)
            self.assertEqual(status, "AFFINE_SUBSPACE")
            self.assertEqual(len(basis), 2)
            assert x_part is not None
            xn, xd = x_part
            for i in range(len(A_sub)):
                self.assertEqual(
                    sum(A_sub[i][j] * xn[j] for j in range(4)), b_sub[i] * xd
                )
            for v in basis:
                # Every nullspace vector must be primitive, canonical (first nonzero > 0), and satisfy A v == 0
                first_nz = next(val for val in v if val != 0)
                self.assertGreater(first_nz, 0)
                for i in range(len(A_sub)):
                    self.assertEqual(sum(A_sub[i][j] * v[j] for j in range(4)), 0)

            # Perturb b_sub[1] from 5 -> 6 to create exact rank inconsistency
            status_inc, x_inc, ns_inc = solve_linear_system_int(A_sub, [10, 6, 15])
            self.assertEqual(status_inc, "INCONSISTENT")
            self.assertIsNone(x_inc)
            self.assertEqual(ns_inc, [])

            # Fixed-point solver (I - M/M_den) x = c/c_den with eigenvalue 1 (unit-eigenvalue Jordan trap!)
            # M / 2 = [[1, 1/2], [0, 1/2]] => I - M/2 = [[0, -1/2], [0, 1/2]]
            st_fp_inc, _, _ = solve_one_minus_m_q(
                [[2, 1], [0, 1]], 2, [1, 1], 3
            )
            self.assertEqual(st_fp_inc, "INCONSISTENT")

            st_fp_aff, x_fp, ns_fp = solve_one_minus_m_q(
                [[2, 1], [0, 1]], 2, [-1, 1], 3
            )
            self.assertEqual(st_fp_aff, "AFFINE_SUBSPACE")
            self.assertEqual(x_fp, ((0, 2), 3))
            self.assertEqual(ns_fp, [(1, 0)])

    def test_10_ct_bab_planet_soundness_vs_exact_pwa_max(self) -> None:
        """10. Multi-layer CT-BaB Planet relaxation soundness against exact enumerated cell maxima."""
        with WatchdogTimer(15.0):
            cell_box = HPolytope.from_box([((-1, 1), (3, 2)), ((-2, 3), (1, 1))])
            v_net = ReLUNetwork(
                in_dim=2,
                out_dim=1,
                layers=(
                    ReLULayer(
                        AffineMapQ.create(
                            2, 3, [[2, -1], [-1, 3], [1, 1]], [1, -1, 0], 2
                        ),
                        use_relu=True,
                    ),
                    ReLULayer(
                        AffineMapQ.create(
                            3, 2, [[3, -2, 1], [-1, 2, 2]], [-1, 1], 3
                        ),
                        use_relu=True,
                    ),
                    ReLULayer(
                        AffineMapQ.create(2, 1, [[4, 3]], [0], 1),
                        use_relu=False,
                    ),
                ),
                residual=AffineMapQ.create(2, 1, [[1, -1]], [0], 5),
                name="two_layer_v",
            )
            f_map = AffineMapQ.create(2, 2, [[2, 1], [-1, 2]], [1, -1], 3)
            sub_map = AffineMapQ.create(2, 1, [[3, 2]], [1], 4)

            planet = ct_bab_planet_upper_bound(
                v_net, f_map, cell_box, linear_subtract_map=sub_map
            )
            self.assertIsNotNone(planet)
            assert planet is not None
            ub_planet, _, dual_scale = planet
            self.assertGreater(dual_scale, 0)

            # Compute exact maximum of v_net(f_map(x)) - sub_map(x) by exact PWA enumeration over cell_box
            exact_pwa = v_net.enumerate_pwa(cell_box, input_map=f_map, merge_bft=False)
            self.assertGreater(len(exact_pwa.cells), 1)
            best_exact = None
            for sc in exact_pwa.cells:
                diff_aff = sc.affine_map.sub(sub_map)
                res_cell = solve_lp_bareiss(
                    sc.polytope.A, sc.polytope.b, c=diff_aff.W[0]
                )
                self.assertEqual(res_cell.status, LPStatus.OPTIMAL)
                assert res_cell.obj_val is not None
                cell_max = q_norm(
                    res_cell.obj_val[0] + diff_aff.c[0] * res_cell.obj_val[1],
                    res_cell.obj_val[1] * diff_aff.denom,
                )
                if best_exact is None or q_lt(best_exact, cell_max):
                    best_exact = cell_max

            assert best_exact is not None
            # Soundness invariant: exact_max <= planet_upper_bound!
            self.assertTrue(q_le(best_exact, ub_planet))

    def test_11_concurrent_and_tangent_hyperplane_arrangements(self) -> None:
        """11. Concurrent 6-line off-origin arrangement + vertex/edge tangent hyperplanes (12 wedges, not 64)."""
        with WatchdogTimer(15.0):
            dom = HPolytope.symmetric_box(2, (1, 1))
            # 6 lines concurring at rational interior point (1/3, -1/5), plus exterior and tangent hyperplanes
            hps = [
                ([3, 0], 1),       # 3*x1 = 1
                ([-6, 0], -2),     # scaled duplicate of 3*x1 = 1
                ([0, 5], -1),      # 5*x2 = -1
                ([3, 5], 0),       # 3*x1 + 5*x2 = 0
                ([3, -5], 2),      # 3*x1 - 5*x2 = 2
                ([6, 5], 1),       # 6*x1 + 5*x2 = 1
                ([3, 10], -1),     # 3*x1 + 10*x2 = -1
                ([1, 0], 1),       # boundary edge tangent x1 = 1 (1D touch -> no split)
                ([1, 1], 2),       # corner vertex tangent x1 + x2 = 2 (0D touch -> no split)
                ([1, 0], 5),       # exterior hyperplane x1 = 5 (no split)
            ]
            cells = enumerate_hyperplane_arrangement(dom, hps)
            self.assertEqual(len(cells), 12)
            conc_pt: QVec = qvec_norm([5, -3], 15)  # (1/3, -1/5)
            for cell in cells:
                ok_fd, r_fd, _ = cell.is_full_dimensional()
                self.assertTrue(ok_fd)
                assert r_fd is not None and r_fd[0] > 0
                # All 12 wedges meet at the exact rational concurrency vertex (1/3, -1/5)
                self.assertTrue(cell.contains(conc_pt))


if __name__ == "__main__":
    unittest.main()
