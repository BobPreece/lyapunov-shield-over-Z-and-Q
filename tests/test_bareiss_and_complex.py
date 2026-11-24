from __future__ import annotations

import time
import unittest

from lyapunov_shield import (
    AffineMapQ,
    FarkasBoundCert,
    HPolytope,
    LPStatus,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
    Verdict,
    bareiss_det,
    bareiss_nullspace,
    bareiss_rank,
    certify_linear_upper_bound,
    certify_relu_upper_bound_ct_bab,
    chebyshev_ball_linf,
    check_affine_neg_farkas,
    check_affine_nonpos_farkas,
    check_farkas_infeasible,
    check_primal_witness,
    compose_pullback_pwa,
    ct_bab_planet_upper_bound,
    enumerate_hyperplane_arrangement,
    q_add,
    q_div,
    q_mul,
    q_norm,
    q_sub,
    qvec_add,
    qvec_dot,
    qvec_l1,
    qvec_linf,
    qvec_norm,
    solve_linear_system_int,
    solve_lp_bareiss,
    zaslavsky_upper_bound,
)


class TestBareissAndComplex(unittest.TestCase):
    def test_rat_and_qvec_canonical_arithmetic(self) -> None:
        a = q_norm(-6, -9)
        b = q_norm(4, -10)
        self.assertEqual(a, (2, 3))
        self.assertEqual(b, (-2, 5))
        self.assertEqual(q_add(a, b), (4, 15))
        self.assertEqual(q_sub(a, b), (16, 15))
        self.assertEqual(q_mul(a, b), (-4, 15))
        self.assertEqual(q_div(a, b), (-5, 3))

        v1 = qvec_norm([6, -9, 12], -15)
        self.assertEqual(v1, ((-2, 3, -4), 5))
        v2 = qvec_norm([1, 2, 3], 5)
        self.assertEqual(qvec_add(v1, v2), ((-1, 5, -1), 5))
        self.assertEqual(qvec_dot(v1, v2), (-8, 25))
        self.assertEqual(qvec_l1(v1), (9, 5))
        self.assertEqual(qvec_linf(v1), (4, 5))

    def test_bareiss_det_rank_nullspace_and_linear_systems(self) -> None:
        M = [[12, 6, 4], [6, 4, 3], [4, 3, 2]]
        self.assertEqual(bareiss_det(M), -4)
        self.assertEqual(bareiss_rank(M), 3)
        st, x_u, ns_u = solve_linear_system_int(M, [22, 13, 9])
        self.assertEqual(st, "UNIQUE")
        self.assertEqual(x_u, ((1, 1, 1), 1))
        self.assertEqual(ns_u, [])

        S = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
        self.assertEqual(bareiss_rank(S), 2)
        self.assertEqual(bareiss_nullspace(S), [(1, -2, 1)])
        st_sub, x_p, ns_s = solve_linear_system_int(S, [6, 12, 3])
        self.assertEqual(st_sub, "AFFINE_SUBSPACE")
        self.assertIsNotNone(x_p)
        self.assertEqual(len(ns_s), 1)

        st_inc, x_inc, _ = solve_linear_system_int([[1, 1], [2, 2]], [1, 3])
        self.assertEqual(st_inc, "INCONSISTENT")
        self.assertIsNone(x_inc)

    def test_standalone_pure_int_farkas_checkers(self) -> None:
        # 1) Infeasibility: x1 + x2 <= -2, -x1 <= 0, -x2 <= 0 => y = (1, 1, 1)
        A_inf = [[1, 1], [-1, 0], [0, -1]]
        d_inf = [-2, 0, 0]
        self.assertTrue(check_farkas_infeasible(A_inf, d_inf, [1, 1, 1]))
        self.assertFalse(check_farkas_infeasible(A_inf, [0, 0, 0], [1, 1, 1]))
        self.assertFalse(check_farkas_infeasible(A_inf, d_inf, [1, -1, 1]))

        # 2) Non-strict affine bound: 2 x1 + 2 x2 - 4 <= 0 on {x1 <= 1, x2 <= 1}
        A_box = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        d_box = [1, 1, 1, 1]
        self.assertTrue(
            check_affine_nonpos_farkas(A_box, d_box, [2, 2], c0=-4, y=[2, 0, 2, 0], q=1)
        )
        # Strict fails at c0=-4 (attains 0 at (1,1)), succeeds at c0=-5!
        self.assertFalse(
            check_affine_neg_farkas(A_box, d_box, [2, 2], c0=-4, y=[2, 0, 2, 0], q=1)
        )
        self.assertTrue(
            check_affine_neg_farkas(A_box, d_box, [2, 2], c0=-5, y=[2, 0, 2, 0], q=1)
        )

    def test_avis_lexicographic_simplex_and_chebyshev_linf(self) -> None:
        # Highly degenerate 3D pyramid apex at (0, 0, 0)
        A_pyr = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, 2], [-1, -1, 2], [0, 0, -1]]
        b_pyr = [0, 0, 0, 0, 0, 0, 1]
        res = solve_lp_bareiss(A_pyr, b_pyr, c=[0, 0, 1])
        self.assertEqual(res.status, LPStatus.OPTIMAL)
        self.assertEqual(res.obj_val, (0, 1))
        assert res.x_opt is not None and res.dual_y is not None
        self.assertTrue(check_primal_witness(A_pyr, b_pyr, res.x_opt))
        bcert = FarkasBoundCert(res.dual_y, res.dual_scale, (0, 0, 1), 1, 0, 1)
        self.assertTrue(check_affine_nonpos_farkas(A_pyr, b_pyr, bcert))

        r_star, x_c, _ = chebyshev_ball_linf(A_pyr, b_pyr, r_cap=10)
        self.assertIsNotNone(r_star)
        assert r_star is not None and x_c is not None
        self.assertGreater(r_star[0], 0)
        self.assertTrue(check_primal_witness(A_pyr, b_pyr, x_c))

    def test_incenu_hyperplane_arrangement_and_zaslavsky(self) -> None:
        self.assertEqual(zaslavsky_upper_bound(2, 4), 11)
        self.assertEqual(zaslavsky_upper_bound(6, 64), 83_278_001)
        dom = HPolytope.symmetric_box(2, (1, 1))
        # 4 lines through origin split [-1, 1]^2 into exactly 8 full-dim cones (not 11 or 16)
        planes = [([1, 0], 0), ([0, 1], 0), ([1, 1], 0), ([1, -1], 0)]
        cells = enumerate_hyperplane_arrangement(dom, planes)
        self.assertEqual(len(cells), 8)
        for c in cells:
            ok_fd, r_q, _ = c.is_full_dimensional()
            self.assertTrue(ok_fd)
            assert r_q is not None and r_q[0] > 0

    def test_bft_fake_crease_merger_and_pullback_composition(self) -> None:
        dom = HPolytope.symmetric_box(2, (1, 1))
        # ReLU(x1 + x2) - ReLU(-x1 - x2) == x1 + x2 (single global affine map across crease!)
        net = ReLUNetwork(
            in_dim=2,
            out_dim=1,
            layers=(
                ReLULayer(AffineMapQ.create(2, 2, [[1, 1], [-1, -1]], [0, 0], 1), True),
                ReLULayer(AffineMapQ.create(2, 1, [[1, -1]], [0], 1), False),
            ),
        )
        unmerged = net.enumerate_pwa(dom, merge_bft=False)
        merged = net.enumerate_pwa(dom, merge_bft=True)
        self.assertEqual(len(unmerged.cells), 2)
        self.assertEqual(len(merged.cells), 1)

        plant = PWAFunction(
            2, 2, dom, (PWACell(0, dom, AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2)),)
        )
        rep = compose_pullback_pwa(net, plant, merge_bft=True)
        self.assertEqual(rep.final_shards, 1)

    def test_ct_bab_crown_planet_pruning_soundness(self) -> None:
        # Off-origin box [1, 2]^2: V(1/2 x) - 3/4 V(x) <= -1/2 < 0 certified at root with 0 splits!
        box = HPolytope.from_box([((1, 1), (2, 1)), ((1, 1), (2, 1))])
        v_net = ReLUNetwork(
            2,
            1,
            (
                ReLULayer(AffineMapQ.create(2, 4, [[1, 0], [-1, 0], [0, 1], [0, -1]], [0] * 4, 1), True),
                ReLULayer(AffineMapQ.create(4, 1, [[1, 1, 1, 1]], [0], 1), False),
            ),
        )
        f_half = AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2)
        sub_map = AffineMapQ.create(2, 1, [[3, 3]], [0], 4)
        pl = ct_bab_planet_upper_bound(v_net, f_half, box, sub_map)
        self.assertIsNotNone(pl)
        assert pl is not None
        self.assertEqual(pl[0], (-1, 2))

        verdict, leaves, pruned, zb = certify_relu_upper_bound_ct_bab(
            v_net, box, rhs=(0, 1), input_map=f_half, subtract_map=sub_map, max_cells=16
        )
        self.assertEqual(verdict, Verdict.CERTIFIED_EXACT)
        self.assertEqual(pruned, 1)
        self.assertEqual(leaves, 0)
        self.assertEqual(zb, 11)

    def test_monotonic_deadline_and_bit_length_fail_closed(self) -> None:
        A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        b = [1, 1, 1, 1]
        expired = time.monotonic() - 1.0
        res_dl = solve_lp_bareiss(A, b, c=[1, 1], deadline=expired)
        self.assertEqual(res_dl.status, LPStatus.BUDGET_EXCEEDED)

        v_str, _, _, _ = certify_linear_upper_bound(A, b, [1, 1], 1, 2, 1, deadline=expired)
        self.assertEqual(v_str, "OPEN_BUDGET_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
