from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import (
    FarkasBoundCert,
    LPStatus,
    certify_linear_upper_bound,
    chebyshev_ball_linf,
    check_primal_witness,
    solve_lp_bareiss,
)
from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    Verdict,
    check_budget,
    gcd_list,
    lcm_list,
    lcm_pair,
    primitive_ineq_row,
    q_add,
    q_cmp,
    q_le,
    q_lt,
    q_mul,
    q_norm,
    q_sub,
    qvec_add,
    qvec_norm,
    qvec_sub,
)


@dataclasses.dataclass(frozen=True)
class HPolytope:
    """Exact H-polytope {x in R^dim : A x <= b} stored as primitive integer tuples."""

    dim: int
    A: Tuple[Tuple[int, ...], ...]
    b: Tuple[int, ...]

    @staticmethod
    def create(
        dim: int, A: Sequence[Sequence[int]], b: Sequence[int]
    ) -> HPolytope:
        if len(A) != len(b):
            raise ValueError("Mismatch between len(A) and len(b)")
        seen = {}
        infeasible = False
        for i, row in enumerate(A):
            if len(row) != dim:
                raise ValueError(f"Row {i} length {len(row)} != dim {dim}")
            pa, pb, _ = primitive_ineq_row(row, b[i])
            if all(v == 0 for v in pa):
                if pb < 0:
                    infeasible = True
                continue
            # Keep tightest RHS if duplicate normal appears
            if pa in seen:
                if pb < seen[pa]:
                    seen[pa] = pb
            else:
                seen[pa] = pb
        if infeasible:
            return HPolytope(
                dim=dim,
                A=(tuple(0 for _ in range(dim)),),
                b=(-1,),
            )
        rows_a = tuple(seen.keys())
        rows_b = tuple(seen[k] for k in rows_a)
        return HPolytope(dim=dim, A=rows_a, b=rows_b)

    @staticmethod
    def from_box(bounds: Sequence[Tuple[Rat, Rat]]) -> HPolytope:
        dim = len(bounds)
        A: List[List[int]] = []
        b: List[int] = []
        for j, (lo, hi) in enumerate(bounds):
            lo_n, lo_d = q_norm(lo[0], lo[1])
            hi_n, hi_d = q_norm(hi[0], hi[1])
            # x_j <= hi_n / hi_d  <=>  hi_d * x_j <= hi_n
            row_hi = [0] * dim
            row_hi[j] = hi_d
            A.append(row_hi)
            b.append(hi_n)
            # x_j >= lo_n / lo_d  <=>  -lo_d * x_j <= -lo_n
            row_lo = [0] * dim
            row_lo[j] = -lo_d
            A.append(row_lo)
            b.append(-lo_n)
        return HPolytope.create(dim, A, b)

    @staticmethod
    def symmetric_box(dim: int, radius: Rat = (1, 1)) -> HPolytope:
        rn, rd = q_norm(radius[0], radius[1])
        return HPolytope.from_box([((-rn, rd), (rn, rd)) for _ in range(dim)])

    def contains(self, x_q: QVec) -> bool:
        return check_primal_witness(self.A, self.b, x_q)

    def with_halfspace(self, a_row: Sequence[int], b_val: int) -> HPolytope:
        return HPolytope.create(
            self.dim,
            list(self.A) + [list(a_row)],
            list(self.b) + [int(b_val)],
        )

    def intersect(self, other: HPolytope) -> HPolytope:
        if self.dim != other.dim:
            raise ValueError("Dimension mismatch in HPolytope.intersect")
        return HPolytope.create(
            self.dim,
            list(self.A) + list(other.A),
            list(self.b) + list(other.b),
        )

    def is_feasible(
        self,
        max_pivots: int = 4000,
        deadline: Optional[float] = None,
    ) -> Tuple[bool, Optional[QVec]]:
        res = solve_lp_bareiss(
            self.A, self.b, c=None, max_pivots=max_pivots, deadline=deadline
        )
        if res.status == LPStatus.BUDGET_EXCEEDED:
            raise BudgetExceededError("LP budget exceeded in is_feasible")
        if res.status == LPStatus.INFEASIBLE:
            return (False, None)
        return (True, res.x_opt)

    def chebyshev_ball(
        self,
        r_cap: Optional[int] = 1000,
        row_l1_norm: bool = True,
        max_pivots: int = 4000,
        deadline: Optional[float] = None,
    ) -> Tuple[Optional[Rat], Optional[QVec]]:
        r_q, xc_q, res = chebyshev_ball_linf(
            self.A,
            self.b,
            r_cap=r_cap,
            row_l1_norm=row_l1_norm,
            max_pivots=max_pivots,
            deadline=deadline,
        )
        if res.status == LPStatus.BUDGET_EXCEEDED:
            raise BudgetExceededError("LP budget exceeded in chebyshev_ball")
        return (r_q, xc_q)

    def is_full_dimensional(
        self,
        max_pivots: int = 4000,
        deadline: Optional[float] = None,
    ) -> Tuple[bool, Optional[Rat], Optional[QVec]]:
        """Strict l_infinity Chebyshev check r* > 0 (filters empty, 0D points, and 1D..n-1D slivers)."""
        # Quick algebraic check: if both a^T x <= b and -a^T x <= -b exist, width is 0 immediately!
        row_map = {self.A[i]: self.b[i] for i in range(len(self.A))}
        for a_row, b_val in row_map.items():
            neg_a = tuple(-v for v in a_row)
            if neg_a in row_map and row_map[neg_a] <= -b_val:
                return (False, (0, 1), None)
        r_q, xc_q = self.chebyshev_ball(
            r_cap=1000, max_pivots=max_pivots, deadline=deadline
        )
        if r_q is None or r_q[0] <= 0:
            return (False, (0, 1), xc_q)
        return (True, r_q, xc_q)

    def prune_redundant(
        self,
        max_pivots: int = 3000,
        deadline: Optional[float] = None,
    ) -> HPolytope:
        """Eliminate redundant linear inequalities via exact Bareiss LP."""
        m = len(self.A)
        if m <= 1:
            return self
        keep_A = list(self.A)
        keep_b = list(self.b)
        idx = 0
        while idx < len(keep_A):
            check_budget(deadline=deadline)
            a_i = keep_A[idx]
            b_i = keep_b[idx]
            other_A = keep_A[:idx] + keep_A[idx + 1 :]
            other_b = keep_b[:idx] + keep_b[idx + 1 :]
            if not other_A:
                idx += 1
                continue
            res = solve_lp_bareiss(
                other_A,
                other_b,
                c=a_i,
                max_pivots=max_pivots,
                deadline=deadline,
            )
            if res.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in prune_redundant")
            if (
                res.status == LPStatus.OPTIMAL
                and res.obj_val is not None
                and res.obj_val[0] <= b_i * res.obj_val[1]
            ):
                keep_A.pop(idx)
                keep_b.pop(idx)
            else:
                idx += 1
        return HPolytope(dim=self.dim, A=tuple(keep_A), b=tuple(keep_b))

    def bounding_box(
        self,
        max_pivots: int = 3000,
        deadline: Optional[float] = None,
    ) -> Optional[List[Tuple[Rat, Rat]]]:
        bounds: List[Tuple[Rat, Rat]] = []
        for j in range(self.dim):
            e_pos = [1 if k == j else 0 for k in range(self.dim)]
            e_neg = [-1 if k == j else 0 for k in range(self.dim)]
            r_hi = solve_lp_bareiss(
                self.A, self.b, c=e_pos, max_pivots=max_pivots, deadline=deadline
            )
            if r_hi.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in bounding_box")
            if r_hi.status != LPStatus.OPTIMAL or r_hi.obj_val is None:
                return None
            r_lo = solve_lp_bareiss(
                self.A, self.b, c=e_neg, max_pivots=max_pivots, deadline=deadline
            )
            if r_lo.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in bounding_box")
            if r_lo.status != LPStatus.OPTIMAL or r_lo.obj_val is None:
                return None
            lo_q = (-r_lo.obj_val[0], r_lo.obj_val[1])
            hi_q = r_hi.obj_val
            bounds.append((lo_q, hi_q))
        return bounds


@dataclasses.dataclass(frozen=True)
class AffineMapQ:
    """Exact rational affine map f(x) = (W x + c) / denom with primitive (W, c, denom), denom > 0."""

    in_dim: int
    out_dim: int
    W: Tuple[Tuple[int, ...], ...]
    c: Tuple[int, ...]
    denom: int = 1

    @staticmethod
    def create(
        in_dim: int,
        out_dim: int,
        W: Sequence[Sequence[int]],
        c: Sequence[int],
        denom: int = 1,
    ) -> AffineMapQ:
        if denom == 0:
            raise ZeroDivisionError("AffineMapQ denominator cannot be 0")
        sign = -1 if denom < 0 else 1
        d = abs(int(denom))
        w_rows = [tuple(sign * int(v) for v in row) for row in W]
        c_tup = tuple(sign * int(v) for v in c)
        if len(w_rows) != out_dim or len(c_tup) != out_dim:
            raise ValueError("Row dimension mismatch in AffineMapQ")
        for r in w_rows:
            if len(r) != in_dim:
                raise ValueError("Column dimension mismatch in AffineMapQ")
        flat = [v for r in w_rows for v in r] + list(c_tup)
        if all(v == 0 for v in flat):
            return AffineMapQ(
                in_dim=in_dim,
                out_dim=out_dim,
                W=tuple(tuple(0 for _ in range(in_dim)) for _ in range(out_dim)),
                c=tuple(0 for _ in range(out_dim)),
                denom=1,
            )
        g = gcd_list(flat + [d])
        if g > 1:
            w_rows = [tuple(v // g for v in r) for r in w_rows]
            c_tup = tuple(v // g for v in c_tup)
            d //= g
        return AffineMapQ(
            in_dim=in_dim,
            out_dim=out_dim,
            W=tuple(w_rows),
            c=c_tup,
            denom=d,
        )

    @staticmethod
    def zero(in_dim: int, out_dim: int) -> AffineMapQ:
        return AffineMapQ.create(
            in_dim,
            out_dim,
            [[0] * in_dim for _ in range(out_dim)],
            [0] * out_dim,
            1,
        )

    @staticmethod
    def identity(dim: int) -> AffineMapQ:
        return AffineMapQ.create(
            dim,
            dim,
            [[1 if i == j else 0 for j in range(dim)] for i in range(dim)],
            [0] * dim,
            1,
        )

    def eval_qvec(self, x_q: QVec) -> QVec:
        xn, xd = x_q
        if len(xn) != self.in_dim:
            raise ValueError("Dimension mismatch in AffineMapQ.eval_qvec")
        out_nums = [
            sum(self.W[i][j] * xn[j] for j in range(self.in_dim))
            + self.c[i] * xd
            for i in range(self.out_dim)
        ]
        return qvec_norm(out_nums, self.denom * xd)

    def eval_scalar(self, x_q: QVec) -> Rat:
        if self.out_dim != 1:
            raise ValueError("eval_scalar requires out_dim == 1")
        qv = self.eval_qvec(x_q)
        return q_norm(qv[0][0], qv[1])

    def row_map(self, i: int) -> AffineMapQ:
        return AffineMapQ.create(
            self.in_dim, 1, [self.W[i]], [self.c[i]], self.denom
        )

    def compose(self, inner: AffineMapQ) -> AffineMapQ:
        """Compute self(inner(x)) = (W_out * ((W_in x + c_in)/d_in) + c_out) / d_out."""
        if self.in_dim != inner.out_dim:
            raise ValueError("Dimension mismatch in AffineMapQ.compose")
        new_W: List[List[int]] = []
        new_c: List[int] = []
        for i in range(self.out_dim):
            row = [
                sum(self.W[i][k] * inner.W[k][j] for k in range(self.in_dim))
                for j in range(inner.in_dim)
            ]
            ci = (
                sum(self.W[i][k] * inner.c[k] for k in range(self.in_dim))
                + self.c[i] * inner.denom
            )
            new_W.append(row)
            new_c.append(ci)
        return AffineMapQ.create(
            inner.in_dim,
            self.out_dim,
            new_W,
            new_c,
            self.denom * inner.denom,
        )

    def add(self, other: AffineMapQ) -> AffineMapQ:
        if self.in_dim != other.in_dim or self.out_dim != other.out_dim:
            raise ValueError("Dimension mismatch in AffineMapQ.add")
        cd = lcm_pair(self.denom, other.denom)
        s1 = cd // self.denom
        s2 = cd // other.denom
        new_W = [
            [
                s1 * self.W[i][j] + s2 * other.W[i][j]
                for j in range(self.in_dim)
            ]
            for i in range(self.out_dim)
        ]
        new_c = [s1 * self.c[i] + s2 * other.c[i] for i in range(self.out_dim)]
        return AffineMapQ.create(self.in_dim, self.out_dim, new_W, new_c, cd)

    def sub(self, other: AffineMapQ) -> AffineMapQ:
        return self.add(other.scale((-1, 1)))

    def scale(self, factor: Rat) -> AffineMapQ:
        fn, fd = q_norm(factor[0], factor[1])
        if fn == 0:
            return AffineMapQ.zero(self.in_dim, self.out_dim)
        new_W = [[fn * v for v in row] for row in self.W]
        new_c = [fn * v for v in self.c]
        return AffineMapQ.create(
            self.in_dim, self.out_dim, new_W, new_c, self.denom * fd
        )

    def shift_output(self, delta: QVec) -> AffineMapQ:
        dn, dd = delta
        if len(dn) != self.out_dim:
            raise ValueError("Output dimension mismatch in shift_output")
        cd = lcm_pair(self.denom, dd)
        s1 = cd // self.denom
        s2 = cd // dd
        new_W = [[s1 * v for v in row] for row in self.W]
        new_c = [s1 * self.c[i] + s2 * dn[i] for i in range(self.out_dim)]
        return AffineMapQ.create(self.in_dim, self.out_dim, new_W, new_c, cd)


@dataclasses.dataclass(frozen=True)
class PWACell:
    cell_id: int
    polytope: HPolytope
    affine_map: AffineMapQ
    tag: str = ""


@dataclasses.dataclass(frozen=True)
class PWAFunction:
    in_dim: int
    out_dim: int
    domain: HPolytope
    cells: Tuple[PWACell, ...]
    name: str = ""

    def eval_qvec(self, x_q: QVec) -> QVec:
        for cell in self.cells:
            if cell.polytope.contains(x_q):
                return cell.affine_map.eval_qvec(x_q)
        raise ValueError(f"State {x_q} outside all PWA cells of {self.name}")

    def eval_scalar(self, x_q: QVec) -> Rat:
        qv = self.eval_qvec(x_q)
        return q_norm(qv[0][0], qv[1])


@dataclasses.dataclass(frozen=True)
class ReLULayer:
    affine: AffineMapQ
    use_relu: bool = True


@dataclasses.dataclass(frozen=True)
class ReLUNetwork:
    """Exact rational multi-layer ReLU MLP with optional skip-linear map and anchor offset:
    y(x) = MLP(x) + residual(x) + post_offset.
    """

    in_dim: int
    out_dim: int
    layers: Tuple[ReLULayer, ...]
    residual: Optional[AffineMapQ] = None
    post_offset: Optional[QVec] = None
    name: str = ""

    def num_relu_neurons(self) -> int:
        return sum(
            layer.affine.out_dim for layer in self.layers if layer.use_relu
        )

    def eval_qvec(self, x_q: QVec) -> QVec:
        cur = x_q
        for layer in self.layers:
            cur = layer.affine.eval_qvec(cur)
            if layer.use_relu:
                cn, cd = cur
                cur = qvec_norm([v if v > 0 else 0 for v in cn], cd)
        if self.residual is not None:
            cur = qvec_add(cur, self.residual.eval_qvec(x_q))
        if self.post_offset is not None:
            cur = qvec_add(cur, self.post_offset)
        return cur

    def eval_scalar(self, x_q: QVec) -> Rat:
        if self.out_dim != 1:
            raise ValueError("eval_scalar requires out_dim == 1")
        qv = self.eval_qvec(x_q)
        return q_norm(qv[0][0], qv[1])

    def with_anchor_zero(self, x_eq: QVec) -> ReLUNetwork:
        """Apply exact post-rounding equilibrium anchor subtraction: y_Q(x) - y_Q(x_eq)."""
        base_net = ReLUNetwork(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=self.layers,
            residual=self.residual,
            post_offset=None,
            name=self.name,
        )
        val_eq = base_net.eval_qvec(x_eq)
        neg_eq = (tuple(-v for v in val_eq[0]), val_eq[1])
        return ReLUNetwork(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=self.layers,
            residual=self.residual,
            post_offset=neg_eq,
            name=self.name + "_anchored" if self.name else "anchored",
        )

    def enumerate_pwa(
        self,
        domain: HPolytope,
        input_map: Optional[AffineMapQ] = None,
        merge_bft: bool = True,
        prune_redundant: bool = True,
        max_cells: int = 4096,
        deadline: Optional[float] = None,
    ) -> PWAFunction:
        """Rada-Cerny incremental cell enumeration with strict r* > 0 Chebyshev filter + BFT merger."""
        src_dim = domain.dim
        cur_init = (
            input_map
            if input_map is not None
            else AffineMapQ.identity(src_dim)
        )
        ok_dom, _, _ = domain.is_full_dimensional(deadline=deadline)
        if not ok_dom:
            return PWAFunction(
                in_dim=src_dim,
                out_dim=self.out_dim,
                domain=domain,
                cells=(),
                name=self.name,
            )

        # Active items: (HPolytope, current_layer_affine_map, signature_str)
        active: List[Tuple[HPolytope, AffineMapQ, str]] = [
            (domain, cur_init, "")
        ]

        for l_idx, layer in enumerate(self.layers):
            check_budget(deadline=deadline)
            # Pre-activation map z(x) = layer.affine(cur_map(x))
            if not layer.use_relu:
                active = [
                    (poly, layer.affine.compose(cmap), sig)
                    for poly, cmap, sig in active
                ]
                continue

            out_m = layer.affine.out_dim
            # Split incrementally neuron by neuron using r* > 0 Chebyshev filter
            neuron_items: List[Tuple[HPolytope, List[Tuple[Tuple[int, ...], int]], int, str]] = []
            for poly, cmap, sig in active:
                pre_map = layer.affine.compose(cmap)
                rows_list = [
                    (pre_map.W[r], pre_map.c[r]) for r in range(out_m)
                ]
                neuron_items.append((poly, rows_list, pre_map.denom, sig))

            for r in range(out_m):
                check_budget(deadline=deadline)
                next_items: List[
                    Tuple[HPolytope, List[Tuple[Tuple[int, ...], int]], int, str]
                ] = []
                for poly, rows_list, denom, sig in neuron_items:
                    w_r, c_r = rows_list[r]
                    if all(v == 0 for v in w_r):
                        if c_r <= 0:
                            new_rows = list(rows_list)
                            new_rows[r] = (tuple(0 for _ in range(src_dim)), 0)
                            next_items.append((poly, new_rows, denom, sig + "0"))
                        else:
                            next_items.append((poly, rows_list, denom, sig + "1"))
                        continue

                    # Active halfspace: w_r^T x + c_r >= 0  <=>  -w_r^T x <= c_r
                    # Inactive halfspace: w_r^T x + c_r <= 0  <=>  w_r^T x <= -c_r
                    neg_w = [-v for v in w_r]
                    pos_poly = poly.with_halfspace(neg_w, c_r)
                    neg_poly = poly.with_halfspace(w_r, -c_r)

                    pos_ok, _, _ = pos_poly.is_full_dimensional(deadline=deadline)
                    neg_ok, _, _ = neg_poly.is_full_dimensional(deadline=deadline)

                    if pos_ok and not neg_ok:
                        next_items.append((poly, rows_list, denom, sig + "1"))
                    elif neg_ok and not pos_ok:
                        new_rows = list(rows_list)
                        new_rows[r] = (tuple(0 for _ in range(src_dim)), 0)
                        next_items.append((poly, new_rows, denom, sig + "0"))
                    elif pos_ok and neg_ok:
                        if prune_redundant and len(pos_poly.A) > 2 * src_dim + 4:
                            pos_poly = pos_poly.prune_redundant(deadline=deadline)
                            neg_poly = neg_poly.prune_redundant(deadline=deadline)
                        next_items.append((pos_poly, rows_list, denom, sig + "1"))
                        new_rows = list(rows_list)
                        new_rows[r] = (tuple(0 for _ in range(src_dim)), 0)
                        next_items.append((neg_poly, new_rows, denom, sig + "0"))

                    if len(next_items) > max_cells:
                        raise BudgetExceededError(
                            f"Active cell count {len(next_items)} exceeded max_cells={max_cells}"
                        )
                neuron_items = next_items

            active = []
            for poly, rows_list, denom, sig in neuron_items:
                W_mat = [r_pair[0] for r_pair in rows_list]
                c_vec = [r_pair[1] for r_pair in rows_list]
                amap = AffineMapQ.create(src_dim, out_m, W_mat, c_vec, denom)
                active.append((poly, amap, sig + ("|" if l_idx + 1 < len(self.layers) else "")))

        raw_cells: List[PWACell] = []
        res_map = None
        if self.residual is not None:
            res_map = (
                self.residual.compose(cur_init)
                if input_map is not None
                else self.residual
            )
        for idx, (poly, amap, sig) in enumerate(active):
            final_map = amap
            if res_map is not None:
                final_map = final_map.add(res_map)
            if self.post_offset is not None:
                final_map = final_map.shift_output(self.post_offset)
            if prune_redundant:
                poly = poly.prune_redundant(deadline=deadline)
            raw_cells.append(
                PWACell(cell_id=idx, polytope=poly, affine_map=final_map, tag=sig)
            )

        if merge_bft and len(raw_cells) > 1:
            raw_cells = bft_merge_cells(
                raw_cells, prune_redundant=prune_redundant, deadline=deadline
            )

        reindexed = tuple(
            PWACell(
                cell_id=i,
                polytope=c.polytope,
                affine_map=c.affine_map,
                tag=c.tag,
            )
            for i, c in enumerate(raw_cells)
        )
        return PWAFunction(
            in_dim=src_dim,
            out_dim=self.out_dim,
            domain=domain,
            cells=reindexed,
            name=self.name,
        )


def enumerate_hyperplane_arrangement(
    domain: HPolytope,
    hyperplanes: Sequence[Tuple[Sequence[int], int]],
    prune_redundant: bool = True,
    max_cells: int = 4096,
    deadline: Optional[float] = None,
) -> List[HPolytope]:
    """Rada-Cerny IncEnu cell enumeration for {a_k^T x = b_k} inside domain with r* > 0 filter."""
    ok, _, _ = domain.is_full_dimensional(deadline=deadline)
    if not ok:
        return []
    cells = [domain]
    for a_row, b_val in hyperplanes:
        check_budget(deadline=deadline)
        neg_a = [-int(v) for v in a_row]
        pos_a = [int(v) for v in a_row]
        b_int = int(b_val)
        next_cells: List[HPolytope] = []
        for poly in cells:
            c_le = poly.with_halfspace(pos_a, b_int)
            c_ge = poly.with_halfspace(neg_a, -b_int)
            ok_le, _, _ = c_le.is_full_dimensional(deadline=deadline)
            ok_ge, _, _ = c_ge.is_full_dimensional(deadline=deadline)
            if ok_le and not ok_ge:
                next_cells.append(poly)
            elif ok_ge and not ok_le:
                next_cells.append(poly)
            elif ok_le and ok_ge:
                if prune_redundant and len(c_le.A) > 2 * domain.dim + 4:
                    c_le = c_le.prune_redundant(deadline=deadline)
                    c_ge = c_ge.prune_redundant(deadline=deadline)
                next_cells.append(c_le)
                next_cells.append(c_ge)
            if len(next_cells) > max_cells:
                raise BudgetExceededError(
                    f"Arrangement cell count {len(next_cells)} exceeded max_cells={max_cells}"
                )
        cells = next_cells
    if prune_redundant:
        cells = [c.prune_redundant(deadline=deadline) for c in cells]
    return cells


def _try_convex_union(
    p1: HPolytope,
    p2: HPolytope,
    deadline: Optional[float] = None,
) -> Optional[HPolytope]:
    """Bemporad-Fukuda-Torrisi (2001) exact algebraic convexity test for C_1 U C_2.

    Checks if C_1 and C_2 share an opposing splitting hyperplane a^T x <= beta vs -a^T x <= -beta
    and their common valid inequalities define P_cand with P_cand == C_1 U C_2.
    """
    map1 = {p1.A[i]: p1.b[i] for i in range(len(p1.A))}
    map2 = {p2.A[i]: p2.b[i] for i in range(len(p2.A))}

    split_candidates: List[Tuple[Tuple[int, ...], int]] = []
    for a1, b1 in map1.items():
        neg_a1 = tuple(-v for v in a1)
        if neg_a1 in map2 and map2[neg_a1] == -b1:
            split_candidates.append((a1, b1))

    if not split_candidates:
        return None

    for a_split, b_split in split_candidates:
        check_budget(deadline=deadline)
        neg_split = tuple(-v for v in a_split)
        # Collect constraints from p1 (except a_split) that hold on all of p2,
        # and constraints from p2 (except neg_split) that hold on all of p1.
        cand_A: List[Tuple[int, ...]] = []
        cand_b: List[int] = []

        for a1, b1 in map1.items():
            if a1 == a_split:
                continue
            if a1 in map2 and map2[a1] <= b1:
                cand_A.append(a1)
                cand_b.append(b1)
                continue
            v_str, _, _, _ = certify_linear_upper_bound(
                p2.A, p2.b, a1, 1, b1, 1, deadline=deadline
            )
            if v_str == "CERTIFIED_EXACT":
                cand_A.append(a1)
                cand_b.append(b1)

        for a2, b2 in map2.items():
            if a2 == neg_split:
                continue
            if a2 in map1 and map1[a2] <= b2:
                cand_A.append(a2)
                cand_b.append(b2)
                continue
            v_str, _, _, _ = certify_linear_upper_bound(
                p1.A, p1.b, a2, 1, b2, 1, deadline=deadline
            )
            if v_str == "CERTIFIED_EXACT":
                cand_A.append(a2)
                cand_b.append(b2)

        p_cand = HPolytope.create(p1.dim, cand_A, cand_b)
        half1 = p_cand.with_halfspace(a_split, b_split)
        half2 = p_cand.with_halfspace(neg_split, -b_split)

        # Verify half1 subset p1 and half2 subset p2
        ok1 = True
        for a1, b1 in map1.items():
            if a1 == a_split:
                continue
            v_str, _, _, _ = certify_linear_upper_bound(
                half1.A, half1.b, a1, 1, b1, 1, deadline=deadline
            )
            if v_str != "CERTIFIED_EXACT":
                ok1 = False
                break
        if not ok1:
            continue

        ok2 = True
        for a2, b2 in map2.items():
            if a2 == neg_split:
                continue
            v_str, _, _, _ = certify_linear_upper_bound(
                half2.A, half2.b, a2, 1, b2, 1, deadline=deadline
            )
            if v_str != "CERTIFIED_EXACT":
                ok2 = False
                break
        if ok2:
            return p_cand

    return None


def bft_merge_cells(
    cells: Sequence[PWACell],
    prune_redundant: bool = True,
    max_passes: int = 8,
    deadline: Optional[float] = None,
) -> List[PWACell]:
    """Bemporad-Fukuda-Torrisi (BFT) fake-crease merger across cells with identical AffineMapQ."""
    by_map: Dict[AffineMapQ, List[PWACell]] = {}
    for c in cells:
        by_map.setdefault(c.affine_map, []).append(c)

    merged_all: List[PWACell] = []
    for amap, group in by_map.items():
        cur = list(group)
        for _ in range(max_passes):
            check_budget(deadline=deadline)
            if len(cur) <= 1:
                break
            changed = False
            i = 0
            while i < len(cur):
                j = i + 1
                merged_here = False
                while j < len(cur):
                    u_poly = _try_convex_union(
                        cur[i].polytope, cur[j].polytope, deadline=deadline
                    )
                    if u_poly is not None:
                        if prune_redundant:
                            u_poly = u_poly.prune_redundant(deadline=deadline)
                        cur[i] = PWACell(
                            cell_id=cur[i].cell_id,
                            polytope=u_poly,
                            affine_map=amap,
                            tag=f"{cur[i].tag}+{cur[j].tag}",
                        )
                        cur.pop(j)
                        changed = True
                        merged_here = True
                        break
                    j += 1
                if not merged_here:
                    i += 1
            if not changed:
                break
        merged_all.extend(cur)

    return merged_all


@dataclasses.dataclass(frozen=True)
class PullbackCompositionReport:
    pwa: PWAFunction
    combinatorial_candidates: int
    point_feasible_shards: int
    pruned_degenerate_shards: int
    pre_bft_full_dim_shards: int
    final_shards: int


def compose_pullback_pwa(
    outer: Union[PWAFunction, ReLUNetwork],
    inner_pwa: PWAFunction,
    merge_bft: bool = True,
    prune_redundant: bool = True,
    count_point_feasible: bool = False,
    max_cells: int = 4096,
    deadline: Optional[float] = None,
) -> PullbackCompositionReport:
    """Compute exact full-dimensional shards of outer(inner_pwa(x)) using r* > 0 Chebyshev pruning."""
    dim = inner_pwa.in_dim
    out_dim = outer.out_dim
    pre_bft_cells: List[PWACell] = []
    point_feasible = 0

    if isinstance(outer, ReLUNetwork):
        k_neurons = outer.num_relu_neurons()
        comb_total = len(inner_pwa.cells) * (1 << min(k_neurons, 30))
        for p_cell in inner_pwa.cells:
            check_budget(deadline=deadline)
            sub_pwa = outer.enumerate_pwa(
                domain=p_cell.polytope,
                input_map=p_cell.affine_map,
                merge_bft=False,
                prune_redundant=prune_redundant,
                max_cells=max_cells,
                deadline=deadline,
            )
            for sc in sub_pwa.cells:
                pre_bft_cells.append(
                    PWACell(
                        cell_id=len(pre_bft_cells),
                        polytope=sc.polytope,
                        affine_map=sc.affine_map,
                        tag=f"c{p_cell.cell_id}:{sc.tag}",
                    )
                )
                if len(pre_bft_cells) > max_cells:
                    raise BudgetExceededError(
                        f"Pullback shards {len(pre_bft_cells)} > max_cells={max_cells}"
                    )
        if (
            count_point_feasible
            and k_neurons <= 12
            and len(outer.layers) in (1, 2)
            and outer.layers[0].use_relu
            and (len(outer.layers) == 1 or not outer.layers[1].use_relu)
        ):
            # Exact count of combinatorial activation patterns feasible at >= 1 point (including 0D/1D!)
            layer0 = outer.layers[0].affine
            for p_cell in inner_pwa.cells:
                comp = layer0.compose(p_cell.affine_map)
                for mask in range(1 << k_neurons):
                    check_budget(deadline=deadline)
                    extra_A: List[List[int]] = list(p_cell.polytope.A)
                    extra_b: List[int] = list(p_cell.polytope.b)
                    for r in range(k_neurons):
                        wr = comp.W[r]
                        cr = comp.c[r]
                        if (mask >> r) & 1:
                            extra_A.append([-v for v in wr])
                            extra_b.append(cr)
                        else:
                            extra_A.append(list(wr))
                            extra_b.append(-cr)
                    poly_cand = HPolytope.create(dim, extra_A, extra_b)
                    feas, _ = poly_cand.is_feasible(deadline=deadline)
                    if feas:
                        point_feasible += 1
        else:
            point_feasible = comb_total
    else:
        comb_total = len(inner_pwa.cells) * len(outer.cells)
        for p_cell in inner_pwa.cells:
            for o_cell in outer.cells:
                check_budget(deadline=deadline)
                # Pull back o_cell.polytope: A_o * ((W_p x + c_p)/d_p) <= b_o
                # <=> (A_o W_p) x <= d_p b_o - A_o c_p
                fm = p_cell.affine_map
                op = o_cell.polytope
                pb_A: List[List[int]] = list(p_cell.polytope.A)
                pb_b: List[int] = list(p_cell.polytope.b)
                for r in range(len(op.A)):
                    ar = op.A[r]
                    br = op.b[r]
                    row = [
                        sum(ar[k] * fm.W[k][j] for k in range(fm.out_dim))
                        for j in range(dim)
                    ]
                    rhs = fm.denom * br - sum(
                        ar[k] * fm.c[k] for k in range(fm.out_dim)
                    )
                    pb_A.append(row)
                    pb_b.append(rhs)
                cand = HPolytope.create(dim, pb_A, pb_b)
                if count_point_feasible:
                    feas, _ = cand.is_feasible(deadline=deadline)
                    if feas:
                        point_feasible += 1
                ok_fd, _, _ = cand.is_full_dimensional(deadline=deadline)
                if ok_fd:
                    if prune_redundant:
                        cand = cand.prune_redundant(deadline=deadline)
                    comp_map = o_cell.affine_map.compose(fm)
                    pre_bft_cells.append(
                        PWACell(
                            cell_id=len(pre_bft_cells),
                            polytope=cand,
                            affine_map=comp_map,
                            tag=f"c{p_cell.cell_id}->o{o_cell.cell_id}",
                        )
                    )
                    if len(pre_bft_cells) > max_cells:
                        raise BudgetExceededError(
                            f"Pullback shards {len(pre_bft_cells)} > max_cells={max_cells}"
                        )
        if not count_point_feasible:
            point_feasible = comb_total

    pre_cnt = len(pre_bft_cells)
    final_cells = (
        bft_merge_cells(
            pre_bft_cells, prune_redundant=prune_redundant, deadline=deadline
        )
        if (merge_bft and pre_cnt > 1)
        else pre_bft_cells
    )
    reindexed = tuple(
        PWACell(cell_id=i, polytope=c.polytope, affine_map=c.affine_map, tag=c.tag)
        for i, c in enumerate(final_cells)
    )
    pwa_out = PWAFunction(
        in_dim=dim,
        out_dim=out_dim,
        domain=inner_pwa.domain,
        cells=reindexed,
        name=f"{outer.name}_o_{inner_pwa.name}",
    )
    pruned_deg = max(0, comb_total - pre_cnt)
    return PullbackCompositionReport(
        pwa=pwa_out,
        combinatorial_candidates=comb_total,
        point_feasible_shards=point_feasible,
        pruned_degenerate_shards=pruned_deg,
        pre_bft_full_dim_shards=pre_cnt,
        final_shards=len(reindexed),
    )


def ct_bab_planet_upper_bound(
    v_net: ReLUNetwork,
    f_map: AffineMapQ,
    cell_poly: HPolytope,
    linear_subtract_map: Optional[AffineMapQ] = None,
    deadline: Optional[float] = None,
) -> Optional[Tuple[Rat, Tuple[int, ...], int]]:
    """Hierarchical CT-BaB exact rational Planet/Triangle LP upper bound over cell_poly:
    upper-bounds v_net(f_map(x)) - linear_subtract_map(x) on bounded cell_poly WITHOUT splitting 2^K neurons!

    Returns (upper_bound_q, dual_y_lifted, dual_scale) or None if cell_poly is unbounded.
    """
    if v_net.out_dim != 1:
        raise ValueError("ct_bab_planet_upper_bound requires scalar v_net (out_dim == 1)")

    dim_x = cell_poly.dim
    # Build lifted LP in variables [x (dim_x), layer_1_post, layer_2_post, ...]
    # Propagate exact LP bounds layer by layer
    A_lift: List[List[int]] = [list(r) for r in cell_poly.A]
    b_lift: List[int] = list(cell_poly.b)
    total_vars = dim_x

    def pad_existing(extra: int) -> None:
        nonlocal total_vars
        for r in A_lift:
            r.extend([0] * extra)
        total_vars += extra

    # Current layer affine expression in terms of lifted variables:
    # represented as AffineMapQ from total_vars -> cur_dim
    cur_expr = f_map  # maps x (0..dim_x-1) -> f_map.out_dim

    for layer in v_net.layers:
        check_budget(deadline=deadline)
        # Embed cur_expr to current total_vars if total_vars > cur_expr.in_dim
        if cur_expr.in_dim < total_vars:
            ext_W = [
                list(row) + [0] * (total_vars - cur_expr.in_dim)
                for row in cur_expr.W
            ]
            cur_expr = AffineMapQ.create(
                total_vars, cur_expr.out_dim, ext_W, cur_expr.c, cur_expr.denom
            )
        pre = layer.affine.compose(cur_expr)
        if not layer.use_relu:
            cur_expr = pre
            continue

        m_neu = pre.out_dim
        # Compute exact rational bounds [l_j, u_j] for each pre-activation neuron j on current lifted polytope
        bounds_lu: List[Tuple[Rat, Rat]] = []
        for j in range(m_neu):
            wj = list(pre.W[j])
            cj = pre.c[j]
            dj = pre.denom
            res_hi = solve_lp_bareiss(A_lift, b_lift, c=wj, deadline=deadline)
            if res_hi.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in CT-BaB bound propagation")
            if res_hi.status != LPStatus.OPTIMAL or res_hi.obj_val is None:
                return None
            res_lo = solve_lp_bareiss(
                A_lift, b_lift, c=[-v for v in wj], deadline=deadline
            )
            if res_lo.status == LPStatus.BUDGET_EXCEEDED:
                raise BudgetExceededError("LP budget exceeded in CT-BaB bound propagation")
            if res_lo.status != LPStatus.OPTIMAL or res_lo.obj_val is None:
                return None
            # hi = (res_hi / dj) + (cj / dj)
            hi_q = q_add(
                q_norm(res_hi.obj_val[0], res_hi.obj_val[1] * dj),
                q_norm(cj, dj),
            )
            lo_q = q_add(
                q_norm(-res_lo.obj_val[0], res_lo.obj_val[1] * dj),
                q_norm(cj, dj),
            )
            bounds_lu.append((lo_q, hi_q))

        # Add post-activation variables only for ambiguous neurons l_j < 0 < u_j!
        next_W: List[List[int]] = []
        next_c: List[int] = []
        # Use common denominator pre.denom for pass-through rows
        ambig_indices: Dict[int, int] = {}
        for j, (lo_q, hi_q) in enumerate(bounds_lu):
            if lo_q[0] < 0 and hi_q[0] > 0:
                ambig_indices[j] = total_vars + len(ambig_indices)

        num_ambig = len(ambig_indices)
        prev_vars = total_vars
        if num_ambig > 0:
            pad_existing(num_ambig)

        for j, (lo_q, hi_q) in enumerate(bounds_lu):
            if hi_q[0] <= 0:
                # Stably inactive: post == 0
                next_W.append([0] * total_vars)
                next_c.append(0)
            elif lo_q[0] >= 0:
                # Stably active: post == pre[j]
                next_W.append(list(pre.W[j]) + [0] * num_ambig)
                next_c.append(pre.c[j])
            else:
                # Ambiguous l < 0 < u: introduce z_var = ambig_indices[j]
                z_col = ambig_indices[j]
                # 1) z >= 0  <=>  -z <= 0
                r1 = [0] * total_vars
                r1[z_col] = -1
                A_lift.append(r1)
                b_lift.append(0)
                # 2) z >= (wj^T v + cj) / dj  <=>  wj^T v - dj * z <= -cj
                r2 = list(pre.W[j]) + [0] * num_ambig
                r2[z_col] = -pre.denom
                A_lift.append(r2)
                b_lift.append(-pre.c[j])
                # 3) Planet chord: z <= (u / (u - l)) * ((wj^T v + cj)/dj - l)
                # Write l = ln/ld, u = un/ud, diff = u - l = dn_diff / dd_diff > 0
                diff_q = q_sub(hi_q, lo_q)  # > 0
                # Slope s = u / (u - l) = sn / sd > 0
                sn, sd = q_norm(
                    hi_q[0] * diff_q[1], hi_q[1] * diff_q[0]
                )
                # z <= (sn / sd) * ((wj^T v + cj)/dj - ln/ld)
                # Multiply by (sd * dj * ld) > 0:
                # (sd * dj * ld) z - (sn * ld) wj^T v <= sn * (ld * cj - dj * ln)
                ld = lo_q[1]
                ln = lo_q[0]
                dj = pre.denom
                r3 = [-(sn * ld) * pre.W[j][k] for k in range(prev_vars)] + [
                    0
                ] * num_ambig
                r3[z_col] = sd * dj * ld
                rhs3 = sn * (ld * pre.c[j] - dj * ln)
                A_lift.append(r3)
                b_lift.append(rhs3)

                row_z = [0] * total_vars
                row_z[z_col] = pre.denom
                next_W.append(row_z)
                next_c.append(0)

        cur_expr = AffineMapQ.create(
            total_vars, m_neu, next_W, next_c, pre.denom
        )

    # Add residual and post_offset of v_net, then subtract linear_subtract_map(x)
    if cur_expr.in_dim < total_vars:
        ext_W = [
            list(row) + [0] * (total_vars - cur_expr.in_dim)
            for row in cur_expr.W
        ]
        cur_expr = AffineMapQ.create(
            total_vars, 1, ext_W, cur_expr.c, cur_expr.denom
        )
    if v_net.residual is not None:
        res_comp = v_net.residual.compose(f_map)
        ext_res = AffineMapQ.create(
            total_vars,
            1,
            [list(res_comp.W[0]) + [0] * (total_vars - dim_x)],
            res_comp.c,
            res_comp.denom,
        )
        cur_expr = cur_expr.add(ext_res)
    if v_net.post_offset is not None:
        cur_expr = cur_expr.shift_output(v_net.post_offset)
    if linear_subtract_map is not None:
        ext_sub = AffineMapQ.create(
            total_vars,
            1,
            [list(linear_subtract_map.W[0]) + [0] * (total_vars - dim_x)],
            linear_subtract_map.c,
            linear_subtract_map.denom,
        )
        cur_expr = cur_expr.sub(ext_sub)

    obj_row = list(cur_expr.W[0])
    res_final = solve_lp_bareiss(
        A_lift, b_lift, c=obj_row, deadline=deadline
    )
    if res_final.status == LPStatus.BUDGET_EXCEEDED:
        raise BudgetExceededError("LP budget exceeded in CT-BaB final solve")
    if res_final.status != LPStatus.OPTIMAL or res_final.obj_val is None or res_final.dual_y is None:
        return None
    ub_q = q_add(
        q_norm(res_final.obj_val[0], res_final.obj_val[1] * cur_expr.denom),
        q_norm(cur_expr.c[0], cur_expr.denom),
    )
    return (ub_q, res_final.dual_y, res_final.dual_scale)


def zaslavsky_upper_bound(dim: int, num_hyperplanes: int) -> int:
    """Pre-flight Zaslavsky upper bound sum_{j=0}^{dim} comb(H, j) on hyperplane arrangement regions."""
    if dim < 0 or num_hyperplanes < 0:
        raise ValueError("dim and num_hyperplanes must be non-negative")
    return sum(
        math.comb(num_hyperplanes, j)
        for j in range(min(dim, num_hyperplanes) + 1)
    )


def _crown_affine_upper_envelope(
    net: ReLUNetwork,
    f_in: AffineMapQ,
    bbox: Sequence[Tuple[Rat, Rat]],
) -> AffineMapQ:
    """Exact rational CROWN / Planet 1-pass affine upper bound U(x) >= net(f_in(x))
    for all x in bbox, computed in O(sum |W_l|) integer arithmetic with zero LPs!
    """
    n = f_in.in_dim

    def interval_bounds_of_affine(amap: AffineMapQ) -> List[Tuple[Rat, Rat]]:
        res_b: List[Tuple[Rat, Rat]] = []
        d = amap.denom
        for r in range(amap.out_dim):
            lo_q = q_norm(amap.c[r], d)
            hi_q = lo_q
            wr = amap.W[r]
            for j in range(n):
                w = wr[j]
                if w > 0:
                    lo_q = q_add(lo_q, q_mul((w, d), bbox[j][0]))
                    hi_q = q_add(hi_q, q_mul((w, d), bbox[j][1]))
                elif w < 0:
                    lo_q = q_add(lo_q, q_mul((w, d), bbox[j][1]))
                    hi_q = q_add(hi_q, q_mul((w, d), bbox[j][0]))
            res_b.append((lo_q, hi_q))
        return res_b

    # Maintain exact affine lower & upper envelopes L(x) <= h(x) <= U(x) for current hidden vector!
    cur_L = f_in
    cur_U = f_in

    for layer in net.layers:
        W = layer.affine.W
        c = layer.affine.c
        d = layer.affine.denom
        out_m = layer.affine.out_dim
        in_m = layer.affine.in_dim

        # Propagate affine lower/upper envelopes through linear map W * h + c:
        # For each row i: if W[i][k] >= 0 use U_k for upper / L_k for lower; if W[i][k] < 0 swap!
        cd_lu = lcm_pair(cur_L.denom, cur_U.denom)
        sL = cd_lu // cur_L.denom
        sU = cd_lu // cur_U.denom

        pre_L_W: List[List[int]] = []
        pre_L_c: List[int] = []
        pre_U_W: List[List[int]] = []
        pre_U_c: List[int] = []

        for i in range(out_m):
            wi = W[i]
            row_l = [0] * n
            row_u = [0] * n
            cl_val = c[i] * cd_lu
            cu_val = c[i] * cd_lu
            for k in range(in_m):
                wk = wi[k]
                if wk > 0:
                    for j in range(n):
                        row_l[j] += wk * sL * cur_L.W[k][j]
                        row_u[j] += wk * sU * cur_U.W[k][j]
                    cl_val += wk * sL * cur_L.c[k]
                    cu_val += wk * sU * cur_U.c[k]
                elif wk < 0:
                    for j in range(n):
                        row_l[j] += wk * sU * cur_U.W[k][j]
                        row_u[j] += wk * sL * cur_L.W[k][j]
                    cl_val += wk * sU * cur_U.c[k]
                    cu_val += wk * sL * cur_L.c[k]
            pre_L_W.append(row_l)
            pre_L_c.append(cl_val)
            pre_U_W.append(row_u)
            pre_U_c.append(cu_val)

        pre_L = AffineMapQ.create(n, out_m, pre_L_W, pre_L_c, d * cd_lu)
        pre_U = AffineMapQ.create(n, out_m, pre_U_W, pre_U_c, d * cd_lu)

        if not layer.use_relu:
            cur_L = pre_L
            cur_U = pre_U
            continue

        b_L = interval_bounds_of_affine(pre_L)
        b_U = interval_bounds_of_affine(pre_U)

        # Relax ReLU neuron by neuron:
        # l_i = b_L[i][0], u_i = b_U[i][1]
        # - u_i <= 0: post_L = 0, post_U = 0
        # - l_i >= 0: post_L = pre_L_i, post_U = pre_U_i
        # - l_i < 0 < u_i:
        #     slope s_i = u_i / (u_i - l_i) in (0, 1)
        #     post_U_i(x) = s_i * (pre_U_i(x) - l_i)  (exact Planet chord!)
        #     post_L_i(x) = s_i * pre_L_i(x)          (valid lower line <= ReLU!)
        rows_L: List[AffineMapQ] = []
        rows_U: List[AffineMapQ] = []
        for i in range(out_m):
            lo_q = b_L[i][0]
            hi_q = b_U[i][1]
            if hi_q[0] <= 0:
                rows_L.append(AffineMapQ.zero(n, 1))
                rows_U.append(AffineMapQ.zero(n, 1))
            elif lo_q[0] >= 0:
                rows_L.append(pre_L.row_map(i))
                rows_U.append(pre_U.row_map(i))
            else:
                diff_q = q_sub(hi_q, lo_q)
                slope_q = q_norm(hi_q[0] * diff_q[1], hi_q[1] * diff_q[0])
                rL_i = pre_L.row_map(i).scale(slope_q)
                rU_i = pre_U.row_map(i).shift_output(((-lo_q[0],), lo_q[1])).scale(slope_q)
                rows_L.append(rL_i)
                rows_U.append(rU_i)

        cd_next_L = lcm_list([rm.denom for rm in rows_L]) if rows_L else 1
        cd_next_U = lcm_list([rm.denom for rm in rows_U]) if rows_U else 1
        cur_L = AffineMapQ.create(
            n,
            out_m,
            [[v * (cd_next_L // rm.denom) for v in rm.W[0]] for rm in rows_L],
            [rm.c[0] * (cd_next_L // rm.denom) for rm in rows_L],
            cd_next_L,
        )
        cur_U = AffineMapQ.create(
            n,
            out_m,
            [[v * (cd_next_U // rm.denom) for v in rm.W[0]] for rm in rows_U],
            [rm.c[0] * (cd_next_U // rm.denom) for rm in rows_U],
            cd_next_U,
        )

    out_U = cur_U
    if net.residual is not None:
        out_U = out_U.add(net.residual.compose(f_in))
    if net.post_offset is not None:
        out_U = out_U.shift_output(net.post_offset)
    return out_U


def certify_relu_upper_bound_ct_bab(
    net: ReLUNetwork,
    domain: HPolytope,
    rhs: Rat = (0, 1),
    input_map: Optional[AffineMapQ] = None,
    subtract_map: Optional[AffineMapQ] = None,
    max_cells: int = 256,
    deadline: Optional[float] = None,
    cert_sink: Optional[
        List[Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...], FarkasBoundCert]]
    ] = None,
    witness_sink: Optional[List[QVec]] = None,
) -> Tuple[Verdict, int, int, int]:
    """Hierarchical CT-BaB parent-box LP bound pruning verifier for net(input_map(x)) - subtract_map(x) <= rhs.

    Before splitting remaining ReLU neurons on a parent polytope, computes the exact rational CROWN/Planet
    affine upper envelope U(x) over the polytope's bounding box and solves a single n-dimensional LP.
    If upper_bound <= rhs, prunes the entire parent polytope immediately without splitting remaining neurons.
    When `cert_sink` is provided, appends `(poly_A, poly_b, FarkasBoundCert)` for every certified parent/leaf.
    When `witness_sink` is provided, appends the falsifying `QVec` counterexample on leaf violation.
    Returns (verdict, certified_leaves, pruned_parents, zaslavsky_bound).
    """
    if net.out_dim != 1:
        raise ValueError("certify_relu_upper_bound_ct_bab requires scalar ReLUNetwork")
    src_dim = domain.dim
    rhs_q = q_norm(rhs[0], rhs[1])
    z_bound = zaslavsky_upper_bound(src_dim, net.num_relu_neurons())
    f_in = input_map if input_map is not None else AffineMapQ.identity(src_dim)

    try:
        ok_dom, _, _ = domain.is_full_dimensional(deadline=deadline)
        if not ok_dom:
            return (Verdict.CERTIFIED_EXACT, 0, 0, z_bound)

        split_planes: List[Tuple[Tuple[int, ...], int]] = []
        prefix_map = f_in
        for layer in net.layers:
            pre = layer.affine.compose(prefix_map)
            if layer.use_relu:
                for r in range(pre.out_dim):
                    if any(v != 0 for v in pre.W[r]):
                        split_planes.append((pre.W[r], pre.c[r]))
                break
            prefix_map = pre

        queue: List[Tuple[HPolytope, int]] = [(domain, 0)]
        certified_leaves = 0
        pruned_parents = 0

        while queue:
            check_budget(deadline=deadline)
            if len(queue) > max_cells:
                if cert_sink is not None:
                    cert_sink.clear()
                return (
                    Verdict.OPEN_BUDGET_EXCEEDED,
                    certified_leaves,
                    pruned_parents,
                    z_bound,
                )
            poly, depth = queue.pop(0)

            bbox = poly.bounding_box(deadline=deadline)
            if bbox is not None:
                u_env = _crown_affine_upper_envelope(net, f_in, bbox)
                expr_env = (
                    u_env.sub(subtract_map) if subtract_map is not None else u_env
                )
                target_env = q_sub(rhs_q, q_norm(expr_env.c[0], expr_env.denom))
                v_env, b_cert_env, _, _ = certify_linear_upper_bound(
                    poly.A,
                    poly.b,
                    expr_env.W[0],
                    expr_env.denom,
                    target_env[0],
                    target_env[1],
                    deadline=deadline,
                )
                if v_env == "CERTIFIED_EXACT":
                    if cert_sink is not None and b_cert_env is not None:
                        cert_sink.append((poly.A, poly.b, b_cert_env))
                    if depth < len(split_planes):
                        pruned_parents += 1
                    else:
                        certified_leaves += 1
                    continue

            if depth >= len(split_planes):
                sub_pwa = net.enumerate_pwa(
                    poly,
                    input_map=f_in,
                    merge_bft=False,
                    max_cells=max_cells,
                    deadline=deadline,
                )
                for sc in sub_pwa.cells:
                    expr = (
                        sc.affine_map.sub(subtract_map)
                        if subtract_map is not None
                        else sc.affine_map
                    )
                    target_rhs = q_sub(rhs_q, q_norm(expr.c[0], expr.denom))
                    v_str, b_cert_leaf, wit_leaf, _ = certify_linear_upper_bound(
                        sc.polytope.A,
                        sc.polytope.b,
                        expr.W[0],
                        expr.denom,
                        target_rhs[0],
                        target_rhs[1],
                        deadline=deadline,
                    )
                    if v_str == "OPEN_BUDGET_EXCEEDED":
                        if cert_sink is not None:
                            cert_sink.clear()
                        return (
                            Verdict.OPEN_BUDGET_EXCEEDED,
                            certified_leaves,
                            pruned_parents,
                            z_bound,
                        )
                    if v_str != "CERTIFIED_EXACT":
                        if cert_sink is not None:
                            cert_sink.clear()
                        if witness_sink is not None and wit_leaf is not None:
                            witness_sink.append(wit_leaf)
                        return (
                            Verdict.FALSIFIED_COUNTEREXAMPLE,
                            certified_leaves,
                            pruned_parents,
                            z_bound,
                        )
                    if cert_sink is not None and b_cert_leaf is not None:
                        cert_sink.append((sc.polytope.A, sc.polytope.b, b_cert_leaf))
                    certified_leaves += 1
                continue

            w_r, c_r = split_planes[depth]
            pos_poly = poly.with_halfspace([-v for v in w_r], c_r)
            neg_poly = poly.with_halfspace(w_r, -c_r)
            pos_ok, _, _ = pos_poly.is_full_dimensional(deadline=deadline)
            neg_ok, _, _ = neg_poly.is_full_dimensional(deadline=deadline)
            if pos_ok and not neg_ok:
                queue.append((poly, depth + 1))
            elif neg_ok and not pos_ok:
                queue.append((poly, depth + 1))
            elif pos_ok and neg_ok:
                if len(queue) + 2 > max_cells:
                    if cert_sink is not None:
                        cert_sink.clear()
                    return (
                        Verdict.OPEN_BUDGET_EXCEEDED,
                        certified_leaves,
                        pruned_parents,
                        z_bound,
                    )
                queue.append((pos_poly, depth + 1))
                queue.append((neg_poly, depth + 1))

        return (Verdict.CERTIFIED_EXACT, certified_leaves, pruned_parents, z_bound)
    except BudgetExceededError:
        if cert_sink is not None:
            cert_sink.clear()
        return (Verdict.OPEN_BUDGET_EXCEEDED, 0, 0, z_bound)
