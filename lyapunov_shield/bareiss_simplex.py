from __future__ import annotations

import dataclasses
import enum
import math
from typing import List, Optional, Sequence, Tuple, Union

from .z_linalg import (
    BudgetExceededError,
    QVec,
    Rat,
    check_budget,
    gcd_list,
    lcm_list,
    lcm_pair,
    primitive_ineq_row,
    q_norm,
    qvec_norm,
)


class LPStatus(str, enum.Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


@dataclasses.dataclass(frozen=True)
class FarkasInfeasibilityCert:
    """Constructive integer Farkas infeasibility certificate for A x <= b:
    y >= 0, y^T A == 0, y^T b < 0.
    """

    y: Tuple[int, ...]
    rhs_dot: int


@dataclasses.dataclass(frozen=True)
class FarkasBoundCert:
    """Constructive integer Farkas positivity/upper-bound certificate for
    (c_num / c_den)^T x <= rhs_num / rhs_den over P = {x : A x <= b}:
      - If is_vacuous is False:
          scale > 0, y >= 0,
          c_den * (y^T A) == scale * c_num^T,
          rhs_den * (y^T b) <= scale * rhs_num (or < if strict).
      - If is_vacuous is True:
          scale == 0, y >= 0, y^T A == 0, y^T b < 0 (P itself is empty).
    """

    y: Tuple[int, ...]
    scale: int
    c_num: Tuple[int, ...]
    c_den: int
    rhs_num: int
    rhs_den: int
    is_vacuous: bool = False


@dataclasses.dataclass(frozen=True)
class LPResult:
    status: LPStatus
    x_opt: Optional[QVec] = None
    obj_val: Optional[Rat] = None
    dual_y: Optional[Tuple[int, ...]] = None
    dual_scale: int = 1
    farkas_infeas: Optional[FarkasInfeasibilityCert] = None
    unbounded_ray: Optional[QVec] = None
    pivots: int = 0
    phase1_pivots: int = 0
    phase2_pivots: int = 0


def _is_strict_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_primal_witness(
    A: Sequence[Sequence[int]], b: Sequence[int], x_q: QVec
) -> bool:
    """Zero-trust O(m * n) pure-integer primal feasibility checker for A x <= b."""
    nums, den = x_q
    if not _is_strict_int(den) or den <= 0:
        return False
    if any(not _is_strict_int(v) for v in nums):
        return False
    m = len(A)
    if m != len(b):
        return False
    for i in range(m):
        row = A[i]
        if len(row) != len(nums) or not _is_strict_int(b[i]):
            return False
        if any(not _is_strict_int(rj) for rj in row):
            return False
        lhs = sum(row[j] * nums[j] for j in range(len(nums)))
        if lhs > b[i] * den:
            return False
    return True


def check_farkas_certificate(
    A: Sequence[Sequence[int]], b: Sequence[int], y: Sequence[int]
) -> bool:
    """Zero-trust O(m * n) pure-integer Farkas infeasibility checker:
    verifies y >= 0, y^T A == 0, and y^T b < 0.
    """
    m = len(A)
    if len(b) != m or len(y) != m:
        return False
    for yi in y:
        if not _is_strict_int(yi) or yi < 0:
            return False
    if m == 0:
        return False
    n = len(A[0])
    for j in range(n):
        col_sum = sum(y[i] * int(A[i][j]) for i in range(m))
        if col_sum != 0:
            return False
    rhs_sum = sum(y[i] * int(b[i]) for i in range(m))
    return rhs_sum < 0


def check_positivity_farkas(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    cert: FarkasBoundCert,
    strict: bool = False,
) -> bool:
    """Zero-trust O(m * n) pure-integer Farkas upper-bound / positivity checker.

    Certifies (c_num / c_den)^T x <= rhs_num / rhs_den (or < if strict=True)
    for all x in {A x <= b}.
    """
    m = len(A)
    if len(b) != m or len(cert.y) != m:
        return False
    if (
        not _is_strict_int(cert.c_den)
        or cert.c_den <= 0
        or not _is_strict_int(cert.rhs_den)
        or cert.rhs_den <= 0
        or not _is_strict_int(cert.rhs_num)
    ):
        return False
    if any(not _is_strict_int(v) for v in cert.c_num):
        return False
    for yi in cert.y:
        if not _is_strict_int(yi) or yi < 0:
            return False
    if cert.is_vacuous:
        return (
            _is_strict_int(cert.scale)
            and cert.scale == 0
            and check_farkas_certificate(A, b, cert.y)
        )
    if not _is_strict_int(cert.scale) or cert.scale <= 0:
        return False
    n = len(cert.c_num)
    if m > 0 and len(A[0]) != n:
        return False
    for j in range(n):
        lhs = cert.c_den * sum(cert.y[i] * int(A[i][j]) for i in range(m))
        rhs = cert.scale * cert.c_num[j]
        if lhs != rhs:
            return False
    lhs_b = cert.rhs_den * sum(cert.y[i] * int(b[i]) for i in range(m))
    rhs_b = cert.scale * cert.rhs_num
    if strict:
        return lhs_b < rhs_b
    return lhs_b <= rhs_b


check_farkas_dual = check_positivity_farkas


def check_farkas_infeasible(
    A: Sequence[Sequence[int]],
    d: Sequence[int],
    y: Union[Sequence[int], FarkasInfeasibilityCert],
) -> bool:
    """Standalone O(m * n) pure-int Farkas infeasibility checker for A x <= d:
    verifies y >= 0, A^T y == 0, and d^T y <= -1.
    """
    y_vec = y.y if isinstance(y, FarkasInfeasibilityCert) else y
    return check_farkas_certificate(A, d, y_vec)


def check_affine_nonpos_farkas(
    A: Sequence[Sequence[int]],
    d: Sequence[int],
    c_or_cert: Union[Sequence[int], FarkasBoundCert],
    c0: int = 0,
    y: Optional[Sequence[int]] = None,
    q: int = 1,
) -> bool:
    """Standalone O(m * n) pure-int Farkas checker for non-strict affine bound
    c^T x + c0 <= 0 on nonempty {x : A x <= d}:
    verifies q >= 1, y >= 0, A^T y == q * c, and d^T y + q * c0 <= 0.
    Also accepts a FarkasBoundCert directly as c_or_cert.
    """
    if isinstance(c_or_cert, FarkasBoundCert):
        return check_positivity_farkas(A, d, c_or_cert, strict=False)
    if y is None or not _is_strict_int(q) or q < 1 or not _is_strict_int(c0):
        return False
    m = len(A)
    if len(d) != m or len(y) != m:
        return False
    if any(not _is_strict_int(yi) or yi < 0 for yi in y):
        return False
    if any(not _is_strict_int(cj) for cj in c_or_cert):
        return False
    if any(not _is_strict_int(di) for di in d):
        return False
    n = len(c_or_cert)
    if m > 0 and any(len(row) != n or any(not _is_strict_int(v) for v in row) for row in A):
        return False
    for j in range(n):
        col_sum = sum(y[i] * A[i][j] for i in range(m))
        if col_sum != q * c_or_cert[j]:
            return False
    rhs_val = sum(y[i] * d[i] for i in range(m)) + q * c0
    return rhs_val <= 0


def check_affine_neg_farkas(
    A: Sequence[Sequence[int]],
    d: Sequence[int],
    c_or_cert: Union[Sequence[int], FarkasBoundCert],
    c0: int = 0,
    y: Optional[Sequence[int]] = None,
    q: int = 1,
) -> bool:
    """Standalone O(m * n) pure-int Farkas checker for strict affine negativity
    c^T x + c0 < 0 on nonempty {x : A x <= d}:
    verifies q >= 1, y >= 0, A^T y == q * c, and d^T y + q * c0 <= -1.
    Also accepts a FarkasBoundCert directly as c_or_cert.
    """
    if isinstance(c_or_cert, FarkasBoundCert):
        return check_positivity_farkas(A, d, c_or_cert, strict=True)
    if y is None or not _is_strict_int(q) or q < 1 or not _is_strict_int(c0):
        return False
    m = len(A)
    if len(d) != m or len(y) != m:
        return False
    if any(not _is_strict_int(yi) or yi < 0 for yi in y):
        return False
    if any(not _is_strict_int(cj) for cj in c_or_cert):
        return False
    if any(not _is_strict_int(di) for di in d):
        return False
    n = len(c_or_cert)
    if m > 0 and any(len(row) != n or any(not _is_strict_int(v) for v in row) for row in A):
        return False
    for j in range(n):
        col_sum = sum(y[i] * A[i][j] for i in range(m))
        if col_sum != q * c_or_cert[j]:
            return False
    rhs_val = sum(y[i] * d[i] for i in range(m)) + q * c0
    return rhs_val <= -1


def _lex_ratio_less(
    p1: int, rhs1: List[int], p2: int, rhs2: List[int]
) -> bool:
    """Exact integer comparison of (1 / p1) * rhs1 <_lex (1 / p2) * rhs2 for p1, p2 > 0."""
    for k in range(len(rhs1)):
        v1 = p2 * rhs1[k]
        v2 = p1 * rhs2[k]
        if v1 != v2:
            return v1 < v2
    return False


def _bareiss_simplex_loop(
    T: List[List[int]],
    RHS: List[List[int]],
    OBJ: List[int],
    obj_val: int,
    basis: List[int],
    D: int,
    max_pivots: int,
    max_bit_length: int,
    deadline: Optional[float],
) -> Tuple[str, List[List[int]], List[List[int]], List[int], int, List[int], int, int, Optional[int]]:
    """Run Bareiss fraction-free simplex pivots with Bland entering + Avis lex leaving."""
    m = len(T)
    num_cols = len(OBJ)
    pivots = 0

    while True:
        check_budget(deadline=deadline, val=D, max_bit_length=max_bit_length)
        enter = -1
        for j in range(num_cols):
            if OBJ[j] > 0:
                enter = j
                break
        if enter == -1:
            return ("OPTIMAL", T, RHS, OBJ, obj_val, basis, D, pivots, None)

        if pivots >= max_pivots:
            raise BudgetExceededError(
                f"Simplex pivot cap {max_pivots} exceeded"
            )

        leave = -1
        for i in range(m):
            p_i = T[i][enter]
            if p_i > 0:
                if leave == -1 or _lex_ratio_less(
                    p_i, RHS[i], T[leave][enter], RHS[leave]
                ):
                    leave = i

        if leave == -1:
            return ("UNBOUNDED", T, RHS, OBJ, obj_val, basis, D, pivots, enter)

        piv = T[leave][enter]
        check_budget(deadline=deadline, val=piv, max_bit_length=max_bit_length)
        piv_row_T = T[leave]
        piv_row_RHS = RHS[leave]
        rhs_len = len(piv_row_RHS)

        for i in range(m):
            if i == leave:
                continue
            ti = T[i]
            factor = ti[enter]
            for j in range(num_cols):
                num = piv * ti[j] - factor * piv_row_T[j]
                assert num % D == 0
                ti[j] = num // D
            ri = RHS[i]
            for k in range(rhs_len):
                num_r = piv * ri[k] - factor * piv_row_RHS[k]
                assert num_r % D == 0
                ri[k] = num_r // D
            row_max = max(
                max((abs(v) for v in ti), default=0),
                max((abs(v) for v in ri), default=0),
            )
            if row_max.bit_length() > max_bit_length:
                raise BudgetExceededError(
                    f"Bareiss simplex tableau bit-length {row_max.bit_length()} > {max_bit_length}"
                )

        obj_e = OBJ[enter]
        for j in range(num_cols):
            num_o = piv * OBJ[j] - obj_e * piv_row_T[j]
            assert num_o % D == 0
            OBJ[j] = num_o // D
        num_ov = piv * obj_val + obj_e * piv_row_RHS[0]
        assert num_ov % D == 0
        obj_val = num_ov // D

        basis[leave] = enter
        D = piv
        pivots += 1


def _lift_reduced_dual(
    y_red: Sequence[int],
    scale_red: int,
    m_orig: int,
    kept_orig_indices: Sequence[int],
    row_scales: Sequence[int],
) -> Tuple[Tuple[int, ...], int]:
    """Lift dual multipliers on primitive-reduced active rows back to original (A, b)."""
    active_scales = [
        row_scales[idx] for idx, yi in enumerate(y_red) if yi != 0
    ]
    G = lcm_list(active_scales) if active_scales else 1
    y_full = [0] * m_orig
    for idx, yi in enumerate(y_red):
        if yi != 0:
            orig_i = kept_orig_indices[idx]
            y_full[orig_i] = yi * (G // row_scales[idx])
    scale_full = scale_red * G
    g = gcd_list(y_full + ([scale_full] if scale_full != 0 else []))
    if g > 1:
        y_full = [v // g for v in y_full]
        scale_full //= g
    return (tuple(y_full), scale_full)


def solve_lp_bareiss(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    c: Optional[Sequence[int]] = None,
    max_pivots: int = 5000,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> LPResult:
    """Solve max { c^T x : A x <= b, x in R^n } in 100% pure-integer Two-Phase Bareiss Simplex
    with Avis lexicographic anti-cycling perturbation on [b, I_m].

    Zero floats, zero fractions.Fraction, zero silent truncation.
    """
    try:
        check_budget(deadline=deadline)
        m_orig = len(A)
        if m_orig != len(b):
            raise ValueError("Row mismatch between A and b")
        if c is not None:
            n = len(c)
            c_vec = [int(v) for v in c]
        else:
            n = len(A[0]) if m_orig > 0 else 0
            c_vec = [0] * n

        # Handle m_orig == 0 (unconstrained R^n)
        if m_orig == 0:
            if all(v == 0 for v in c_vec):
                return LPResult(
                    status=LPStatus.OPTIMAL,
                    x_opt=(tuple(0 for _ in range(n)), 1),
                    obj_val=(0, 1),
                    dual_y=(),
                    dual_scale=1,
                    pivots=0,
                )
            return LPResult(
                status=LPStatus.UNBOUNDED,
                x_opt=(tuple(0 for _ in range(n)), 1),
                unbounded_ray=qvec_norm(c_vec, 1),
                pivots=0,
            )

        # Normalize rows by positive GCD and screen zero rows
        A_red: List[Tuple[int, ...]] = []
        b_red: List[int] = []
        kept_indices: List[int] = []
        row_scales: List[int] = []

        for i in range(m_orig):
            if len(A[i]) != n:
                raise ValueError("Column mismatch in A")
            pa, pb, sc = primitive_ineq_row(A[i], b[i])
            if all(v == 0 for v in pa):
                if pb < 0:
                    y_infeas = tuple(1 if k == i else 0 for k in range(m_orig))
                    cert = FarkasInfeasibilityCert(y=y_infeas, rhs_dot=int(b[i]))
                    assert check_farkas_certificate(A, b, cert.y)
                    return LPResult(
                        status=LPStatus.INFEASIBLE,
                        farkas_infeas=cert,
                        pivots=0,
                    )
                # 0 <= b_i >= 0 is tautological; omit from active tableau
                continue
            A_red.append(pa)
            b_red.append(pb)
            kept_indices.append(i)
            row_scales.append(sc)

        m = len(A_red)
        if m == 0:
            if all(v == 0 for v in c_vec):
                return LPResult(
                    status=LPStatus.OPTIMAL,
                    x_opt=(tuple(0 for _ in range(n)), 1),
                    obj_val=(0, 1),
                    dual_y=tuple(0 for _ in range(m_orig)),
                    dual_scale=1,
                    pivots=0,
                )
            return LPResult(
                status=LPStatus.UNBOUNDED,
                x_opt=(tuple(0 for _ in range(n)), 1),
                unbounded_ray=qvec_norm(c_vec, 1),
                pivots=0,
            )

        # Standard-form lifting of free x in R^n via x = u - v0 * 1, u in R_+^n, v0 in R_+:
        # z = (u_0, ..., u_{n-1}, v0) >= 0 (p = n + 1 columns).
        p = n + 1
        A_tilde: List[List[int]] = []
        for i in range(m):
            row = list(A_red[i])
            row.append(-sum(row))
            A_tilde.append(row)
        c_tilde = list(c_vec) + [-sum(c_vec)]

        min_b = min(b_red)
        p1_pivots = 0

        if min_b < 0:
            # Phase I with scalar artificial variable z0 >= 0 at col p, slacks s_0..s_{m-1} at cols p+1..p+m
            # Choose largest row index i_star tied for min_b so RHS stays strictly lex-positive!
            i_star = 0
            for i in range(m):
                if b_red[i] == min_b:
                    i_star = i

            num_cols_p1 = p + 1 + m
            z0_col = p
            s_offset_p1 = p + 1

            T_p1: List[List[int]] = []
            RHS_p1: List[List[int]] = []
            basis_p1: List[int] = []

            for i in range(m):
                if i == i_star:
                    # Negate row i_star: -a_tilde_{i*}^T z + z0 - s_{i*} = -b_{i*} - eps_{i*}
                    t_row = [-v for v in A_tilde[i_star]] + [1] + [0] * m
                    t_row[s_offset_p1 + i_star] = -1
                    rhs_row = [-b_red[i_star]] + [0] * m
                    rhs_row[1 + i_star] = -1
                    T_p1.append(t_row)
                    RHS_p1.append(rhs_row)
                    basis_p1.append(z0_col)
                else:
                    # Subtract row i_star from row i:
                    # (a_tilde_i - a_tilde_{i*})^T z + s_i - s_{i*} = (b_i - b_{i*}) + eps_i - eps_{i*}
                    t_row = (
                        [A_tilde[i][j] - A_tilde[i_star][j] for j in range(p)]
                        + [0]
                        + [0] * m
                    )
                    t_row[s_offset_p1 + i] = 1
                    t_row[s_offset_p1 + i_star] = -1
                    rhs_row = [b_red[i] - b_red[i_star]] + [0] * m
                    rhs_row[1 + i] = 1
                    rhs_row[1 + i_star] = -1
                    T_p1.append(t_row)
                    RHS_p1.append(rhs_row)
                    basis_p1.append(s_offset_p1 + i)

            # Phase I objective: max (-z0) = b_{i*} - a_tilde_{i*}^T z - s_{i*}
            OBJ_p1 = [-v for v in A_tilde[i_star]] + [0] + [0] * m
            OBJ_p1[s_offset_p1 + i_star] = -1
            obj_val_p1 = b_red[i_star]

            (
                status_p1,
                T_p1,
                RHS_p1,
                OBJ_p1,
                obj_val_p1,
                basis_p1,
                D,
                piv_cnt_1,
                _,
            ) = _bareiss_simplex_loop(
                T_p1,
                RHS_p1,
                OBJ_p1,
                obj_val_p1,
                basis_p1,
                1,
                max_pivots=max_pivots,
                max_bit_length=max_bit_length,
                deadline=deadline,
            )
            p1_pivots = piv_cnt_1 + 1

            if obj_val_p1 < 0:
                y_red = [-OBJ_p1[s_offset_p1 + i] for i in range(m)]
                y_orig, _ = _lift_reduced_dual(
                    y_red, 0, m_orig, kept_indices, row_scales
                )
                rhs_dot = sum(y_orig[i] * int(b[i]) for i in range(m_orig))
                cert = FarkasInfeasibilityCert(y=y_orig, rhs_dot=rhs_dot)
                assert check_farkas_certificate(A, b, cert.y), (
                    "Internal Bareiss Phase-I Farkas invariant failed"
                )
                return LPResult(
                    status=LPStatus.INFEASIBLE,
                    farkas_infeas=cert,
                    pivots=p1_pivots,
                    phase1_pivots=p1_pivots,
                )

            # By our Avis lex-perturbation theorem, z0_col MUST have left basis_p1!
            assert z0_col not in basis_p1, "z0 remained basic despite lex perturbation"

            # Strip z0_col to obtain clean Phase II tableau with cols 0..p-1 (z) and p..p+m-1 (s)
            T = [row[:z0_col] + row[z0_col + 1 :] for row in T_p1]
            RHS = RHS_p1
            basis = [
                col if col < z0_col else col - 1 for col in basis_p1
            ]
        else:
            # b_red >= 0: initial basis is s_0 .. s_{m-1} with D = 1 and RHS = [b_red, I_m]
            D = 1
            T = []
            RHS = []
            for i in range(m):
                t_row = list(A_tilde[i]) + [0] * m
                t_row[p + i] = 1
                rhs_row = [b_red[i]] + [0] * m
                rhs_row[1 + i] = 1
                T.append(t_row)
                RHS.append(rhs_row)
            basis = [p + i for i in range(m)]

        # Phase II objective setup: d_w = c_tilde on z (0..p-1), 0 on slacks (p..p+m-1)
        num_cols_p2 = p + m
        d_w = c_tilde + [0] * m
        OBJ = [0] * num_cols_p2
        for j in range(num_cols_p2):
            acc = D * d_w[j]
            for i in range(m):
                b_col = basis[i]
                if d_w[b_col] != 0:
                    acc -= d_w[b_col] * T[i][j]
            OBJ[j] = acc
        obj_val = sum(d_w[basis[i]] * RHS[i][0] for i in range(m))

        (
            status_p2,
            T,
            RHS,
            OBJ,
            obj_val,
            basis,
            D,
            p2_pivots,
            enter_unb,
        ) = _bareiss_simplex_loop(
            T,
            RHS,
            OBJ,
            obj_val,
            basis,
            D,
            max_pivots=max(0, max_pivots - p1_pivots),
            max_bit_length=max_bit_length,
            deadline=deadline,
        )

        # Extract primal point x = u - v0 * 1 from current basis
        z_vals = [0] * p
        for i in range(m):
            if basis[i] < p:
                z_vals[basis[i]] = RHS[i][0]
        v0 = z_vals[n]
        x_nums = [z_vals[j] - v0 for j in range(n)]
        x_opt = qvec_norm(x_nums, D)
        assert check_primal_witness(A, b, x_opt), (
            "Internal Bareiss primal witness invariant failed"
        )

        if status_p2 == "UNBOUNDED":
            assert enter_unb is not None
            # Construct ray in w: w[enter_unb] = D, w[basis[i]] = -T[i][enter_unb]
            z_ray = [0] * p
            if enter_unb < p:
                z_ray[enter_unb] = D
            for i in range(m):
                if basis[i] < p:
                    z_ray[basis[i]] = -T[i][enter_unb]
            v0_ray = z_ray[n]
            ray_nums = [z_ray[j] - v0_ray for j in range(n)]
            ray_q = qvec_norm(ray_nums, D)
            return LPResult(
                status=LPStatus.UNBOUNDED,
                x_opt=x_opt,
                unbounded_ray=ray_q,
                pivots=p1_pivots + p2_pivots,
                phase1_pivots=p1_pivots,
                phase2_pivots=p2_pivots,
            )

        # OPTIMAL: extract dual y_red[i] = -OBJ[p + i] >= 0 and lift to original rows
        y_red = [-OBJ[p + i] for i in range(m)]
        y_orig, scale_orig = _lift_reduced_dual(
            y_red, D, m_orig, kept_indices, row_scales
        )
        opt_q = q_norm(obj_val, D)

        # Verify zero-trust Farkas optimality bound on original (A, b)
        bound_cert = FarkasBoundCert(
            y=y_orig,
            scale=scale_orig,
            c_num=tuple(c_vec),
            c_den=1,
            rhs_num=opt_q[0],
            rhs_den=opt_q[1],
            is_vacuous=False,
        )
        assert check_positivity_farkas(A, b, bound_cert, strict=False), (
            "Internal Bareiss Phase-II Farkas optimality invariant failed"
        )

        return LPResult(
            status=LPStatus.OPTIMAL,
            x_opt=x_opt,
            obj_val=opt_q,
            dual_y=y_orig,
            dual_scale=scale_orig,
            pivots=p1_pivots + p2_pivots,
            phase1_pivots=p1_pivots,
            phase2_pivots=p2_pivots,
        )
    except BudgetExceededError:
        return LPResult(status=LPStatus.BUDGET_EXCEEDED)


def certify_linear_upper_bound(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    c_num: Sequence[int],
    c_den: int,
    rhs_num: int,
    rhs_den: int,
    strict: bool = False,
    max_pivots: int = 5000,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> Tuple[Verdict_Like, Optional[FarkasBoundCert], Optional[QVec], int]:
    """Certify (c_num / c_den)^T x <= rhs_num / rhs_den (or < if strict) over {A x <= b}.

    Returns (verdict_str, FarkasBoundCert | None, counterexample_x | None, pivots).
    """
    if c_den <= 0 or rhs_den <= 0:
        raise ValueError("Denominators must be positive")
    res = solve_lp_bareiss(
        A,
        b,
        c=c_num,
        max_pivots=max_pivots,
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    if res.status == LPStatus.BUDGET_EXCEEDED:
        return ("OPEN_BUDGET_EXCEEDED", None, None, res.pivots)
    if res.status == LPStatus.INFEASIBLE:
        assert res.farkas_infeas is not None
        cert = FarkasBoundCert(
            y=res.farkas_infeas.y,
            scale=0,
            c_num=tuple(int(v) for v in c_num),
            c_den=int(c_den),
            rhs_num=int(rhs_num),
            rhs_den=int(rhs_den),
            is_vacuous=True,
        )
        assert check_positivity_farkas(A, b, cert, strict=strict)
        return ("CERTIFIED_EXACT", cert, None, res.pivots)

    if res.status == LPStatus.UNBOUNDED:
        # Step along unbounded ray from x_opt until c^T x > rhs
        assert res.x_opt is not None and res.unbounded_ray is not None
        xn, xd = res.x_opt
        rn, rd = res.unbounded_ray
        n = len(c_num)
        cx_num = sum(int(c_num[j]) * xn[j] for j in range(n))
        cr_num = sum(int(c_num[j]) * rn[j] for j in range(n))
        # Need (cx_num / (c_den * xd)) + step * (cr_num / (c_den * rd)) > rhs_num / rhs_den
        # Pick integer step >= 1 sufficiently large:
        cd = lcm_pair(xd, rd)
        base_v = cx_num * (cd // xd)
        step_v = cr_num * (cd // rd)
        # Want rhs_den * (base_v + k * step_v) > c_den * cd * rhs_num
        target = c_den * cd * rhs_num - rhs_den * base_v
        if target < 0:
            k = 1
        else:
            k = (target // (rhs_den * step_v)) + 2
        wit_nums = [
            xn[j] * (cd // xd) + k * rn[j] * (cd // rd) for j in range(n)
        ]
        wit = qvec_norm(wit_nums, cd)
        assert check_primal_witness(A, b, wit)
        return ("FALSIFIED_COUNTEREXAMPLE", None, wit, res.pivots)

    # OPTIMAL: res.obj_val is max (c_num^T x) = opt_num / opt_den
    assert res.obj_val is not None and res.dual_y is not None and res.x_opt is not None
    opt_num, opt_den = res.obj_val
    # Compare (opt_num / (opt_den * c_den)) vs (rhs_num / rhs_den)
    lhs_cmp = opt_num * rhs_den
    rhs_cmp = rhs_num * opt_den * c_den
    holds = (lhs_cmp < rhs_cmp) if strict else (lhs_cmp <= rhs_cmp)
    if holds:
        # In solve_lp_bareiss: y^T A == dual_scale * c_num^T and y^T b == dual_scale * (opt_num / opt_den).
        # For FarkasBoundCert with c_den: we want c_den * (y_new^T A) == scale_new * c_num^T.
        # Choosing y_new = dual_y and scale_new = c_den * dual_scale satisfies:
        #   c_den * (dual_y^T A) == c_den * dual_scale * c_num^T == scale_new * c_num^T
        #   rhs_den * (dual_y^T b) == rhs_den * dual_scale * (opt_num / opt_den)
        #                           <= c_den * dual_scale * rhs_num == scale_new * rhs_num!
        y_scaled = list(res.dual_y)
        sc_scaled = int(c_den) * int(res.dual_scale)
        g = gcd_list(y_scaled + [sc_scaled])
        if g > 1:
            y_scaled = [v // g for v in y_scaled]
            sc_scaled //= g
        cert = FarkasBoundCert(
            y=tuple(y_scaled),
            scale=sc_scaled,
            c_num=tuple(int(v) for v in c_num),
            c_den=int(c_den),
            rhs_num=int(rhs_num),
            rhs_den=int(rhs_den),
            is_vacuous=False,
        )
        assert check_positivity_farkas(A, b, cert, strict=strict)
        return ("CERTIFIED_EXACT", cert, None, res.pivots)

    return ("FALSIFIED_COUNTEREXAMPLE", None, res.x_opt, res.pivots)


Verdict_Like = str


def chebyshev_ball_linf(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    r_cap: Optional[int] = 1000,
    row_l1_norm: bool = True,
    max_pivots: int = 5000,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> Tuple[Optional[Rat], Optional[QVec], LPResult]:
    """Compute exact l_infinity Chebyshev ball max r s.t. A x + w_i * r <= b, r >= 0.

    When row_l1_norm=True (geometric l_inf cube B_inf(x_c, r) subset P), w_i = ||a_i||_1 in Z
    (avoiding irrational Euclidean row norms!). When row_l1_norm=False, w_i = 1 (max r s.t. A x + r * 1 <= b).
    If r_cap is an int > 0, caps r <= r_cap so unbounded cones return finite r* > 0.
    Returns (r_star, x_center, lp_result). If infeasible or r* == 0, r_star is (0, 1) or None.
    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    aug_A: List[List[int]] = []
    aug_b: List[int] = []
    for i in range(m):
        row = [int(v) for v in A[i]]
        if row_l1_norm:
            wi = sum(abs(v) for v in row)
        else:
            wi = 1 if any(v != 0 for v in row) else 0
        aug_A.append(row + [wi])
        aug_b.append(int(b[i]))
    # Enforce r >= 0 -> -r <= 0
    aug_A.append([0] * n + [-1])
    aug_b.append(0)
    if r_cap is not None:
        aug_A.append([0] * n + [1])
        aug_b.append(int(r_cap))

    c_obj = [0] * n + [1]
    res = solve_lp_bareiss(
        aug_A,
        aug_b,
        c=c_obj,
        max_pivots=max_pivots,
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    if res.status in (LPStatus.INFEASIBLE, LPStatus.BUDGET_EXCEEDED):
        return (None, None, res)
    if res.status == LPStatus.UNBOUNDED:
        assert res.x_opt is not None and res.unbounded_ray is not None
        # Step along ray so r > 0
        xn, xd = res.x_opt
        rn, rd = res.unbounded_ray
        cd = lcm_pair(xd, rd)
        comb = [xn[j] * (cd // xd) + rn[j] * (cd // rd) for j in range(n + 1)]
        r_q = q_norm(comb[n], cd)
        xc_q = qvec_norm(comb[:n], cd)
        return (r_q, xc_q, res)

    assert res.x_opt is not None and res.obj_val is not None
    xn, xd = res.x_opt
    xc_q = qvec_norm(xn[:n], xd)
    return (res.obj_val, xc_q, res)
