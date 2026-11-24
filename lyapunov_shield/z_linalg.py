from __future__ import annotations

import enum
import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple


class Verdict(str, enum.Enum):
    """Three-valued constructive verification verdict (fail-closed)."""

    CERTIFIED_EXACT = "CERTIFIED_EXACT"
    FALSIFIED_COUNTEREXAMPLE = "FALSIFIED_COUNTEREXAMPLE"
    OPEN_BUDGET_EXCEEDED = "OPEN_BUDGET_EXCEEDED"


class BudgetExceededError(RuntimeError):
    """Raised when deterministic iteration, shard, bit-length, or wall-clock caps trigger."""


Rat = Tuple[int, int]
QVec = Tuple[Tuple[int, ...], int]


class WatchdogTimer:
    """Deterministic monotonic wall-clock deadline watchdog (never uses signal.alarm)."""

    def __init__(self, timeout_sec: float = 20.0, use_sigalrm: bool = False) -> None:
        self.timeout_sec = float(timeout_sec)
        self.use_sigalrm = False
        self.start_time = 0.0
        self.deadline = 0.0

    def __enter__(self) -> WatchdogTimer:
        self.start_time = time.monotonic()
        self.deadline = self.start_time + self.timeout_sec
        return self

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise BudgetExceededError(
                f"Wall-clock deadline exceeded ({self.timeout_sec:.2f}s)"
            )

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


def check_budget(
    deadline: Optional[float] = None,
    val: Optional[int] = None,
    max_bit_length: int = 4096,
) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise BudgetExceededError("Wall-clock deadline exceeded inside exact kernel")
    if val is not None and val.bit_length() > max_bit_length:
        raise BudgetExceededError(
            f"Integer bit-length {val.bit_length()} exceeded cap {max_bit_length}"
        )


def gcd_list(vals: Iterable[int]) -> int:
    g = 0
    for v in vals:
        if v != 0:
            g = math.gcd(g, abs(int(v)))
            if g == 1:
                return 1
    return g


def lcm_pair(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return math.lcm(abs(int(a)), abs(int(b)))


def lcm_list(vals: Iterable[int]) -> int:
    res = 1
    for v in vals:
        iv = abs(int(v))
        if iv == 0:
            return 0
        res = math.lcm(res, iv)
    return res


def q_norm(num: int, den: int = 1) -> Rat:
    if den == 0:
        raise ZeroDivisionError("Zero denominator in exact rational")
    if den < 0:
        num = -num
        den = -den
    if num == 0:
        return (0, 1)
    g = math.gcd(abs(num), den)
    if g > 1:
        return (num // g, den // g)
    return (num, den)


def q_add(a: Rat, b: Rat) -> Rat:
    return q_norm(a[0] * b[1] + b[0] * a[1], a[1] * b[1])


def q_sub(a: Rat, b: Rat) -> Rat:
    return q_norm(a[0] * b[1] - b[0] * a[1], a[1] * b[1])


def q_mul(a: Rat, b: Rat) -> Rat:
    if a[0] == 0 or b[0] == 0:
        return (0, 1)
    return q_norm(a[0] * b[0], a[1] * b[1])


def q_div(a: Rat, b: Rat) -> Rat:
    if b[0] == 0:
        raise ZeroDivisionError("Division by zero rational")
    return q_norm(a[0] * b[1], a[1] * b[0])


def q_neg(a: Rat) -> Rat:
    return (-a[0], a[1])


def q_abs(a: Rat) -> Rat:
    return (abs(a[0]), a[1])


def q_cmp(a: Rat, b: Rat) -> int:
    diff = a[0] * b[1] - b[0] * a[1]
    if diff < 0:
        return -1
    if diff > 0:
        return 1
    return 0


def q_le(a: Rat, b: Rat) -> bool:
    return a[0] * b[1] <= b[0] * a[1]


def q_lt(a: Rat, b: Rat) -> bool:
    return a[0] * b[1] < b[0] * a[1]


def q_eq(a: Rat, b: Rat) -> bool:
    return a[0] * b[1] == b[0] * a[1]


def q_min(a: Rat, b: Rat) -> Rat:
    return a if q_le(a, b) else b


def q_max(a: Rat, b: Rat) -> Rat:
    return a if q_le(b, a) else b


def q_str(a: Rat) -> str:
    n, d = q_norm(a[0], a[1])
    return str(n) if d == 1 else f"{n}/{d}"


def q_to_float(a: Rat) -> float:
    return a[0] / a[1]


def qvec_norm(nums: Sequence[int], den: int = 1) -> QVec:
    if den == 0:
        raise ZeroDivisionError("Zero denominator in QVec")
    if den < 0:
        nums = tuple(-int(v) for v in nums)
        den = -int(den)
    else:
        nums = tuple(int(v) for v in nums)
        den = int(den)
    if all(v == 0 for v in nums):
        return (tuple(0 for _ in nums), 1)
    g = math.gcd(gcd_list(nums), den)
    if g > 1:
        return (tuple(v // g for v in nums), den // g)
    return (nums, den)


def qvec_from_rats(rats: Sequence[Rat]) -> QVec:
    if not rats:
        return ((), 1)
    common_d = lcm_list(r[1] for r in rats)
    nums = [r[0] * (common_d // r[1]) for r in rats]
    return qvec_norm(nums, common_d)


def qvec_to_rats(x: QVec) -> Tuple[Rat, ...]:
    nums, den = x
    return tuple(q_norm(v, den) for v in nums)


def qvec_zero(dim: int) -> QVec:
    return (tuple(0 for _ in range(dim)), 1)


def qvec_add(a: QVec, b: QVec) -> QVec:
    an, ad = a
    bn, bd = b
    cd = lcm_pair(ad, bd)
    sa = cd // ad
    sb = cd // bd
    return qvec_norm([an[i] * sa + bn[i] * sb for i in range(len(an))], cd)


def qvec_sub(a: QVec, b: QVec) -> QVec:
    an, ad = a
    bn, bd = b
    cd = lcm_pair(ad, bd)
    sa = cd // ad
    sb = cd // bd
    return qvec_norm([an[i] * sa - bn[i] * sb for i in range(len(an))], cd)


def qvec_scale(scale: Rat, x: QVec) -> QVec:
    sn, sd = scale
    xn, xd = x
    if sn == 0:
        return qvec_zero(len(xn))
    return qvec_norm([sn * v for v in xn], sd * xd)


def qvec_dot(a: QVec, b: QVec) -> Rat:
    an, ad = a
    bn, bd = b
    s = sum(an[i] * bn[i] for i in range(len(an)))
    return q_norm(s, ad * bd)


def qvec_l1(x: QVec) -> Rat:
    xn, xd = x
    return q_norm(sum(abs(v) for v in xn), xd)


def qvec_linf(x: QVec) -> Rat:
    xn, xd = x
    if not xn:
        return (0, 1)
    return q_norm(max(abs(v) for v in xn), xd)


def qvec_eq(a: QVec, b: QVec) -> bool:
    na = qvec_norm(a[0], a[1])
    nb = qvec_norm(b[0], b[1])
    return na == nb


def qvec_str(x: QVec) -> str:
    rats = qvec_to_rats(x)
    return "(" + ", ".join(q_str(r) for r in rats) + ")"


def primitive_ineq_row(a_row: Sequence[int], b_val: int) -> Tuple[Tuple[int, ...], int, int]:
    """Normalize inequality a^T x <= b by dividing by positive gcd(a_row, b_val).

    Preserves inequality direction strictly (scale >= 1).
    Returns (prim_a, prim_b, scale_div).
    """
    a_tup = tuple(int(v) for v in a_row)
    b_int = int(b_val)
    g = gcd_list(a_tup + (b_int,))
    if g <= 1:
        if all(v == 0 for v in a_tup):
            if b_int < 0:
                return (a_tup, -1, abs(b_int))
            if b_int > 0:
                return (a_tup, 1, b_int)
            return (a_tup, 0, 1)
        return (a_tup, b_int, 1)
    return (tuple(v // g for v in a_tup), b_int // g, g)


def canonical_hyperplane(a_row: Sequence[int], b_val: int) -> Tuple[Tuple[int, ...], int, int]:
    """Canonicalize hyperplane a^T x = b so gcd == 1 and first nonzero of a is > 0.

    Returns (canon_a, canon_b, sign_flipped_flag_pm1).
    """
    a_tup = tuple(int(v) for v in a_row)
    b_int = int(b_val)
    g = gcd_list(a_tup + (b_int,))
    if g > 1:
        a_tup = tuple(v // g for v in a_tup)
        b_int //= g
    sign = 1
    for v in a_tup:
        if v != 0:
            if v < 0:
                sign = -1
            break
    if sign == -1:
        a_tup = tuple(-v for v in a_tup)
        b_int = -b_int
    return (a_tup, b_int, sign)


def bareiss_echelon(
    matrix: Sequence[Sequence[int]],
    num_piv_cols: Optional[int] = None,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> Tuple[List[List[int]], Tuple[int, ...], int, int]:
    """Exact fraction-free Gauss-Jordan Bareiss elimination over Z.

    Eliminates both above and below each pivot using Sylvester's identity.
    After completion:
      - Every pivot row r has M[r][pivot_cols[r]] == det_sub != 0 and 0 in all
        other pivot columns.
      - Non-pivot rows r >= rank have M[r][:num_piv_cols] == 0.
    Returns (M_reduced, pivot_cols, det_sub, swap_sign).
    """
    m = len(matrix)
    if m == 0:
        return ([], (), 1, 1)
    n = len(matrix[0])
    if num_piv_cols is None:
        num_piv_cols = n
    M = [[int(v) for v in row] for row in matrix]
    pivot_cols: List[int] = []
    prev_piv = 1
    swap_sign = 1
    piv_row = 0

    for col in range(num_piv_cols):
        check_budget(deadline=deadline)
        if piv_row >= m:
            break
        sel = -1
        for r in range(piv_row, m):
            if M[r][col] != 0:
                sel = r
                break
        if sel == -1:
            continue
        if sel != piv_row:
            M[piv_row], M[sel] = M[sel], M[piv_row]
            swap_sign = -swap_sign

        piv = M[piv_row][col]
        check_budget(deadline=deadline, val=piv, max_bit_length=max_bit_length)
        pivot_cols.append(col)

        piv_row_vals = M[piv_row]
        for r in range(m):
            if r == piv_row:
                continue
            r_vals = M[r]
            factor = r_vals[col]
            for c in range(n):
                num = piv * r_vals[c] - factor * piv_row_vals[c]
                assert num % prev_piv == 0
                val = num // prev_piv
                r_vals[c] = val
            # Check bit length on max entry in row
            row_max = max((abs(x) for x in r_vals), default=0)
            if row_max.bit_length() > max_bit_length:
                raise BudgetExceededError(
                    f"Bareiss echelon bit-length {row_max.bit_length()} > {max_bit_length}"
                )

        # Update earlier pivot entries to match current piv!
        # Wait: when factor == 0 (which holds in earlier pivot columns since piv_row has 0 in earlier pivot cols!),
        # num = piv * M[r][earlier_piv_col] - 0 = piv * prev_piv, divided by prev_piv == piv!
        # So earlier pivot entries automatically became `piv` in the loop above!
        prev_piv = piv
        piv_row += 1

    return (M, tuple(pivot_cols), prev_piv, swap_sign)


def bareiss_det(
    square_matrix: Sequence[Sequence[int]],
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> int:
    n = len(square_matrix)
    if n == 0:
        return 1
    _, piv_cols, det_sub, swap_sign = bareiss_echelon(
        square_matrix,
        num_piv_cols=n,
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    if len(piv_cols) < n:
        return 0
    return swap_sign * det_sub


def bareiss_rank(
    matrix: Sequence[Sequence[int]],
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> int:
    if not matrix or not matrix[0]:
        return 0
    _, piv_cols, _, _ = bareiss_echelon(
        matrix,
        num_piv_cols=len(matrix[0]),
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    return len(piv_cols)


def bareiss_nullspace(
    matrix: Sequence[Sequence[int]],
    num_vars: Optional[int] = None,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> List[Tuple[int, ...]]:
    """Compute an exact primitive integer basis for {v in Z^n : A v = 0}."""
    if not matrix:
        if num_vars is None:
            return []
        return [
            tuple(1 if j == i else 0 for j in range(num_vars))
            for i in range(num_vars)
        ]
    n = len(matrix[0])
    M, piv_cols, det_sub, _ = bareiss_echelon(
        matrix,
        num_piv_cols=n,
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    piv_set = set(piv_cols)
    piv_map = {col: idx for idx, col in enumerate(piv_cols)}
    basis: List[Tuple[int, ...]] = []
    for free_col in range(n):
        if free_col in piv_set:
            continue
        vec = [0] * n
        vec[free_col] = det_sub
        for pcol, prow in piv_map.items():
            vec[pcol] = -M[prow][free_col]
        g = gcd_list(vec)
        if g > 1:
            vec = [v // g for v in vec]
        # Canonicalize sign: first nonzero entry > 0
        for v in vec:
            if v != 0:
                if v < 0:
                    vec = [-x for x in vec]
                break
        basis.append(tuple(vec))
    return basis


def solve_linear_system_int(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> Tuple[str, Optional[QVec], List[Tuple[int, ...]]]:
    """Solve exact integer linear system A x = b over Q^n.

    Returns (status, x_particular, nullspace_basis) where status is one of:
      - "UNIQUE": unique solution x_particular in Q^n, nullspace_basis == []
      - "AFFINE_SUBSPACE": x_particular + span(nullspace_basis)
      - "INCONSISTENT": (None, [])
    """
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    if m == 0:
        return ("UNIQUE", ((), 1), [])
    aug = [list(A[i]) + [int(b[i])] for i in range(m)]
    M, piv_cols, det_sub, _ = bareiss_echelon(
        aug,
        num_piv_cols=n,
        max_bit_length=max_bit_length,
        deadline=deadline,
    )
    rank = len(piv_cols)
    for r in range(rank, m):
        if M[r][n] != 0:
            return ("INCONSISTENT", None, [])

    x_nums = [0] * n
    for idx, pcol in enumerate(piv_cols):
        x_nums[pcol] = M[idx][n]
    x_part = qvec_norm(x_nums, det_sub)

    if rank == n:
        return ("UNIQUE", x_part, [])

    piv_set = set(piv_cols)
    piv_map = {col: idx for idx, col in enumerate(piv_cols)}
    basis: List[Tuple[int, ...]] = []
    for free_col in range(n):
        if free_col in piv_set:
            continue
        vec = [0] * n
        vec[free_col] = det_sub
        for pcol, prow in piv_map.items():
            vec[pcol] = -M[prow][free_col]
        g = gcd_list(vec)
        if g > 1:
            vec = [v // g for v in vec]
        for v in vec:
            if v != 0:
                if v < 0:
                    vec = [-x for x in vec]
                break
        basis.append(tuple(vec))
    return ("AFFINE_SUBSPACE", x_part, basis)


def solve_one_minus_m_q(
    M_num: Sequence[Sequence[int]],
    M_den: int,
    c_num: Sequence[int],
    c_den: int,
    max_bit_length: int = 4096,
    deadline: Optional[float] = None,
) -> Tuple[str, Optional[QVec], List[Tuple[int, ...]]]:
    """Solve fixed-point / unrolled cycle system (I - M/M_den) x = c/c_den over Q^n."""
    n = len(M_num)
    if M_den <= 0 or c_den <= 0:
        raise ValueError("Denominators must be positive integers")
    common_d = lcm_pair(M_den, c_den)
    sm = common_d // M_den
    sc = common_d // c_den
    A_int: List[List[int]] = []
    b_int: List[int] = []
    for i in range(n):
        row = [-sm * int(M_num[i][j]) for j in range(n)]
        row[i] += common_d
        A_int.append(row)
        b_int.append(sc * int(c_num[i]))
    return solve_linear_system_int(
        A_int, b_int, max_bit_length=max_bit_length, deadline=deadline
    )
