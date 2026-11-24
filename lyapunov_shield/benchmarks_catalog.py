from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import (
    LPStatus,
    certify_linear_upper_bound,
    chebyshev_ball_linf,
    check_positivity_farkas,
    check_primal_witness,
    solve_lp_bareiss,
)
from .barrier_verifier import (
    BarrierVerificationResult,
    ControlAffineCell,
    ControlAffinePWAPlant,
    MultiFacetBarrier,
    check_barrier_certificate_bundle,
    check_corner_deadlock_cert,
    eval_heaviside_formal_lie_derivative,
    verify_barrier,
)
from .filippov_crease import check_filippov_witness
from .lattice_cegis import (
    compose_linear_plant_with_relu_controller,
    snap_relu_network_to_lattice,
)
from .lyapunov_verifier import (
    LyapunovVerificationResult,
    SectorPWACell,
    SectorPWAPlant,
    check_lyapunov_certificate_bundle,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
)
from .orbit_detector import (
    check_non_fixed_point_certs,
    check_periodic_orbit_witness,
    detect_periodic_orbits_and_fixed_points,
)
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
    certify_relu_upper_bound_ct_bab,
    compose_pullback_pwa,
    zaslavsky_upper_bound,
)
from .runtime_shield import (
    RuntimeCBFShield,
    ShieldStepResult,
    check_shield_step_result,
)
from .z_linalg import (
    QVec,
    Rat,
    Verdict,
    q_lt,
    q_norm,
    qvec_add,
    qvec_eq,
    qvec_norm,
    qvec_scale,
)


@dataclasses.dataclass(frozen=True)
class TrapAuditEntry:
    trap_id: int
    name: str
    passed: bool
    verdict: Verdict
    elapsed_ms: float
    summary: str


# ==============================================================================
# PART 1: NINE ADVERSARIAL CONTROL TRAPS (EXACT CLOSED-FORM CONSTRUCTORS)
# ==============================================================================


def build_l1_norm_relu(dim: int, name: str = "v_l1") -> ReLUNetwork:
    """Exact ReLU network computing V(x) = ||x||_1 = sum_{j=1}^n (ReLU(x_j) + ReLU(-x_j))."""
    W_in: List[List[int]] = []
    for j in range(dim):
        pos = [1 if q == j else 0 for q in range(dim)]
        neg = [-1 if q == j else 0 for q in range(dim)]
        W_in.append(pos)
        W_in.append(neg)
    return ReLUNetwork(
        in_dim=dim,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(dim, 2 * dim, W_in, [0] * (2 * dim), 1), use_relu=True),
            ReLULayer(AffineMapQ.create(2 * dim, 1, [[1] * (2 * dim)], [0], 1), use_relu=False),
        ),
        name=name,
    )


def build_trap1_filippov_uphill_sliding() -> Tuple[PWAFunction, PWAFunction, HPolytope]:
    """TRAP 1: Filippov Uphill Sliding Crease on D = [-1, 1]^2, x* = (0,0), V(x) = |x_1| + |x_2|.
    f(x) = (-2|x_2| - x_1, x_2) for x_1 > 0 and f(x) = (+2|x_2| - x_1, x_2) for x_1 < 0.
    In all 4 open quadrants g_q^T f_q(x) = -||x||_1 < 0, yet on x_1 = 0 (x_2 > 0)
    both sides push inward driving f_S(0, 1/2) = (0, +1/2) with D^+_{f_S} V = +1/2 > 0.
    """
    dom = HPolytope.symmetric_box(2, (1, 1))
    q_polys = [
        dom.with_halfspace([-1, 0], 0).with_halfspace([0, -1], 0),  # Q1: x1>=0, x2>=0
        dom.with_halfspace([1, 0], 0).with_halfspace([0, -1], 0),   # Q2: x1<=0, x2>=0
        dom.with_halfspace([1, 0], 0).with_halfspace([0, 1], 0),    # Q3: x1<=0, x2<=0
        dom.with_halfspace([-1, 0], 0).with_halfspace([0, 1], 0),   # Q4: x1>=0, x2<=0
    ]
    v_grads = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    v_pwa = PWAFunction(
        in_dim=2,
        out_dim=1,
        domain=dom,
        cells=tuple(
            PWACell(
                cell_id=i,
                polytope=q_polys[i],
                affine_map=AffineMapQ.create(2, 1, [v_grads[i]], [0], 1),
                tag=f"Q{i+1}",
            )
            for i in range(4)
        ),
        name="trap1_v_l1",
    )
    # Linear matrices per quadrant:
    # Q1 (x1>=0, x2>=0): (-x1 - 2x2, x2)
    # Q2 (x1<=0, x2>=0): (-x1 + 2x2, x2)
    # Q3 (x1<=0, x2<=0): (-x1 - 2x2, x2)  (since +2|x2| = -2x2)
    # Q4 (x1>=0, x2<=0): (-x1 + 2x2, x2)  (since -2|x2| = +2x2)
    f_mats = [
        [[-1, -2], [0, 1]],
        [[-1, 2], [0, 1]],
        [[-1, -2], [0, 1]],
        [[-1, 2], [0, 1]],
    ]
    plant_pwa = PWAFunction(
        in_dim=2,
        out_dim=2,
        domain=dom,
        cells=tuple(
            PWACell(
                cell_id=i,
                polytope=q_polys[i],
                affine_map=AffineMapQ.create(2, 2, f_mats[i], [0, 0], 1),
                tag=f"Q{i+1}",
            )
            for i in range(4)
        ),
        name="trap1_filippov_plant",
    )
    return (v_pwa, plant_pwa, dom)


def build_trap2_bouligand_corner_escape() -> Tuple[ReLUNetwork, ControlAffinePWAPlant, ControlAffinePWAPlant, HPolytope]:
    """TRAP 2: Safe cone C = {x in [-1, 1]^2 : h(x) = x_2 - ReLU(x_1) - ReLU(-x_1) >= 0} at apex x* = (0,0).
    Returns (h_net, plant_autonomous_2a, plant_controlled_2b, domain).
    """
    dom = HPolytope.symmetric_box(2, (1, 1))
    h_net = ReLUNetwork(
        in_dim=2,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(2, 2, [[1, 0], [-1, 0]], [0, 0], 1), use_relu=True),
            ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [0], 1), use_relu=False),
        ),
        residual=AffineMapQ.create(2, 1, [[0, 1]], [0], 1),
        name="trap2_cone_barrier",
    )
    # (2a) Autonomous f(x) = (1, 0)
    plant_2a = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=0,
        domain=dom,
        f_map=AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [1, 0], 1),
        B_num=[[], []],
        B_denom=1,
        u_poly=None,
        name="trap2a_autonomous",
    )
    # (2b) Controlled dx/dt = (0, -1) + (1, 0) u, u in [-2, 2]
    plant_2b = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=1,
        domain=dom,
        f_map=AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [0, -1], 1),
        B_num=[[1], [0]],
        B_denom=1,
        u_poly=HPolytope.symmetric_box(1, (2, 1)),
        name="trap2b_controlled",
    )
    return (h_net, plant_2a, plant_2b, dom)


def build_trap3_hidden_periodic_orbits() -> Tuple[ReLUNetwork, HPolytope, ReLUNetwork, HPolytope]:
    """TRAP 3: Hidden Rational Period-2 & Period-3 Limit Cycles with Unique Fixed Point 0.
    - (3a) F(x) = (1/2)x - (5/2)ReLU(x - 1/9) + (3/2)ReLU(x - 1/3) + (5/2)ReLU(-x - 1/9) - (3/2)ReLU(-x - 1/3) on [-1, 1]
    - (3b) 2D 6-neuron ReLU plateau map on [-3, 3]^2 with s(z) = 2*ReLU(z - 1/2) - 2*ReLU(z - 1),
           z_1 = x_1 - (1/2)x_2, z_2 = x_2 - (1/2)x_1, z_3 = -(1/2)x_1 - (1/2)x_2,
           F(x) = (0, 2)*s(z_1) + (-2, -2)*s(z_2) + (2, 0)*s(z_3).
    """
    dom_3a = HPolytope.symmetric_box(1, (1, 1))
    net_3a = ReLUNetwork(
        in_dim=1,
        out_dim=1,
        layers=(
            ReLULayer(
                AffineMapQ.create(1, 4, [[9], [9], [-9], [-9]], [-1, -3, -1, -3], 9),
                use_relu=True,
            ),
            ReLULayer(
                AffineMapQ.create(4, 1, [[-5, 3, 5, -3]], [0], 2),
                use_relu=False,
            ),
        ),
        residual=AffineMapQ.create(1, 1, [[1]], [0], 2),
        name="trap3a_odd_period2",
    )

    dom_3b = HPolytope.symmetric_box(2, (3, 1))
    # 6 neurons: z_1 - 1/2, z_1 - 1, z_2 - 1/2, z_2 - 1, z_3 - 1/2, z_3 - 1
    # Multiply by denom=2:
    # 2 z_1 = 2 x_1 - x_2 -> biases -1, -2
    # 2 z_2 = -x_1 + 2 x_2 -> biases -1, -2
    # 2 z_3 = -x_1 - x_2 -> biases -1, -2
    W_3b = [
        [2, -1],
        [2, -1],
        [-1, 2],
        [-1, 2],
        [-1, -1],
        [-1, -1],
    ]
    b_3b = [-1, -2, -1, -2, -1, -2]
    # s(z_k) = 2 * u_{2k} - 2 * u_{2k+1}
    # F_1 = 0 * s(z_1) - 2 * s(z_2) + 2 * s(z_3) = -4 u_2 + 4 u_3 + 4 u_4 - 4 u_5
    # F_2 = 2 * s(z_1) - 2 * s(z_2) + 0 * s(z_3) = 4 u_0 - 4 u_1 - 4 u_2 + 4 u_3
    W_out_3b = [
        [0, 0, -4, 4, 4, -4],
        [4, -4, -4, 4, 0, 0],
    ]
    net_3b = ReLUNetwork(
        in_dim=2,
        out_dim=2,
        layers=(
            ReLULayer(AffineMapQ.create(2, 6, W_3b, b_3b, 2), use_relu=True),
            ReLULayer(AffineMapQ.create(6, 2, W_out_3b, [0, 0], 1), use_relu=False),
        ),
        name="trap3b_plateau_period3",
    )
    return (net_3a, dom_3a, net_3b, dom_3b)


def build_trap4_deadlocked_concave_corner() -> Tuple[MultiFacetBarrier, ControlAffinePWAPlant, ControlAffinePWAPlant, HPolytope]:
    """TRAP 4: Concave barrier h(x) = min(3x_1 - x_2, -x_1 + 3x_2) on [-1, 1]^2 at x* = (0,0)
    under dx/dt = (-7/4, -7/4) + I_2 u with u in [-1, 1]^2.
    """
    dom = HPolytope.symmetric_box(2, (1, 1))
    h1 = AffineMapQ.create(2, 1, [[3, -1]], [0], 1)
    h2 = AffineMapQ.create(2, 1, [[-1, 3]], [0], 1)
    barrier = MultiFacetBarrier(dim=2, domain=dom, facets=(h1, h2), name="trap4_concave_corner")
    f_drift = AffineMapQ.create(2, 2, [[0, 0], [0, 0]], [-7, -7], 4)
    plant_sat = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=2,
        domain=dom,
        f_map=f_drift,
        B_num=[[1, 0], [0, 1]],
        B_denom=1,
        u_poly=HPolytope.symmetric_box(2, (1, 1)),
        name="trap4_saturated",
    )
    plant_unsat = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=2,
        domain=dom,
        f_map=f_drift,
        B_num=[[1, 0], [0, 1]],
        B_denom=1,
        u_poly=None,
        name="trap4_unsaturated",
    )
    return (barrier, plant_sat, plant_unsat, dom)


def build_trap5_sub_tolerance_and_sliver() -> Tuple[HPolytope, HPolytope, HPolytope]:
    """TRAP 5: 5e-8 Sub-Tolerance Leak (large 10^7 shard + [0,1]^3 shard) & 1e-14 2D Sliver."""
    # (5a-1) P_large = {x >= 0 : x_1 + x_2 <= 10^7, x_2 + x_3 <= 1}
    A_large = [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [1, 1, 0],
        [0, 1, 1],
    ]
    b_large = [0, 0, 0, 10_000_000, 1]
    p_large = HPolytope.create(3, A_large, b_large)

    # (5a-2) Unit cube variant [0, 1]^3 with 20_000_000 x_3 - x_1 <= 0
    A_cube = [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
        [-1, 0, 20_000_000],
    ]
    b_cube = [0, 1, 0, 1, 0, 1, 0]
    p_cube = HPolytope.create(3, A_cube, b_cube)

    # (5b) 2D sliver {0 <= x_1 <= 1, x_1 <= x_2 <= x_1 + 10^-14}
    s14 = 10**14
    A_sliver = [
        [-1, 0],
        [1, 0],
        [1, -1],
        [-s14, s14],
    ]
    b_sliver = [0, 1, 0, 1]
    p_sliver = HPolytope.create(2, A_sliver, b_sliver)
    return (p_large, p_cube, p_sliver)


def build_trap6_rounding_drift_vs_anchor() -> Tuple[ReLUNetwork, ReLUNetwork, PWAFunction, PWAFunction, ReLUNetwork, HPolytope]:
    """TRAP 6: Discrete plant x_{t+1} = x_t + (1/2) u_t on [-1, 1] with
    pi(x) = -x + (1/3)ReLU(x + 1/2) - 1/6, V(x) = |x|, lattice 1/16.
    """
    dom = HPolytope.symmetric_box(1, (1, 1))
    origin = ((0,), 1)
    A_plant = AffineMapQ.identity(1)
    B_num = [[1]]
    B_denom = 2

    # Float specification of pi(x) = -x + (1/3) ReLU(x + 1/2) - 1/6
    layer_specs = [
        ([[1.0]], [0.5], True),
        ([[1.0 / 3.0]], [-1.0 / 6.0], False),
    ]
    res_spec = ([[-1.0]], [0.0])

    pi_naive = snap_relu_network_to_lattice(
        layer_specs,
        scale_denom=16,
        anchor_equilibrium=None,
        residual_spec=res_spec,
        name="trap6_pi_naive",
    )
    pi_anchored = snap_relu_network_to_lattice(
        layer_specs,
        scale_denom=16,
        anchor_equilibrium=origin,
        residual_spec=res_spec,
        name="trap6_pi_anchored",
    )
    cl_naive = compose_linear_plant_with_relu_controller(
        A_plant, B_num, B_denom, pi_naive, dom
    )
    cl_anchored = compose_linear_plant_with_relu_controller(
        A_plant, B_num, B_denom, pi_anchored, dom
    )
    v_abs = build_l1_norm_relu(1, name="trap6_v_abs")
    return (pi_naive, pi_anchored, cl_naive, cl_anchored, v_abs, dom)


def build_trap7_composition_shard_blowup() -> Tuple[ReLUNetwork, PWAFunction, HPolytope]:
    """TRAP 7: 2D quadrant-split contraction F(x) = (1/2)x on [-1, 1]^2 composed with
    4-crease octagonal V(x) = |x_1| + |x_2| + (1/2)|x_1 + x_2| + (1/2)|x_1 - x_2|.
    """
    dom = HPolytope.symmetric_box(2, (1, 1))
    half_id = AffineMapQ.create(2, 2, [[1, 0], [0, 1]], [0, 0], 2)
    q_polys = [
        dom.with_halfspace([-1, 0], 0).with_halfspace([0, -1], 0),
        dom.with_halfspace([1, 0], 0).with_halfspace([0, -1], 0),
        dom.with_halfspace([1, 0], 0).with_halfspace([0, 1], 0),
        dom.with_halfspace([-1, 0], 0).with_halfspace([0, 1], 0),
    ]
    plant_pwa = PWAFunction(
        in_dim=2,
        out_dim=2,
        domain=dom,
        cells=tuple(
            PWACell(cell_id=i, polytope=q_polys[i], affine_map=half_id, tag=f"Q{i+1}")
            for i in range(4)
        ),
        name="trap7_quad_contraction",
    )
    normals_8 = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
    ]
    weights_8 = [[2, 2, 2, 2, 1, 1, 1, 1]]
    v_oct = ReLUNetwork(
        in_dim=2,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(2, 8, normals_8, [0] * 8, 1), use_relu=True),
            ReLULayer(AffineMapQ.create(8, 1, weights_8, [0], 2), use_relu=False),
        ),
        name="trap7_octagonal_v",
    )
    return (v_oct, plant_pwa, dom)


def build_trap8_degenerate_origin_cone_3d() -> Tuple[ReLUNetwork, PWAFunction, HPolytope, Tuple[List[List[int]], List[int]]]:
    """TRAP 8: 12-Hyperplane Degenerate Origin Cone in 3D on [-1, 1]^3."""
    dom = HPolytope.symmetric_box(3, (1, 1))
    planes_12 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
    ]
    # Symmetrized positive ReLU sum over all 12 origin hyperplanes using 12 neurons via |z| = 2*ReLU(z) - z:
    # V(x) = sum_{k=1}^{12} |n_k^T x| = 2 * sum_{k=1}^{12} ReLU(n_k^T x) - sum_{k=1}^{12} n_k^T x!
    sum_n = [sum(p[j] for p in planes_12) for j in range(3)]
    v_12_3d = ReLUNetwork(
        in_dim=3,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(3, 12, planes_12, [0] * 12, 1), use_relu=True),
            ReLULayer(AffineMapQ.create(12, 1, [[2] * 12], [0], 1), use_relu=False),
        ),
        residual=AffineMapQ.create(3, 1, [[-v for v in sum_n]], [0], 1),
        name="trap8_12hyperplane_v3d",
    )
    plant_3d = PWAFunction(
        in_dim=3,
        out_dim=3,
        domain=dom,
        cells=(
            PWACell(
                cell_id=0,
                polytope=dom,
                affine_map=AffineMapQ.create(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], 2),
                tag="global_half",
            ),
        ),
        name="trap8_half_3d",
    )
    # Direct 12-hyperplane degenerate origin cone LP (all 12 active at (0,0,0))
    cone_normals = [
        [1, 0, 1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, 1],
        [1, 1, 2],
        [-1, 1, 2],
        [1, -1, 2],
        [-1, -1, 2],
        [2, 1, 3],
        [-2, 1, 3],
        [1, 2, 3],
        [-1, -2, 3],
    ]
    A_cone = cone_normals + [[0, 0, -1]]
    b_cone = [0] * 12 + [1]
    return (v_12_3d, plant_3d, dom, (A_cone, b_cone))


def build_trap9_6d_64neuron_budget_guard() -> Tuple[ReLUNetwork, PWAFunction, HPolytope]:
    """TRAP 9: 6D 64-Neuron XP Arrangement Budget Guard on [-1, 1]^6."""
    dom = HPolytope.symmetric_box(6, (1, 1))
    W_64: List[List[int]] = []
    b_64: List[int] = []
    # Deterministic general-position integer normals in Z^6 with small shifts so 64 hyperplanes split [-1, 1]^6
    primes = [2, 3, 5, 7, 11, 13]
    for k in range(64):
        row = [
            ((k + 1) * primes[j] + (j + 1) * (k % 5 + 1)) % 11 - 5
            for j in range(6)
        ]
        if all(v == 0 for v in row):
            row[k % 6] = 1
        W_64.append(row)
        b_64.append((k % 7) - 3)
    v_64 = ReLUNetwork(
        in_dim=6,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(6, 64, W_64, b_64, 8), use_relu=True),
            ReLULayer(AffineMapQ.create(64, 1, [[1] * 64], [0], 8), use_relu=False),
        ),
        name="trap9_6d_64neuron",
    ).with_anchor_zero(((0, 0, 0, 0, 0, 0), 1))
    plant_6d = PWAFunction(
        in_dim=6,
        out_dim=6,
        domain=dom,
        cells=(
            PWACell(
                cell_id=0,
                polytope=dom,
                affine_map=AffineMapQ.create(
                    6,
                    6,
                    [[1 if i == j else 0 for j in range(6)] for i in range(6)],
                    [0] * 6,
                    2,
                ),
                tag="half_6d",
            ),
        ),
        name="trap9_plant_6d",
    )
    return (v_64, plant_6d, dom)


def run_all_adversarial_traps() -> List[TrapAuditEntry]:
    """Execute and independently verify all 9 Adversarial Control Traps in exact Z/Q."""
    results: List[TrapAuditEntry] = []

    # TRAP 1
    t0 = time.perf_counter()
    v1, p1, dom1 = build_trap1_filippov_uphill_sliding()
    r1_int = verify_continuous_lyapunov(v1, p1, dom1, decay_lambda=(1, 2), check_filippov=False)
    r1_full = verify_continuous_lyapunov(v1, p1, dom1, decay_lambda=(1, 2), check_filippov=True)
    ok1 = (
        r1_int.verdict == Verdict.CERTIFIED_EXACT
        and r1_full.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r1_full.filippov_witness is not None
        and r1_full.filippov_witness.x_star[0][0] == 0
        and r1_full.filippov_witness.dini_deriv[0] > 0
        and check_filippov_witness(
            r1_full.filippov_witness,
            v1.cells,
            {c.cell_id: (c.affine_map,) for c in p1.cells},
        )
    )
    results.append(
        TrapAuditEntry(
            trap_id=1,
            name="Filippov Uphill Sliding Crease",
            passed=ok1,
            verdict=r1_full.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary=f"Caught uphill sliding at x*={r1_full.filippov_witness.x_star if r1_full.filippov_witness else None}",
        )
    )

    # TRAP 2
    t0 = time.perf_counter()
    h2, p2a, p2b, dom2 = build_trap2_bouligand_corner_escape()
    origin2 = ((0, 0), 1)
    heaviside_ok = all(
        eval_heaviside_formal_lie_derivative(h2, origin2, ((1, 0), 1), H) == (0, 1)
        for H in ((0, 1), (1, 2), (1, 1))
    )
    r2a = verify_barrier(h2, p2a, domain=dom2)
    r2b = verify_barrier(h2, p2b, domain=dom2)
    ok2 = (
        heaviside_ok
        and r2a.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r2b.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r2a.counterexample is not None
        and r2b.counterexample is not None
        and r2a.counterexample.x_star == origin2
        and r2b.counterexample.x_star == origin2
        and all(check_corner_deadlock_cert(s) for s in r2a.counterexample.deadlock_subcerts)
        and all(check_corner_deadlock_cert(s) for s in r2b.counterexample.deadlock_subcerts)
    )
    results.append(
        TrapAuditEntry(
            trap_id=2,
            name="Bouligand Corner Escape Defeating Heaviside",
            passed=ok2,
            verdict=r2b.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary="Falsified both autonomous & controlled apex escapes at x*=(0,0)",
        )
    )

    # TRAP 3
    t0 = time.perf_counter()
    net3a, dom3a, net3b, dom3b = build_trap3_hidden_periodic_orbits()
    r3a = detect_periodic_orbits_and_fixed_points(net3a, domain=dom3a, max_period=3)
    r3b = detect_periodic_orbits_and_fixed_points(net3b, domain=dom3b, max_period=3)
    p2_found = any(
        orb.period == 2 and orb.points == (((-4,), 9), ((4,), 9))
        and check_periodic_orbit_witness(orb, net3a)
        for orb in r3a.periodic_orbits
    )
    p3_found = any(
        orb.period == 3
        and orb.points == (((-2, -2), 1), ((2, 0), 1), ((0, 2), 1))
        and orb.stability == "SUPER_STABLE"
        and check_periodic_orbit_witness(orb, net3b)
        for orb in r3b.periodic_orbits
    )
    ok3 = (
        r3a.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r3b.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r3a.unique_origin_fixed_point_rn
        and r3a.origin_locally_exp_stable
        and r3b.unique_origin_fixed_point_rn
        and r3b.origin_locally_exp_stable
        and check_non_fixed_point_certs(r3a.non_fixed_cell_certs)
        and check_non_fixed_point_certs(r3b.non_fixed_cell_certs)
        and p2_found
        and p3_found
    )
    results.append(
        TrapAuditEntry(
            trap_id=3,
            name="Hidden Rational Period-2 & Period-3 Limit Cycles",
            passed=ok3,
            verdict=r3b.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary="Extracted {-4/9, +4/9} (1D) and ((-2,-2), (2,0), (0,2)) (2D)",
        )
    )

    # TRAP 4
    t0 = time.perf_counter()
    b4, p4_sat, p4_unsat, dom4 = build_trap4_deadlocked_concave_corner()
    r4_f1 = verify_barrier(MultiFacetBarrier(2, dom4, (b4.facets[0],)), p4_sat)
    r4_f2 = verify_barrier(MultiFacetBarrier(2, dom4, (b4.facets[1],)), p4_sat)
    r4_unsat = verify_barrier(b4, p4_unsat)
    r4_sat = verify_barrier(b4, p4_sat)
    ok4 = (
        r4_f1.verdict == Verdict.CERTIFIED_EXACT
        and r4_f2.verdict == Verdict.CERTIFIED_EXACT
        and r4_unsat.verdict == Verdict.CERTIFIED_EXACT
        and r4_sat.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and r4_sat.counterexample is not None
        and r4_sat.counterexample.x_star == ((0, 0), 1)
        and len(r4_sat.counterexample.deadlock_subcerts) == 1
        and check_corner_deadlock_cert(r4_sat.counterexample.deadlock_subcerts[0])
        and all(v > 0 for v in r4_sat.counterexample.deadlock_subcerts[0].lambda_tan)
        and any(v > 0 for v in r4_sat.counterexample.deadlock_subcerts[0].eta_sat)
    )
    results.append(
        TrapAuditEntry(
            trap_id=4,
            name="Actuator-Saturated Deadlocked Concave Corner",
            passed=ok4,
            verdict=r4_sat.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary="Emitted Farkas deadlock dual (lambda > 0, eta > 0) at x*=(0,0)",
        )
    )

    # TRAP 5
    t0 = time.perf_counter()
    p_large, p_cube, p_sliver = build_trap5_sub_tolerance_and_sliver()
    v5_l, _, wit5_l, _ = certify_linear_upper_bound(
        p_large.A, p_large.b, [1, 0, 0], 20_000_000, 1, 4
    )
    v5_c, _, wit5_c, _ = certify_linear_upper_bound(
        p_cube.A, p_cube.b, [0, 0, 1], 1, 0, 1
    )
    r_sliv, xc_sliv, lp_sliv = chebyshev_ball_linf(p_sliver.A, p_sliver.b, r_cap=10)
    ok5 = (
        v5_l == "FALSIFIED_COUNTEREXAMPLE"
        and wit5_l is not None
        and check_primal_witness(p_large.A, p_large.b, wit5_l)
        and v5_c == "FALSIFIED_COUNTEREXAMPLE"
        and wit5_c is not None
        and check_primal_witness(p_cube.A, p_cube.b, wit5_c)
        and lp_sliv.status == LPStatus.OPTIMAL
        and r_sliv == (1, 4 * 10**14)
        and xc_sliv is not None
    )
    results.append(
        TrapAuditEntry(
            trap_id=5,
            name="5e-8 Sub-Tolerance Leak & 1e-14 Rational Sliver",
            passed=ok5,
            verdict=Verdict.FALSIFIED_COUNTEREXAMPLE,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary=f"Caught 5e-8 leak & certified sliver r*={r_sliv}",
        )
    )

    # TRAP 6
    t0 = time.perf_counter()
    pi_n, pi_a, cl_n, cl_a, v6, dom6 = build_trap6_rounding_drift_vs_anchor()
    fp_n = detect_periodic_orbits_and_fixed_points(cl_n, domain=dom6, max_period=2, check_global_rn_uniqueness=False)
    r6_n = verify_discrete_lyapunov(v6, cl_n, dom6, epsilon=(1, 4))
    r6_a = verify_discrete_lyapunov(v6, cl_a, dom6, epsilon=(1, 4))
    ok6 = (
        pi_n.eval_qvec(((0,), 1)) == ((-1,), 32)
        and any(qvec_eq(fp.point, ((-1,), 22)) for fp in fp_n.fixed_points)
        and r6_n.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE
        and pi_a.eval_qvec(((0,), 1)) == ((0,), 1)
        and r6_a.verdict == Verdict.CERTIFIED_EXACT
        and check_lyapunov_certificate_bundle(r6_a)
    )
    results.append(
        TrapAuditEntry(
            trap_id=6,
            name="Post-Rounding Equilibrium Drift vs. Anchor Subtraction",
            passed=ok6,
            verdict=r6_a.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary="pi_hat(0)=-1/32 shifted x*=-1/22; anchor restored CERTIFIED_EXACT",
        )
    )

    # TRAP 7
    t0 = time.perf_counter()
    v7, p7, dom7 = build_trap7_composition_shard_blowup()
    rep7 = compose_pullback_pwa(v7, p7, merge_bft=False, count_point_feasible=True)
    r7 = verify_discrete_lyapunov(v7, p7, dom7, epsilon=(1, 4))
    ok7 = (
        rep7.combinatorial_candidates == 1024
        and rep7.point_feasible_shards == 1024
        and rep7.pruned_degenerate_shards == 1016
        and rep7.pre_bft_full_dim_shards == 8
        and r7.verdict == Verdict.CERTIFIED_EXACT
        and check_lyapunov_certificate_bundle(r7)
    )
    results.append(
        TrapAuditEntry(
            trap_id=7,
            name="1,016 / 1,024 Two-Stage Composition Shard Blowup",
            passed=ok7,
            verdict=r7.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary="Pruned 1,016 degenerate 0D/1D shards; certified exact 8 2D cones",
        )
    )

    # TRAP 8
    t0 = time.perf_counter()
    v8, p8, dom8, (A_cone, b_cone) = build_trap8_degenerate_origin_cone_3d()
    lp8_ok = True
    for obj in ([0, 0, 1], [1, 1, 3], [-2, 1, 4]):
        res_c = solve_lp_bareiss(A_cone, b_cone, c=obj, max_pivots=200)
        if res_c.status != LPStatus.OPTIMAL or res_c.obj_val != (0, 1):
            lp8_ok = False
    # Also verify full 12-hyperplane 3D anchored ReLU V(x) on [-1, 1]^3 via Avis lex-positive simplex!
    r8 = verify_discrete_lyapunov(v8, p8, dom8, epsilon=(1, 4), max_cells=512)
    ok8 = lp8_ok and r8.verdict == Verdict.CERTIFIED_EXACT and check_lyapunov_certificate_bundle(r8)
    results.append(
        TrapAuditEntry(
            trap_id=8,
            name="12-Hyperplane Degenerate Origin Cone in 3D",
            passed=ok8,
            verdict=r8.verdict,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            summary=f"Certified {r8.shards_certified} 3D origin cones with zero cycling",
        )
    )

    # TRAP 9
    t0 = time.perf_counter()
    v9, p9, dom9 = build_trap9_6d_64neuron_budget_guard()
    z_bound = zaslavsky_upper_bound(6, 64)
    v9_bab, _, _, z_rep = certify_relu_upper_bound_ct_bab(
        v9, dom9, rhs=(0, 1), max_cells=24
    )
    dt9_ms = (time.perf_counter() - t0) * 1000.0
    ok9 = (
        z_bound == 83_278_001
        and z_rep == z_bound
        and v9_bab == Verdict.OPEN_BUDGET_EXCEEDED
        and dt9_ms < 500.0
    )
    results.append(
        TrapAuditEntry(
            trap_id=9,
            name="6D 64-Neuron XP Arrangement Budget Guard",
            passed=ok9,
            verdict=v9_bab,
            elapsed_ms=dt9_ms,
            summary=f"Zaslavsky={z_bound}; fail-closed OPEN_BUDGET_EXCEEDED in {dt9_ms:.1f}ms",
        )
    )

    return results


# ==============================================================================
# PART 2: EXACT RATIONAL / SECTOR-BOUNDED PHYSICAL BENCHMARKS
# ==============================================================================


def build_inverted_pendulum_sector_2d(
    torque_limit: Optional[Rat] = (3, 2),
) -> Tuple[ReLUNetwork, SectorPWAPlant, HPolytope]:
    """Physical Benchmark 1: 2D Inverted Pendulum x = (theta, omega) on D = [-1, 1] x [-2, 2]
    with certified rational sin(theta) sector wedge [(7/11)*theta, theta] on |theta| <= 1
    and PD feedback u = -3*theta - 2*omega (optionally torque-saturated at |u| <= torque_limit).

    Zero-hole polyhedral Lyapunov candidate: V(theta, omega) = |2*theta + omega| + |theta + omega|.
    """
    dom = HPolytope.from_box([((-1, 1), (1, 1)), ((-2, 1), (2, 1))])
    W_v = [
        [2, 1],
        [-2, -1],
        [1, 1],
        [-1, -1],
    ]
    v_net = ReLUNetwork(
        in_dim=2,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(2, 4, W_v, [0, 0, 0, 0], 1), use_relu=True),
            ReLULayer(AffineMapQ.create(4, 1, [[1, 1, 1, 1]], [0], 1), use_relu=False),
        ),
        name="pendulum_v_polyhedral",
    )

    # Sector generators for sin(theta) in conv{(7/11)*theta, 1*theta}
    # Unsaturated d/dt (theta, omega) = (omega, s*theta - omega + u)
    if torque_limit is None:
        g_hi = AffineMapQ.create(2, 2, [[0, 1], [-2, -3]], [0, 0], 1)
        g_lo = AffineMapQ.create(2, 2, [[0, 11], [-26, -33]], [0, 0], 11)
        sec_plant = SectorPWAPlant(
            dim=2,
            domain=dom,
            cells=(
                SectorPWACell(
                    cell_id=0,
                    polytope=dom,
                    generators=(g_lo, g_hi),
                    tag="global_sector",
                ),
            ),
            is_discrete=False,
            name="inverted_pendulum_sector",
        )
        return (v_net, sec_plant, dom)

    un, ud = q_norm(torque_limit[0], torque_limit[1])
    # Split domain into 3 feedback regions along z = 3*theta + 2*omega:
    # 1) Linear unsaturated: -un/ud <= 3*theta + 2*omega <= un/ud  <=>  +- (3*ud*theta + 2*ud*omega) <= un
    # 2) Positive saturation (z <= -un/ud => u = +un/ud): -(3*ud*theta + 2*ud*omega) >= un  <=>  3*ud*theta + 2*ud*omega <= -un
    # 3) Negative saturation (z >= +un/ud => u = -un/ud): 3*ud*theta + 2*ud*omega >= un  <=>  -3*ud*theta - 2*ud*omega <= -un
    row_z = [3 * ud, 2 * ud]
    neg_z = [-3 * ud, -2 * ud]

    p_lin = dom.with_halfspace(row_z, un).with_halfspace(neg_z, un)
    p_pos_sat = dom.with_halfspace(row_z, -un)
    p_neg_sat = dom.with_halfspace(neg_z, -un)

    cells: List[SectorPWACell] = []
    # Unsaturated cell: u = -3*theta - 2*omega
    g_lin_hi = AffineMapQ.create(2, 2, [[0, 1], [-2, -3]], [0, 0], 1)
    g_lin_lo = AffineMapQ.create(2, 2, [[0, 11], [-26, -33]], [0, 0], 11)
    cells.append(SectorPWACell(0, p_lin, (g_lin_lo, g_lin_hi), "unsaturated"))

    # Positive torque saturation u = +un/ud: d(omega)/dt = s*theta - omega + un/ud
    ok_pos, _, _ = p_pos_sat.is_full_dimensional()
    if ok_pos:
        g_pos_hi = AffineMapQ.create(2, 2, [[0, ud], [ud, -ud]], [0, un], ud)
        g_pos_lo = AffineMapQ.create(2, 2, [[0, 11 * ud], [7 * ud, -11 * ud]], [0, 11 * un], 11 * ud)
        cells.append(SectorPWACell(len(cells), p_pos_sat, (g_pos_lo, g_pos_hi), "sat_plus"))

    # Negative torque saturation u = -un/ud: d(omega)/dt = s*theta - omega - un/ud
    ok_neg, _, _ = p_neg_sat.is_full_dimensional()
    if ok_neg:
        g_neg_hi = AffineMapQ.create(2, 2, [[0, ud], [ud, -ud]], [0, -un], ud)
        g_neg_lo = AffineMapQ.create(2, 2, [[0, 11 * ud], [7 * ud, -11 * ud]], [0, -11 * un], 11 * ud)
        cells.append(SectorPWACell(len(cells), p_neg_sat, (g_neg_lo, g_neg_hi), "sat_minus"))

    sec_plant = SectorPWAPlant(
        dim=2,
        domain=dom,
        cells=tuple(cells),
        is_discrete=False,
        name="inverted_pendulum_torque_saturated",
    )
    return (v_net, sec_plant, dom)


def build_path_following_car_2d() -> Tuple[ReLUNetwork, ControlAffinePWAPlant, HPolytope, Tuple[HPolytope, ...], Tuple[HPolytope, ...]]:
    """Physical Benchmark 2a: 2D Path-Following Car x = (e_y, e_psi) on [-2, 2] x [-1, 1],
    de_y/dt = e_psi, de_psi/dt = u, u in [-2, 2], with ReLU NCBF h(x) = 1 - |e_y + (1/2) e_psi|.
    """
    dom = HPolytope.from_box([((-2, 1), (2, 1)), ((-1, 1), (1, 1))])
    h_net = ReLUNetwork(
        in_dim=2,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(2, 2, [[2, 1], [-2, -1]], [0, 0], 2), use_relu=True),
            ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [1], 1), use_relu=False),
        ),
        name="car_lane_ncbf",
    )
    plant = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=1,
        domain=dom,
        f_map=AffineMapQ.create(2, 2, [[0, 1], [0, 0]], [0, 0], 1),
        B_num=[[0], [1]],
        B_denom=1,
        u_poly=HPolytope.symmetric_box(1, (2, 1)),
        name="path_following_car_2d",
    )
    init_set = (HPolytope.from_box([((-1, 4), (1, 4)), ((-1, 4), (1, 4))]),)
    unsafe_set = (HPolytope.from_box([((7, 4), (2, 1)), ((1, 2), (1, 1))]),)
    return (h_net, plant, dom, init_set, unsafe_set)


def build_dubins_3sector_obstacle_3d() -> Tuple[ReLUNetwork, ControlAffinePWAPlant, HPolytope, Tuple[HPolytope, ...], Tuple[HPolytope, ...]]:
    """Physical Benchmark 2b: 3D 3-Sector Dubins Obstacle Avoidance x = (p_x, p_y, theta)
    on D = [-2, 2] x [-2, 2] x [-1, 1] across 3 heading sectors [-1, -1/3], [-1/3, 1/3], [1/3, 1]
    with ReLU reflex-obstacle NCBF h(x) = p_y - (1/4) ReLU(1 - p_x) - (1/4) ReLU(1 + p_x) + (1/2) theta + 1
    under steering rate saturation u in [-3, 3].
    """
    dom = HPolytope.from_box([((-2, 1), (2, 1)), ((-2, 1), (2, 1)), ((-1, 1), (1, 1))])
    # 3 heading sectors in theta:
    # S0: theta <= -1/3  (3*theta <= -1): v_x = 7/6 + (1/2)*theta, v_y = theta
    # S1: -1/3 <= theta <= 1/3:           v_x = 1,                 v_y = theta
    # S2: theta >= 1/3   (-3*theta <= -1): v_x = 7/6 - (1/2)*theta, v_y = theta
    s0_poly = dom.with_halfspace([0, 0, 3], -1)
    s1_poly = dom.with_halfspace([0, 0, -3], 1).with_halfspace([0, 0, 3], 1)
    s2_poly = dom.with_halfspace([0, 0, -3], -1)

    f_s0 = AffineMapQ.create(3, 3, [[0, 0, 3], [0, 0, 6], [0, 0, 0]], [7, 0, 0], 6)
    f_s1 = AffineMapQ.create(3, 3, [[0, 0, 0], [0, 0, 1], [0, 0, 0]], [1, 0, 0], 1)
    f_s2 = AffineMapQ.create(3, 3, [[0, 0, -3], [0, 0, 6], [0, 0, 0]], [7, 0, 0], 6)

    B_dubins = ((0,), (0,), (1,))
    plant = ControlAffinePWAPlant(
        dim=3,
        control_dim=1,
        domain=dom,
        cells=(
            ControlAffineCell(0, s0_poly, f_s0, B_dubins, 1, "sector_left"),
            ControlAffineCell(1, s1_poly, f_s1, B_dubins, 1, "sector_center"),
            ControlAffineCell(2, s2_poly, f_s2, B_dubins, 1, "sector_right"),
        ),
        u_poly=HPolytope.symmetric_box(1, (3, 1)),
        name="dubins_3sector_3d",
    )

    h_net = ReLUNetwork(
        in_dim=3,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(3, 2, [[-1, 0, 0], [1, 0, 0]], [1, 1], 1), use_relu=True),
            ReLULayer(AffineMapQ.create(2, 1, [[-1, -1]], [4], 4), use_relu=False),
        ),
        residual=AffineMapQ.create(3, 1, [[0, 2, 1]], [0], 2),
        name="dubins_reflex_obstacle_ncbf",
    )
    init_set = (
        HPolytope.from_box([((-2, 1), (-3, 2)), ((1, 2), (1, 1)), ((0, 1), (1, 4))]),
    )
    unsafe_set = (
        HPolytope.from_box([((-1, 2), (1, 2)), ((-2, 1), (-3, 2)), ((-1, 2), (1, 2))]),
    )
    return (h_net, plant, dom, init_set, unsafe_set)


def build_double_integrator_2d_corridor_diamond() -> Tuple[MultiFacetBarrier, ReLUNetwork, ControlAffinePWAPlant, HPolytope]:
    """Physical Benchmark 3a: 2D Double Integrator x = (p, v), dp/dt = v, dv/dt = u, |u| <= 1
    with braking-aware concave-corner corridor AND 2D planar corridor + disconnected diamond obstacle.
    Returns (corridor_2d, diamond_corridor_relu_2d, plant_2d_di, domain_2d_di).
    """
    dom = HPolytope.from_box([((-5, 1), (5, 1)), ((-2, 1), (2, 1))])
    # Braking-aware safe corridor: 2 - p - 2v >= 0, 1 - v >= 0, 2 + p + 2v >= 0, 1 + v >= 0
    facets = (
        AffineMapQ.create(2, 1, [[-1, -2]], [2], 1),
        AffineMapQ.create(2, 1, [[0, -1]], [1], 1),
        AffineMapQ.create(2, 1, [[1, 2]], [2], 1),
        AffineMapQ.create(2, 1, [[0, 1]], [1], 1),
    )
    corridor = MultiFacetBarrier(2, dom, facets, name="di_2d_braking_corridor")
    plant_di = ControlAffinePWAPlant.single_cell(
        dim=2,
        control_dim=1,
        domain=dom,
        f_map=AffineMapQ.create(2, 2, [[0, 1], [0, 0]], [0, 0], 1),
        B_num=[[0], [1]],
        B_denom=1,
        u_poly=HPolytope.symmetric_box(1, (1, 1)),
        name="double_integrator_2d",
    )

    # Disconnected inner exclusion zone in stopping coordinate q = p + 2v: |p + 2v| >= 1/2
    # Combined with speed limit |v| <= 1 via h(p,v) = min(|p + 2v| - 1/2, 1 - |v|) = a - ReLU(a - b)!
    # Layer 0: [q, -q, v, -v] -> ReLU -> Layer 1: [a - b = (|q| - 1/2) - (1 - |v|) = |q| + |v| - 3/2] -> ReLU
    # Output: (|q| - 1/2) - ReLU(|q| + |v| - 3/2) = min(|p + 2v| - 1/2, 1 - |v|)!
    # Let's represent min(|p + 2v| - 1/2, 1 - |v|) cleanly as a 2-hidden-layer ReLUNetwork:
    l0 = ReLULayer(
        AffineMapQ.create(2, 4, [[1, 2], [-1, -2], [0, 1], [0, -1]], [0, 0, 0, 0], 1),
        use_relu=True,
    )
    # Pass |q| = u0 + u1 (>= 0) through neuron 0 of layer 1, and compute u0 + u1 + u2 + u3 - 3/2 in neuron 1!
    l1 = ReLULayer(
        AffineMapQ.create(4, 2, [[2, 2, 0, 0], [2, 2, 2, 2]], [0, -3], 2),
        use_relu=True,
    )
    l2 = ReLULayer(
        AffineMapQ.create(2, 1, [[2, -2]], [-1], 2),
        use_relu=False,
    )
    diamond_relu = ReLUNetwork(
        in_dim=2,
        out_dim=1,
        layers=(l0, l1, l2),
        name="di_2d_obstacle_and_speed_ncbf",
    )
    return (corridor, diamond_relu, plant_di, dom)


def build_double_integrator_4d_corridor_diamond() -> Tuple[
    MultiFacetBarrier,
    ReLUNetwork,
    ControlAffinePWAPlant,
    ControlAffinePWAPlant,
    HPolytope,
    Tuple[HPolytope, ...],
    Tuple[HPolytope, ...],
]:
    """Physical Benchmark 3b: 4D Planar Double Integrator x = (p_x, p_y, v_x, v_y),
    dp/dt = v, dv/dt = u under box saturation ||u||_inf <= 1:
      - corridor_4d: 8-facet braking-aware corridor (concave corridor corners in 4D),
      - diamond_obs_4d: 4D ReLU disconnected stopping-point diamond obstacle
        h_diam(x) = |p_x + 2 v_x| + |p_y + 2 v_y| - 1/2 >= 0 (4 non-convex reflex obstacle corners/ridges).
    """
    dom_full = HPolytope.from_box(
        [((-5, 1), (5, 1)), ((-5, 1), (5, 1)), ((-2, 1), (2, 1)), ((-2, 1), (2, 1))]
    )
    # Velocity-bounded operational domain |v_x| <= 1, |v_y| <= 1 for standalone reflex diamond obstacle audit
    dom_vel1 = HPolytope.from_box(
        [((-4, 1), (4, 1)), ((-4, 1), (4, 1)), ((-1, 1), (1, 1)), ((-1, 1), (1, 1))]
    )
    f_4d = AffineMapQ.create(
        4,
        4,
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [0, 0, 0, 0],
        1,
    )
    B_4d = [[0, 0], [0, 0], [1, 0], [0, 1]]
    u_box = HPolytope.symmetric_box(2, (1, 1))

    plant_corridor = ControlAffinePWAPlant.single_cell(
        dim=4,
        control_dim=2,
        domain=dom_full,
        f_map=f_4d,
        B_num=B_4d,
        B_denom=1,
        u_poly=u_box,
        name="di_4d_corridor_plant",
    )
    plant_diamond = ControlAffinePWAPlant.single_cell(
        dim=4,
        control_dim=2,
        domain=dom_vel1,
        f_map=f_4d,
        B_num=B_4d,
        B_denom=1,
        u_poly=u_box,
        name="di_4d_diamond_plant",
    )

    facets_8 = (
        AffineMapQ.create(4, 1, [[-1, 0, -2, 0]], [2], 1),
        AffineMapQ.create(4, 1, [[0, 0, -1, 0]], [1], 1),
        AffineMapQ.create(4, 1, [[1, 0, 2, 0]], [2], 1),
        AffineMapQ.create(4, 1, [[0, 0, 1, 0]], [1], 1),
        AffineMapQ.create(4, 1, [[0, -1, 0, -2]], [2], 1),
        AffineMapQ.create(4, 1, [[0, 0, 0, -1]], [1], 1),
        AffineMapQ.create(4, 1, [[0, 1, 0, 2]], [2], 1),
        AffineMapQ.create(4, 1, [[0, 0, 0, 1]], [1], 1),
    )
    corridor_4d = MultiFacetBarrier(4, dom_full, facets_8, name="di_4d_corridor")

    # Disconnected 4D diamond obstacle exterior: |p_x + 2 v_x| + |p_y + 2 v_y| - 1/2 >= 0
    W_diam = [
        [1, 0, 2, 0],
        [-1, 0, -2, 0],
        [0, 1, 0, 2],
        [0, -1, 0, -2],
    ]
    diamond_obs_4d = ReLUNetwork(
        in_dim=4,
        out_dim=1,
        layers=(
            ReLULayer(AffineMapQ.create(4, 4, W_diam, [0, 0, 0, 0], 1), use_relu=True),
            ReLULayer(AffineMapQ.create(4, 1, [[2, 2, 2, 2]], [-1], 2), use_relu=False),
        ),
        name="di_4d_diamond_obstacle_ncbf",
    )

    init_set = (
        HPolytope.from_box(
            [
                ((-3, 2), (-1, 1)),
                ((1, 2), (1, 1)),
                ((0, 1), (1, 4)),
                ((-1, 4), (0, 1)),
            ]
        ),
    )
    unsafe_set = (
        HPolytope.from_box(
            [
                ((-1, 16), (1, 16)),
                ((-1, 16), (1, 16)),
                ((-1, 32), (1, 32)),
                ((-1, 32), (1, 32)),
            ]
        ),
    )
    return (
        corridor_4d,
        diamond_obs_4d,
        plant_corridor,
        plant_diamond,
        dom_vel1,
        init_set,
        unsafe_set,
    )


def simulate_shielded_4d_double_integrator(
    steps: int = 12,
    dt: Rat = (1, 10),
) -> List[Tuple[QVec, ShieldStepResult, Rat, Rat]]:
    """Run an online exact-Q CBF-LP trajectory rollout in 4D navigating around the central diamond
    obstacle while remaining strictly inside the outer braking corridor under ||u||_inf <= 1.
    Returns list of (state_x, shield_result, h_corridor, h_diamond).
    """
    (
        corridor_4d,
        diamond_4d,
        plant_corr,
        _,
        _,
        _,
        _,
    ) = build_double_integrator_4d_corridor_diamond()

    # Combine corridor facets + active diamond branch in a joint online safety shield step
    shield_corr = RuntimeCBFShield(corridor_4d, plant_corr, alpha_gain=(1, 1))
    shield_diam = RuntimeCBFShield(diamond_4d, plant_corr, alpha_gain=(2, 1))
    diam_pwa = diamond_4d.enumerate_pwa(plant_corr.domain)

    # Start at x_0 = (-3/2, 1/4, 1/2, 0) inside S_I (h_corr = 1/2 > 0, h_diam = 1/4 > 0),
    # heading directly toward the central diamond obstacle (0, 0)!
    x_cur: QVec = ((-6, 1, 2, 0), 4)
    trace: List[Tuple[QVec, ShieldStepResult, Rat, Rat]] = []
    f_map = plant_corr.cells[0].f_map
    B_map = AffineMapQ.create(2, 4, plant_corr.cells[0].B_num, [0, 0, 0, 0], 1)

    for _ in range(steps):
        # Nominal proportional pursuit controller trying to drive straight through (0, 0) toward (+1, 0, 0, 0)
        xn, xd = x_cur
        # u_nom = (clip(1 - p_x - v_x), clip(-p_y - v_y)) pushing directly into diamond!
        u_nom = qvec_norm([xd - xn[0] - xn[2], -xn[1] - xn[3]], 2 * xd)

        # Pick active valid diamond branch and build combined MultiFacetBarrier with all 8 corridor facets!
        cand_d = [
            hc for hc in diam_pwa.cells
            if hc.polytope.contains(x_cur) and hc.affine_map.eval_scalar(x_cur)[0] >= 0
        ]
        if not cand_d:
            cand_d = [diam_pwa.cells[0]]

        best_step: Optional[ShieldStepResult] = None
        for hc in cand_d:
            joint_b = MultiFacetBarrier(
                4,
                plant_corr.domain,
                corridor_4d.facets + (hc.affine_map,),
                name="joint_4d_corr_diam",
            )
            sh = RuntimeCBFShield(joint_b, plant_corr, alpha_gain=(1, 1))
            sr = sh.step(x_cur, u_nom)
            if sr.verdict == Verdict.CERTIFIED_EXACT:
                if (
                    best_step is None
                    or best_step.verdict != Verdict.CERTIFIED_EXACT
                    or (
                        sr.l1_intervention is not None
                        and best_step.l1_intervention is not None
                        and q_lt(sr.l1_intervention, best_step.l1_intervention)
                    )
                ):
                    best_step = sr
            elif best_step is None:
                best_step = sr

        assert best_step is not None and best_step.u_safe is not None
        h_c = corridor_4d.eval_scalar(x_cur)
        h_d = shield_diam.eval_barrier(x_cur)
        trace.append((x_cur, best_step, h_c, h_d))

        # Exact rational Euler integration x_{t+1} = x_t + dt * (f(x_t) + B u_safe)
        dx = qvec_add(f_map.eval_qvec(x_cur), B_map.eval_qvec(best_step.u_safe))
        x_cur = qvec_add(x_cur, qvec_scale(dt, dx))

    return trace


def build_coupled_platoon_microgrid_4d() -> Tuple[ReLUNetwork, ReLUNetwork, MultiFacetBarrier, ControlAffinePWAPlant, HPolytope]:
    """Physical Benchmark 4: 4D Coupled Platoon / Microgrid Oscillator Damping on D = [-2, 2]^4.
    Certifies BOTH:
      1) Zero-hole l_1 Lyapunov contraction V(x) = ||x||_1 on nonlinear ReLU droop dynamics, AND
      2) Safe inter-vehicle headway / frequency barrier invariance h(x) >= 0 on [-1, 1]^4 subset D.
    """
    dom = HPolytope.symmetric_box(4, (2, 1))
    v_l1_4d = build_l1_norm_relu(4, name="platoon_4d_l1_lyapunov")

    # Continuous-time strictly column-diagonally-dominant 4D coupled platoon / microgrid
    # with nonlinear deadzone bumper damping on x_1: -(1/2) ReLU(x_1 - 1/2) + (1/2) ReLU(-x_1 - 1/2)
    A_lin = AffineMapQ.create(
        4,
        4,
        [
            [-8, 2, 1, 0],
            [2, -12, 0, 1],
            [1, 0, -8, 2],
            [0, 1, 2, -12],
        ],
        [0, 0, 0, 0],
        4,
    )
    platoon_net = ReLUNetwork(
        in_dim=4,
        out_dim=4,
        layers=(
            ReLULayer(
                AffineMapQ.create(4, 2, [[2, 0, 0, 0], [-2, 0, 0, 0]], [-1, -1], 2),
                use_relu=True,
            ),
            ReLULayer(
                AffineMapQ.create(
                    2,
                    4,
                    [[-1, 1], [0, 0], [0, 0], [0, 0]],
                    [0, 0, 0, 0],
                    2,
                ),
                use_relu=False,
            ),
        ),
        residual=A_lin,
        name="coupled_platoon_4d_plant",
    )

    # Safe headway & frequency box C = [-1, 1]^4: on any facet x_i = +-1, diagonal decay -2 or -3
    # strictly dominates coupling (<= 3/4), so autonomous & disturbance-bounded invariance holds!
    headway_facets: List[AffineMapQ] = []
    for j in range(4):
        pos_r = [1 if q == j else 0 for q in range(4)]
        neg_r = [-1 if q == j else 0 for q in range(4)]
        headway_facets.append(AffineMapQ.create(4, 1, [pos_r], [1], 1))
        headway_facets.append(AffineMapQ.create(4, 1, [neg_r], [1], 1))
    headway_barrier = MultiFacetBarrier(
        4, dom, tuple(headway_facets), name="platoon_4d_headway_barrier"
    )
    platoon_pwa = platoon_net.enumerate_pwa(dom)
    platoon_barrier_plant = ControlAffinePWAPlant.autonomous_from_pwa(platoon_pwa)
    return (v_l1_4d, platoon_net, headway_barrier, platoon_barrier_plant, dom)
