from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence

from .barrier_verifier import ControlAffinePWAPlant, verify_barrier
from .benchmarks_catalog import (
    build_coupled_platoon_microgrid_4d,
    build_double_integrator_2d_corridor_diamond,
    build_double_integrator_4d_corridor_diamond,
    build_dubins_3sector_obstacle_3d,
    build_inverted_pendulum_sector_2d,
    build_path_following_car_2d,
    build_trap1_filippov_uphill_sliding,
    build_trap2_bouligand_corner_escape,
    build_trap3_hidden_periodic_orbits,
    build_trap4_deadlocked_concave_corner,
    build_trap6_rounding_drift_vs_anchor,
    build_trap7_composition_shard_blowup,
    build_trap8_degenerate_origin_cone_3d,
    run_all_adversarial_traps,
    simulate_shielded_4d_double_integrator,
)
from .filippov_crease import audit_filippov_creases
from .io_schema import (
    control_affine_plant_from_dict,
    export_signed_certificate_bundle,
    hpolytope_from_dict,
    load_function_or_barrier_dict,
    load_plant_dict,
    parse_qvec,
    parse_rat,
    serialize_obj,
)
from .lattice_cegis import synthesize_lyapunov_cegis
from .lyapunov_verifier import (
    SectorPWAPlant,
    verify_continuous_lyapunov,
    verify_discrete_lyapunov,
)
from .orbit_detector import detect_periodic_orbits_and_fixed_points
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
)
from .runtime_shield import RuntimeCBFShield


def _read_json(path_str: str) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path_str).read_text(encoding="utf-8"))


def _cmd_verify_lyapunov(args: argparse.Namespace) -> int:
    if args.input:
        spec = _read_json(args.input)
        dom = hpolytope_from_dict(spec["domain"])
        x_eq = parse_qvec(spec["x_eq"]) if "x_eq" in spec else None
        scale_d = int(spec["scale_denom"]) if "scale_denom" in spec else None
        v_fn = load_function_or_barrier_dict(
            spec["lyapunov"], scale_denom=scale_d, anchor_equilibrium=x_eq
        )
        assert isinstance(v_fn, (ReLUNetwork, PWAFunction))
        plant = load_plant_dict(spec["plant"])
        assert isinstance(plant, (ReLUNetwork, PWAFunction, SectorPWAPlant))
        is_disc = bool(spec.get("is_discrete", not args.continuous))
        rate = parse_rat(spec.get("rate", args.rate))
        roa = bool(spec.get("compute_maximal_roa", args.roa))
        if is_disc:
            res = verify_discrete_lyapunov(
                v_fn, plant, dom, x_eq=x_eq, epsilon=rate, compute_maximal_roa=roa
            )
            kind = "lyapunov_discrete"
        else:
            res = verify_continuous_lyapunov(
                v_fn, plant, dom, x_eq=x_eq, decay_lambda=rate, compute_maximal_roa=roa
            )
            kind = "lyapunov_continuous"
    else:
        bench = args.benchmark or "inverted_pendulum"
        rate = parse_rat(args.rate)
        if bench == "inverted_pendulum":
            v_fn, plant, dom = build_inverted_pendulum_sector_2d(torque_limit=None)
            res = verify_continuous_lyapunov(v_fn, plant, dom, decay_lambda=rate)
            kind = "lyapunov_continuous"
        elif bench == "inverted_pendulum_roa":
            v_fn, plant, dom = build_inverted_pendulum_sector_2d(torque_limit=(3, 2))
            res = verify_continuous_lyapunov(
                v_fn, plant, dom, decay_lambda=rate, compute_maximal_roa=True
            )
            kind = "lyapunov_continuous"
        elif bench == "platoon_4d":
            v_fn, p_net, _, _, dom = build_coupled_platoon_microgrid_4d()
            res = verify_continuous_lyapunov(v_fn, p_net, dom, decay_lambda=rate)
            kind = "lyapunov_continuous"
        elif bench == "trap1":
            v1, p1, d1 = build_trap1_filippov_uphill_sliding()
            res = verify_continuous_lyapunov(v1, p1, d1, decay_lambda=rate)
            kind = "lyapunov_continuous"
        elif bench == "trap6":
            _, _, _, cl_a, v6, d6 = build_trap6_rounding_drift_vs_anchor()
            res = verify_discrete_lyapunov(v6, cl_a, d6, epsilon=rate)
            kind = "lyapunov_discrete"
        elif bench == "trap7":
            v7, p7, d7 = build_trap7_composition_shard_blowup()
            res = verify_discrete_lyapunov(v7, p7, d7, epsilon=rate)
            kind = "lyapunov_discrete"
        elif bench == "trap8":
            v8, p8, d8, _ = build_trap8_degenerate_origin_cone_3d()
            res = verify_discrete_lyapunov(v8, p8, d8, epsilon=rate)
            kind = "lyapunov_discrete"
        else:
            raise ValueError(f"Unknown lyapunov benchmark: {bench}")

    bundle = export_signed_certificate_bundle(res, kind=kind, output_path=args.output)
    summary = {
        "command": "verify-lyapunov",
        "verdict": res.verdict.value,
        "is_discrete": res.is_discrete,
        "alpha_1": res.alpha_1,
        "epsilon": res.epsilon,
        "rho_star": res.rho_star,
        "shards_certified": res.shards_certified,
        "sha256": bundle["sha256"],
        "message": res.message,
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_verify_barrier(args: argparse.Namespace) -> int:
    if args.input:
        spec = _read_json(args.input)
        dom = hpolytope_from_dict(spec["domain"]) if "domain" in spec else None
        barrier = load_function_or_barrier_dict(spec["barrier"])
        plant = control_affine_plant_from_dict(spec["plant"])
        init_s = tuple(hpolytope_from_dict(p) for p in spec.get("initial_set", ()))
        uns_s = tuple(hpolytope_from_dict(p) for p in spec.get("unsafe_set", ()))
        res = verify_barrier(
            barrier, plant, domain=dom, initial_set=init_s, unsafe_set=uns_s
        )
    else:
        bench = args.benchmark or "dubins_3d"
        if bench == "path_car_2d":
            b, p, d, i_s, u_s = build_path_following_car_2d()
            res = verify_barrier(b, p, domain=d, initial_set=i_s, unsafe_set=u_s)
        elif bench == "dubins_3d":
            b, p, d, i_s, u_s = build_dubins_3sector_obstacle_3d()
            res = verify_barrier(b, p, domain=d, initial_set=i_s, unsafe_set=u_s)
        elif bench == "di_2d_corridor":
            c2, _, p2, d2 = build_double_integrator_2d_corridor_diamond()
            res = verify_barrier(c2, p2, domain=d2)
        elif bench == "di_2d_diamond":
            _, dr2, p2, d2 = build_double_integrator_2d_corridor_diamond()
            res = verify_barrier(dr2, p2, domain=d2)
        elif bench == "di_4d_corridor":
            c4, _, pc4, _, _, _, _ = build_double_integrator_4d_corridor_diamond()
            res = verify_barrier(c4, pc4)
        elif bench == "di_4d_diamond":
            _, do4, _, pd4, dv4, i4, u4 = build_double_integrator_4d_corridor_diamond()
            res = verify_barrier(do4, pd4, domain=dv4, initial_set=i4, unsafe_set=u4)
        elif bench == "platoon_4d_headway":
            _, _, hb4, pbp4, d4 = build_coupled_platoon_microgrid_4d()
            res = verify_barrier(hb4, pbp4, domain=d4)
        elif bench == "trap2":
            h2, _, p2b, d2 = build_trap2_bouligand_corner_escape()
            res = verify_barrier(h2, p2b, domain=d2)
        elif bench == "trap4":
            b4, p4_sat, _, _ = build_trap4_deadlocked_concave_corner()
            res = verify_barrier(b4, p4_sat)
        else:
            raise ValueError(f"Unknown barrier benchmark: {bench}")

    bundle = export_signed_certificate_bundle(res, kind="barrier", output_path=args.output)
    summary = {
        "command": "verify-barrier",
        "verdict": res.verdict.value,
        "faces_certified": res.faces_certified,
        "corners_checked": res.corners_checked,
        "sha256": bundle["sha256"],
        "message": res.message,
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_audit_filippov(args: argparse.Namespace) -> int:
    eps_s = parse_rat(args.eps_slide)
    if args.input:
        spec = _read_json(args.input)
        dom = hpolytope_from_dict(spec["domain"])
        v_raw = load_function_or_barrier_dict(spec["lyapunov"])
        assert isinstance(v_raw, (ReLUNetwork, PWAFunction))
        v_pwa = v_raw.enumerate_pwa(dom) if isinstance(v_raw, ReLUNetwork) else v_raw
        p_raw = load_plant_dict(spec["plant"])
        if isinstance(p_raw, ReLUNetwork):
            p_pwa = p_raw.enumerate_pwa(dom)
            sec = SectorPWAPlant.from_pwa(p_pwa, is_discrete=False)
        elif isinstance(p_raw, PWAFunction):
            sec = SectorPWAPlant.from_pwa(p_raw, is_discrete=False)
        elif isinstance(p_raw, SectorPWAPlant):
            sec = p_raw
        else:
            raise ValueError("Unsupported plant type for audit-filippov")
        # Build joint partition
        joint_cells: List[PWACell] = []
        f_map_dict: Dict[int, Sequence[AffineMapQ]] = {}
        for vc in v_pwa.cells:
            for pc in sec.cells:
                inter = vc.polytope.intersect(pc.polytope)
                ok_fd, _, _ = inter.is_full_dimensional()
                if ok_fd:
                    cid = len(joint_cells)
                    joint_cells.append(
                        PWACell(cid, inter.prune_redundant(), vc.affine_map, f"v{vc.cell_id}_p{pc.cell_id}")
                    )
                    f_map_dict[cid] = pc.generators
        joint_pwa = PWAFunction(dom.dim, 1, dom, tuple(joint_cells), "joint_filippov")
        res = audit_filippov_creases(joint_pwa, f_map_dict, eps_slide=eps_s)
    else:
        bench = args.benchmark or "trap1"
        if bench == "trap1":
            v1, p1, _ = build_trap1_filippov_uphill_sliding()
            res = audit_filippov_creases(
                v1, {c.cell_id: (c.affine_map,) for c in p1.cells}, eps_slide=eps_s
            )
        elif bench == "inverted_pendulum":
            vp, sp, dp = build_inverted_pendulum_sector_2d(torque_limit=None)
            vp_pwa = vp.enumerate_pwa(dp)
            res = audit_filippov_creases(
                vp_pwa, {c.cell_id: sp.cells[0].generators for c in vp_pwa.cells}, eps_slide=eps_s
            )
        else:
            raise ValueError(f"Unknown filippov benchmark: {bench}")

    bundle = export_signed_certificate_bundle(res, kind="filippov", output_path=args.output)
    summary = {
        "command": "audit-filippov",
        "verdict": res.verdict.value,
        "creases_audited": res.creases_audited,
        "codim1_creases": res.codim1_creases,
        "codim_k_creases": res.codim_k_creases,
        "counterexample": res.counterexample,
        "sha256": bundle["sha256"],
        "message": res.message,
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_detect_orbits(args: argparse.Namespace) -> int:
    if args.input:
        spec = _read_json(args.input)
        dom = hpolytope_from_dict(spec["domain"]) if "domain" in spec else None
        plant = load_plant_dict(spec["plant"])
        assert isinstance(plant, (ReLUNetwork, PWAFunction))
        res = detect_periodic_orbits_and_fixed_points(
            plant, domain=dom, max_period=args.max_period
        )
    else:
        bench = args.benchmark or "trap3b"
        net3a, dom3a, net3b, dom3b = build_trap3_hidden_periodic_orbits()
        if bench == "trap3a":
            res = detect_periodic_orbits_and_fixed_points(
                net3a, domain=dom3a, max_period=args.max_period
            )
        elif bench == "trap3b":
            res = detect_periodic_orbits_and_fixed_points(
                net3b, domain=dom3b, max_period=args.max_period
            )
        elif bench == "trap6_naive":
            _, _, cl_n, _, _, dom6 = build_trap6_rounding_drift_vs_anchor()
            res = detect_periodic_orbits_and_fixed_points(
                cl_n, domain=dom6, max_period=args.max_period, check_global_rn_uniqueness=False
            )
        else:
            raise ValueError(f"Unknown orbit benchmark: {bench}")

    bundle = export_signed_certificate_bundle(res, kind="orbits", output_path=args.output)
    summary = {
        "command": "detect-orbits",
        "verdict": res.verdict.value,
        "num_cells": res.num_cells,
        "fixed_points": len(res.fixed_points),
        "periodic_orbits": res.periodic_orbits,
        "unique_origin_fixed_point_rn": res.unique_origin_fixed_point_rn,
        "origin_locally_exp_stable": res.origin_locally_exp_stable,
        "sha256": bundle["sha256"],
        "message": res.message,
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_shield_step(args: argparse.Namespace) -> int:
    alpha_q = parse_rat(args.alpha)
    if args.input:
        spec = _read_json(args.input)
        barrier = load_function_or_barrier_dict(spec["barrier"])
        plant = control_affine_plant_from_dict(spec["plant"])
        x_q = parse_qvec(json.loads(args.state) if args.state else spec["state"])
        u_nom = parse_qvec(json.loads(args.u_nominal) if args.u_nominal else spec["u_nominal"])
        shield = RuntimeCBFShield(barrier, plant, alpha_gain=alpha_q)
        res = shield.step(x_q, u_nom)
    else:
        bench = args.benchmark or "dubins_3d"
        if bench == "path_car_2d":
            b, p, _, _, _ = build_path_following_car_2d()
            shield = RuntimeCBFShield(b, p, alpha_gain=alpha_q)
            x_q = parse_qvec(json.loads(args.state)) if args.state else ((3, 1), 4)
            u_nom = parse_qvec(json.loads(args.u_nominal)) if args.u_nominal else ((2,), 1)
            res = shield.step(x_q, u_nom)
        elif bench == "dubins_3d":
            b, p, _, _, _ = build_dubins_3sector_obstacle_3d()
            shield = RuntimeCBFShield(b, p, alpha_gain=alpha_q)
            x_q = parse_qvec(json.loads(args.state)) if args.state else ((0, -1, 0), 2)
            u_nom = parse_qvec(json.loads(args.u_nominal)) if args.u_nominal else ((-3,), 1)
            res = shield.step(x_q, u_nom)
        elif bench == "di_4d_rollout":
            trace = simulate_shielded_4d_double_integrator(steps=6)
            res = trace[0][1]
        else:
            raise ValueError(f"Unknown shield benchmark: {bench}")

    bundle = export_signed_certificate_bundle(res, kind="shield", output_path=args.output)
    summary = {
        "command": "shield-step",
        "verdict": res.verdict.value,
        "u_nominal": res.u_nominal,
        "u_safe": res.u_safe,
        "intervened": res.intervened,
        "l1_intervention": res.l1_intervention,
        "solve_time_us": res.solve_time_us,
        "sha256": bundle["sha256"],
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_synthesize_cegis(args: argparse.Namespace) -> int:
    rate_q = parse_rat(args.rate)
    if args.input:
        spec = _read_json(args.input)
        dom = hpolytope_from_dict(spec["domain"])
        plant = load_plant_dict(spec["plant"])
        assert isinstance(plant, (ReLUNetwork, PWAFunction, SectorPWAPlant))
        is_disc = bool(spec.get("is_discrete", not args.continuous))
        res = synthesize_lyapunov_cegis(
            plant,
            dom,
            is_discrete=is_disc,
            rate_target=rate_q,
            lattice_denom=args.lattice_denom,
            max_iters=args.max_iters,
            timeout_sec=args.timeout,
        )
    else:
        dom = HPolytope.symmetric_box(2, (1, 1))
        plant = ReLUNetwork(
            in_dim=2,
            out_dim=2,
            layers=(
                ReLULayer(AffineMapQ.create(2, 2, [[2, 3], [0, 2]], [0, 0], 4), False),
            ),
            name="coupled_non_normal_2d",
        )
        res = synthesize_lyapunov_cegis(
            plant,
            dom,
            is_discrete=True,
            rate_target=rate_q,
            lattice_denom=args.lattice_denom,
            max_iters=args.max_iters,
            timeout_sec=args.timeout,
        )

    export_payload = res.verification if res.verification is not None else res
    export_kind = "lyapunov_discrete" if (res.verification is not None and res.verification.is_discrete) else "cegis"
    bundle = export_signed_certificate_bundle(export_payload, kind=export_kind, output_path=args.output)
    summary = {
        "command": "synthesize-cegis",
        "verdict": res.verdict.value,
        "iterations": res.iterations,
        "witness_count": res.witness_count,
        "weights_q": res.weights_q,
        "elapsed_sec": res.elapsed_sec,
        "sha256": bundle["sha256"],
        "message": res.message,
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_audit_traps(args: argparse.Namespace) -> int:
    entries = run_all_adversarial_traps()
    all_passed = all(e.passed for e in entries)
    bundle = export_signed_certificate_bundle(
        entries, kind="adversarial_traps_audit", output_path=args.output
    )
    summary = {
        "command": "audit-traps",
        "all_passed": all_passed,
        "passed_count": sum(1 for e in entries if e.passed),
        "total_traps": len(entries),
        "traps": entries,
        "sha256": bundle["sha256"],
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0 if all_passed else 1


def _cmd_tournament(args: argparse.Namespace) -> int:
    from benchmarks.run_grand_tournament import run_grand_tournament

    out_path = (
        pathlib.Path(args.output)
        if args.output
        else pathlib.Path(__file__).resolve().parent.parent / "results" / "grand_tournament_summary.json"
    )
    bundle = run_grand_tournament(seed=args.seed, output_path=out_path)
    pl = bundle["payload"]
    summary = {
        "command": "tournament",
        "total_instances": pl["total_instances"],
        "verified": pl["certificates_and_witnesses_verified"],
        "verification_rate": pl["verification_rate"],
        "verdict_counts": pl["verdict_counts"],
        "traps_defended": f"{pl['traps_defended']['passed']}/{pl['traps_defended']['total']}",
        "elapsed_sec": pl["elapsed_sec"],
        "output": str(out_path.name),
        "sha256": bundle["sha256"],
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0


def _cmd_render_atlas(args: argparse.Namespace) -> int:
    from .svg_atlas import render_svg_atlas

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else None
    summary_p = pathlib.Path(args.tournament_summary) if args.tournament_summary else None
    report = render_svg_atlas(output_dir=out_dir, tournament_summary_path=summary_p)
    summary = {
        "command": "render-atlas",
        "svg_count": report["svg_count"],
        "total_oob": report["total_oob"],
        "total_text_overlaps": report["total_text_overlaps"],
        "files": [
            {
                "filename": f["filename"],
                "elements": f["elements"],
                "texts": f["texts"],
                "oob_count": f["oob_count"],
                "text_overlap_count": f["text_overlap_count"],
            }
            for f in report["files"]
        ],
    }
    print(json.dumps(serialize_obj(summary), indent=2))
    return 0 if (report["total_oob"] == 0 and report["total_text_overlaps"] == 0) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyapunov-shield",
        description="Exact-Z/Q Neural Lyapunov, Bouligand Barrier & Filippov Safety Shield (v2.0.0)",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_lyap = sub.add_parser("verify-lyapunov", help="Verify discrete or continuous Lyapunov & ROA over Z/Q")
    p_lyap.add_argument("--input", type=str, default=None, help="Input JSON system bundle")
    p_lyap.add_argument("--benchmark", type=str, default=None, help="Named benchmark or trap")
    p_lyap.add_argument("--continuous", action="store_true", help="Force continuous-time verification")
    p_lyap.add_argument("--roa", action="store_true", help="Compute maximal exact-Q sublevel ROA")
    p_lyap.add_argument("--rate", type=str, default="1/4", help="Rational decay rate")
    p_lyap.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_lyap.set_defaults(func=_cmd_verify_lyapunov)

    p_bar = sub.add_parser("verify-barrier", help="Verify Nagumo-Bouligand NCBF & multi-facet safety")
    p_bar.add_argument("--input", type=str, default=None, help="Input JSON barrier bundle")
    p_bar.add_argument("--benchmark", type=str, default=None, help="Named barrier benchmark or trap")
    p_bar.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_bar.set_defaults(func=_cmd_verify_barrier)

    p_fil = sub.add_parser("audit-filippov", help="Audit codim-1 & codim-k Filippov sliding creases")
    p_fil.add_argument("--input", type=str, default=None, help="Input JSON system bundle")
    p_fil.add_argument("--benchmark", type=str, default=None, help="Named Filippov benchmark")
    p_fil.add_argument("--eps-slide", type=str, default="1/4", help="Rational sliding decay rate")
    p_fil.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_fil.set_defaults(func=_cmd_audit_filippov)

    p_orb = sub.add_parser("detect-orbits", help="Detect fixed points & exact rational Period-k limit cycles")
    p_orb.add_argument("--input", type=str, default=None, help="Input JSON discrete system bundle")
    p_orb.add_argument("--benchmark", type=str, default=None, help="Named orbit benchmark")
    p_orb.add_argument("--max-period", type=int, default=3, help="Max cycle period k")
    p_orb.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_orb.set_defaults(func=_cmd_detect_orbits)

    p_sh = sub.add_parser("shield-step", help="Execute online exact-Q Runtime CBF-LP safety filter step")
    p_sh.add_argument("--input", type=str, default=None, help="Input JSON barrier+plant bundle")
    p_sh.add_argument("--benchmark", type=str, default=None, help="Named shield benchmark")
    p_sh.add_argument("--state", type=str, default=None, help="JSON rational state vector")
    p_sh.add_argument("--u-nominal", type=str, default=None, help="JSON rational nominal control vector")
    p_sh.add_argument("--alpha", type=str, default="1", help="Rational CBF class-K gain alpha")
    p_sh.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_sh.set_defaults(func=_cmd_shield_step)

    p_ceg = sub.add_parser("synthesize-cegis", help="Run bounded exact-Q anchor-safe CEGIS Lyapunov learner")
    p_ceg.add_argument("--input", type=str, default=None, help="Input JSON plant bundle")
    p_ceg.add_argument("--benchmark", type=str, default=None, help="Named CEGIS benchmark")
    p_ceg.add_argument("--continuous", action="store_true", help="Synthesize continuous-time V")
    p_ceg.add_argument("--rate", type=str, default="1/10", help="Target rational decay rate")
    p_ceg.add_argument("--lattice-denom", type=int, default=16, help="Weight lattice denominator S")
    p_ceg.add_argument("--max-iters", type=int, default=20, help="Max CEGIS rounds (<= 25)")
    p_ceg.add_argument("--timeout", type=float, default=2.5, help="Monotonic wall-clock cap (<= 3.0s)")
    p_ceg.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_ceg.set_defaults(func=_cmd_synthesize_cegis)

    p_tr = sub.add_parser("audit-traps", help="Run and verify all 9/9 Adversarial Control Traps")
    p_tr.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON bundle path")
    p_tr.set_defaults(func=_cmd_audit_traps)

    p_tour = sub.add_parser("tournament", help="Run 1,000+ instance Grand Control Verification Tournament")
    p_tour.add_argument("--seed", type=int, default=20261123, help="Deterministic tournament seed")
    p_tour.add_argument("--output", type=str, default=None, help="Output SHA-256 signed JSON summary path")
    p_tour.set_defaults(func=_cmd_tournament)

    p_atl = sub.add_parser("render-atlas", help="Render 10 publication-grade SVGs in figures/ with zero-defect audit")
    p_atl.add_argument("--output-dir", type=str, default=None, help="Target SVG directory (default: figures/)")
    p_atl.add_argument("--tournament-summary", type=str, default=None, help="Signed tournament summary JSON path")
    p_atl.set_defaults(func=_cmd_render_atlas)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
