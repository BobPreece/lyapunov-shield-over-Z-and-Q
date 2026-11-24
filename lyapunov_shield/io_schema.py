from __future__ import annotations

import dataclasses
import enum
from fractions import Fraction
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .bareiss_simplex import FarkasBoundCert, FarkasInfeasibilityCert
from .barrier_verifier import (
    BarrierCounterexample,
    BarrierVerificationResult,
    BarrierViabilityFaceCert,
    ControlAffineCell,
    ControlAffinePWAPlant,
    CornerDeadlockSubCert,
    MultiFacetBarrier,
    check_barrier_certificate_bundle,
    check_corner_deadlock_cert,
)
from .filippov_crease import (
    FilippovAuditResult,
    FilippovCreaseCert,
    FilippovSlidingWitness,
    check_filippov_crease_cert,
)
from .lyapunov_verifier import (
    BoundaryLevelCert,
    CoercivityCellCert,
    LyapunovCounterexample,
    LyapunovVerificationResult,
    SectorPWACell,
    SectorPWAPlant,
    ShardDecayCert,
    check_lyapunov_certificate_bundle,
)
from .orbit_detector import (
    NonFixedPointCellCert,
    check_non_fixed_point_certs,
)
from .polyhedral_complex import (
    AffineMapQ,
    HPolytope,
    PWACell,
    PWAFunction,
    ReLULayer,
    ReLUNetwork,
)
from .runtime_shield import ShieldStepResult, check_shield_step_result
from .z_linalg import QVec, Rat, Verdict, lcm_list, q_norm, qvec_from_rats, qvec_norm


SCHEMA_VERSION = "1.0.0"


def parse_rat(val: Any) -> Rat:
    """Parse int, float, rational string ('-4/9', '0.25'), [num, den], or {'num':..,'den':..} into exact Rat."""
    if isinstance(val, bool):
        raise TypeError("Boolean is not a valid rational scalar")
    if isinstance(val, int):
        return (int(val), 1)
    if isinstance(val, float):
        fr = Fraction(str(val))
        return q_norm(fr.numerator, fr.denominator)
    if isinstance(val, str):
        fr = Fraction(val.strip())
        return q_norm(fr.numerator, fr.denominator)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return q_norm(int(val[0]), int(val[1]))
    if isinstance(val, dict) and "num" in val and "den" in val:
        return q_norm(int(val["num"]), int(val["den"]))
    raise ValueError(f"Cannot parse exact rational from {val!r}")


def parse_qvec(val: Any) -> QVec:
    """Parse QVec from {'nums': [...], 'den': d}, [[n0, n1, ...], d], or list of rational scalars."""
    if isinstance(val, dict) and "nums" in val and "den" in val:
        return qvec_norm([int(x) for x in val["nums"]], int(val["den"]))
    if (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[0], (list, tuple))
        and isinstance(val[1], int)
        and not isinstance(val[1], bool)
        and all(isinstance(v, int) and not isinstance(v, bool) for v in val[0])
    ):
        return qvec_norm([int(v) for v in val[0]], int(val[1]))
    if isinstance(val, (list, tuple)):
        return qvec_from_rats([parse_rat(x) for x in val])
    raise ValueError(f"Cannot parse exact QVec from {val!r}")


def hpolytope_to_dict(poly: HPolytope) -> Dict[str, Any]:
    return {
        "dim": poly.dim,
        "A": [list(row) for row in poly.A],
        "b": list(poly.b),
    }


def hpolytope_from_dict(data: Dict[str, Any]) -> HPolytope:
    if "box" in data:
        bounds = [(parse_rat(lo), parse_rat(hi)) for lo, hi in data["box"]]
        return HPolytope.from_box(bounds)
    return HPolytope.create(
        int(data["dim"]),
        [[int(v) for v in row] for row in data["A"]],
        [int(v) for v in data["b"]],
    )


def affine_map_to_dict(amap: AffineMapQ) -> Dict[str, Any]:
    return {
        "in_dim": amap.in_dim,
        "out_dim": amap.out_dim,
        "W": [list(row) for row in amap.W],
        "c": list(amap.c),
        "denom": amap.denom,
    }


def affine_map_from_dict(data: Dict[str, Any]) -> AffineMapQ:
    if "denom" in data:
        return AffineMapQ.create(
            int(data["in_dim"]),
            int(data["out_dim"]),
            [[int(v) for v in row] for row in data["W"]],
            [int(v) for v in data["c"]],
            int(data["denom"]),
        )
    # Parse general rational matrix W_q and vector c_q via Fraction
    W_rats = [[parse_rat(v) for v in row] for row in data["W"]]
    c_rats = [parse_rat(v) for v in data["c"]]
    out_dim = len(W_rats)
    in_dim = len(W_rats[0]) if out_dim > 0 else int(data.get("in_dim", 0))
    denoms = [r[1] for row in W_rats for r in row] + [r[1] for r in c_rats]
    cd = lcm_list(denoms) if denoms else 1
    W_int = [[r[0] * (cd // r[1]) for r in row] for row in W_rats]
    c_int = [r[0] * (cd // r[1]) for r in c_rats]
    return AffineMapQ.create(in_dim, out_dim, W_int, c_int, cd)


def pwa_to_dict(pwa: PWAFunction) -> Dict[str, Any]:
    return {
        "type": "PWAFunction",
        "in_dim": pwa.in_dim,
        "out_dim": pwa.out_dim,
        "domain": hpolytope_to_dict(pwa.domain),
        "cells": [
            {
                "cell_id": c.cell_id,
                "polytope": hpolytope_to_dict(c.polytope),
                "affine_map": affine_map_to_dict(c.affine_map),
                "tag": c.tag,
            }
            for c in pwa.cells
        ],
        "name": pwa.name,
    }


def pwa_from_dict(data: Dict[str, Any]) -> PWAFunction:
    dom = hpolytope_from_dict(data["domain"])
    cells = tuple(
        PWACell(
            cell_id=int(c["cell_id"]),
            polytope=hpolytope_from_dict(c["polytope"]),
            affine_map=affine_map_from_dict(c["affine_map"]),
            tag=str(c.get("tag", "")),
        )
        for c in data["cells"]
    )
    return PWAFunction(
        in_dim=int(data["in_dim"]),
        out_dim=int(data["out_dim"]),
        domain=dom,
        cells=cells,
        name=str(data.get("name", "")),
    )


def relu_net_to_dict(net: ReLUNetwork) -> Dict[str, Any]:
    return {
        "type": "ReLUNetwork",
        "in_dim": net.in_dim,
        "out_dim": net.out_dim,
        "layers": [
            {
                "affine": affine_map_to_dict(layer.affine),
                "use_relu": layer.use_relu,
            }
            for layer in net.layers
        ],
        "residual": affine_map_to_dict(net.residual) if net.residual is not None else None,
        "post_offset": [list(net.post_offset[0]), net.post_offset[1]]
        if net.post_offset is not None
        else None,
        "name": net.name,
    }


def relu_net_from_dict(data: Dict[str, Any]) -> ReLUNetwork:
    layers = tuple(
        ReLULayer(
            affine=affine_map_from_dict(ld["affine"]),
            use_relu=bool(ld.get("use_relu", True)),
        )
        for ld in data["layers"]
    )
    res = (
        affine_map_from_dict(data["residual"])
        if data.get("residual") is not None
        else None
    )
    post = (
        parse_qvec(data["post_offset"])
        if data.get("post_offset") is not None
        else None
    )
    return ReLUNetwork(
        in_dim=int(data["in_dim"]),
        out_dim=int(data["out_dim"]),
        layers=layers,
        residual=res,
        post_offset=post,
        name=str(data.get("name", "")),
    )


def load_pytorch_state_dict_json(
    state_dict: Dict[str, Any],
    scale_denom: Optional[int] = None,
    anchor_equilibrium: Optional[QVec] = None,
    name: str = "pytorch_state_dict_net",
) -> ReLUNetwork:
    """Load a PyTorch-styleSequential MLP state_dict JSON (e.g., {'0.weight': ..., '0.bias': ..., '2.weight': ...})
    into an exact ReLUNetwork (either exact Fraction conversion or snapped to Z/scale_denom + equilibrium anchor).
    """
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    if not weight_keys:
        raise ValueError("No '*.weight' tensors found in state_dict")

    def _layer_sort_key(k: str) -> Tuple[int, str]:
        prefix = k[: -len(".weight")]
        parts = prefix.split(".")
        for p in reversed(parts):
            if p.isdigit():
                return (int(p), prefix)
        return (0, prefix)

    weight_keys.sort(key=_layer_sort_key)
    layers: List[ReLULayer] = []
    num_w = len(weight_keys)

    for idx, wk in enumerate(weight_keys):
        prefix = wk[: -len(".weight")]
        bk = prefix + ".bias"
        W_raw = state_dict[wk]
        out_dim = len(W_raw)
        in_dim = len(W_raw[0]) if out_dim > 0 else 0
        b_raw = state_dict.get(bk, [0] * out_dim)

        if scale_denom is not None:
            if scale_denom <= 0:
                raise ValueError("scale_denom must be positive")
            W_int = [
                [int(round(float(Fraction(str(v))) * scale_denom)) for v in row]
                for row in W_raw
            ]
            b_int = [int(round(float(Fraction(str(v))) * scale_denom)) for v in b_raw]
            amap = AffineMapQ.create(in_dim, out_dim, W_int, b_int, scale_denom)
        else:
            amap = affine_map_from_dict(
                {"in_dim": in_dim, "out_dim": out_dim, "W": W_raw, "c": b_raw}
            )
        use_relu = idx < (num_w - 1)
        layers.append(ReLULayer(affine=amap, use_relu=use_relu))

    net = ReLUNetwork(
        in_dim=layers[0].affine.in_dim,
        out_dim=layers[-1].affine.out_dim,
        layers=tuple(layers),
        name=name,
    )
    if anchor_equilibrium is not None:
        net = net.with_anchor_zero(anchor_equilibrium)
    return net


def sector_plant_to_dict(plant: SectorPWAPlant) -> Dict[str, Any]:
    return {
        "type": "SectorPWAPlant",
        "dim": plant.dim,
        "domain": hpolytope_to_dict(plant.domain),
        "is_discrete": plant.is_discrete,
        "cells": [
            {
                "cell_id": c.cell_id,
                "polytope": hpolytope_to_dict(c.polytope),
                "generators": [affine_map_to_dict(g) for g in c.generators],
                "tag": c.tag,
            }
            for c in plant.cells
        ],
        "name": plant.name,
    }


def sector_plant_from_dict(data: Dict[str, Any]) -> SectorPWAPlant:
    dom = hpolytope_from_dict(data["domain"])
    cells = tuple(
        SectorPWACell(
            cell_id=int(c["cell_id"]),
            polytope=hpolytope_from_dict(c["polytope"]),
            generators=tuple(affine_map_from_dict(g) for g in c["generators"]),
            tag=str(c.get("tag", "")),
        )
        for c in data["cells"]
    )
    return SectorPWAPlant(
        dim=int(data["dim"]),
        domain=dom,
        cells=cells,
        is_discrete=bool(data.get("is_discrete", True)),
        name=str(data.get("name", "")),
    )


def control_affine_plant_to_dict(plant: ControlAffinePWAPlant) -> Dict[str, Any]:
    return {
        "type": "ControlAffinePWAPlant",
        "dim": plant.dim,
        "control_dim": plant.control_dim,
        "domain": hpolytope_to_dict(plant.domain),
        "cells": [
            {
                "cell_id": c.cell_id,
                "polytope": hpolytope_to_dict(c.polytope),
                "f_map": affine_map_to_dict(c.f_map),
                "B_num": [list(r) for r in c.B_num],
                "B_denom": c.B_denom,
                "tag": c.tag,
            }
            for c in plant.cells
        ],
        "u_poly": hpolytope_to_dict(plant.u_poly) if plant.u_poly is not None else None,
        "name": plant.name,
    }


def control_affine_plant_from_dict(data: Dict[str, Any]) -> ControlAffinePWAPlant:
    dom = hpolytope_from_dict(data["domain"])
    u_poly = (
        hpolytope_from_dict(data["u_poly"])
        if data.get("u_poly") is not None
        else None
    )
    cells = tuple(
        ControlAffineCell(
            cell_id=int(c["cell_id"]),
            polytope=hpolytope_from_dict(c["polytope"]),
            f_map=affine_map_from_dict(c["f_map"]),
            B_num=tuple(tuple(int(v) for v in r) for r in c.get("B_num", ())),
            B_denom=int(c.get("B_denom", 1)),
            tag=str(c.get("tag", "")),
        )
        for c in data["cells"]
    )
    return ControlAffinePWAPlant(
        dim=int(data["dim"]),
        control_dim=int(data["control_dim"]),
        domain=dom,
        cells=cells,
        u_poly=u_poly,
        name=str(data.get("name", "")),
    )


def multi_facet_barrier_to_dict(barrier: MultiFacetBarrier) -> Dict[str, Any]:
    return {
        "type": "MultiFacetBarrier",
        "dim": barrier.dim,
        "domain": hpolytope_to_dict(barrier.domain),
        "facets": [affine_map_to_dict(f) for f in barrier.facets],
        "name": barrier.name,
    }


def multi_facet_barrier_from_dict(data: Dict[str, Any]) -> MultiFacetBarrier:
    return MultiFacetBarrier(
        dim=int(data["dim"]),
        domain=hpolytope_from_dict(data["domain"]),
        facets=tuple(affine_map_from_dict(f) for f in data["facets"]),
        name=str(data.get("name", "multi_facet_barrier")),
    )


def load_function_or_barrier_dict(
    data: Dict[str, Any],
    scale_denom: Optional[int] = None,
    anchor_equilibrium: Optional[QVec] = None,
) -> Union[ReLUNetwork, PWAFunction, MultiFacetBarrier]:
    tp = data.get("type", "")
    if tp == "PWAFunction":
        return pwa_from_dict(data)
    if tp == "MultiFacetBarrier":
        return multi_facet_barrier_from_dict(data)
    if tp == "PyTorchStateDict" or "state_dict" in data:
        sd = data["state_dict"] if "state_dict" in data else data
        return load_pytorch_state_dict_json(
            sd, scale_denom=scale_denom, anchor_equilibrium=anchor_equilibrium
        )
    return relu_net_from_dict(data)


def load_plant_dict(
    data: Dict[str, Any],
) -> Union[ReLUNetwork, PWAFunction, SectorPWAPlant, ControlAffinePWAPlant]:
    tp = data.get("type", "")
    if tp == "PWAFunction":
        return pwa_from_dict(data)
    if tp == "SectorPWAPlant":
        return sector_plant_from_dict(data)
    if tp == "ControlAffinePWAPlant":
        return control_affine_plant_from_dict(data)
    if tp == "PyTorchStateDict" or "state_dict" in data:
        sd = data["state_dict"] if "state_dict" in data else data
        return load_pytorch_state_dict_json(sd)
    return relu_net_from_dict(data)


def serialize_obj(obj: Any) -> Any:
    """Recursively convert dataclasses, Enums, tuples, and lists into deterministic JSON primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            out[f.name] = serialize_obj(getattr(obj, f.name))
        return out
    if isinstance(obj, (list, tuple)):
        return [serialize_obj(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_obj(v) for k, v in obj.items()}
    raise TypeError(f"Unsupported type in serialize_obj: {type(obj)!r}")


def compute_bundle_sha256(envelope: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 digest over {'schema_version', 'kind', 'metadata', 'payload'}."""
    canonical_bytes = json.dumps(
        {
            "schema_version": envelope["schema_version"],
            "kind": envelope["kind"],
            "metadata": envelope.get("metadata", {}),
            "payload": envelope["payload"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def export_signed_certificate_bundle(
    result: Any,
    kind: str,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize any verification result, compute SHA-256 cryptographic signature, and optionally save JSON."""
    envelope: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "metadata": serialize_obj(metadata or {}),
        "payload": serialize_obj(result),
    }
    envelope["sha256"] = compute_bundle_sha256(envelope)
    if output_path is not None:
        p = pathlib.Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return envelope


# ==============================================================================
# ZERO-TRUST CERTIFICATE DESERIALIZERS & BUNDLE VERIFICATION
# ==============================================================================


def _farkas_infeas_from_dict(d: Optional[Dict[str, Any]]) -> Optional[FarkasInfeasibilityCert]:
    if d is None:
        return None
    return FarkasInfeasibilityCert(
        y=tuple(int(v) for v in d["y"]),
        rhs_dot=int(d["rhs_dot"]),
    )


def _farkas_bound_from_dict(d: Optional[Dict[str, Any]]) -> Optional[FarkasBoundCert]:
    if d is None:
        return None
    return FarkasBoundCert(
        y=tuple(int(v) for v in d["y"]),
        scale=int(d["scale"]),
        c_num=tuple(int(v) for v in d["c_num"]),
        c_den=int(d["c_den"]),
        rhs_num=int(d["rhs_num"]),
        rhs_den=int(d["rhs_den"]),
        is_vacuous=bool(d.get("is_vacuous", False)),
    )


def _filippov_cert_from_dict(d: Dict[str, Any]) -> FilippovCreaseCert:
    return FilippovCreaseCert(
        face_id=int(d["face_id"]),
        codim=int(d["codim"]),
        cell_ids=tuple(int(v) for v in d["cell_ids"]),
        normals=tuple(tuple(int(v) for v in r) for r in d["normals"]),
        rhss=tuple(int(v) for v in d["rhss"]),
        gamma_kink=parse_rat(d["gamma_kink"]) if d.get("gamma_kink") is not None else None,
        mode=str(d["mode"]),
        slide_poly_A=tuple(tuple(int(v) for v in r) for r in d["slide_poly_A"]),
        slide_poly_b=tuple(int(v) for v in d["slide_poly_b"]),
        infeas_cert=_farkas_infeas_from_dict(d.get("infeas_cert")),
        eq_only_certs=tuple(_farkas_bound_from_dict(c) for c in d.get("eq_only_certs", ()) if c is not None),
        continuity_certs=tuple(_farkas_bound_from_dict(c) for c in d.get("continuity_certs", ()) if c is not None),
        diff_map=affine_map_from_dict(d["diff_map"]) if d.get("diff_map") is not None else None,
        tau_int=tuple(int(v) for v in d.get("tau_int", ())),
        y_per_gen=tuple(tuple(int(v) for v in r) for r in d.get("y_per_gen", ())),
        scale=int(d.get("scale", 1)),
        gen_h_rows=tuple(tuple(int(v) for v in r) for r in d.get("gen_h_rows", ())),
        gen_kappas=tuple(int(v) for v in d.get("gen_kappas", ())),
        gen_Q_mats=tuple(
            tuple(tuple(int(v) for v in row) for row in mat)
            for mat in d.get("gen_Q_mats", ())
        ),
        gen_s_vecs=tuple(tuple(int(v) for v in r) for r in d.get("gen_s_vecs", ())),
    )


def _filippov_audit_from_dict(d: Optional[Dict[str, Any]]) -> Optional[FilippovAuditResult]:
    if d is None:
        return None
    wit = None
    if d.get("counterexample") is not None:
        wd = d["counterexample"]
        wit = FilippovSlidingWitness(
            face_id=int(wd["face_id"]),
            codim=int(wd["codim"]),
            cell_ids=tuple(int(v) for v in wd["cell_ids"]),
            normals=tuple(tuple(int(v) for v in r) for r in wd["normals"]),
            rhss=tuple(int(v) for v in wd["rhss"]),
            gamma_kink=parse_rat(wd["gamma_kink"]) if wd.get("gamma_kink") is not None else None,
            x_star=parse_qvec(wd["x_star"]),
            alpha_weights=tuple(parse_rat(a) for a in wd["alpha_weights"]),
            f_sliding=parse_qvec(wd["f_sliding"]),
            dini_deriv=parse_rat(wd["dini_deriv"]),
            v_at_x=parse_rat(wd["v_at_x"]),
            nu_normals=tuple(parse_rat(nu) for nu in wd["nu_normals"]),
        )
    return FilippovAuditResult(
        verdict=Verdict(d["verdict"]),
        creases_audited=int(d["creases_audited"]),
        codim1_creases=int(d["codim1_creases"]),
        codim_k_creases=int(d["codim_k_creases"]),
        certificates=tuple(_filippov_cert_from_dict(c) for c in d.get("certificates", ())),
        counterexample=wit,
        message=str(d.get("message", "")),
    )


def lyapunov_result_from_dict(d: Dict[str, Any]) -> LyapunovVerificationResult:
    coerc = tuple(
        CoercivityCellCert(
            cell_id=int(c["cell_id"]),
            cell_A=tuple(tuple(int(v) for v in r) for r in c["cell_A"]),
            cell_b=tuple(int(v) for v in c["cell_b"]),
            v_map=affine_map_from_dict(c["v_map"]),
            mu_bounds=tuple(parse_rat(m) for m in c["mu_bounds"]),
            alpha_cell=parse_rat(c["alpha_cell"]),
            bound_certs=tuple(_farkas_bound_from_dict(bc) for bc in c["bound_certs"] if bc is not None),
            strict_pos_cert=_farkas_bound_from_dict(c.get("strict_pos_cert")),
        )
        for c in d.get("coercivity_certs", ())
    )
    decay = tuple(
        ShardDecayCert(
            shard_id=int(dc["shard_id"]),
            poly_A=tuple(tuple(int(v) for v in r) for r in dc["poly_A"]),
            poly_b=tuple(int(v) for v in dc["poly_b"]),
            bound_cert=_farkas_bound_from_dict(dc["bound_cert"]),  # type: ignore[arg-type]
            tag=str(dc.get("tag", "")),
            is_ct_bab_lifted=bool(dc.get("is_ct_bab_lifted", False)),
        )
        for dc in d.get("decay_certs", ())
    )
    bdry = tuple(
        BoundaryLevelCert(
            cell_id=int(bc["cell_id"]),
            boundary_idx=int(bc["boundary_idx"]),
            facet_A=tuple(tuple(int(v) for v in r) for r in bc["facet_A"]),
            facet_b=tuple(int(v) for v in bc["facet_b"]),
            bound_cert=_farkas_bound_from_dict(bc["bound_cert"]),  # type: ignore[arg-type]
        )
        for bc in d.get("boundary_certs", ())
    )
    cex = None
    if d.get("counterexample") is not None:
        cd = d["counterexample"]
        cex = LyapunovCounterexample(
            reason=str(cd["reason"]),
            x_witness=parse_qvec(cd["x_witness"]),
            v_at_x=parse_rat(cd["v_at_x"]),
            v_next_or_deriv=parse_rat(cd["v_next_or_deriv"]),
            residual_violation=parse_rat(cd["residual_violation"]),
        )
    f_aud = _filippov_audit_from_dict(d.get("filippov_audit"))
    return LyapunovVerificationResult(
        verdict=Verdict(d["verdict"]),
        is_discrete=bool(d["is_discrete"]),
        alpha_1=parse_rat(d["alpha_1"]) if d.get("alpha_1") is not None else None,
        epsilon=parse_rat(d["epsilon"]) if d.get("epsilon") is not None else None,
        rho_star=parse_rat(d["rho_star"]) if d.get("rho_star") is not None else None,
        shards_certified=int(d.get("shards_certified", 0)),
        ct_bab_pruned_cells=int(d.get("ct_bab_pruned_cells", 0)),
        coercivity_certs=coerc,
        decay_certs=decay,
        boundary_certs=bdry,
        filippov_audit=f_aud,
        counterexample=cex,
        filippov_witness=f_aud.counterexample if f_aud is not None else None,
        message=str(d.get("message", "")),
    )


def barrier_result_from_dict(d: Dict[str, Any]) -> BarrierVerificationResult:
    viab = tuple(
        BarrierViabilityFaceCert(
            face_id=int(vc["face_id"]),
            valid_cell_id=int(vc["valid_cell_id"]),
            face_A=tuple(tuple(int(v) for v in r) for r in vc["face_A"]),
            face_b=tuple(int(v) for v in vc["face_b"]),
            K_num=tuple(tuple(int(v) for v in r) for r in vc.get("K_num", ())),
            u0_num=tuple(int(v) for v in vc.get("u0_num", ())),
            u_scale=int(vc.get("u_scale", 1)),
            tangent_certs=tuple(_farkas_bound_from_dict(tc) for tc in vc["tangent_certs"] if tc is not None),
            sat_certs=tuple(_farkas_bound_from_dict(sc) for sc in vc.get("sat_certs", ()) if sc is not None),
        )
        for vc in d.get("viability_certs", ())
    )
    cex = None
    if d.get("counterexample") is not None:
        cd = d["counterexample"]
        subs = tuple(
            CornerDeadlockSubCert(
                valid_cell_id=int(s["valid_cell_id"]),
                M_act=tuple(tuple(int(v) for v in r) for r in s["M_act"]),
                control_lp_A=tuple(tuple(int(v) for v in r) for r in s["control_lp_A"]),
                control_lp_b=tuple(int(v) for v in s["control_lp_b"]),
                lambda_tan=tuple(int(v) for v in s["lambda_tan"]),
                eta_sat=tuple(int(v) for v in s["eta_sat"]),
                rhs_dot=int(s["rhs_dot"]),
            )
            for s in cd.get("deadlock_subcerts", ())
        )
        cex = BarrierCounterexample(
            reason=str(cd["reason"]),
            x_star=parse_qvec(cd["x_star"]),
            h_at_x=parse_rat(cd["h_at_x"]),
            deadlock_subcerts=subs,
        )
    return BarrierVerificationResult(
        verdict=Verdict(d["verdict"]),
        faces_certified=int(d.get("faces_certified", 0)),
        corners_checked=int(d.get("corners_checked", 0)),
        init_certs=tuple(_farkas_bound_from_dict(ic) for ic in d.get("init_certs", ()) if ic is not None),
        init_polys=tuple(
            (
                tuple(tuple(int(v) for v in r) for r in p[0]),
                tuple(int(v) for v in p[1]),
            )
            for p in d.get("init_polys", ())
        ),
        unsafe_certs=tuple(_farkas_infeas_from_dict(uc) for uc in d.get("unsafe_certs", ()) if uc is not None),
        unsafe_polys=tuple(
            (
                tuple(tuple(int(v) for v in r) for r in p[0]),
                tuple(int(v) for v in p[1]),
            )
            for p in d.get("unsafe_polys", ())
        ),
        viability_certs=viab,
        counterexample=cex,
        message=str(d.get("message", "")),
    )


def shield_result_from_dict(d: Dict[str, Any]) -> ShieldStepResult:
    return ShieldStepResult(
        verdict=Verdict(d["verdict"]),
        u_safe=parse_qvec(d["u_safe"]) if d.get("u_safe") is not None else None,
        u_nominal=parse_qvec(d["u_nominal"]),
        l1_intervention=parse_rat(d["l1_intervention"]) if d.get("l1_intervention") is not None else None,
        intervened=bool(d["intervened"]),
        active_barrier_indices=tuple(int(v) for v in d.get("active_barrier_indices", ())),
        lp_A=tuple(tuple(int(v) for v in r) for r in d.get("lp_A", ())),
        lp_b=tuple(int(v) for v in d.get("lp_b", ())),
        opt_cert=_farkas_bound_from_dict(d.get("opt_cert")),
        deadlock_cert=_farkas_infeas_from_dict(d.get("deadlock_cert")),
        pivots=int(d.get("pivots", 0)),
        solve_time_us=float(d.get("solve_time_us", 0.0)),
    )


def verify_signed_certificate_bundle(
    path_or_bundle: Union[str, pathlib.Path, Dict[str, Any]],
) -> Tuple[bool, str]:
    """Verify both SHA-256 digest integrity and standalone pure-int Farkas / witness proofs."""
    if isinstance(path_or_bundle, (str, pathlib.Path)):
        raw = json.loads(pathlib.Path(path_or_bundle).read_text(encoding="utf-8"))
    else:
        raw = path_or_bundle

    if "sha256" not in raw or "payload" not in raw or "kind" not in raw:
        return (False, "Missing required envelope fields (kind, payload, sha256)")

    expected_hash = compute_bundle_sha256(raw)
    if raw["sha256"] != expected_hash:
        return (False, "SHA-256 signature mismatch (tampered bundle)")

    kind = raw["kind"]
    payload = raw["payload"]
    meta = raw.get("metadata", {})
    x_eq = parse_qvec(meta["x_eq"]) if "x_eq" in meta and meta["x_eq"] is not None else None

    if kind in ("lyapunov", "lyapunov_discrete", "lyapunov_continuous"):
        lres = lyapunov_result_from_dict(payload)
        if lres.verdict == Verdict.CERTIFIED_EXACT:
            ok = check_lyapunov_certificate_bundle(lres, x_eq=x_eq)
            return (ok, "Verified exact Lyapunov Farkas bundle" if ok else "Lyapunov Farkas check failed")
        return (True, f"Signed Lyapunov bundle ({lres.verdict.value})")

    if kind == "barrier":
        bres = barrier_result_from_dict(payload)
        if bres.verdict == Verdict.CERTIFIED_EXACT:
            ok = check_barrier_certificate_bundle(bres)
            return (ok, "Verified exact Barrier Farkas bundle" if ok else "Barrier Farkas check failed")
        if bres.verdict == Verdict.FALSIFIED_COUNTEREXAMPLE and bres.counterexample is not None:
            if bres.counterexample.deadlock_subcerts:
                ok_d = all(check_corner_deadlock_cert(s) for s in bres.counterexample.deadlock_subcerts)
                return (ok_d, "Verified deadlock Farkas duals" if ok_d else "Deadlock dual check failed")
        return (True, f"Signed Barrier bundle ({bres.verdict.value})")

    if kind == "filippov":
        fres = _filippov_audit_from_dict(payload)
        if fres is None:
            return (False, "Invalid Filippov payload")
        if fres.verdict == Verdict.CERTIFIED_EXACT:
            n_dim = len(fres.certificates[0].slide_poly_A[0]) if fres.certificates else 1
            eq_use = x_eq if x_eq is not None else (tuple(0 for _ in range(n_dim)), 1)
            ok = all(check_filippov_crease_cert(fc, x_eq=eq_use) for fc in fres.certificates)
            return (ok, "Verified Filippov crease certificates" if ok else "Filippov crease check failed")
        return (True, f"Signed Filippov bundle ({fres.verdict.value})")

    if kind == "shield":
        sres = shield_result_from_dict(payload)
        ok = check_shield_step_result(sres)
        return (ok, "Verified Runtime CBF-LP Shield step" if ok else "Shield step check failed")

    if kind == "orbits":
        nf_raw = payload.get("non_fixed_cell_certs", ())
        nf_certs = [
            NonFixedPointCellCert(
                cell_id=int(c["cell_id"]),
                aug_A=tuple(tuple(int(v) for v in r) for r in c["aug_A"]),
                aug_b=tuple(int(v) for v in c["aug_b"]),
                infeas_cert=_farkas_infeas_from_dict(c["infeas_cert"]),  # type: ignore[arg-type]
            )
            for c in nf_raw
        ]
        ok = check_non_fixed_point_certs(nf_certs)
        return (ok, "Verified Orbit non-fixed-point certificates" if ok else "Orbit certificate check failed")

    return (True, f"Verified SHA-256 signature for kind={kind}")
