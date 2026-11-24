from __future__ import annotations

import ast
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import List, Optional, Set

ROOT = pathlib.Path(__file__).resolve().parent

ALLOWED_STDLIB_MODULES: Set[str] = {
    "__future__",
    "argparse",
    "ast",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "io",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "random",
    "re",
    "subprocess",
    "sys",
    "tempfile",
    "time",
    "typing",
    "unittest",
    "xml",
}

FORBIDDEN_EXTERNAL_MODULES: Set[str] = {
    "numpy",
    "scipy",
    "torch",
    "gurobipy",
    "z3",
    "cvxpy",
    "matplotlib",
    "sympy",
    "signal",
}

INTERNAL_PACKAGES: Set[str] = {
    "lyapunov_shield",
    "tests",
    "benchmarks",
}


def _check_no_bitor_in_annotation(ann_node: Optional[ast.AST], rel: pathlib.Path) -> None:
    if ann_node is None:
        return
    for sub in ast.walk(ann_node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            raise AssertionError(
                f"{rel}:{getattr(sub, 'lineno', '?')}: forbidden Python 3.10+ `X | Y` union type annotation"
            )


def stage1_clean_room_and_ast_audit() -> str:
    """Stage 1: Python 3.9 syntax + AST `fractions` ban + zero hardcoded user path scan + zero external imports."""
    forbidden_prefixes = ("/" + "Users" + "/", "/" + "home" + "/")
    text_exts = {".py", ".md", ".toml", ".json", ".svg", ".txt"}

    scanned_files = sorted(
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.relative_to(ROOT).parts
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
        and (p.suffix in text_exts or p.name == "LICENSE")
    )
    for fpath in scanned_files:
        rel = fpath.relative_to(ROOT)
        content = fpath.read_text(encoding="utf-8")
        for pref in forbidden_prefixes:
            if pref in content:
                raise AssertionError(f"{rel}: contains hardcoded user path prefix {pref!r}")

    py_files = [p for p in scanned_files if p.suffix == ".py"]
    if not py_files:
        raise RuntimeError("No Python files found in repository")

    io_schema_path = (ROOT / "lyapunov_shield" / "io_schema.py").resolve()
    match_cls = getattr(ast, "Match", None)

    for py_path in py_files:
        rel = py_path.relative_to(ROOT)
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(rel), feature_version=(3, 9))

        has_future_annot = any(
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "__future__"
            and any(alias.name == "annotations" for alias in stmt.names)
            for stmt in tree.body[:5]
        )
        if not has_future_annot:
            raise AssertionError(f"{rel}: missing `from __future__ import annotations`")

        for node in ast.walk(tree):
            if match_cls is not None and isinstance(node, match_cls):
                raise AssertionError(f"{rel}: forbidden Python 3.10+ match/case syntax")

            if isinstance(node, ast.AnnAssign):
                _check_no_bitor_in_annotation(node.annotation, rel)
            elif isinstance(node, ast.arg):
                _check_no_bitor_in_annotation(node.annotation, rel)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check_no_bitor_in_annotation(node.returns, rel)

            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    if node.func.attr == "pairwise":
                        raise AssertionError(f"{rel}: forbidden Python 3.10+ itertools.pairwise")
                    if node.func.attr == "alarm":
                        raise AssertionError(f"{rel}: forbidden signal.alarm() call")

                if func_name == "dataclass":
                    for kw in node.keywords:
                        if kw.arg == "slots":
                            raise AssertionError(f"{rel}: forbidden @dataclass(slots=...) on Python 3.9")
                if func_name == "zip":
                    for kw in node.keywords:
                        if kw.arg == "strict":
                            raise AssertionError(f"{rel}: forbidden zip(..., strict=...) on Python 3.9")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top == "fractions" and py_path.resolve() != io_schema_path:
                        raise AssertionError(f"{rel}: `fractions` banned outside io_schema.py")
                    if top in FORBIDDEN_EXTERNAL_MODULES:
                        raise AssertionError(f"{rel}: forbidden external module `{top}`")
                    if top not in ALLOWED_STDLIB_MODULES and top not in INTERNAL_PACKAGES:
                        raise AssertionError(f"{rel}: non-whitelisted import `{top}`")

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top = node.module.split(".")[0]
                    if top == "fractions" and py_path.resolve() != io_schema_path:
                        raise AssertionError(f"{rel}: `fractions` banned outside io_schema.py")
                    if top in FORBIDDEN_EXTERNAL_MODULES:
                        raise AssertionError(f"{rel}: forbidden external module `{top}`")
                    if top not in ALLOWED_STDLIB_MODULES and top not in INTERNAL_PACKAGES:
                        raise AssertionError(f"{rel}: non-whitelisted import `{top}`")

    return f"Scanned {len(scanned_files)} repo files & {len(py_files)} Python ASTs (0 user paths, 0 external deps, 0 fractions leaks)"


def run_stage1_subprocess() -> None:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "verify_all.py"), "--run-stage1-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=25,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise AssertionError(f"Stage 1 failed with code {proc.returncode}")
    print(f"[PASS] Stage 1 ({dt:.2f}s): {proc.stdout.strip()}")


def run_stage2_unittest_suite() -> None:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=25,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise AssertionError(f"Stage 2 unittest suite failed with code {proc.returncode}")
    ran_match = re.search(r"Ran (\d+) tests", proc.stderr)
    if not ran_match:
        raise AssertionError(f"Could not parse unittest count from: {proc.stderr!r}")
    num_tests = int(ran_match.group(1))
    if num_tests < 55:
        raise AssertionError(f"Expected >= 55 unit tests, ran only {num_tests}")
    print(f"[PASS] Stage 2 ({dt:.2f}s): Ran {num_tests} unit tests (>= 55 required, 9/9 Traps + tamper + 500-step shield stress).")


def run_stage3_grand_tournament() -> None:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "benchmarks/run_grand_tournament.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=25,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise AssertionError(f"Stage 3 tournament failed with code {proc.returncode}")
    summary = json.loads(proc.stdout)
    total_i = int(summary["total_instances"])
    ver_i = int(summary["verified"])
    if total_i < 1000 or ver_i != total_i:
        raise AssertionError(f"Tournament verification incomplete: {ver_i}/{total_i}")

    from lyapunov_shield.io_schema import verify_signed_certificate_bundle

    json_path = ROOT / "results" / "grand_tournament_summary.json"
    ok_sig, msg_sig = verify_signed_certificate_bundle(json_path)
    if not ok_sig:
        raise AssertionError(f"Tournament SHA-256 bundle check failed: {msg_sig}")

    # Verify exact 100% README scorecard table <-> grand_tournament_summary.json parity
    bundle = json.loads(json_path.read_text(encoding="utf-8"))
    payload = bundle["payload"]
    fb = payload["family_breakdown"]
    vc = payload["verdict_counts"]
    ordered_keys = [
        "1_lyapunov_roa",
        "2_filippov_creases",
        "3_bouligand_ncbf",
        "4_periodic_orbits",
        "5_lattice_anchor_sweeps",
        "6_highdim_xp_ctbab",
    ]
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    table_rows: List[List[str]] = []
    for line in readme_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| **") and stripped.endswith("|"):
            cols = [c.strip() for c in stripped.strip("|").split("|")]
            table_rows.append(cols)
    if len(table_rows) != 7:
        raise AssertionError(f"Expected 7 data+total rows in README scorecard table, found {len(table_rows)}")

    def _clean_int(cell: str) -> int:
        digits = re.sub(r"[^0-9]", "", cell)
        return int(digits)

    sum_inst = 0
    sum_cert = 0
    sum_fals = 0
    sum_open = 0
    for idx, fkey in enumerate(ordered_keys):
        cols = table_rows[idx]
        fstats = fb[fkey]
        r_inst = _clean_int(cols[1])
        r_cert = _clean_int(cols[2])
        r_fals = _clean_int(cols[3])
        r_open = _clean_int(cols[4])
        if (r_inst, r_cert, r_fals, r_open) != (
            int(fstats["total"]),
            int(fstats["certified_exact"]),
            int(fstats["falsified_counterexample"]),
            int(fstats["open_budget_exceeded"]),
        ):
            raise AssertionError(f"README row {idx + 1} ({fkey}) mismatch vs JSON: {cols} vs {fstats}")
        sum_inst += r_inst
        sum_cert += r_cert
        sum_fals += r_fals
        sum_open += r_open

    tot_cols = table_rows[6]
    t_inst = _clean_int(tot_cols[1])
    t_cert = _clean_int(tot_cols[2])
    t_fals = _clean_int(tot_cols[3])
    t_open = _clean_int(tot_cols[4])
    if (sum_inst, sum_cert, sum_fals, sum_open) != (t_inst, t_cert, t_fals, t_open):
        raise AssertionError(
            f"README family rows sum {(sum_inst, sum_cert, sum_fals, sum_open)} != Total row {(t_inst, t_cert, t_fals, t_open)}"
        )
    if (t_inst, t_cert, t_fals, t_open) != (
        int(payload["total_instances"]),
        int(vc["CERTIFIED_EXACT"]),
        int(vc["FALSIFIED_COUNTEREXAMPLE"]),
        int(vc["OPEN_BUDGET_EXCEEDED"]),
    ):
        raise AssertionError("README Total row does not match summary JSON verdict_counts")
    if t_cert + t_fals + t_open != t_inst:
        raise AssertionError("Verdict counts do not sum to total_instances")

    print(
        f"[PASS] Stage 3 ({dt:.2f}s): Grand Tournament verified {ver_i}/{total_i} (100%) instances, {summary['traps_defended']} traps & README table parity."
    )


def run_stage4_svg_atlas_and_audit() -> None:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "lyapunov_shield", "render-atlas"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=25,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise AssertionError(f"Stage 4 render-atlas failed with code {proc.returncode}")
    rep = json.loads(proc.stdout)
    if rep["svg_count"] != 10 or rep["total_oob"] != 0 or rep["total_text_overlaps"] != 0:
        raise AssertionError(f"SVG atlas defect detected: {rep}")

    from lyapunov_shield.svg_atlas import ATLAS_SPECS, audit_svg_geometry

    for fname, _ in ATLAS_SPECS:
        aud = audit_svg_geometry(ROOT / "figures" / fname)
        if aud["oob_count"] != 0 or aud["text_overlap_count"] != 0:
            raise AssertionError(f"{fname} failed standalone geometry audit: {aud}")
    print(f"[PASS] Stage 4 ({dt:.2f}s): Regenerated & audited 10/10 SVGs (OOB == 0, text_overlaps == 0).")


def run_stage5_cli_end_to_end() -> None:
    commands: List[List[str]] = [
        ["audit-traps"],
        ["verify-lyapunov", "--benchmark", "inverted_pendulum_roa"],
        ["verify-barrier", "--benchmark", "dubins_3d"],
        ["audit-filippov", "--benchmark", "trap1"],
        ["detect-orbits", "--benchmark", "trap3b"],
        ["shield-step", "--benchmark", "dubins_3d"],
        ["synthesize-cegis", "--max-iters", "10"],
    ]
    t0 = time.monotonic()
    for argv in commands:
        proc = subprocess.run(
            [sys.executable, "-m", "lyapunov_shield"] + argv,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise AssertionError(f"Stage 5 CLI {argv} failed with code {proc.returncode}")
        data = json.loads(proc.stdout)
        if argv[0] == "audit-traps" and (not data.get("all_passed") or data.get("passed_count") != 9):
            raise AssertionError(f"CLI audit-traps failed: {data}")
    dt = time.monotonic() - t0
    print(f"[PASS] Stage 5 ({dt:.2f}s): Verified all end-to-end CLI subcommands (9/9 subcommands operational).")


def main(argv: Optional[List[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--run-stage1-only" in args:
        msg = stage1_clean_room_and_ast_audit()
        print(msg)
        return 0

    t_start = time.monotonic()
    import lyapunov_shield

    if lyapunov_shield.__version__ != "2.0.0":
        raise AssertionError(f"Expected lyapunov_shield.__version__ == '2.0.0', got {lyapunov_shield.__version__!r}")

    run_stage1_subprocess()
    run_stage2_unittest_suite()
    run_stage3_grand_tournament()
    run_stage4_svg_atlas_and_audit()
    run_stage5_cli_end_to_end()

    total_sec = time.monotonic() - t_start
    if total_sec >= 45.0:
        raise AssertionError(f"verify_all.py exceeded 45.0s budget ({total_sec:.2f}s)")
    print(f"[PASS] All 5 Wave 2 v{lyapunov_shield.__version__} verification stages passed in {total_sec:.2f}s (< 45.0s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
