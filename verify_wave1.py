from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys
import time
from typing import List, Set

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


def run_ast_firewall() -> None:
    py_files = sorted(
        p
        for p in ROOT.rglob("*.py")
        if "__pycache__" not in p.relative_to(ROOT).parts
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
    )
    if not py_files:
        raise RuntimeError("No Python files found in repository")

    io_schema_path = (ROOT / "lyapunov_shield" / "io_schema.py").resolve()

    for py_path in py_files:
        rel = py_path.relative_to(ROOT)
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(rel))

        # 1. Check `from __future__ import annotations` at top of file
        has_future_annot = False
        for stmt in tree.body[:5]:
            if (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module == "__future__"
                and any(alias.name == "annotations" for alias in stmt.names)
            ):
                has_future_annot = True
                break
        if not has_future_annot:
            raise AssertionError(f"{rel}: missing `from __future__ import annotations`")

        # 2. Walk AST for forbidden Python 3.10+ features, `signal.alarm`, and `fractions` firewall
        match_cls = getattr(ast, "Match", None)
        for node in ast.walk(tree):
            if match_cls is not None and isinstance(node, match_cls):
                raise AssertionError(f"{rel}: forbidden Python 3.10+ match/case syntax")

            if isinstance(node, ast.Call):
                # Forbid @dataclass(slots=True) and zip(..., strict=True)
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
                        raise AssertionError(
                            f"{rel}: AST fractions firewall violation (`fractions` only allowed in io_schema.py)"
                        )
                    if top in FORBIDDEN_EXTERNAL_MODULES:
                        raise AssertionError(f"{rel}: forbidden module import `{top}`")
                    if top not in ALLOWED_STDLIB_MODULES and top not in ("lyapunov_shield", "tests", "benchmarks"):
                        raise AssertionError(f"{rel}: non-whitelisted import `{top}`")

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top = node.module.split(".")[0]
                    if top == "fractions" and py_path.resolve() != io_schema_path:
                        raise AssertionError(
                            f"{rel}: AST fractions firewall violation (`fractions` only allowed in io_schema.py)"
                        )
                    if top in FORBIDDEN_EXTERNAL_MODULES:
                        raise AssertionError(f"{rel}: forbidden module import `{top}`")
                    if top not in ALLOWED_STDLIB_MODULES and top not in ("lyapunov_shield", "tests", "benchmarks"):
                        raise AssertionError(f"{rel}: non-whitelisted import `{top}`")

    print(f"[PASS] AST Python 3.9 & `fractions` firewall verified across {len(py_files)} files.")


def run_unittest_suite() -> None:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=20,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise AssertionError(f"Unittest suite failed with exit code {proc.returncode}")
    summary_line = [ln for ln in proc.stderr.splitlines() if ln.startswith("Ran ")]
    info = summary_line[-1] if summary_line else "unittest OK"
    print(f"[PASS] Unit test suite: {info} ({dt:.2f}s).")


def run_cli_smoke_checks() -> None:
    commands: List[List[str]] = [
        ["audit-traps"],
        ["verify-lyapunov", "--benchmark", "inverted_pendulum"],
        ["verify-barrier", "--benchmark", "dubins_3d"],
        ["audit-filippov", "--benchmark", "trap1"],
        ["detect-orbits", "--benchmark", "trap3b"],
        ["shield-step", "--benchmark", "path_car_2d"],
        ["synthesize-cegis", "--max-iters", "10"],
    ]
    t0 = time.monotonic()
    for argv in commands:
        proc = subprocess.run(
            [sys.executable, "-m", "lyapunov_shield"] + argv,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise AssertionError(f"CLI command {argv} failed with code {proc.returncode}")
        payload = json.loads(proc.stdout)
        if argv[0] == "audit-traps":
            if not payload.get("all_passed") or payload.get("passed_count") != 9:
                raise AssertionError(f"audit-traps did not pass 9/9: {payload}")
    dt = time.monotonic() - t0
    print(f"[PASS] All {len(commands)} CLI subcommands verified ({dt:.2f}s).")


def main() -> int:
    start = time.monotonic()
    import lyapunov_shield

    if lyapunov_shield.__version__ not in ("1.0.0", "2.0.0"):
        raise AssertionError(f"Expected __version__ in ('1.0.0', '2.0.0'), got {lyapunov_shield.__version__!r}")

    run_ast_firewall()
    run_unittest_suite()
    run_cli_smoke_checks()

    total_sec = time.monotonic() - start
    if total_sec >= 25.0:
        raise AssertionError(f"verify_wave1.py exceeded 25s budget ({total_sec:.2f}s)")
    print(f"[PASS] Wave 1 v{lyapunov_shield.__version__} verification completed in {total_sec:.2f}s (< 25.0s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
