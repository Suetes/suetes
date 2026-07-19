"""Run the SUETES dynamical-core verification ladder in isolated processes.

This is an orchestration entry point, not a replacement for the individual
diagnostic scripts.  Each stage runs in a fresh process so XLA releases native
memory on exit, and its complete output is retained under
``output/verification``.

Quick gate (``--profile quick``):
  1. Equation-level manufactured tendency convergence.
  2. Pure temporal convergence of both time integrators.
  3. Combined space-time and cross-core rising-bubble convergence.

Full gate (``--profile full``) additionally runs:
  4. Forced dual-core MMS integration.
  5. SISL component/operator tests.
  6. Conservation, balance, terrain, solver, and autodiff diagnostics.

The convergence programs remain quantitative diagnostics: inspect their rates
in the logs.  A zero process exit status means execution/assertions succeeded;
it does not turn a poor printed convergence rate into a pass.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "output" / "verification"

STAGES = {
    "equations": {
        "title": "Equation-level manufactured tendency audit",
        "script": "benchmarks/test_equation_correctness_mms.py",
        "purpose": "Checks the discrete Euler RHS and its spatial order.",
        "profiles": {"quick", "full"},
    },
    "temporal": {
        "title": "Pure temporal convergence",
        "script": "benchmarks/test_temporal_convergence_sisl.py",
        "purpose": "Checks SISL and split-explicit time integration at fixed dx.",
        "profiles": {"quick", "full"},
    },
    "bubble": {
        "title": "Combined rising-bubble convergence",
        "script": "benchmarks/test_dual_core_bubble_convergence.py",
        "purpose": "Checks nonlinear combined space-time convergence of both cores.",
        "profiles": {"quick", "full"},
    },
    "dual_mms": {
        "title": "Forced dual-core MMS integration",
        "script": "benchmarks/test_dual_core_mms_convergence.py",
        "purpose": "Checks integrated spatial convergence under manufactured forcing.",
        "profiles": {"full"},
    },
    "components": {
        "title": "SISL component and operator convergence",
        "script": "benchmarks/test_sisl_components_convergence.py",
        "purpose": "Checks interpolation, trajectories, GMRES, operators, and FFSL.",
        "profiles": {"full"},
    },
    "invariants": {
        "title": "Conservation, balance, terrain, and solver diagnostics",
        "script": "tests/test_dynamical_core_3d.py",
        "purpose": "Checks closed-box mass, balance, boundaries, terrain, and AD.",
        "profiles": {"full"},
    },
}


def selected_stages(profile: str, only: list[str] | None) -> list[str]:
    if only:
        unknown = sorted(set(only) - STAGES.keys())
        if unknown:
            raise ValueError(f"Unknown stages: {', '.join(unknown)}")
        return only
    return [name for name, cfg in STAGES.items() if profile in cfg["profiles"]]


def run_stage(name: str, log_dir: Path) -> tuple[bool, float, Path]:
    cfg = STAGES[name]
    script = REPO_ROOT / cfg["script"]
    log_path = log_dir / f"{name}.log"
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("JAX_ENABLE_X64", "true")

    print(f"\n{'=' * 78}")
    print(f"[{name}] {cfg['title']}")
    print(f"Purpose: {cfg['purpose']}")
    print(f"Command: {sys.executable} {script.relative_to(REPO_ROOT)}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    print(f"{'=' * 78}")

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()

    elapsed = time.monotonic() - start
    passed = return_code == 0
    print(
        f"[{name}] {'COMPLETED' if passed else 'FAILED'} "
        f"in {elapsed:.1f}s (exit={return_code})"
    )
    return passed, elapsed, log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SUETES core verification stages in isolated processes."
    )
    parser.add_argument(
        "--profile", choices=("quick", "full"), default="quick",
        help="quick runs the main convergence gate; full adds expensive audits.",
    )
    parser.add_argument(
        "--only", nargs="+", choices=tuple(STAGES),
        help="Run only the named stages, in the supplied order.",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="Continue after a stage exits unsuccessfully.",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR,
        help="Directory for complete per-stage logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = selected_stages(args.profile, args.only)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    print("SUETES dynamical-core verification")
    print(f"Profile: {args.profile}")
    print(f"Stages: {', '.join(names)}")
    print("Each stage uses a fresh process to guarantee XLA memory release.")

    results = []
    for name in names:
        passed, elapsed, log_path = run_stage(name, log_dir)
        results.append((name, passed, elapsed, log_path))
        if not passed and not args.keep_going:
            break

    print(f"\n{'=' * 78}")
    print("Verification execution summary")
    print(f"{'=' * 78}")
    for name, passed, elapsed, log_path in results:
        print(
            f"{name:12s} {'OK' if passed else 'FAILED':6s} "
            f"{elapsed:8.1f}s  {log_path}"
        )

    incomplete = len(results) != len(names)
    failed = any(not passed for _, passed, _, _ in results)
    if incomplete:
        print("Some stages were skipped after the first failure.")
    print(
        "Review the reported convergence rates in the logs; successful execution "
        "alone is not a numerical-order assertion."
    )
    return 1 if failed or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
