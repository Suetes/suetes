#!/usr/bin/env python3
"""Run, render, and audit the experiments used by the manuscript."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "paper_suite.toml"


def load_manifest(path: Path) -> dict:
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    cases = manifest.get("case", [])
    identifiers = [case.get("id") for case in cases]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every [[case]] requires a non-empty id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Case ids must be unique")
    for case in cases:
        if case.get("group") not in {"table", "appendix"}:
            raise ValueError(
                f"{case['id']}: group must be 'table' or 'appendix'"
            )
        for stage in case.get("stage", []):
            if stage.get("kind") not in {"run", "render"}:
                raise ValueError(
                    f"{case['id']}: stage kind must be 'run' or 'render'"
                )
            if not stage.get("command"):
                raise ValueError(f"{case['id']}: stage command is required")
    return manifest


def context(output_root: Path) -> dict[str, str]:
    return {
        "python": str(REPO_ROOT / ".venv" / "bin" / "python3"),
        "repo": str(REPO_ROOT),
        "output_root": str(output_root),
    }


def expand(values: list[str], substitutions: dict[str, str]) -> list[str]:
    return [value.format_map(substitutions) for value in values]


def selected_cases(
    manifest: dict, identifiers: list[str], group: str | None
) -> list[dict]:
    cases = manifest["case"]
    by_id = {case["id"]: case for case in cases}
    missing = sorted(set(identifiers) - set(by_id))
    if missing:
        raise ValueError(f"Unknown case(s): {', '.join(missing)}")
    selected = [by_id[value] for value in identifiers] if identifiers else cases
    if group:
        selected = [case for case in selected if case["group"] == group]
    return selected


def stage_complete(stage: dict, substitutions: dict[str, str]) -> bool:
    outputs = expand(stage.get("outputs", []), substitutions)
    return bool(outputs) and all(Path(path).exists() for path in outputs)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def run_stage(
    case: dict,
    stage: dict,
    substitutions: dict[str, str],
    *,
    dry_run: bool,
    force: bool,
) -> bool:
    command = expand(stage["command"], substitutions)
    label = f"{case['id']}:{stage['name']}"
    if stage_complete(stage, substitutions) and not force:
        print(f"SKIP {label} (declared outputs exist)")
        return True
    print(f"RUN  {label}")
    print("     " + " ".join(command))
    if dry_run:
        return True

    for output in expand(stage.get("outputs", []), substitutions):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(substitutions["output_root"]) / "logs" / case["id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage['name']}.log"
    started = datetime.now(timezone.utc)
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(substitutions["output_root"]) / ".matplotlib"),
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    ended = datetime.now(timezone.utc)
    record = {
        "case": case["id"],
        "stage": stage["name"],
        "kind": stage["kind"],
        "command": command,
        "git_commit": git_commit(),
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "return_code": return_code,
        "log": str(log_path),
    }
    with log_path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    if return_code:
        print(f"FAIL {label} (exit={return_code}; log={log_path})")
        return False
    if not stage_complete(stage, substitutions):
        print(f"FAIL {label} (command succeeded but declared outputs are missing)")
        return False
    print(f"DONE {label}")
    return True


def write_suite_record(
    manifest_path: Path, output_root: Path, selected: list[dict]
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = manifest_path.read_bytes()
    record = {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_cases": [case["id"] for case in selected],
    }
    with (output_root / "suite.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    (output_root / "paper_suite.toml").write_bytes(manifest_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("list", "status", "run", "render", "all")
    )
    parser.add_argument("cases", nargs="*", help="Case ids; default: all")
    parser.add_argument("--group", choices=("table", "appendix"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "output" / "paper"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(manifest_path)
    cases = selected_cases(manifest, args.cases, args.group)
    substitutions = context(output_root)

    if args.action == "list":
        for case in cases:
            print(f"{case['id']:30s} {case['group']:8s} {case['description']}")
        return 0
    if args.action == "status":
        for case in cases:
            states = [
                f"{stage['name']}="
                f"{'done' if stage_complete(stage, substitutions) else 'missing'}"
                for stage in case.get("stage", [])
            ]
            print(f"{case['id']:30s} " + ", ".join(states))
        return 0

    if not args.dry_run:
        write_suite_record(manifest_path, output_root, cases)
    allowed_kinds = {
        "run": {"run"},
        "render": {"render"},
        "all": {"run", "render"},
    }[args.action]
    for case in cases:
        for stage in case.get("stage", []):
            if stage["kind"] not in allowed_kinds:
                continue
            if not run_stage(
                case,
                stage,
                substitutions,
                dry_run=args.dry_run,
                force=args.force,
            ):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
