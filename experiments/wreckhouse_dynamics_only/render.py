#!/usr/bin/env python3
"""Render a dynamics-only Wreckhouse artifact with the publication dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_RENDERER = REPOSITORY / "runs" / "plot_wreckhouse_worst_case.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-timeseries",
        action="store_true",
        help="Plot the full time series from beginning to end.",
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        parser.error(f"artifact does not exist: {args.artifact}")
    default_suffix = (
        "_full_timeseries.png" if args.full_timeseries else ".png"
    )
    if args.output:
        output = args.output
    else:
        output = args.artifact.with_name(
            args.artifact.stem.replace("_plot_data", "_dashboard") + default_suffix
        )
    cmd = [
        sys.executable,
        str(SHARED_RENDERER),
        str(args.artifact),
        "--output",
        str(output),
    ]
    if args.full_timeseries:
        cmd.append("--full-timeseries")
    subprocess.run(
        cmd,
        check=True,
    )


if __name__ == "__main__":
    main()
