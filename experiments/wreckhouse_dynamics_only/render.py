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
    args = parser.parse_args()

    if not args.artifact.is_file():
        parser.error(f"artifact does not exist: {args.artifact}")
    output = args.output or args.artifact.with_name(
        args.artifact.stem.replace("_plot_data", "_dashboard") + ".png"
    )
    subprocess.run(
        [
            sys.executable,
            str(SHARED_RENDERER),
            str(args.artifact),
            "--output",
            str(output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
