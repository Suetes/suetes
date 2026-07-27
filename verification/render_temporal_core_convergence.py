#!/usr/bin/env python3
"""Render fixed-grid temporal convergence from its JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convergence_plotting import save_four_panel_convergence
from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path,
        default=Path("output/verification/temporal_core_convergence/default"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path = artifact_from_bundle(args.source, pattern="summary.json")
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    output_dir = args.output_dir or figure_dir_for(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "temporal_self_convergence.png"
    save_four_panel_convergence(
        summary["coarse_dt_s"],
        [
            ("SISL", summary["cores"]["sisl"]["self_errors"]),
            (
                "Split-Explicit",
                summary["cores"]["split-explicit"]["self_errors"],
            ),
        ],
        {
            "u": r"$\|\Delta u\|_2$ [m s$^{-1}$]",
            "w": r"$\|\Delta w\|_2$ [m s$^{-1}$]",
            "pi": r"$\|\Delta\pi\|_2$ [1]",
            "th_v": r"$\|\Delta\theta_v\|_2$ [K]",
        },
        r"Coarse timestep $\Delta t$ [s]",
        "Rising bubble temporal self-convergence at $t=24$ s",
        output,
        reference_order=2,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
