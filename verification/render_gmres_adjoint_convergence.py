#!/usr/bin/env python3
"""Render the SISL adjoint GMRES convergence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'font.family': 'sans-serif'
})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path,
        default=Path("output/verification/gmres_adjoint_convergence/default"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path = artifact_from_bundle(args.source, pattern="summary.json")
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    output_dir = args.output_dir or figure_dir_for(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for result in summary["results"].values():
        axis.semilogy(
            result["iters"], result["l2_errors"], "o-",
            color=result["color"], label=result["label"],
            linewidth=2.5, markersize=6,
        )
    axis.axhline(
        1.0e-4, color="gray", linestyle=":", alpha=0.7,
        label=r"Target accuracy ($10^{-4}$)",
    )
    axis.axhline(
        1.0e-6, color="black", linestyle="-.", alpha=0.7,
        label=r"High accuracy ($10^{-6}$)",
    )
    axis.set_xlabel("GMRES iterations per timestep")
    axis.set_ylabel(r"Relative adjoint error")
    axis.grid(True, which="both", ls="--", alpha=0.4)
    
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output = output_dir / "gmres_adjoint_convergence.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
