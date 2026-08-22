#!/usr/bin/env python3
"""Render dual-core CPU--GPU scaling results from a saved benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "benchmarks" / "cpu_gpu_short_scaling"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "font.family": "sans-serif",
    }
)


def load_results(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        results = list(csv.DictReader(stream))
    required = {"platform", "core", "workload", "grid_cells", "median_s"}
    missing = required - set(results[0] if results else ())
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    return results


def render(results: list[dict], output_path: Path):
    cores = [core for core in ("split-explicit", "sisl") if any(row["core"] == core for row in results)]
    if not cores:
        raise ValueError("CSV contains no split-explicit or SISL results")

    fig, axes = plt.subplots(1, 2 * len(cores), figsize=(14.4, 3.6), sharex=True, squeeze=False)
    device_labels = {"cpu": "CPU", "gpu": "GPU"}
    colors = {"cpu": "#377eb8", "gpu": "#e41a1c"}
    core_labels = {"split-explicit": "Split-Explicit", "sisl": "SISL"}
    workload_labels = {"forward": "forward", "value_and_gradient": "value + grad."}
    panel_letters = iter("abcd")
    grid_style = {"ls": "--", "lw": 0.6, "alpha": 0.4, "which": "both"}

    for row_index, core in enumerate(cores):
        for column_index, workload in enumerate(("forward", "value_and_gradient")):
            panel_index = 2 * row_index + column_index
            axis = axes[0, panel_index]
            for platform in ("cpu", "gpu"):
                selected = sorted(
                    (
                        row
                        for row in results
                        if row["platform"] == platform and row["core"] == core and row["workload"] == workload
                    ),
                    key=lambda row: int(row["grid_cells"]),
                )
                if selected:
                    axis.plot(
                        [int(row["grid_cells"]) for row in selected],
                        [1000.0 * float(row["median_s"]) for row in selected],
                        "o-",
                        color=colors[platform],
                        lw=1.2,
                        ms=4,
                        label=device_labels[platform],
                    )

            panel = next(panel_letters)
            axis.text(
                0.04,
                0.95,
                f"({panel}) {core_labels[core]} {workload_labels[workload]}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, **grid_style)
            axis.set_xlabel(r"Grid cells $N^3$")
    axes[0, 0].set_ylabel("Execution time (ms)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.97, bottom=0.27, wspace=0.22)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "cpu_gpu_short_scaling.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "figure_cpu_gpu_scaling.png")
    parser.add_argument("--paper-figure", type=Path)
    args = parser.parse_args()

    results = load_results(args.input)
    render(results, args.output)
    if args.paper_figure:
        render(results, args.paper_figure)


if __name__ == "__main__":
    main()
