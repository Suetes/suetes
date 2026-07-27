#!/usr/bin/env python3
"""Render reverse-mode scaling diagnostics from the benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from suetes.shared.artifacts import figure_dir_for

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "font.family": "sans-serif",
    }
)


def render(source: Path, output_dir: Path | None = None) -> Path:
    csv_path = source / "data" / "single_gpu_domain_scaling.csv" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    required = {
        "mode",
        "grid_cells",
        "forward_time_ms",
        "estimated_adjoint_time_ms",
        "value_and_grad_time_ms",
        "adjoint_to_forward_ratio",
        "forward_peak_mib",
        "value_and_grad_peak_mib",
        "peak_difference_mib",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.0))
    line_styles = ["-", "--", "-.", ":"]
    for mode_idx, (mode, group) in enumerate(data.groupby("mode", sort=False)):
        ls = line_styles[mode_idx % len(line_styles)]
        group = group.sort_values("grid_cells")
        x = group["grid_cells"]
        axes[0].plot(x, group["value_and_grad_time_ms"], marker="o", ls=ls, linewidth=2.0, ms=8, label=mode)
        axes[0].plot(
            x, group["forward_time_ms"], marker="", ls=":", color="grey", alpha=0.65, label=f"Forward ({mode})"
        )
        axes[1].plot(x, group["adjoint_to_forward_ratio"], marker="o", ls=ls, linewidth=2.0, ms=8, label=mode)
        axes[2].plot(x, group["value_and_grad_peak_mib"], marker="o", ls=ls, linewidth=2.0, ms=8, label=mode)
        axes[2].plot(
            x, group["forward_peak_mib"], marker="", ls=":", color="grey", alpha=0.65, label=f"Forward ({mode})"
        )
        axes[3].plot(x, group["peak_difference_mib"], marker="o", ls=ls, linewidth=2.0, ms=8, label=mode)
    labels = (
        ("Execution time", "Time (ms)"),
        ("Adjoint-to-forward cost", "Ratio"),
        ("Absolute peak allocation", "MiB"),
        ("Peak-allocation difference", "MiB"),
    )
    props = dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    for i, (axis, (title, ylabel)) in enumerate(zip(axes, labels)):
        axis.set(xlabel="Grid cells", ylabel=ylabel)
        axis.text(
            0.05,
            0.95,
            f"({chr(97 + i)}) {title}",
            transform=axis.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props,
        )
        axis.grid(True, ls="--", alpha=0.4)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(legend_labels))

    output = output_dir / "single_gpu_domain_scaling.png"
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.source, args.output_dir)}")
