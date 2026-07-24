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


def render(source: Path, output_dir: Path | None = None) -> Path:
    csv_path = (
        source / "data" / "single_gpu_domain_scaling.csv"
        if source.is_dir() else source
    )
    output_dir = output_dir or figure_dir_for(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(csv_path)
    required = {
        "mode", "grid_cells", "forward_time_ms",
        "estimated_adjoint_time_ms", "value_and_grad_time_ms",
        "adjoint_to_forward_ratio", "forward_peak_mib",
        "value_and_grad_peak_mib", "peak_difference_mib",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for mode, group in data.groupby("mode", sort=False):
        group = group.sort_values("grid_cells")
        x = group["grid_cells"]
        axes[0].plot(x, group["value_and_grad_time_ms"], "o-", label=mode)
        axes[0].plot(x, group["forward_time_ms"], "--", alpha=0.65)
        axes[1].plot(x, group["adjoint_to_forward_ratio"], "o-", label=mode)
        axes[2].plot(x, group["value_and_grad_peak_mib"], "o-", label=mode)
        axes[2].plot(x, group["forward_peak_mib"], "--", alpha=0.65)
        axes[3].plot(x, group["peak_difference_mib"], "o-", label=mode)
    labels = (
        ("Execution time", "Time (ms)"),
        ("Adjoint-to-forward cost", "Ratio"),
        ("Absolute peak allocation", "MiB"),
        ("Peak-allocation difference", "MiB"),
    )
    for axis, (title, ylabel) in zip(axes, labels):
        axis.set(xlabel="Grid cells", ylabel=ylabel, title=title)
        axis.grid(True, ls="--", alpha=0.4)
        axis.legend(fontsize=8)
    output = output_dir / "single_gpu_domain_scaling.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.source, args.output_dir)}")
