#!/usr/bin/env python3
"""Render self- and cross-convergence from a long-horizon audit bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {
                **row,
                **{
                    key: float(row[key])
                    for key in ("coarse_dx_m", "fine_dx_m", "time_s", "value")
                },
            }
            for row in csv.DictReader(stream)
        ]


def render(metrics_path: Path, output_dir: Path | None = None) -> list[Path]:
    rows = _read(metrics_path)
    output_dir = output_dir or figure_dir_for(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cross = [row for row in rows if row["metric"] == "cross_core_relative_l2"]
    self_rows = [row for row in rows if row["metric"] == "self_relative_l2"]
    times = sorted({row["time_s"] for row in cross if row["time_s"] > 0})
    dx_values = sorted({row["coarse_dx_m"] for row in cross}, reverse=True)
    selected_times = sorted(set((times[0], times[len(times) // 2], times[-1])))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for time in selected_times:
        values = [
            next(
                row["value"] for row in cross
                if np.isclose(row["time_s"], time)
                and np.isclose(row["coarse_dx_m"], dx)
            )
            for dx in dx_values
        ]
        axes[0].loglog(dx_values, values, "o-", lw=2, label=f"{time:g} s")
    reference = np.asarray(dx_values) ** 2
    reference *= axes[0].lines[0].get_ydata()[0] / reference[0]
    axes[0].loglog(dx_values, reference, "k--", alpha=0.7, label="Order 2")
    props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
    axes[0].text(0.05, 0.95, "(a) Cross-core refinement", transform=axes[0].transAxes, fontsize=16, verticalalignment='top', bbox=props)
    axes[0].set_xlabel(r"Grid spacing $\Delta x$ [m]")
    axes[0].set_ylabel(r"Relative $L_2$ difference")
    axes[0].invert_xaxis()
    handles0, labels0 = axes[0].get_legend_handles_labels()

    final_time = times[-1]
    styles = {
        "sisl": ("o", "#0072B2", "SISL"),
        "split-explicit": ("s", "#D55E00", "Split-Explicit"),
    }
    for core, (marker, color, label) in styles.items():
        selected = sorted(
            (
                row for row in self_rows
                if row["core"] == core and np.isclose(row["time_s"], final_time)
            ),
            key=lambda row: row["coarse_dx_m"],
            reverse=True,
        )
        axes[1].loglog(
            [row["coarse_dx_m"] for row in selected],
            [row["value"] for row in selected],
            marker=marker, color=color, lw=2, label=label,
        )
    axes[1].text(0.05, 0.95, f"(b) Self-convergence at {final_time:g} s", transform=axes[1].transAxes, fontsize=16, verticalalignment='top', bbox=props)
    axes[1].set_xlabel(r"Coarse-grid spacing $\Delta x$ [m]")
    axes[1].set_ylabel(r"Relative $L_2$ difference")
    axes[1].invert_xaxis()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    import matplotlib.ticker as ticker
    for axis in axes:
        axis.grid(True, which="both", ls="--", alpha=0.4)
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())
        axis.tick_params(axis="x", which="both", rotation=0)
        
    ticks0 = sorted(list(set([dx_values[0], dx_values[len(dx_values)//2], dx_values[-1]])))
    axes[0].set_xticks(ticks0)
    axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    dx_self = sorted(list({row["coarse_dx_m"] for row in self_rows}), reverse=True)
    ticks1 = sorted(list(set([dx_self[0], dx_self[len(dx_self)//2], dx_self[-1]])))
    axes[1].set_xticks(ticks1)
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
        
    fig.legend(handles0 + handles1, labels0 + labels1, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels0) + len(labels1))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    convergence_path = output_dir / "long_horizon_convergence.png"
    fig.savefig(convergence_path, dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for dx in dx_values:
        selected = sorted(
            (row for row in cross if np.isclose(row["coarse_dx_m"], dx)),
            key=lambda row: row["time_s"],
        )
        axis.plot(
            [row["time_s"] for row in selected],
            [row["value"] for row in selected],
            "o-", lw=2, label=rf"$\Delta x={dx:g}$ m",
        )
    axis.set_xlabel("Simulation time [s]")
    axis.set_ylabel(r"Relative $L_2$ difference")
    axis.grid(True, ls="--", alpha=0.4)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis.tick_params(axis="x", which="both", rotation=0)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    evolution_path = output_dir / "cross_core_error_evolution.png"
    fig.savefig(evolution_path, dpi=300)
    plt.close(fig)

    for path in (convergence_path, evolution_path):
        print(f"Saved {path}")
    return [convergence_path, evolution_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Metrics CSV, data directory, or bundle")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    metrics = artifact_from_bundle(args.source, pattern="metrics.csv")
    render(metrics, args.output_dir)


if __name__ == "__main__":
    main()
