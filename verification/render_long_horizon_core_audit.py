#!/usr/bin/env python3
"""Render self- and cross-convergence from a long-horizon audit bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for


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
    axes[0].set_title("(a) Cross-core refinement")
    axes[0].set_xlabel(r"Grid spacing $\Delta x$ (m)")
    axes[0].set_ylabel("SISL–Split relative $L_2$ difference")
    axes[0].invert_xaxis()
    axes[0].legend()

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
    axes[1].set_title(f"(b) Self-convergence at {final_time:g} s")
    axes[1].set_xlabel(r"Coarse-grid spacing $\Delta x$ (m)")
    axes[1].set_ylabel("Successive-grid relative $L_2$ error")
    axes[1].invert_xaxis()
    axes[1].legend()
    for axis in axes:
        axis.grid(True, which="both", ls="--", alpha=0.4)
    fig.suptitle("Long-horizon rising-bubble convergence")
    fig.tight_layout()
    convergence_path = output_dir / "long_horizon_convergence.png"
    fig.savefig(convergence_path, dpi=300, bbox_inches="tight")
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
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("SISL–Split relative $L_2$ difference")
    axis.set_title("Cross-core disagreement through time")
    axis.grid(True, ls="--", alpha=0.4)
    axis.legend()
    fig.tight_layout()
    evolution_path = output_dir / "cross_core_error_evolution.png"
    fig.savefig(evolution_path, dpi=300, bbox_inches="tight")
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
