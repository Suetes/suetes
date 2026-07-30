#!/usr/bin/env python3
# Training renderer.
"""Render NEUVE and SLEVE training diagnostics from saved data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _directories(source: Path, output_dir: Path | None) -> tuple[Path, Path]:
    data_dir = source / "data" if source.is_dir() and (source / "data").is_dir() else source
    if output_dir is not None:
        figures = output_dir
    elif data_dir.name == "data":
        figures = data_dir.parent / "figures"
    else:
        figures = data_dir
    figures.mkdir(parents=True, exist_ok=True)
    return data_dir, figures


def render(source: Path, output_dir: Path | None = None) -> list[Path]:
    data_dir, figures = _directories(source, output_dir)
    outputs: list[Path] = []

    history_path = data_dir / "training_history.npz"
    if not history_path.exists():
        history_path = data_dir / "neuve_training_history.npz"
    if history_path.exists():
        with np.load(history_path) as archive:
            history = {key: archive[key] for key in archive.files}
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
        epoch = history["epoch"]
        axes[0].semilogy(epoch, history["mean_tke"], label="Current metric", color="C0", linewidth=2)
        if "best_objective" in history:
            axes[0].semilogy(epoch, history["best_objective"], "--", label="Best feasible objective", color="C1", linewidth=2)
        elif "objective" in history:
            axes[0].semilogy(epoch, history["objective"], "--", label="Penalized objective", color="C1", linewidth=2)
        axes[0].set(xlabel="Epoch", ylabel="Objective (Mean TKE)")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)

        # Legend outside
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)

        axes[1].plot(epoch, history["minimum_layer_m"], color="purple", linewidth=2)
        axes[1].set(xlabel="Epoch", ylabel="Minimum layer thickness (m)")
        axes[1].grid(True, ls="--", alpha=0.4)

        output = figures / "neuve_training.png"
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(output, dpi=300)
        plt.close(fig)
        outputs.append(output)

    search_path = data_dir / "sleve_grid_search.csv"
    config_path = data_dir / "best_sleve.json"
    if search_path.exists() and config_path.exists():
        with search_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with config_path.open() as stream:
            best = json.load(stream)
        feasible = [row for row in rows if row["feasible"].strip().lower() in {"true", "1"}]
        fig, axis = plt.subplots(figsize=(8.0, 6.0))
        metric_key = "mean_metric" if "mean_metric" in rows[0] else "mean_physical_tke"
        axis.scatter(
            [float(row["scale_s"]) / 1000.0 for row in feasible],
            [float(row[metric_key]) for row in feasible],
            color="C0",
            s=80,
            alpha=0.7,
            label="Evaluated points",
        )
        best_metric = best["mean_metric"] if "mean_metric" in best else best["mean_physical_tke"]
        axis.scatter(
            float(best["scale_s"]) / 1000.0,
            float(best_metric),
            marker="*",
            s=300,
            facecolor="none",
            edgecolor="red",
            linewidth=2.0,
            label="Best configuration",
        )
        axis.set(xlabel="SLEVE scale (km)", ylabel="Objective (Mean TKE)")
        axis.grid(True, ls="--", alpha=0.4)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
        output = figures / "sleve_tuning.png"
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(output, dpi=300)
        plt.close(fig)
        outputs.append(output)

    if not outputs:
        raise FileNotFoundError(f"No NEUVE training diagnostics found in {data_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Execution bundle or data directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for output in render(args.source, args.output_dir):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
