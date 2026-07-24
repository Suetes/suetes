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

    history_path = data_dir / "neuve_training_history.npz"
    if history_path.exists():
        with np.load(history_path) as archive:
            history = {key: archive[key] for key in archive.files}
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        epoch = history["epoch"]
        axes[0].semilogy(epoch, history["mean_tke"], label="Current metric")
        axes[0].semilogy(
            epoch, history["best_objective"], "--",
            label="Best feasible objective",
        )
        axes[0].set(xlabel="Epoch", ylabel="Objective")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        axes[1].plot(epoch, history["minimum_layer_m"])
        axes[1].set(xlabel="Epoch", ylabel="Minimum layer thickness (m)")
        axes[1].grid(True, alpha=0.3)
        output = figures / "neuve_training.png"
        fig.tight_layout()
        fig.savefig(output, dpi=250)
        plt.close(fig)
        outputs.append(output)

    search_path = data_dir / "sleve_grid_search.csv"
    config_path = data_dir / "best_sleve.json"
    if search_path.exists() and config_path.exists():
        with search_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with config_path.open() as stream:
            best = json.load(stream)
        feasible = [
            row for row in rows
            if row["feasible"].strip().lower() in {"true", "1"}
        ]
        fig, axis = plt.subplots(figsize=(6.2, 4.5))
        axis.scatter(
            [float(row["scale_s"]) / 1000.0 for row in feasible],
            [float(row["mean_metric"]) for row in feasible],
        )
        axis.scatter(
            float(best["scale_s"]) / 1000.0, float(best["mean_metric"]),
            marker="*", s=160, facecolor="none", edgecolor="red",
        )
        axis.set(xlabel="SLEVE scale (km)", ylabel="Objective")
        axis.grid(True, alpha=0.3)
        output = figures / "sleve_tuning.png"
        fig.tight_layout()
        fig.savefig(output, dpi=250)
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
