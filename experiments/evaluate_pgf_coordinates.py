#!/usr/bin/env python3
"""Evaluate frozen NEUVE, tuned SLEVE, and Gal-Chen on M terrains."""

import argparse
import csv
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments.neuve_pgf_common import (
    DEFAULT_STEPS, load_dataset, load_neuve, make_dataset, parse_seeds,
    run_case, terrains_from_dataset,
)
from suetes.shared.transforms import GalChenSigma, SleveSimple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Existing testing dataset manifest")
    parser.add_argument("--terrain-family", choices=["ridge", "random3d"], default="ridge")
    parser.add_argument("--seeds", default="999", help="Testing seeds when no manifest is supplied")
    parser.add_argument("--neuve-weights")
    parser.add_argument("--sleve-config")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument(
        "--plot-samples", type=int, default=3,
        help="Number of testing terrains included in the geometry/TKE gallery",
    )
    parser.add_argument("--output-dir", default="output/neuve_pgf_experiment")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.neuve_weights = args.neuve_weights or os.path.join(args.output_dir, "neuve_pgf.npz")
    args.sleve_config = args.sleve_config or os.path.join(args.output_dir, "best_sleve.json")
    dataset = (
        load_dataset(args.dataset) if args.dataset
        else make_dataset(args.terrain_family, parse_seeds(args.seeds), "testing")
    )
    make_dataset(
        dataset["terrain_family"], dataset["seeds"], "testing",
        os.path.join(args.output_dir, "testing_dataset.json"),
    )
    terrains = terrains_from_dataset(dataset)
    with open(args.sleve_config) as stream:
        sleve = json.load(stream)
    coordinates = {
        "Gal-Chen": GalChenSigma(),
        "Tuned SLEVE": SleveSimple(
            scale_s=float(sleve["scale_s"]), scale_l=15000.0, n=float(sleve["n"])
        ),
        "NEUVE": load_neuve(args.neuve_weights),
    }
    rows, representative = [], {}
    gallery = {}
    gallery_count = min(max(args.plot_samples, 0), len(terrains))
    for sample_index, (seed, terrain) in enumerate(zip(dataset["seeds"], terrains)):
        print(f"Terrain {sample_index + 1}/{len(terrains)}: seed={seed}")
        for name, transform in coordinates.items():
            final, series, grid, diagnostics = run_case(transform, terrain, args.steps)
            tke, max_u, max_w = series
            row = {
                "sample": sample_index, "seed": seed, "coordinate": name,
                "mean_physical_tke": float(jnp.mean(tke)),
                "peak_u": float(jnp.max(max_u)), "peak_w": float(jnp.max(max_w)),
                "minimum_layer_m": float(diagnostics[0]),
                "upper_slope_squared": float(diagnostics[1]),
                "dimensionless_curvature": float(diagnostics[2]),
            }
            rows.append(row)
            if sample_index == 0:
                representative[name] = (np.asarray(tke), grid)
            if sample_index < gallery_count:
                gallery.setdefault(sample_index, {})[name] = (
                    np.asarray(tke), grid
                )

    csv_path = os.path.join(args.output_dir, "testing_metrics.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {}
    print("\nTesting-dataset summary")
    for name in coordinates:
        selected = [row for row in rows if row["coordinate"] == name]
        values = np.asarray([row["mean_physical_tke"] for row in selected])
        summary[name] = {
            "mean_physical_tke": float(np.mean(values)),
            "std_physical_tke": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum_layer_m": min(row["minimum_layer_m"] for row in selected),
            "mean_upper_slope_squared": float(np.mean([row["upper_slope_squared"] for row in selected])),
            "mean_dimensionless_curvature": float(np.mean([row["dimensionless_curvature"] for row in selected])),
        }
        print(f"{name:<12}: {summary[name]['mean_physical_tke']:.6e} +/- {summary[name]['std_physical_tke']:.2e}")
    gal = np.asarray([row["mean_physical_tke"] for row in rows if row["coordinate"] == "Gal-Chen"])
    slv = np.asarray([row["mean_physical_tke"] for row in rows if row["coordinate"] == "Tuned SLEVE"])
    neu = np.asarray([row["mean_physical_tke"] for row in rows if row["coordinate"] == "NEUVE"])
    reduction_gal = 100.0 * (1.0 - neu / gal)
    reduction_sleve = 100.0 * (1.0 - neu / slv)

    def paired_interval(values):
        if len(values) == 1:
            return [float(values[0]), float(values[0])]
        generator = np.random.default_rng(args.bootstrap_seed)
        indices = generator.integers(
            0, len(values), size=(args.bootstrap_samples, len(values))
        )
        bootstrap_means = np.mean(values[indices], axis=1)
        return [float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5])]

    summary["paired"] = {
        "neuve_reduction_vs_galchen_percent": float(np.mean(reduction_gal)),
        "neuve_reduction_vs_galchen_95pct_bootstrap_ci": paired_interval(reduction_gal),
        "neuve_reduction_vs_sleve_percent": float(np.mean(reduction_sleve)),
        "neuve_reduction_vs_sleve_95pct_bootstrap_ci": paired_interval(reduction_sleve),
        "neuve_wins_vs_sleve": int(np.count_nonzero(neu < slv)),
        "samples": len(neu),
        "bootstrap_samples": args.bootstrap_samples,
    }
    with open(os.path.join(args.output_dir, "testing_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["paired"], indent=2))

    time = (np.arange(args.steps) + 1) * 0.8
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for name, (series, _) in representative.items():
        axis.semilogy(time, series, label=name)
    axis.set(xlabel="Time (s)", ylabel=r"Physical spurious TKE (m$^2$ s$^{-2}$)")
    axis.grid(True, which="both", alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "representative_tke.png"), dpi=250)
    plt.close(fig)

    # Coordinate geometry on the representative terrain.  Plot physical layer
    # interfaces, including the true terrain-following lower boundary.
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True, sharey=True)
    for axis, (name, (_, grid)) in zip(axes, representative.items()):
        x = np.asarray(grid.x_m) / 1000.0
        z = np.asarray(grid.Z_w)[:, grid.ny // 2, :] / 1000.0
        for level in range(z.shape[-1]):
            axis.plot(x, z[:, level], color="C0", linewidth=0.7)
        axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)
        axis.set_title(name)
        axis.set_xlabel("x (km)")
        axis.grid(True, alpha=0.15)
    axes[0].set_ylabel("Height (km)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "coordinate_grids.png"), dpi=250)
    plt.close(fig)

    if gallery_count:
        names = list(coordinates)
        fig, axes = plt.subplots(
            gallery_count, len(names),
            figsize=(11.5, 3.0 * gallery_count),
            sharex=True, sharey=True, squeeze=False,
        )
        for sample_index in range(gallery_count):
            seed = dataset["seeds"][sample_index]
            for column, name in enumerate(names):
                axis = axes[sample_index, column]
                _, grid = gallery[sample_index][name]
                x = np.asarray(grid.x_m) / 1000.0
                z = np.asarray(grid.Z_w)[:, grid.ny // 2, :] / 1000.0
                for level in range(z.shape[-1]):
                    axis.plot(x, z[:, level], color="C0", linewidth=0.65)
                axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)
                axis.grid(True, alpha=0.12)
                if sample_index == 0:
                    axis.set_title(name)
                if column == 0:
                    axis.set_ylabel(f"Seed {seed}\nHeight (km)")
                if sample_index == gallery_count - 1:
                    axis.set_xlabel("x (km)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.output_dir, "coordinate_sample_gallery.png"),
            dpi=250,
        )
        plt.close(fig)

        fig, axes = plt.subplots(
            1, gallery_count, figsize=(4.0 * gallery_count, 3.5),
            sharex=True, sharey=True, squeeze=False,
        )
        for sample_index in range(gallery_count):
            axis = axes[0, sample_index]
            cumulative = {}
            for name in names:
                series, _ = gallery[sample_index][name]
                cumulative[name] = np.cumsum(series) / np.arange(
                    1, len(series) + 1
                )
            reference = cumulative["Tuned SLEVE"]
            valid = reference > np.finfo(float).tiny
            for name in names:
                ratio = np.full_like(reference, np.nan)
                ratio[valid] = cumulative[name][valid] / reference[valid]
                axis.plot(time, ratio, label=name)
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.8)
            axis.set_xlabel("Time (s)")
            axis.set_xlim(0.0, time[-1])
            axis.grid(True, alpha=0.25)
        axes[0, 0].set_ylabel("Cumulative mean TKE / tuned SLEVE")
        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.output_dir, "tke_sample_gallery.png"), dpi=250
        )
        plt.close(fig)
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
