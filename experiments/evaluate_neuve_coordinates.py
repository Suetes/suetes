#!/usr/bin/env python3
"""Evaluate Gal--Chen, tuned SLEVE, and NEUVE on held-out terrains."""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from experiments.neuve_coordinate_common import (
    TARGET_DEFAULTS, dataset_target, load_dataset, load_neuve, make_dataset,
    neuve_template, parse_seeds, run_target_case, terrains_from_dataset,
)
from suetes.shared.transforms import GalChenSigma, SleveSimple
from suetes.shared.artifacts import save_plot_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["pgf_rest", "tracer_reversibility", "mountain_flux"],
        default="pgf_rest",
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--terrain-family",
        choices=["ridge", "random3d", "multiscale3d"],
        default="ridge",
    )
    parser.add_argument("--seeds", default="999")
    parser.add_argument("--neuve-weights")
    parser.add_argument("--sleve-config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument(
        "--include-untrained", action="store_true",
        help="Include the untrained NEUVE initialization as an ablation",
    )
    parser.add_argument("--output-dir", default="output/neuve_coordinate_experiment")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.neuve_weights = args.neuve_weights or os.path.join(
        args.output_dir, "neuve_coordinate.npz"
    )
    args.sleve_config = args.sleve_config or os.path.join(
        args.output_dir, "best_sleve.json"
    )
    default_steps = TARGET_DEFAULTS[args.target]["steps"]
    dataset = (
        load_dataset(args.dataset) if args.dataset else make_dataset(
            args.terrain_family, parse_seeds(args.seeds), "testing",
            target=args.target, steps=args.steps or default_steps,
        )
    )
    target = dataset_target(dataset)
    if args.dataset and target != args.target:
        parser.error(f"dataset target is {target!r}, not {args.target!r}")
    steps = args.steps or int(dataset["integration"]["steps"])
    make_dataset(
        dataset["terrain_family"], dataset["seeds"], "testing",
        os.path.join(args.output_dir, "testing_dataset.json"),
        target=target, steps=steps,
    )
    terrains = terrains_from_dataset(dataset)
    with open(args.sleve_config) as stream:
        sleve = json.load(stream)
    if sleve.get("target", "pgf_rest") != target:
        parser.error("SLEVE configuration and evaluation targets differ")
    coordinates = {
        "Gal-Chen": GalChenSigma(),
        "Tuned SLEVE": SleveSimple(
            scale_s=float(sleve["scale_s"]), scale_l=15000.0,
            n=float(sleve["n"]),
        ),
    }
    if target == "tracer_reversibility" and args.include_untrained:
        coordinates["Untrained NEUVE"] = neuve_template()
    coordinates["NEUVE"] = load_neuve(args.neuve_weights)

    rows, gallery, artifact_series, artifact_z = [], {}, {}, {}
    first_grid = None
    gallery_count = min(max(args.plot_samples, 0), len(terrains))
    for sample_index, (seed, terrain) in enumerate(zip(dataset["seeds"], terrains)):
        print(f"Terrain {sample_index + 1}/{len(terrains)}: seed={seed}")
        for name, transform in coordinates.items():
            metric, series, grid, diagnostics = run_target_case(
                target, transform, terrain, steps
            )
            if first_grid is None:
                first_grid = grid
            row = {
                "sample": sample_index, "seed": seed, "coordinate": name,
                "metric": float(metric),
                "minimum_layer_m": float(diagnostics[0]),
                "upper_slope_squared": float(diagnostics[1]),
                "dimensionless_curvature": float(diagnostics[2]),
            }
            if target == "tracer_reversibility":
                row["tracer_mass_drift"] = float(diagnostics[3])
            elif target == "mountain_flux":
                row["momentum_flux_rms"] = float(diagnostics[3])
            rows.append(row)
            artifact_series[(sample_index, name)] = np.asarray(series)
            artifact_z[(sample_index, name)] = np.asarray(
                grid.Z_w[:, grid.ny // 2, :]
            )
            if sample_index < gallery_count:
                gallery.setdefault(sample_index, {})[name] = (
                    np.asarray(series), grid
                )

    with open(os.path.join(args.output_dir, "testing_metrics.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    print("\nTesting-dataset summary")
    for name in coordinates:
        selected = [row for row in rows if row["coordinate"] == name]
        values = np.asarray([row["metric"] for row in selected])
        summary[name] = {
            "mean_metric": float(np.mean(values)),
            "std_metric": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum_layer_m": min(row["minimum_layer_m"] for row in selected),
        }
        print(
            f"{name:<12}: {summary[name]['mean_metric']:.6e} "
            f"+/- {summary[name]['std_metric']:.2e}"
        )
    arrays = {
        name: np.asarray([
            row["metric"] for row in rows if row["coordinate"] == name
        ]) for name in coordinates
    }
    gal, slv, neu = arrays["Gal-Chen"], arrays["Tuned SLEVE"], arrays["NEUVE"]

    def paired_interval(values):
        if len(values) == 1:
            return [float(values[0]), float(values[0])]
        rng = np.random.default_rng(args.bootstrap_seed)
        index = rng.integers(
            0, len(values), size=(args.bootstrap_samples, len(values))
        )
        return [
            float(value) for value in
            np.percentile(np.mean(values[index], axis=1), [2.5, 97.5])
        ]

    reduction_gal = 100.0 * (1.0 - neu / gal)
    reduction_sleve = 100.0 * (1.0 - neu / slv)
    summary["target"] = target
    summary["paired"] = {
        "neuve_reduction_vs_galchen_percent": float(np.mean(reduction_gal)),
        "neuve_reduction_vs_galchen_95pct_bootstrap_ci": paired_interval(reduction_gal),
        "neuve_reduction_vs_sleve_percent": float(np.mean(reduction_sleve)),
        "neuve_reduction_vs_sleve_95pct_bootstrap_ci": paired_interval(reduction_sleve),
        "neuve_wins_vs_sleve": int(np.count_nonzero(neu < slv)),
        "samples": len(neu),
    }
    with open(os.path.join(args.output_dir, "testing_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["paired"], indent=2))

    names = list(coordinates)
    sample_count = len(terrains)
    series_data = np.stack([
        np.stack([artifact_series[(sample, name)] for name in names])
        for sample in range(sample_count)
    ])
    z_data = np.stack([
        np.stack([artifact_z[(sample, name)] for name in names])
        for sample in range(sample_count)
    ])
    metric_data = np.asarray([
        [next(
            row["metric"] for row in rows
            if row["sample"] == sample and row["coordinate"] == name
        ) for name in names]
        for sample in range(sample_count)
    ])
    plot_dataset = xr.Dataset(
        data_vars={
            "diagnostic_series": (
                ("sample", "coordinate", "time"), series_data
            ),
            "metric": (("sample", "coordinate"), metric_data),
            "z_interface": (
                ("sample", "coordinate", "x", "z_interface_level"), z_data
            ),
            "terrain": (
                ("sample", "x"), z_data[:, 0, :, 0]
            ),
        },
        coords={
            "sample": np.arange(sample_count),
            "seed": ("sample", np.asarray(dataset["seeds"], dtype=int)),
            "coordinate": names,
            "time": (np.arange(steps) + 1) * float(dataset["integration"]["dt"]),
            "x": np.asarray(first_grid.x_m),
            "z_interface_level": np.arange(z_data.shape[-1]),
        },
        attrs={
            "target": target,
            "terrain_family": dataset["terrain_family"],
            "diagnostic_description": (
                "instantaneous volume-weighted TKE" if target == "pgf_rest"
                else target
            ),
        },
    )
    artifact_path = save_plot_dataset(
        plot_dataset,
        os.path.join(args.output_dir, "coordinate_evaluation.nc"),
        experiment=f"neuve_{target}_evaluation",
        metadata={
            "neuve_weights": os.path.abspath(args.neuve_weights),
            "sleve_config": os.path.abspath(args.sleve_config),
        },
    )
    print(f"Saved plot-ready evaluation artifact to {artifact_path}")

    if gallery_count:
        fig, axes = plt.subplots(
            gallery_count, len(names),
            figsize=(3.7 * len(names), 3.0 * gallery_count),
            sharex=True, sharey=True, squeeze=False,
        )
        for sample in range(gallery_count):
            for column, name in enumerate(names):
                axis = axes[sample, column]
                _, grid = gallery[sample][name]
                x = np.asarray(grid.x_m) / 1000.0
                z = np.asarray(grid.Z_w)[:, grid.ny // 2, :] / 1000.0
                for level in range(z.shape[-1]):
                    axis.plot(x, z[:, level], color="C0", linewidth=0.65)
                axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)
                if sample == 0:
                    axis.set_title(name)
                if column == 0:
                    axis.set_ylabel("Height (km)")
                if sample == gallery_count - 1:
                    axis.set_xlabel("x (km)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "coordinate_sample_gallery.png"), dpi=250)
        plt.close(fig)

        dt = float(dataset["integration"]["dt"])
        time = (np.arange(steps) + 1) * dt
        fig, axes = plt.subplots(
            1, gallery_count, figsize=(4.0 * gallery_count, 3.5),
            sharex=True, sharey=True, squeeze=False,
        )
        for sample in range(gallery_count):
            axis = axes[0, sample]
            plotted = {}
            for name in names:
                series, _ = gallery[sample][name]
                plotted[name] = (
                    np.cumsum(series) / np.arange(1, len(series) + 1)
                    if target == "pgf_rest" else series
                )
            if target == "mountain_flux":
                for name in names:
                    axis.plot(time, plotted[name], label=name)
            else:
                reference = plotted["Tuned SLEVE"]
                valid = reference > np.finfo(float).tiny
                for name in names:
                    ratio = np.full_like(reference, np.nan)
                    ratio[valid] = plotted[name][valid] / reference[valid]
                    axis.plot(time, ratio, label=name)
                axis.axhline(
                    1.0, color="0.25", linestyle="--", linewidth=0.8
                )
            time_start = (
                0.5 * time[-1] if target == "mountain_flux" else 0.0
            )
            axis.set(xlabel="Time (s)", xlim=(time_start, time[-1]))
            axis.grid(True, alpha=0.25)
        axes[0, 0].set_ylabel(
            "Cumulative mean TKE / tuned SLEVE"
            if target == "pgf_rest"
            else (
                "Tracer return error / tuned SLEVE"
                if target == "tracer_reversibility"
                else "Momentum-flux transmission error"
            )
        )
        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "metric_sample_gallery.png"), dpi=250)
        plt.close(fig)
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
