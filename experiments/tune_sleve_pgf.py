#!/usr/bin/env python3
"""Geometry-screened constrained SLEVE tuning on a NEUVE dataset."""

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
    DEFAULT_STEPS, DX, DY, DZ, NX, NY, NZ, load_dataset, run_case,
    terrains_from_dataset,
)
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.shared.transforms import SleveSimple


def values(text):
    return [float(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="NEUVE training_dataset.json")
    parser.add_argument(
        "--scale-values",
        default="2000,2500,3000,3500,4000,4500,5000,5750,6500,7500",
        help="Comma-separated SLEVE decay scales in metres",
    )
    parser.add_argument("--n-min", type=float, default=0.5)
    parser.add_argument("--n-max", type=float, default=4.0)
    parser.add_argument("--geometry-iterations", type=int, default=18)
    parser.add_argument(
        "--geometry-margin-m", type=float, default=2.0,
        help="Keep this margin above the dynamical minimum-layer constraint",
    )
    parser.add_argument("--minimum-layer-m", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--output-dir",
        help="Shared experiment directory (defaults to the dataset directory)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.dataset))
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset(args.dataset)
    terrains = terrains_from_dataset(dataset)

    def geometry_minimum(scale, exponent):
        minima = []
        for terrain in terrains:
            try:
                grid = RegionalGrid3D(
                    NX, NY, NZ, DX, DY, DZ,
                    lat_center=45.0, lon_center=0.0,
                    h_func=terrain,
                    transform=SleveSimple(
                        scale_s=scale, scale_l=15000.0, n=exponent
                    ),
                )
                value = float(jnp.min(grid.dz_m_full))
            except (ValueError, FloatingPointError):
                return -np.inf
            if not np.isfinite(value):
                return -np.inf
            minima.append(value)
        return min(minima)

    target_layer = args.minimum_layer_m + args.geometry_margin_m
    boundary_candidates = []
    print(
        "Geometry screening: locating the strongest feasible exponent for "
        f"each scale (target min_dz={target_layer:.1f} m)"
    )
    for scale in values(args.scale_values):
        low, high = args.n_min, args.n_max
        if geometry_minimum(scale, low) < target_layer:
            print(f"  scale={scale:.0f} m: no feasible exponent in range")
            continue
        if geometry_minimum(scale, high) >= target_layer:
            exponent = high
            bounded = True
        else:
            bounded = False
            for _ in range(args.geometry_iterations):
                middle = 0.5 * (low + high)
                if geometry_minimum(scale, middle) >= target_layer:
                    low = middle
                else:
                    high = middle
            exponent = low
        minimum = geometry_minimum(scale, exponent)
        boundary_candidates.append((scale, exponent, minimum))
        suffix = " (n_max reached)" if bounded else ""
        print(
            f"  scale={scale:.0f} m: n={exponent:.5f}, "
            f"min_dz={minimum:.2f} m{suffix}"
        )

    rows = []
    print(f"Running only {len(boundary_candidates)} dynamical candidates")
    for scale, exponent, _ in boundary_candidates:
        transform = SleveSimple(scale_s=scale, scale_l=15000.0, n=exponent)
        losses, minimum_layers = [], []
        print(f"SLEVE scale={scale:.0f} m, constrained n={exponent:.5f}")
        feasible = True
        for terrain in terrains:
            try:
                _, series, _, diagnostics = run_case(transform, terrain, args.steps)
                losses.append(float(jnp.mean(series[0])))
                minimum_layers.append(float(diagnostics[0]))
            except (ValueError, FloatingPointError):
                feasible = False
                break
        feasible = (
            feasible and len(losses) == len(terrains)
            and np.all(np.isfinite(losses))
            and min(minimum_layers) >= args.minimum_layer_m
        )
        row = {
            "scale_s": scale, "n": exponent,
            "mean_physical_tke": float(np.mean(losses)) if losses else np.inf,
            "std_physical_tke": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "minimum_layer_m": min(minimum_layers) if minimum_layers else np.nan,
            "feasible": bool(feasible),
        }
        rows.append(row)
        print(f"  mean={row['mean_physical_tke']:.6e}, min_dz={row['minimum_layer_m']:.2f}, feasible={feasible}")
    candidates = [row for row in rows if row["feasible"]]
    if not candidates:
        raise RuntimeError("No feasible SLEVE candidate")
    best = min(candidates, key=lambda row: row["mean_physical_tke"])
    with open(os.path.join(args.output_dir, "sleve_grid_search.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    config = {**best, "training_dataset": os.path.abspath(args.dataset)}
    with open(os.path.join(args.output_dir, "best_sleve.json"), "w") as stream:
        json.dump(config, stream, indent=2)
    fig, axis = plt.subplots(figsize=(6.2, 4.5))
    feasible_rows = [row for row in rows if row["feasible"]]
    axis.plot(
        [row["scale_s"] / 1000.0 for row in feasible_rows],
        [row["mean_physical_tke"] for row in feasible_rows], "o-",
    )
    axis.scatter(best["scale_s"] / 1000.0, best["mean_physical_tke"],
                 marker="*", s=160, facecolor="none", edgecolor="red", linewidth=1.5)
    axis.set(xlabel=r"SLEVE scale $s$ (km)",
             ylabel=r"Mean spurious TKE (m$^2$ s$^{-2}$)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sleve_tuning.png"), dpi=250)
    plt.close(fig)
    print("Best SLEVE:\n" + json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
