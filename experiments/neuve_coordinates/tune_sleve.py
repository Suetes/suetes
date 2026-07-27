#!/usr/bin/env python3
# SLEVE tuning entry point.
"""Geometry-screened SLEVE tuning for a neural-coordinate experiment."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.neuve_coordinate import (
    DX,
    DY,
    DZ,
    NX,
    NY,
    NZ,
    dataset_target,
    load_dataset,
    run_target_case,
    terrains_from_dataset,
)
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.shared.transforms import SleveSimple
from suetes.shared.experiment import figure_dir_for, add_experiment_args, setup_experiment_directories


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
        "--transport-n-samples",
        type=int,
        default=5,
        help=("Number of exponents evaluated from n-min to the geometric boundary at each scale for non-PGF targets"),
    )
    parser.add_argument(
        "--geometry-margin-m",
        type=float,
        default=2.0,
        help="Keep this margin above the dynamical minimum-layer constraint",
    )
    parser.add_argument("--minimum-layer-m", type=float, default=100.0)
    parser.add_argument("--steps", type=int)
    add_experiment_args(parser)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--worker-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-n", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.dataset))
    os.makedirs(args.output_dir, exist_ok=True)
    figure_dir = figure_dir_for(Path(args.output_dir) / "artifact.nc")
    figure_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    target = dataset_target(dataset)
    steps = args.steps or int(dataset["integration"]["steps"])
    terrains = terrains_from_dataset(dataset)

    def geometry_minimum(scale, exponent):
        minima = []
        for terrain in terrains:
            try:
                grid = RegionalGrid3D(
                    NX,
                    NY,
                    NZ,
                    DX,
                    DY,
                    DZ,
                    lat_center=45.0,
                    lon_center=0.0,
                    h_func=terrain,
                    transform=SleveSimple(scale_s=scale, scale_l=15000.0, n=exponent),
                )
                value = float(jnp.min(grid.dz_m_full))
            except (ValueError, FloatingPointError):
                return -np.inf
            if not np.isfinite(value):
                return -np.inf
            minima.append(value)
        return min(minima)

    def evaluate_candidate(scale, exponent):
        transform = SleveSimple(scale_s=scale, scale_l=15000.0, n=exponent)
        losses, minimum_layers = [], []
        feasible = True
        for terrain in terrains:
            try:
                metric, _, _, diagnostics = run_target_case(target, transform, terrain, steps)
                losses.append(float(metric))
                minimum_layers.append(float(diagnostics[0]))
                if args.worker_scale is not None:
                    # Python 3.14/JAX can otherwise retain executable-cache
                    # state across terrain-specific grid constants until a
                    # worker exits.
                    jax.clear_caches()
                    gc.collect()
            except (ValueError, FloatingPointError):
                feasible = False
                break
        feasible = (
            feasible
            and len(losses) == len(terrains)
            and np.all(np.isfinite(losses))
            and min(minimum_layers) >= args.minimum_layer_m
        )
        return {
            "scale_s": scale,
            "n": exponent,
            "mean_metric": (float(np.mean(losses)) if losses else np.inf),
            "std_metric": (float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0),
            "minimum_layer_m": (min(minimum_layers) if minimum_layers else np.nan),
            "feasible": bool(feasible),
        }

    if args.worker_scale is not None:
        if args.worker_n is None or args.worker_result is None:
            parser.error("incomplete internal worker arguments")
        row = evaluate_candidate(args.worker_scale, args.worker_n)
        with open(args.worker_result, "w") as stream:
            json.dump(row, stream)
        return

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
        print(f"  scale={scale:.0f} m: n={exponent:.5f}, min_dz={minimum:.2f} m{suffix}")

    if target != "pgf_rest":
        # Reversibility is not monotone in terrain decay.  Unlike the resting
        # PGF objective, its optimum need not lie on the minimum-layer
        # constraint, so sample the feasible interior as well as the boundary.
        dynamical_candidates = []
        for scale, boundary_exponent, _ in boundary_candidates:
            for exponent in np.linspace(args.n_min, boundary_exponent, args.transport_n_samples):
                minimum = geometry_minimum(scale, float(exponent))
                if minimum >= target_layer:
                    dynamical_candidates.append((scale, float(exponent), minimum))
    else:
        dynamical_candidates = boundary_candidates

    csv_path = os.path.join(args.output_dir, "sleve_grid_search.csv")
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as stream:
            for raw in csv.DictReader(stream):
                row = {
                    "scale_s": float(raw["scale_s"]),
                    "n": float(raw["n"]),
                    "mean_metric": float(raw["mean_metric"]),
                    "std_metric": float(raw["std_metric"]),
                    "minimum_layer_m": float(raw["minimum_layer_m"]),
                    "feasible": raw["feasible"].lower() == "true",
                }
                existing[(row["scale_s"], round(row["n"], 10))] = row

    def save_progress(current_rows):
        if not current_rows:
            return
        with open(csv_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(current_rows[0]))
            writer.writeheader()
            writer.writerows(current_rows)

    rows = []
    print(f"Running {len(dynamical_candidates)} dynamical candidates")
    with tempfile.TemporaryDirectory(prefix="suetes-sleve-") as worker_dir:
        for candidate, (scale, exponent, _) in enumerate(dynamical_candidates):
            print(f"SLEVE scale={scale:.0f} m, constrained n={exponent:.5f}")
            key = (float(scale), round(float(exponent), 10))
            if key in existing:
                row = existing[key]
                rows.append(row)
                print(
                    f"  resumed mean={row['mean_metric']:.6e}, "
                    f"min_dz={row['minimum_layer_m']:.2f}, "
                    f"feasible={row['feasible']}"
                )
                continue
            result_path = os.path.join(worker_dir, f"candidate_{candidate}.json")
            command = [
                sys.executable,
                os.path.abspath(__file__),
                "--dataset",
                os.path.abspath(args.dataset),
                "--minimum-layer-m",
                str(args.minimum_layer_m),
                "--steps",
                str(steps),
                "--worker-scale",
                str(scale),
                "--worker-n",
                str(exponent),
                "--worker-result",
                result_path,
            ]
            environment = os.environ.copy()
            environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            completed = None
            worker_succeeded = False
            for attempt in range(1, 4):
                if os.path.exists(result_path):
                    os.unlink(result_path)
                completed = subprocess.run(command, check=False, env=environment)
                if completed.returncode == 0 and os.path.exists(result_path):
                    worker_succeeded = True
                    break
                print(f"  worker failed (attempt {attempt}/3, exit={completed.returncode}); retrying")
            if not worker_succeeded:
                save_progress(rows)
                raise RuntimeError(f"SLEVE worker failed repeatedly for scale={scale}, n={exponent}")
            with open(result_path) as stream:
                row = json.load(stream)
            rows.append(row)
            save_progress(rows)
            print(f"  mean={row['mean_metric']:.6e}, min_dz={row['minimum_layer_m']:.2f}, feasible={row['feasible']}")
    candidates = [row for row in rows if row["feasible"]]
    if not candidates:
        raise RuntimeError("No feasible SLEVE candidate")
    best = min(candidates, key=lambda row: row["mean_metric"])
    save_progress(rows)
    config = {**best, "training_dataset": os.path.abspath(args.dataset), "target": target, "steps": steps}
    with open(os.path.join(args.output_dir, "best_sleve.json"), "w") as stream:
        json.dump(config, stream, indent=2)
    if not args.no_render:
        fig, axis = plt.subplots(figsize=(6.2, 4.5))
        feasible_rows = [row for row in rows if row["feasible"]]
        if target != "pgf_rest":
            points = axis.scatter(
                [row["scale_s"] / 1000.0 for row in feasible_rows],
                [row["n"] for row in feasible_rows],
                c=[row["mean_metric"] for row in feasible_rows],
                cmap="viridis",
                s=45,
            )
            axis.scatter(
                best["scale_s"] / 1000.0, best["n"], marker="*", s=180, facecolor="none", edgecolor="red", linewidth=1.5
            )
            axis.set(xlabel=r"SLEVE scale $s$ (km)", ylabel=r"SLEVE exponent $n$")
            fig.colorbar(
                points,
                ax=axis,
                label=(
                    r"Relative tracer round-trip $L_2$ error"
                    if target == "tracer_reversibility"
                    else r"Normalized momentum-flux non-uniformity"
                ),
            )
        else:
            axis.plot(
                [row["scale_s"] / 1000.0 for row in feasible_rows], [row["mean_metric"] for row in feasible_rows], "o-"
            )
            axis.scatter(
                best["scale_s"] / 1000.0,
                best["mean_metric"],
                marker="*",
                s=160,
                facecolor="none",
                edgecolor="red",
                linewidth=1.5,
            )
            axis.set(xlabel=r"SLEVE scale $s$ (km)", ylabel=r"Mean spurious TKE (m$^2$ s$^{-2}$)")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_dir / "sleve_tuning.png", dpi=250)
        plt.close(fig)
    print("Best SLEVE:\n" + json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
