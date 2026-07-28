#!/usr/bin/env python3
"""Train the learned-density NEUVE coordinate in one end-to-end stage."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments._shared.neuve_coordinate import (
    DX,
    DY,
    dataset_target,
    load_dataset,
    run_target_case,
    terrain_factory,
    terrains_from_dataset,
)
from experiments._shared.neuve_learned_coordinate import (
    LearnedDensityCoordinate,
    save_learned_coordinate,
    scalar_density_params,
)
from suetes.regional3d.geometry import RegionalGrid3D


def _copy_tree(tree):
    return jax.tree.map(jnp.array, tree)


def _grid(coordinate, terrain):
    return RegionalGrid3D(
        32,
        12,
        16,
        DX,
        DY,
        1000.0,
        lat_center=45.0,
        lon_center=0.0,
        h_func=terrain,
        transform=coordinate,
    )


def _initial_amplitude(terrains, basis_count, target_minimum, iterations=24):
    def minimum_layer(amplitude):
        coordinate = LearnedDensityCoordinate(
            scalar_density_params(amplitude, basis_count)
        )
        minima = []
        for terrain in terrains:
            try:
                minima.append(float(jnp.min(_grid(coordinate, terrain).dz_m_full)))
            except (ValueError, FloatingPointError):
                return -np.inf
        return min(minima)

    low, high = 0.0, 6.0
    if minimum_layer(low) < target_minimum:
        raise ValueError(
            "The neutral coordinate already violates the initialization "
            "minimum-layer target"
        )
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if minimum_layer(middle) >= target_minimum:
            low = middle
        else:
            high = middle
    return low, minimum_layer(low)


def _conditioned_params(global_params, basis_count, hidden, seed):
    key = jax.random.PRNGKey(seed)
    return {
        "global": global_params["global"],
        "conditioner": {
            "w1": 0.2 * jax.random.normal(key, (3, hidden)),
            "b1": jnp.zeros(hidden),
            "w2": jnp.zeros((hidden, basis_count)),
            "b2": jnp.zeros(basis_count),
        },
    }


def _optimizer(params, global_lr, conditioner_lr, epochs):
    global_schedule = optax.cosine_decay_schedule(global_lr, epochs, alpha=0.02)
    conditioner_schedule = optax.cosine_decay_schedule(
        conditioner_lr, epochs, alpha=0.02
    )
    labels = {
        "global": "global",
        "conditioner": jax.tree.map(lambda _: "conditioner", params["conditioner"]),
    }
    updates = optax.multi_transform(
        {
            "global": optax.adam(global_schedule),
            "conditioner": optax.adam(conditioner_schedule),
        },
        labels,
    )
    return optax.chain(optax.clip_by_global_norm(1.0), updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--basis-count", type=int, default=16)
    parser.add_argument("--conditioner-hidden", type=int, default=24)
    parser.add_argument("--global-lr", type=float, default=1.0e-3)
    parser.add_argument("--conditioner-lr", type=float, default=3.0e-3)
    parser.add_argument("--residual-scale", type=float, default=1.5)
    parser.add_argument("--minimum-layer-m", type=float, default=150.0)
    parser.add_argument(
        "--initial-minimum-layer-m",
        type=float,
        default=175.0,
        help="Geometry-only initialization target",
    )
    parser.add_argument("--barrier-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.initial_minimum_layer_m <= args.minimum_layer_m:
        parser.error("--initial-minimum-layer-m must exceed --minimum-layer-m")

    dataset = load_dataset(args.dataset)
    if dataset_target(dataset) != "pgf_rest":
        parser.error("Learned-density training currently supports pgf_rest")
    terrains = terrains_from_dataset(dataset)
    generator = terrain_factory(dataset["terrain_family"])
    seeds = tuple(dataset["seeds"])
    steps = int(dataset["integration"]["steps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    amplitude, initial_minimum = _initial_amplitude(
        terrains,
        args.basis_count,
        args.initial_minimum_layer_m,
    )
    global_params = scalar_density_params(jnp.asarray(amplitude), args.basis_count)
    params = _conditioned_params(
        global_params,
        args.basis_count,
        args.conditioner_hidden,
        args.seed,
    )
    print(f"Geometry initialization: a={amplitude:.6f}, min_dz={initial_minimum:.2f} m")

    def objective(current_params, terrain_seed):
        coordinate = LearnedDensityCoordinate(current_params, args.residual_scale)
        metric, _, grid, _ = run_target_case(
            "pgf_rest", coordinate, generator(terrain_seed), steps
        )
        minimum = jnp.min(grid.dz_m_full)
        violation = jax.nn.relu((args.minimum_layer_m - minimum) / args.minimum_layer_m)
        total = metric + args.barrier_weight * violation**2
        return total, (metric, minimum)

    value_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    optimizer = _optimizer(params, args.global_lr, args.conditioner_lr, args.epochs)
    optimizer_state = optimizer.init(params)
    best_params = _copy_tree(params)
    best_metric = np.inf
    history = []

    print(f"End-to-end NEUVE training on {len(seeds)} terrain sample(s)")
    print("epoch mean_TKE objective min_dz[m] time[s]")
    for epoch in range(args.epochs + 1):
        start = time.time()
        gradient_sum = jax.tree.map(jnp.zeros_like, params)
        objectives, metrics, minima = [], [], []
        for terrain_seed in seeds:
            (total, diagnostics), gradient = value_gradient(
                params, jnp.asarray(terrain_seed)
            )
            objectives.append(float(total))
            metrics.append(float(diagnostics[0]))
            minima.append(float(diagnostics[1]))
            gradient_sum = jax.tree.map(
                lambda accumulated, value: accumulated + value,
                gradient_sum,
                gradient,
            )
        mean_objective = float(np.mean(objectives))
        mean_metric = float(np.mean(metrics))
        minimum = float(np.min(minima))
        history.append((epoch, mean_metric, mean_objective, minimum))
        marker = ""
        if minimum >= args.minimum_layer_m and mean_metric < best_metric:
            best_metric = mean_metric
            best_params = _copy_tree(params)
            save_learned_coordinate(
                args.output_dir / "neuve_coordinate.npz",
                best_params,
                residual_scale=args.residual_scale,
                target="pgf_rest",
                training_mode="geometry_initialized_end_to_end",
            )
            marker = " checkpoint"
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"{epoch:5d} {mean_metric:.6e} {mean_objective:.6e} "
                f"{minimum:.2f} {time.time() - start:.2f}{marker}"
            )
        if epoch < args.epochs:
            gradient = jax.tree.map(lambda value: value / len(seeds), gradient_sum)
            updates, optimizer_state = optimizer.update(
                gradient, optimizer_state, params
            )
            params = optax.apply_updates(params, updates)

    history_array = np.asarray(history)
    np.savez(
        args.output_dir / "training_history.npz",
        epoch=history_array[:, 0],
        mean_tke=history_array[:, 1],
        objective=history_array[:, 2],
        minimum_layer_m=history_array[:, 3],
    )
    configuration = dict(vars(args))
    configuration["dataset"] = os.path.abspath(args.dataset)
    configuration["output_dir"] = str(args.output_dir)
    with open(args.output_dir / "training_summary.json", "w") as stream:
        json.dump(
            {
                "best_mean_tke": best_metric,
                "initial_amplitude": amplitude,
                "initial_minimum_layer_m": initial_minimum,
                "configuration": configuration,
            },
            stream,
            indent=2,
        )

    if not args.no_render:
        figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
        axes[0].plot(history_array[:, 0], history_array[:, 1])
        axes[0].set(
            xlabel="Epoch",
            ylabel=r"Spurious TKE (m$^2$ s$^{-2}$)",
        )
        axes[1].plot(history_array[:, 0], history_array[:, 3])
        axes[1].axhline(args.minimum_layer_m, color="k", linestyle="--")
        axes[1].set(xlabel="Epoch", ylabel="Minimum layer thickness (m)")
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(args.output_dir / "training.png", dpi=250)
        plt.close(figure)

    print(
        f"Saved {args.output_dir / 'neuve_coordinate.npz'} "
        f"(best mean TKE={best_metric:.6e})"
    )


if __name__ == "__main__":
    main()
