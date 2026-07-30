#!/usr/bin/env python3
# NEUVE training entry point.
"""Train spatial NEUVE for one of the supported physical objectives."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments._shared.neuve_coordinate import (
    GENERATOR_VERSION,
    TARGET_DEFAULTS,
    dataset_target,
    load_dataset,
    load_neuve,
    make_dataset,
    neuve_template,
    parse_seeds,
    save_neuve,
    run_target_case,
    terrains_from_dataset,
)
from suetes.shared.experiment import figure_dir_for, resolve_data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["pgf_rest", "tracer_reversibility", "mountain_flux"], default="pgf_rest")
    parser.add_argument("--dataset", help="Existing training dataset manifest")
    parser.add_argument("--terrain-family", choices=["ridge", "random3d", "multiscale3d"], default="ridge")
    parser.add_argument("--seeds", default="999", help="Comma-separated training seeds")
    parser.add_argument("--initial-weights")
    parser.add_argument("--epochs", type=int, default=90, help="Total updates (default: 30 discovery + 60 refinement)")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--discovery-epochs", type=int, default=30, help="Updates before resetting Adam for refinement")
    parser.add_argument("--refinement-lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--minimum-layer-m", type=float, default=100.0)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--name", default="default", help="Execution label")
    parser.add_argument("--output-dir", help="Explicit legacy output directory")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.discovery_epochs < 1:
        parser.error("--discovery-epochs must be positive")
    args.output_dir = str(
        resolve_data_dir(
            kind="experiments", case="neuve_coordinates", execution=args.name, output_root=args.output_root, output_dir=args.output_dir
        )
    )
    figure_dir = figure_dir_for(Path(args.output_dir) / "artifact.nc")
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "training_dataset.json")
    steps = args.steps or TARGET_DEFAULTS[args.target]["steps"]
    dataset = (
        load_dataset(args.dataset)
        if args.dataset
        else make_dataset(args.terrain_family, parse_seeds(args.seeds), "training", target=args.target, steps=steps)
    )
    target = dataset_target(dataset)
    if args.dataset and target != args.target:
        parser.error(f"dataset target is {target!r}, not requested {args.target!r}")
    steps = int(dataset["integration"]["steps"])
    make_dataset(dataset["terrain_family"], dataset["seeds"], "training", manifest_path, target=target, steps=steps)
    terrains = terrains_from_dataset(dataset)

    template = neuve_template()
    params = load_neuve(args.initial_weights).params if args.initial_weights else template.params

    def make_objective(terrain):
        def objective(current_params):
            transform = template.with_params(current_params)
            metric, _, grid, _ = run_target_case(target, transform, terrain, steps)
            minimum_layer = jnp.min(grid.dz_m_full)
            thin = jax.nn.relu((args.minimum_layer_m - minimum_layer) / args.minimum_layer_m) ** 2
            return metric + thin, (metric, minimum_layer)

        return jax.jit(jax.value_and_grad(objective, has_aux=True)), jax.jit(objective)

    functions = [make_objective(terrain) for terrain in terrains]
    discovery_epochs = min(args.discovery_epochs, args.epochs)

    def make_optimizer(learning_rate, updates):
        schedule = optax.cosine_decay_schedule(learning_rate, max(updates, 1), alpha=0.01)
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    optimizer = make_optimizer(args.lr, discovery_epochs)
    opt_state = optimizer.init(params)
    checkpoint = os.path.join(args.output_dir, "neuve_coordinate.npz")

    def evaluate(current_params):
        values = [fn(current_params) for _, fn in functions]
        losses = np.asarray([float(value[0]) for value in values])
        tkes = np.asarray([float(value[1][0]) for value in values])
        minimum = min(float(value[1][1]) for value in values)
        return float(np.mean(losses)), float(np.mean(tkes)), minimum

    best_loss, initial_tke, initial_minimum = evaluate(params)
    best_params = jax.tree.map(jnp.array, params)
    best_state = jax.tree.map(jnp.array, opt_state)
    save_neuve(checkpoint, best_params)
    history = {
        "epoch": [0],
        "mean_tke": [initial_tke],
        "current_objective": [best_loss],
        "best_objective": [best_loss],
        "minimum_layer_m": [initial_minimum],
    }
    metric_label = {"pgf_rest": "mean_TKE", "tracer_reversibility": "roundtrip_L2", "mountain_flux": "flux_nonuniformity"}[target]
    print(f"NEUVE {target} tuning on {len(terrains)} terrain sample(s)")
    print(f"epoch {metric_label} current_objective best_objective min_dz[m] time[s]")
    print(f"{0:5d} {initial_tke:.6e} {best_loss:.6e} {best_loss:.6e} {initial_minimum:.2f}")

    for epoch in range(1, args.epochs + 1):
        if epoch == discovery_epochs + 1 and epoch <= args.epochs:
            # The publication-quality fixed-domain solution was obtained by a
            # broad discovery stage followed by a fresh, lower-rate Adam pass.
            # Make that deterministic continuation part of one clean run.
            params = jax.tree.map(jnp.array, best_params)
            optimizer = make_optimizer(args.refinement_lr, args.epochs - discovery_epochs)
            opt_state = optimizer.init(params)
            best_state = jax.tree.map(jnp.array, opt_state)
            print(f"--- refinement stage: reset Adam, lr={args.refinement_lr:.3e}, starting from best checkpoint ---")
        start = time.time()
        gradient_sum = jax.tree.map(jnp.zeros_like, params)
        pre_losses = []
        pre_objectives = []
        pre_minimum_layers = []
        finite = True
        for value_grad, _ in functions:
            (loss, diagnostics), gradient = value_grad(params)
            finite = finite and bool(jnp.isfinite(loss)) and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
            if not finite:
                break
            pre_losses.append(float(diagnostics[0]))
            pre_objectives.append(float(loss))
            pre_minimum_layers.append(float(diagnostics[1]))
            gradient_sum = jax.tree.map(lambda a, b: a + b, gradient_sum, gradient)
        if not finite:
            params, opt_state = best_params, best_state
            history["epoch"].append(epoch)
            history["mean_tke"].append(np.nan)
            history["current_objective"].append(np.nan)
            history["best_objective"].append(best_loss)
            history["minimum_layer_m"].append(np.nan)
            print(f"{epoch:5d} REJECTED non-finite; rollback")
            continue
        gradient = jax.tree.map(lambda value: value / len(functions), gradient_sum)
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        params = optax.apply_updates(params, updates)
        marker = ""
        current_objective = float(np.mean(pre_objectives))
        mean_tke = float(np.mean(pre_losses))
        minimum = float(np.min(pre_minimum_layers))
        if epoch % 5 == 0 or epoch == args.epochs:
            candidate, mean_tke, minimum = evaluate(params)
            current_objective = candidate
            if np.isfinite(candidate) and candidate < best_loss:
                best_loss = candidate
                best_params = jax.tree.map(jnp.array, params)
                best_state = jax.tree.map(jnp.array, opt_state)
                save_neuve(checkpoint, best_params)
                marker = " checkpoint"
        history["epoch"].append(epoch)
        history["mean_tke"].append(mean_tke)
        history["current_objective"].append(current_objective)
        history["best_objective"].append(best_loss)
        history["minimum_layer_m"].append(minimum)
        print(f"{epoch:5d} {mean_tke:.6e} {current_objective:.6e} {best_loss:.6e} {minimum:.2f} {time.time() - start:.2f}{marker}")

    np.savez(
        os.path.join(args.output_dir, "neuve_training_history.npz"),
        **{key: np.asarray(value) for key, value in history.items()},
        seeds=np.asarray(dataset["seeds"]),
        terrain_family=dataset["terrain_family"],
        discovery_epochs=discovery_epochs,
        discovery_lr=args.lr,
        refinement_lr=args.refinement_lr,
        target=target,
        steps=steps,
    )
    with open(os.path.join(args.output_dir, "training_complete.json"), "w") as stream:
        json.dump(
            {
                "generator_version": GENERATOR_VERSION,
                "target": target,
                "terrain_family": dataset["terrain_family"],
                "seeds": dataset["seeds"],
                "steps": steps,
                "epochs": args.epochs,
                "best_objective": best_loss,
            },
            stream,
            indent=2,
        )
    if not args.no_render:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        epoch = np.asarray(history["epoch"])
        axes[0].semilogy(epoch, history["mean_tke"], label="Current metric")
        axes[0].semilogy(epoch, history["best_objective"], "--", label="Best feasible objective")
        metric_ylabel = {
            "pgf_rest": r"Mean spurious TKE (m$^2$ s$^{-2}$)",
            "tracer_reversibility": r"Relative tracer round-trip $L_2$ error",
            "mountain_flux": r"Normalized momentum-flux non-uniformity",
        }[target]
        axes[0].set(xlabel="Epoch", ylabel=metric_ylabel)
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        axes[1].plot(epoch, history["minimum_layer_m"])
        axes[1].axhline(args.minimum_layer_m, color="k", linestyle="--", label="Feasibility threshold")
        axes[1].set(xlabel="Epoch", ylabel="Minimum layer thickness (m)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(figure_dir / "neuve_training.png", dpi=250)
        plt.close(fig)
    print(f"Saved {checkpoint} and {manifest_path}")


if __name__ == "__main__":
    main()
