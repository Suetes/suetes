#!/usr/bin/env python3
"""Tune spatial NEUVE on a manifest containing one or N terrains."""

import argparse
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments.neuve_pgf_common import (
    DEFAULT_STEPS, build_case, grid_diagnostics, load_dataset, load_neuve,
    make_dataset, neuve_template, parse_seeds, physical_tke, save_neuve,
    terrains_from_dataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Existing training dataset manifest")
    parser.add_argument("--terrain-family", choices=["ridge", "random3d"], default="ridge")
    parser.add_argument("--seeds", default="999", help="Comma-separated training seeds")
    parser.add_argument("--initial-weights")
    parser.add_argument(
        "--epochs", type=int, default=90,
        help="Total updates (default: 30 discovery + 60 refinement)",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--discovery-epochs", type=int, default=30,
        help="Updates before resetting Adam for refinement",
    )
    parser.add_argument("--refinement-lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--minimum-layer-m", type=float, default=100.0)
    parser.add_argument("--output-dir", default="output/neuve_pgf_tuning")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.discovery_epochs < 1:
        parser.error("--discovery-epochs must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "training_dataset.json")
    dataset = (
        load_dataset(args.dataset) if args.dataset
        else make_dataset(args.terrain_family, parse_seeds(args.seeds), "training")
    )
    make_dataset(dataset["terrain_family"], dataset["seeds"], "training", manifest_path)
    terrains = terrains_from_dataset(dataset)

    template = neuve_template()
    params = load_neuve(args.initial_weights).params if args.initial_weights else template.params

    def make_objective(terrain):
        def objective(current_params):
            transform = template.with_params(current_params)
            grid, state, stepper, boundary = build_case(transform, terrain)

            @jax.checkpoint
            def scan(current, _):
                next_state = stepper.step(current, 0.0, None, boundary)
                return next_state, physical_tke(next_state, grid)

            _, series = jax.lax.scan(scan, state, jnp.arange(args.steps))
            mean_tke = jnp.mean(series)
            minimum_layer = jnp.min(grid.dz_m_full)
            thin = jax.nn.relu(
                (args.minimum_layer_m - minimum_layer) / args.minimum_layer_m
            ) ** 2
            return mean_tke + thin, (mean_tke, minimum_layer)
        return jax.jit(jax.value_and_grad(objective, has_aux=True)), jax.jit(objective)

    functions = [make_objective(terrain) for terrain in terrains]
    discovery_epochs = min(args.discovery_epochs, args.epochs)

    def make_optimizer(learning_rate, updates):
        schedule = optax.cosine_decay_schedule(
            learning_rate, max(updates, 1), alpha=0.01
        )
        return optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(schedule)
        )

    optimizer = make_optimizer(args.lr, discovery_epochs)
    opt_state = optimizer.init(params)
    checkpoint = os.path.join(args.output_dir, "neuve_pgf.npz")

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
        "epoch": [0], "mean_tke": [initial_tke],
        "current_objective": [best_loss], "best_objective": [best_loss],
        "minimum_layer_m": [initial_minimum],
    }
    print(f"NEUVE tuning on {len(terrains)} terrain sample(s)")
    print("epoch mean_TKE current_objective best_objective min_dz[m] time[s]")
    print(
        f"{0:5d} {initial_tke:.6e} {best_loss:.6e} "
        f"{best_loss:.6e} {initial_minimum:.2f}"
    )

    for epoch in range(1, args.epochs + 1):
        if epoch == discovery_epochs + 1 and epoch <= args.epochs:
            # The publication-quality fixed-domain solution was obtained by a
            # broad discovery stage followed by a fresh, lower-rate Adam pass.
            # Make that deterministic continuation part of one clean run.
            params = jax.tree.map(jnp.array, best_params)
            optimizer = make_optimizer(
                args.refinement_lr, args.epochs - discovery_epochs
            )
            opt_state = optimizer.init(params)
            best_state = jax.tree.map(jnp.array, opt_state)
            print(
                f"--- refinement stage: reset Adam, "
                f"lr={args.refinement_lr:.3e}, starting from best checkpoint ---"
            )
        start = time.time()
        gradient_sum = jax.tree.map(jnp.zeros_like, params)
        pre_losses = []
        pre_objectives = []
        pre_minimum_layers = []
        finite = True
        for value_grad, _ in functions:
            (loss, diagnostics), gradient = value_grad(params)
            finite = finite and bool(jnp.isfinite(loss)) and all(
                bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient)
            )
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
        print(
            f"{epoch:5d} {mean_tke:.6e} {current_objective:.6e} "
            f"{best_loss:.6e} {minimum:.2f} {time.time()-start:.2f}{marker}"
        )

    np.savez(
        os.path.join(args.output_dir, "neuve_training_history.npz"),
        **{key: np.asarray(value) for key, value in history.items()},
        seeds=np.asarray(dataset["seeds"]), terrain_family=dataset["terrain_family"],
        discovery_epochs=discovery_epochs, discovery_lr=args.lr,
        refinement_lr=args.refinement_lr,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    epoch = np.asarray(history["epoch"])
    axes[0].semilogy(epoch, history["mean_tke"], label="Current physical TKE")
    axes[0].semilogy(epoch, history["best_objective"], "--", label="Best feasible objective")
    axes[0].set(xlabel="Epoch", ylabel=r"Mean spurious TKE (m$^2$ s$^{-2}$)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].plot(epoch, history["minimum_layer_m"])
    axes[1].axhline(args.minimum_layer_m, color="k", linestyle="--", label="Feasibility threshold")
    axes[1].set(xlabel="Epoch", ylabel="Minimum layer thickness (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "neuve_training.png"), dpi=250)
    plt.close(fig)
    print(f"Saved {checkpoint} and {manifest_path}")


if __name__ == "__main__":
    main()
