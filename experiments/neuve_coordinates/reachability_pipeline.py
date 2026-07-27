#!/usr/bin/env python3
"""Four-stage NEUVE capacity and optimization diagnostic.

This experiment separates representational capacity from optimizer
reachability:

1. fit the existing NEUVE MLP to the tuned SLEVE geometry;
2. learn a shared vertical density profile using a short PGF integration;
3. continue that profile with the full PGF integration;
4. add a bounded, terrain-conditioned residual and train on the dataset.

The SLEVE fit is an audit only.  None of its parameters or fitted NEUVE
weights initialize the self-supervised stages.
"""

import argparse
import gc
import json
import math
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
    DX,
    DY,
    load_dataset,
    neuve_template,
    run_target_case,
    save_neuve,
    terrains_from_dataset,
)
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.shared.transforms import BaseTransform, GalChenSigma, SleveSimple

_GL_X = jnp.asarray(
    [
        -0.9815606342467192,
        -0.9041172563704749,
        -0.7699026743093193,
        -0.5873179542866175,
        -0.3678314989981802,
        -0.1252334085114689,
        0.1252334085114689,
        0.3678314989981802,
        0.5873179542866175,
        0.7699026743093193,
        0.9041172563704749,
        0.9815606342467192,
    ]
)
_GL_W = jnp.asarray(
    [
        0.0471753363865118,
        0.1069393259953184,
        0.1600783285433462,
        0.2031674267230659,
        0.2334925365383548,
        0.2491470458134028,
        0.2491470458134028,
        0.2334925365383548,
        0.2031674267230659,
        0.1600783285433462,
        0.1069393259953184,
        0.0471753363865118,
    ]
)


def _bernstein(y, count):
    powers = jnp.arange(count)
    coefficients = jnp.asarray([math.comb(count - 1, k) for k in range(count)])
    return (
        coefficients
        * y[..., None] ** powers
        * (1.0 - y[..., None]) ** (count - 1 - powers)
    )


class DirectDensityCoordinate(BaseTransform):
    """Smooth learned coordinate expressed in vertical Bernstein coefficients."""

    def __init__(self, params, residual_scale=1.5):
        self.params = params
        self.residual_scale = residual_scale

    def _column_coefficients(self, xi, h):
        coefficients = self.params["global"]
        if "conditioner" not in self.params or h.ndim != 3:
            return coefficients
        h2 = h[:, :, 0] / 4000.0
        spacing = jnp.maximum(jnp.mean(jnp.abs(jnp.diff(xi[:, 0, 0]))), 1.0)
        hx = jnp.gradient(h[:, :, 0], axis=0) / spacing
        hy = jnp.gradient(h[:, :, 0], axis=1) / spacing
        slope = jnp.sqrt(hx**2 + hy**2 + 1.0e-12)
        curvature = (
            4000.0 * (jnp.gradient(hx, axis=0) + jnp.gradient(hy, axis=1)) / spacing
        )
        features = jnp.stack((h2, slope, curvature), axis=-1)
        network = self.params["conditioner"]
        hidden = jnp.tanh(features @ network["w1"] + network["b1"])
        residual = jnp.tanh(hidden @ network["w2"] + network["b2"])
        # Remove the vertically uniform component, which cancels from the
        # normalized density integral and otherwise creates a null direction.
        residual -= jnp.mean(residual, axis=-1, keepdims=True)
        return coefficients + self.residual_scale * residual

    def __call__(self, xi, zeta, h, Lz):
        eta = jnp.clip(zeta / Lz, 0.0, 1.0)
        coefficients = self._column_coefficients(xi, h)

        def density(y):
            basis = _bernstein(y, coefficients.shape[-1])
            if coefficients.ndim == 1:
                log_density = jnp.sum(basis * coefficients, axis=-1)
            elif y.ndim == 1:
                log_density = jnp.sum(
                    basis[None, None, ...] * coefficients[..., None, :],
                    axis=-1,
                )
            else:
                log_density = jnp.sum(basis * coefficients[..., None, None, :], axis=-1)
            return jnp.exp(jnp.clip(log_density, -6.0, 6.0))

        total_nodes = 0.5 * (_GL_X + 1.0)
        total = 0.5 * jnp.sum(_GL_W * density(total_nodes), axis=-1)
        partial_nodes = 0.5 * eta[..., None] * (_GL_X + 1.0)
        partial = 0.5 * eta * jnp.sum(_GL_W * density(partial_nodes), axis=-1)
        if jnp.ndim(total):
            total = total[..., None]
        cumulative = partial / total
        return zeta + h * (1.0 - cumulative)


def _grid(transform, terrain):
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
        transform=transform,
    )


def _copy_tree(tree):
    return jax.tree.map(lambda value: jnp.array(value), tree)


def _save_tree(path, params, **metadata):
    payload = {"global": np.asarray(params["global"])}
    if "conditioner" in params:
        payload.update(
            {
                f"conditioner_{name}": np.asarray(value)
                for name, value in params["conditioner"].items()
            }
        )
    np.savez(path, **payload, metadata=json.dumps(metadata))


def _load_tree(path):
    with np.load(path) as source:
        params = {"global": jnp.asarray(source["global"])}
        conditioner = {
            name.removeprefix("conditioner_"): jnp.asarray(source[name])
            for name in source.files
            if name.startswith("conditioner_")
        }
    if conditioner:
        params["conditioner"] = conditioner
    return params


def _release_compiled_state(*objects):
    """Release stage-specific XLA executables before compiling the next stage."""
    del objects
    gc.collect()
    jax.clear_caches()
    gc.collect()


def _optimizer(learning_rate, epochs):
    schedule = optax.cosine_decay_schedule(learning_rate, max(epochs, 1), alpha=0.02)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


def _fit_sleve_geometry(terrain, sleve, epochs, learning_rate, output_dir):
    template = neuve_template()
    target = _grid(sleve, terrain)
    target_z = jnp.asarray(target.Z_w)
    height_scale = jnp.maximum(jnp.max(target_z[..., 0]), 1.0)

    def objective(params):
        candidate = _grid(template.with_params(params), terrain)
        return jnp.mean(((candidate.Z_w - target_z) / height_scale) ** 2)

    value_gradient = jax.jit(jax.value_and_grad(objective))
    optimizer = _optimizer(learning_rate, epochs)
    params = template.params
    state = optimizer.init(params)
    history = []
    best_loss = np.inf
    best_params = _copy_tree(params)
    for epoch in range(epochs + 1):
        loss, gradient = value_gradient(params)
        value = float(loss)
        history.append(value)
        if value < best_loss:
            best_loss, best_params = value, _copy_tree(params)
        if epoch < epochs:
            updates, state = optimizer.update(gradient, state, params)
            params = optax.apply_updates(params, updates)
    save_neuve(output_dir / "stage1_mlp_sleve_fit.npz", best_params)
    fitted = template.with_params(best_params)
    fitted_grid = _grid(fitted, terrain)
    geometry_relative_l2 = float(
        jnp.linalg.norm(fitted_grid.Z_w - target_z) / jnp.linalg.norm(target_z)
    )
    return (
        fitted,
        history,
        {
            "fit_loss": best_loss,
            "geometry_relative_l2": geometry_relative_l2,
            "minimum_layer_m": float(jnp.min(fitted_grid.dz_m_full)),
        },
    )


def _train_direct(
    initial_params,
    terrains,
    steps,
    epochs,
    learning_rate,
    minimum_layer,
    barrier_weight,
    residual_scale,
    label,
):
    def make_function(terrain):
        def objective(params):
            transform = DirectDensityCoordinate(params, residual_scale)
            metric, _, grid, _ = run_target_case("pgf_rest", transform, terrain, steps)
            minimum = jnp.min(grid.dz_m_full)
            violation = jax.nn.relu((minimum_layer - minimum) / minimum_layer)
            return metric + barrier_weight * violation**2, (metric, minimum)

        return jax.jit(jax.value_and_grad(objective, has_aux=True))

    functions = [make_function(terrain) for terrain in terrains]
    optimizer = _optimizer(learning_rate, epochs)
    params = _copy_tree(initial_params)
    state = optimizer.init(params)
    best_params = _copy_tree(params)
    best_objective = np.inf
    history = []
    print(f"\n[{label}] steps={steps}, epochs={epochs}, samples={len(terrains)}")
    print("epoch metric objective min_dz[m] time[s]")
    for epoch in range(epochs + 1):
        start = time.time()
        gradient_sum = jax.tree.map(jnp.zeros_like, params)
        objectives, metrics, minima = [], [], []
        for function in functions:
            (objective, diagnostics), gradient = function(params)
            objectives.append(float(objective))
            metrics.append(float(diagnostics[0]))
            minima.append(float(diagnostics[1]))
            gradient_sum = jax.tree.map(
                lambda total, value: total + value,
                gradient_sum,
                gradient,
            )
        mean_objective = float(np.mean(objectives))
        mean_metric = float(np.mean(metrics))
        minimum = float(np.min(minima))
        history.append((epoch, mean_metric, mean_objective, minimum))
        if mean_objective < best_objective:
            best_objective = mean_objective
            best_params = _copy_tree(params)
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"{epoch:5d} {mean_metric:.6e} {mean_objective:.6e} "
                f"{minimum:.2f} {time.time() - start:.2f}"
            )
        if epoch < epochs:
            gradient = jax.tree.map(lambda value: value / len(functions), gradient_sum)
            updates, state = optimizer.update(gradient, state, params)
            params = optax.apply_updates(params, updates)
    return best_params, history


def _conditioned_params(global_params, basis_count, hidden, seed):
    key1 = jax.random.PRNGKey(seed)
    return {
        "global": global_params["global"],
        "conditioner": {
            "w1": 0.2 * jax.random.normal(key1, (3, hidden)),
            "b1": jnp.zeros((hidden,)),
            # Zero output preserves the learned global curve, while gradients
            # through w2 are nonzero on the first update.
            "w2": jnp.zeros((hidden, basis_count)),
            "b2": jnp.zeros((basis_count,)),
        },
    }


def _full_metrics(transforms, terrains, steps):
    result = {}
    for name, transform in transforms.items():
        values, minima = [], []
        for terrain in terrains:
            metric, _, grid, _ = run_target_case("pgf_rest", transform, terrain, steps)
            values.append(float(metric))
            minima.append(float(jnp.min(grid.dz_m_full)))
        result[name] = {
            "mean_tke": float(np.mean(values)),
            "std_tke": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum_layer_m": float(np.min(minima)),
            "samples": len(values),
        }
    return result


def _plot(output_dir, terrain, sleve, stages, histories, metrics):
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    for label, history in histories.items():
        axes[0, 0].plot(
            [row[0] for row in history],
            [row[1] for row in history],
            label=label,
        )
    axes[0, 0].set(xlabel="Epoch", ylabel=r"Spurious TKE (m$^2$ s$^{-2}$)")
    if histories:
        axes[0, 0].legend(frameon=False)
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "Training histories unavailable\nin reporting-only process",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )

    eta = jnp.linspace(0.0, 1.0, 200)
    for label, params in stages.items():
        coefficients = params["global"]
        density = jnp.exp(_bernstein(eta, len(coefficients)) @ coefficients)
        density /= jnp.trapezoid(density, eta)
        axes[0, 1].plot(np.asarray(density), np.asarray(eta), label=label)
    axes[0, 1].set(xlabel=r"Normalized layer density", ylabel=r"$\eta$")
    axes[0, 1].legend(frameon=False)

    transforms = {
        "Gal--Chen": GalChenSigma(),
        "SLEVE": sleve,
        "Global": DirectDensityCoordinate(stages["Stage 3"]),
        "Conditioned": DirectDensityCoordinate(stages["Stage 4"]),
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for coordinate_index, (name, transform) in enumerate(transforms.items()):
        grid = _grid(transform, terrain)
        x_values = np.asarray(grid.x_m)
        x = (x_values[:, grid.ny // 2] if x_values.ndim == 2 else x_values) / 1000.0
        z = np.asarray(grid.Z_w[:, grid.ny // 2, :]) / 1000.0
        for level in range(0, z.shape[-1], 2):
            axes[1, 0].plot(
                x,
                z[:, level],
                color=colors[coordinate_index % len(colors)],
                linewidth=0.6,
                alpha=0.75,
                label=name if level == 0 else None,
            )
    axes[1, 0].set(xlabel="x (km)", ylabel="Height (km)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    names = list(metrics)
    axes[1, 1].bar(
        np.arange(len(names)),
        [metrics[name]["mean_tke"] for name in names],
        yerr=[metrics[name]["std_tke"] for name in names],
    )
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[1, 1].set_ylabel(r"Spurious TKE (m$^2$ s$^{-2}$)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "reachability_pipeline.png", dpi=250)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sleve-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-count", type=int, default=16)
    parser.add_argument("--conditioner-hidden", type=int, default=24)
    parser.add_argument("--minimum-layer-m", type=float, default=150.0)
    parser.add_argument("--barrier-weight", type=float, default=100.0)
    parser.add_argument("--audit-epochs", type=int, default=200)
    parser.add_argument("--audit-lr", type=float, default=3.0e-3)
    parser.add_argument("--short-steps", type=int, default=25)
    parser.add_argument("--short-epochs", type=int, default=30)
    parser.add_argument("--short-lr", type=float, default=5.0e-2)
    parser.add_argument("--long-epochs", type=int, default=45)
    parser.add_argument("--long-lr", type=float, default=1.0e-2)
    parser.add_argument("--conditioned-epochs", type=int, default=45)
    parser.add_argument("--conditioned-lr", type=float, default=3.0e-3)
    parser.add_argument("--residual-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Resume from a saved stage in output-dir",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate final metrics and plots from saved stage checkpoints",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    if dataset.get("target", "pgf_rest") != "pgf_rest":
        parser.error("This diagnostic currently supports only pgf_rest")
    terrains = terrains_from_dataset(dataset)
    full_steps = int(dataset["integration"]["steps"])
    with open(args.sleve_config) as stream:
        sleve_config = json.load(stream)
    sleve = SleveSimple(
        scale_s=float(sleve_config["scale_s"]),
        scale_l=15000.0,
        n=float(sleve_config["n"]),
    )

    audit, audit_history = {}, []
    history2, history3, history4 = [], [], []
    progress_path = args.output_dir / "pipeline_progress.json"
    if args.report_only:
        if progress_path.exists():
            with open(progress_path) as stream:
                audit = json.load(stream).get("capacity_audit", {})
        stage2 = _load_tree(args.output_dir / "stage2_global_short.npz")
        stage3 = _load_tree(args.output_dir / "stage3_global_long.npz")
        stage4 = _load_tree(args.output_dir / "stage4_conditioned.npz")
        transforms = {
            "Gal-Chen": GalChenSigma(),
            "Tuned SLEVE": sleve,
            "Stage 2": DirectDensityCoordinate(stage2),
            "Stage 3": DirectDensityCoordinate(stage3),
            "Stage 4": DirectDensityCoordinate(stage4, args.residual_scale),
        }
        metrics = _full_metrics(transforms, terrains, full_steps)
        configuration = dict(vars(args))
        configuration["output_dir"] = str(args.output_dir)
        summary = {
            "dataset": os.path.abspath(args.dataset),
            "sleve_config": os.path.abspath(args.sleve_config),
            "capacity_audit": audit,
            "metrics": metrics,
            "configuration": configuration,
        }
        with open(args.output_dir / "summary.json", "w") as stream:
            json.dump(summary, stream, indent=2)
        _plot(
            args.output_dir,
            terrains[0],
            sleve,
            {"Stage 2": stage2, "Stage 3": stage3, "Stage 4": stage4},
            {},
            metrics,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Regenerated reports in {args.output_dir}")
        return

    if args.start_stage <= 1:
        print("[Stage 1] Existing-MLP capacity audit against tuned SLEVE")
        fitted, audit_history, audit = _fit_sleve_geometry(
            terrains[0],
            sleve,
            args.audit_epochs,
            args.audit_lr,
            args.output_dir,
        )
        audit_metric, _, audit_grid, _ = run_target_case(
            "pgf_rest", fitted, terrains[0], full_steps
        )
        audit["full_horizon_tke"] = float(audit_metric)
        audit["minimum_layer_m"] = float(jnp.min(audit_grid.dz_m_full))
        print(json.dumps(audit, indent=2))
        with open(progress_path, "w") as stream:
            json.dump({"capacity_audit": audit, "completed_stage": 1}, stream, indent=2)
        del fitted, audit_grid, audit_metric
        _release_compiled_state()
    elif progress_path.exists():
        with open(progress_path) as stream:
            audit = json.load(stream).get("capacity_audit", {})

    stage2_path = args.output_dir / "stage2_global_short.npz"
    if args.start_stage <= 2:
        initial = {"global": jnp.zeros((args.basis_count,))}
        stage2, history2 = _train_direct(
            initial,
            terrains,
            args.short_steps,
            args.short_epochs,
            args.short_lr,
            args.minimum_layer_m,
            args.barrier_weight,
            args.residual_scale,
            "Stage 2: direct global profile / short horizon",
        )
        _save_tree(stage2_path, stage2)
        with open(progress_path, "w") as stream:
            json.dump({"capacity_audit": audit, "completed_stage": 2}, stream, indent=2)
        _release_compiled_state()
    else:
        stage2 = _load_tree(stage2_path)

    stage3_path = args.output_dir / "stage3_global_long.npz"
    if args.start_stage <= 3:
        stage3, history3 = _train_direct(
            stage2,
            terrains,
            full_steps,
            args.long_epochs,
            args.long_lr,
            args.minimum_layer_m,
            args.barrier_weight,
            args.residual_scale,
            "Stage 3: global profile / full horizon",
        )
        _save_tree(stage3_path, stage3)
        with open(progress_path, "w") as stream:
            json.dump({"capacity_audit": audit, "completed_stage": 3}, stream, indent=2)
        _release_compiled_state()
    else:
        stage3 = _load_tree(stage3_path)

    conditioned_initial = _conditioned_params(
        stage3, args.basis_count, args.conditioner_hidden, args.seed
    )
    stage4, history4 = _train_direct(
        conditioned_initial,
        terrains,
        full_steps,
        args.conditioned_epochs,
        args.conditioned_lr,
        args.minimum_layer_m,
        args.barrier_weight,
        args.residual_scale,
        "Stage 4: terrain-conditioned residual",
    )
    _save_tree(args.output_dir / "stage4_conditioned.npz", stage4)
    with open(progress_path, "w") as stream:
        json.dump({"capacity_audit": audit, "completed_stage": 4}, stream, indent=2)
    _release_compiled_state()

    transforms = {
        "Gal-Chen": GalChenSigma(),
        "Tuned SLEVE": sleve,
        "Stage 2": DirectDensityCoordinate(stage2),
        "Stage 3": DirectDensityCoordinate(stage3),
        "Stage 4": DirectDensityCoordinate(stage4, args.residual_scale),
    }
    metrics = _full_metrics(transforms, terrains, full_steps)
    configuration = dict(vars(args))
    configuration["output_dir"] = str(args.output_dir)
    summary = {
        "dataset": os.path.abspath(args.dataset),
        "sleve_config": os.path.abspath(args.sleve_config),
        "capacity_audit": audit,
        "metrics": metrics,
        "configuration": configuration,
    }
    with open(args.output_dir / "summary.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    np.savez(
        args.output_dir / "training_history.npz",
        audit=np.asarray(audit_history),
        stage2=np.asarray(history2),
        stage3=np.asarray(history3),
        stage4=np.asarray(history4),
    )
    _plot(
        args.output_dir,
        terrains[0],
        sleve,
        {"Stage 2": stage2, "Stage 3": stage3, "Stage 4": stage4},
        {"Stage 2": history2, "Stage 3": history3, "Stage 4": history4},
        metrics,
    )
    print("\nFinal comparison")
    print(json.dumps(metrics, indent=2))
    print(f"Saved diagnostic pipeline to {args.output_dir}")


if __name__ == "__main__":
    main()
