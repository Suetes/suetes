#!/usr/bin/env python3
"""
Master Training Pipeline: Physics-Guided Neural Coordinate Discovery (NEUVE)
============================================================================
Discovers optimal vertical coordinate transformations by minimizing forward integration
errors through the Suêtes atmospheric dynamical core over multi-peak rugged terrain.

Supported Optimization Targets (`--target`):
  1. `pgf_rest`:
     Minimizes resting horizontal pressure gradient force (PGF) discretization error.
     Initializes atmosphere completely at rest (u=v=w=0) with strong stratification (N_bv=0.02).
     Objective: Total accumulated 3D spurious kinetic energy density (TKE) over time.
  2. `acoustic_energy`:
     Minimizes spurious acoustic/gravity wave noise aloft (z > 10 km) over steep terrain.
     Initializes atmosphere with strong impinging flow (u=15 m/s) and N_bv=0.01.
     Objective: Spurious vertical velocity variance w^2 aloft + optional metric regularity.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from suetes.shared.transforms import NEUVECoordinate
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite

# Domain configuration
nx, ny, nz = 32, 12, 16
dx, dy, dz = 500.0, 500.0, 1000.0  # Lz = 16,000 m

constants = {
    'g': 9.81, 'cp': 1004.0, 'cv': 717.0, 'Rd': 287.0,
    'p0': 100000.0, 'kappa': 287.0 / 1004.0, 'cvd': 717.0
}


def generate_rugged_terrain(seed_or_key):
    """Return a smooth, genuinely 3-D random multi-peak terrain.

    Peak amplitude, position, horizontal scales, and orientation all vary.
    This prevents NEUVE from memorizing the former fixed, x-only ridge family.
    """
    if isinstance(seed_or_key, int):
        key = jax.random.PRNGKey(seed_or_key)
    else:
        key = seed_or_key
    key_amp, key_x, key_y, key_sx, key_sy, key_angle = jax.random.split(key, 6)
    n_peaks = 4
    amplitudes = jax.random.uniform(key_amp, (n_peaks,), minval=900.0, maxval=2300.0)
    x_centres = jax.random.uniform(key_x, (n_peaks,), minval=-6000.0, maxval=6000.0)
    y_centres = jax.random.uniform(key_y, (n_peaks,), minval=-2200.0, maxval=2200.0)
    sigma_x = jax.random.uniform(key_sx, (n_peaks,), minval=650.0, maxval=1800.0)
    sigma_y = jax.random.uniform(key_sy, (n_peaks,), minval=650.0, maxval=1800.0)
    angles = jax.random.uniform(key_angle, (n_peaks,), minval=-jnp.pi, maxval=jnp.pi)

    def terrain_fn(x, y):
        terrain = jnp.zeros_like(x)
        for peak in range(n_peaks):
            x_rel = x - x_centres[peak]
            y_rel = y - y_centres[peak]
            c = jnp.cos(angles[peak])
            s = jnp.sin(angles[peak])
            x_rot = c * x_rel + s * y_rel
            y_rot = -s * x_rel + c * y_rel
            terrain = terrain + amplitudes[peak] * jnp.exp(
                -0.5 * ((x_rot / sigma_x[peak]) ** 2 + (y_rot / sigma_y[peak]) ** 2)
            )
        # Smoothly bound extreme peak overlap while retaining differentiable
        # slopes and a generous free-atmosphere depth.
        # Together with the learned-density range [0.3, 1.7], h < 2800 m and
        # Lz=16 km guarantee a positive vertical derivative for the decay map.
        return 2800.0 * jnp.tanh(terrain / 2800.0)
    return terrain_fn


def physical_cell_volumes(grid):
    """Physical mass-cell volumes, including map and vertical metrics."""
    return (
        grid.dx * grid.dy * grid.dz_m_full
        / grid.m_factors['m'][..., None] ** 2
    )


def pgf_rest_metric(state, grid, interior_width=4):
    """Volume-weighted mean kinetic energy generated from a resting state."""
    u_c = 0.5 * (state['u'][:-1, :, :] + state['u'][1:, :, :])
    v_c = 0.5 * (state['v'][:, :-1, :] + state['v'][:, 1:, :])
    w_c = 0.5 * (state['w'][:, :, :-1] + state['w'][:, :, 1:])
    energy = 0.5 * (u_c**2 + v_c**2 + w_c**2)
    volumes = physical_cell_volumes(grid)
    mask = jnp.zeros((grid.nx, grid.ny), dtype=volumes.dtype)
    mask = mask.at[
        interior_width:grid.nx-interior_width,
        interior_width:grid.ny-interior_width,
    ].set(1.0)
    weights = volumes * mask[..., None]
    return jnp.sum(weights * energy) / jnp.sum(weights)


def coordinate_regularity_penalty(grid, minimum_ratio=0.20, maximum_ratio=3.0):
    """Penalize excessively compressed or expanded physical layers."""
    column_depth = grid.Lz - grid.Z_w[:, :, 0]
    reference_dz = column_depth[..., None] / grid.nz
    layer_ratio = grid.dz_m_full / reference_dz
    thin = jax.nn.relu(minimum_ratio - layer_ratio)
    thick = jax.nn.relu(layer_ratio - maximum_ratio)
    return jnp.mean(thin**2 + thick**2)


def coordinate_shape_penalties(grid, flatness_start_fraction=0.40):
    """Return resolution-independent upper-level slope and curvature costs.

    The first term discourages terrain imprint above the lower troposphere.
    The second suppresses short horizontal bends throughout the coordinate.
    Both use physical derivatives rather than grid-index differences.
    """
    dz_dx = jnp.gradient(grid.Z_m, axis=0) / grid.dx
    dz_dy = jnp.gradient(grid.Z_m, axis=1) / grid.dy
    first_flat_level = int(flatness_start_fraction * grid.nz)
    flatness = jnp.mean(
        dz_dx[..., first_flat_level:] ** 2
        + dz_dy[..., first_flat_level:] ** 2
    )

    d2z_dx2 = jnp.gradient(dz_dx, axis=0) / grid.dx
    d2z_dy2 = jnp.gradient(dz_dy, axis=1) / grid.dy
    # Multiplication by the terrain scale makes curvature dimensionless.
    curvature = jnp.mean((4000.0 * (d2z_dx2 + d2z_dy2)) ** 2)
    return flatness, curvature


def load_neuve_weights(path, template_params):
    data = np.load(path)
    arrays = [data[f"arr_{index}"] for index in range(len(data.files))]
    flat_template, tree = jax.tree_util.tree_flatten(template_params)
    if len(arrays) != len(flat_template):
        raise ValueError(
            f"Checkpoint has {len(arrays)} arrays; expected {len(flat_template)}"
        )
    return jax.tree_util.tree_unflatten(
        tree, [jnp.asarray(value) for value in arrays]
    )


def make_simulation_case(target, phi_params, terrain_fn, neuve_template):
    """Creates grid, initial base state, stepper, and boundary conditions for target experiment."""
    transform_op = neuve_template.with_params(phi_params)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()

    if target == "pgf_rest":
        dt_val = 0.8
        N_bv_val = 0.02
        u_init = jnp.zeros_like(grid.Z_u)
        v_init = jnp.zeros_like(grid.Z_v)
    else:  # acoustic_energy
        dt_val = 2.5
        N_bv_val = 0.01
        u_init = jnp.full_like(grid.Z_u, 15.0)
        v_init = jnp.zeros_like(grid.Z_v)

    physics = Euler3D(
        grid, op, constants, dt=dt_val, N_bv=N_bv_val, damp_height=12000.0,
        max_damp=0.3, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite
    )

    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    sponge = BenchmarkSponge(nx, ny, sponge_depth=4, axes=('x', 'y'))

    def bc_fn(state_in, forcing=None):
        ext_state = {
            'u': u_init,
            'v': v_init,
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        return sponge.blend(state_in, ext_state)

    base_state = {
        'u': u_init,
        'v': v_init,
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v']
    }

    stepper, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=op, constants=constants,
        initial_state=base_state, physics_suite=suite,
        dt=dt_val, ns=20, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=12000.0, max_damp=0.3, N_bv=N_bv_val
    )
    stepper.use_checkpointing = True
    return grid, base_state, stepper, bc_fn, dt_val


def main():
    parser = argparse.ArgumentParser(description="Master Training Script for NEUVE Coordinate Discovery")
    parser.add_argument("--target", type=str, default="pgf_rest", choices=["pgf_rest", "acoustic_energy"],
                        help="Optimization target: pgf_rest (resting PGF error) or acoustic_energy (wave noise aloft)")
    parser.add_argument("--mode", type=str, default="unconstrained", choices=["unconstrained", "flat_aloft"],
                        help="Metric regularization mode for acoustic_energy target")
    parser.add_argument(
        "--terrain-mode", choices=["ensemble", "fixed"], default="ensemble",
        help=("Train across randomized terrains, or optimize one smooth decay "
              "for a fixed regional domain"),
    )
    parser.add_argument("--fixed-terrain-seed", type=int, default=999,
                        help="Terrain seed used when --terrain-mode=fixed")
    parser.add_argument(
        "--coordinate-mode", choices=["global", "spatial", "spatial-legacy"],
        default="global",
        help="Shared safe profile, or the successful fully spatial NEUVE architecture",
    )
    parser.add_argument("--initial-weights", default=None,
                        help="Optional compatible checkpoint for curriculum fine-tuning")
    parser.add_argument("--num-topographies", type=int, default=4, help="Number of training topographies via key split")
    parser.add_argument("--master-seed", type=int, default=1, help="Master PRNG seed")
    parser.add_argument("--validation-seed", type=int, default=10001,
                        help="Base seed for the fixed validation terrain set")
    parser.add_argument("--num-validation-topographies", type=int, default=4,
                        help="Number of fixed terrains used for checkpoint selection")
    parser.add_argument("--validation-interval", type=int, default=5,
                        help="Evaluate the fixed validation set every N epochs")
    parser.add_argument("--regularity-weight", type=float, default=1.0e-3,
                        help="Weight on the physical layer-thickness regularizer")
    parser.add_argument("--flatness-weight", type=float, default=0.25,
                        help="Weight on terrain imprint above 40%% of model height")
    parser.add_argument("--curvature-weight", type=float, default=1.0e-3,
                        help="Weight on dimensionless horizontal grid curvature")
    parser.add_argument("--minimum-layer-m", type=float, default=100.0,
                        help="Minimum acceptable physical layer thickness")
    parser.add_argument("--minimum-layer-weight", type=float, default=1.0,
                        help="Weight on the minimum-layer feasibility barrier")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (defaults: 140 for pgf_rest, 20 for acoustic_energy)")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate (defaults: 1e-3 for pgf_rest, 3e-3 for acoustic_energy)")
    parser.add_argument("--output-dir", type=str, default="output", help="Root output directory")
    args = parser.parse_args()

    n_epochs = args.epochs if args.epochs is not None else (140 if args.target == "pgf_rest" else 20)
    init_lr = args.lr if args.lr is not None else (1e-3 if args.target == "pgf_rest" else 3e-3)
    num_steps = 500 if args.target == "pgf_rest" else 100

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "weights")
    plots_dir = os.path.join(args.output_dir, "plots", "neuve")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if args.target == "pgf_rest":
        terrain_tag = (
            "ensemble" if args.terrain_mode == "ensemble"
            else f"fixed_seed{args.fixed_terrain_seed}"
        )
        profile_tag = (
            "spatial" if args.coordinate_mode in ("spatial", "spatial-legacy")
            else "global_decay_v5"
        )
        weights_filename = f"neuve_pgf_rest_{profile_tag}_{terrain_tag}.npz"
    else:
        weights_filename = f"neuve_{args.target}_{args.mode}.npz"
    weights_path = os.path.join(weights_dir, weights_filename)

    print("=========================================================================")
    print(f"NEUVE Master Coordinate Training | Target: [{args.target.upper()}]")
    print("=========================================================================")
    effective_topographies = (
        1 if args.terrain_mode == "fixed" else args.num_topographies
    )
    print(f"Domain: nx={nx}, ny={ny}, nz={nz}, Lz={nz*dz:.0f} m | Topographies: {effective_topographies}")
    if args.target == "pgf_rest":
        print("Initial state: AT REST (u=v=w=0) | Stratification: N_bv = 0.02 s^-1 | Steps: 500 (dt=0.8s)")
    else:
        print(f"Initial state: IMPINGING WIND (u=15 m/s) | Mode: {args.mode} | Stratification: N_bv = 0.01 s^-1 | Steps: 100 (dt=2.5s)")
    print(f"Hyperparameters: Epochs = {n_epochs}, Initial LR = {init_lr}, Master Seed = {args.master_seed}")
    profile_description = (
        "fully spatial neural decay"
        if args.coordinate_mode in ("spatial", "spatial-legacy")
        else "globally shared b(eta)"
    )
    print(
        f"Coordinate profile: {profile_description} | "
        f"Terrain mode: {args.terrain_mode}"
        + (f" (seed={args.fixed_terrain_seed})" if args.terrain_mode == "fixed" else "")
    )
    if args.terrain_mode == "ensemble":
        print(
            f"Validation: {args.num_validation_topographies} fixed terrains "
            f"(seed={args.validation_seed}), every {args.validation_interval} epochs"
        )
    else:
        print(
            f"Validation: fixed domain seed={args.fixed_terrain_seed}, "
            f"every {args.validation_interval} epochs"
        )

    neuve_init = NEUVECoordinate(
        hidden_dim=64,
        key_seed=42,
        condition_on_terrain=(
            args.target != "pgf_rest"
            or args.coordinate_mode in ("spatial", "spatial-legacy")
        ),
        legacy_behavior=(
            args.target == "pgf_rest"
            and args.coordinate_mode in ("spatial", "spatial-legacy")
        ),
    )
    if args.initial_weights:
        neuve_init = neuve_init.with_params(
            load_neuve_weights(args.initial_weights, neuve_init.params)
        )
        print(f"Initial checkpoint: {args.initial_weights}")
    master_key = jax.random.PRNGKey(args.master_seed)

    def discovery_loss(phi, key):
        terrain_key = (
            jax.random.PRNGKey(args.fixed_terrain_seed)
            if args.terrain_mode == "fixed" else key
        )
        terrain_fn = generate_rugged_terrain(terrain_key)
        grid, base_state, stepper, bc_fn, dt_val = make_simulation_case(args.target, phi, terrain_fn, neuve_init)

        @jax.checkpoint
        def step_fn(st_curr, _):
            next_st = stepper.step(st_curr, 0.0, None, bc_fn)
            if args.target == "pgf_rest":
                step_metric = pgf_rest_metric(next_st, grid)
            else:  # acoustic_energy
                w_aloft = next_st['w'][:, :, 10:]
                step_metric = jnp.mean(w_aloft ** 2)
            return next_st, step_metric

        _, metric_series = jax.lax.scan(step_fn, base_state, jnp.arange(num_steps))
        mean_metric = jnp.mean(metric_series)

        if args.target == "pgf_rest":
            flatness, curvature = coordinate_shape_penalties(grid)
            minimum_layer = jnp.min(grid.dz_m_full)
            thin_barrier = jax.nn.relu(
                (args.minimum_layer_m - minimum_layer)
                / args.minimum_layer_m
            ) ** 2
            return (
                mean_metric
                + args.regularity_weight * coordinate_regularity_penalty(grid)
                + args.flatness_weight * flatness
                + args.curvature_weight * curvature
                + args.minimum_layer_weight * thin_barrier
            )
        else:  # acoustic_energy
            d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)
            if args.mode == "flat_aloft":
                Z_aloft = grid.Z_m[:, :, 10:]
                dZ_dx_aloft = jnp.gradient(Z_aloft, axis=0)
                dZ_dy_aloft = jnp.gradient(Z_aloft, axis=1)
                metric_penalty_aloft = jnp.mean(dZ_dx_aloft ** 2) + jnp.mean(dZ_dy_aloft ** 2)
                return mean_metric + 0.1 * metric_penalty_aloft + 1e-5 * jnp.sqrt(jnp.mean(d2Z_dx2 ** 2) + 1e-12)
            else:
                d2Z_dz2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=2), axis=2)
                smoothness_loss = jnp.mean(d2Z_dx2 ** 2) + jnp.mean(d2Z_dz2 ** 2)
                return mean_metric + 5e-5 * jnp.sqrt(smoothness_loss + 1e-12)

    val_and_grad_fn = jax.jit(jax.value_and_grad(discovery_loss))
    validation_loss_fn = jax.jit(discovery_loss)

    schedule = optax.cosine_decay_schedule(init_value=init_lr, decay_steps=n_epochs, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule)
    )

    phi = neuve_init.params
    opt_state = optimizer.init(phi)

    loss_history = []
    validation_epochs = []
    validation_history = []
    best_loss = float('inf')
    best_phi = jax.tree.map(lambda value: jnp.array(value), phi)
    best_opt_state = jax.tree.map(lambda value: jnp.array(value), opt_state)
    validation_keys = (
        jnp.asarray([jax.random.PRNGKey(args.fixed_terrain_seed)])
        if args.terrain_mode == "fixed"
        else jax.random.split(
            jax.random.PRNGKey(args.validation_seed),
            args.num_validation_topographies,
        )
    )
    rejected_epochs = []

    metric_label = "Mean Spurious TKE (m^2/s^2)" if args.target == "pgf_rest" else "Mean Aloft Noise w^2 (m^2/s^2)"
    print(f"\nStarting optimization loop...")
    print(f"{'Epoch':<8} | {metric_label:<32} | {'Time (s)':<10}")
    print("-" * 55)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_grad = jax.tree.map(jnp.zeros_like, phi)
        epoch_is_finite = True

        epoch_keys = (
            jnp.asarray([jax.random.PRNGKey(args.fixed_terrain_seed)])
            if args.terrain_mode == "fixed"
            else jax.random.split(
                jax.random.fold_in(master_key, epoch),
                args.num_topographies,
            )
        )
        for key in epoch_keys:
            lval, gval = val_and_grad_fn(phi, key)
            loss_is_finite = bool(np.asarray(jnp.isfinite(lval)))
            grads_are_finite = all(
                bool(np.asarray(jnp.all(jnp.isfinite(leaf))))
                for leaf in jax.tree.leaves(gval)
            )
            if not loss_is_finite or not grads_are_finite:
                epoch_is_finite = False
                break

            epoch_loss += float(lval)
            epoch_grad = jax.tree.map(lambda a, b: a + b, epoch_grad, gval)

        if not epoch_is_finite:
            # Reject the complete stochastic epoch and restore a known-good
            # parameter *and optimizer* checkpoint.  No invalid gradient is
            # masked or partially applied.
            phi = jax.tree.map(lambda value: jnp.array(value), best_phi)
            opt_state = jax.tree.map(lambda value: jnp.array(value), best_opt_state)
            rejected_epochs.append(epoch)
            loss_history.append(np.nan)
            print(
                f"{epoch:<8} | {'REJECTED (non-finite)':<32} | "
                f"{time.time() - t0:<10.2f} | rollback"
            )
            continue

        epoch_loss /= len(epoch_keys)
        epoch_grad = jax.tree.map(lambda g: g / len(epoch_keys), epoch_grad)

        updates, opt_state = optimizer.update(epoch_grad, opt_state, phi)
        phi = optax.apply_updates(phi, updates)

        loss_history.append(epoch_loss)
        dt_epoch = time.time() - t0

        validation_due = (
            epoch == 1
            or epoch == n_epochs
            or epoch % args.validation_interval == 0
        )
        if validation_due:
            validation_values = [
                float(validation_loss_fn(phi, key)) for key in validation_keys
            ]
            if not np.all(np.isfinite(validation_values)):
                raise FloatingPointError(
                    f"Non-finite fixed-validation loss at epoch {epoch}."
                )
            validation_loss = float(np.mean(validation_values))
            validation_epochs.append(epoch)
            validation_history.append(validation_loss)
            validation_text = f" | val={validation_loss:.6e}"
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_phi = jax.tree.map(lambda value: jnp.array(value), phi)
                best_opt_state = jax.tree.map(lambda value: jnp.array(value), opt_state)
                flat_best, _ = jax.tree_util.tree_flatten(best_phi)
                np.savez(weights_path, *[np.array(p) for p in flat_best])
                validation_text += " | checkpoint"
            elif validation_loss > 5.0 * best_loss:
                # A rare terrain/optimizer interaction can produce a formally
                # finite but dynamically useless coordinate.  Reject the whole
                # optimizer state, rather than allowing Adam momentum to carry
                # the excursion into subsequent epochs.
                phi = jax.tree.map(lambda value: jnp.array(value), best_phi)
                opt_state = jax.tree.map(
                    lambda value: jnp.array(value), best_opt_state
                )
                validation_text += " | rollback"
        else:
            validation_text = ""

        print(
            f"{epoch:<8} | {epoch_loss:<32.6e} | {dt_epoch:<10.2f}"
            f"{validation_text}"
        )

    # Save trained weights
    flat_params, _ = jax.tree_util.tree_flatten(best_phi)
    np.savez(weights_path, *[np.array(p) for p in flat_params])
    print(f"\n[SUCCESS] Saved trained NEUVE weights to: {weights_path}")
    history_path = os.path.join(
        weights_dir,
        f"neuve_{args.target}_training_history.npz",
    )
    np.savez(
        history_path,
        training_loss=np.asarray(loss_history),
        validation_epochs=np.asarray(validation_epochs),
        validation_loss=np.asarray(validation_history),
        training_seed=args.master_seed,
        validation_seed=args.validation_seed,
        regularity_weight=args.regularity_weight,
        flatness_weight=args.flatness_weight,
        curvature_weight=args.curvature_weight,
        coordinate_mode=args.coordinate_mode,
        minimum_layer_m=args.minimum_layer_m,
        minimum_layer_weight=args.minimum_layer_weight,
        terrain_mode=args.terrain_mode,
        fixed_terrain_seed=args.fixed_terrain_seed,
        rejected_epochs=np.asarray(rejected_epochs),
    )
    print(f"[SUCCESS] Saved training/validation history to: {history_path}")

    # Retain the acoustic compatibility name only.  PGF weights trained with
    # the former, non-monotone mapping are intentionally not interchangeable.
    if args.target == "acoustic_energy":
        compat_path = f"output/trained_neuve_physics_{args.mode}_weights.bin.npz"
        np.savez(compat_path, *[np.array(p) for p in flat_params])

    # Save training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), loss_history, linewidth=2,
             label="stochastic training", color='#3182CE' if args.target == 'acoustic_energy' else '#38A169')
    if validation_history:
        plt.plot(validation_epochs, validation_history, marker='o', linewidth=2,
                 label="fixed validation", color='#DD6B20')
    plt.title(f"NEUVE Training Curve: {args.target.upper()} Optimization", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel(metric_label, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"train_{args.target}_{args.mode if args.target == 'acoustic_energy' else 'default'}_loss.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved training loss curve to: {plot_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
