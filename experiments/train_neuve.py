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
    """Generates multi-peak rugged terrain with sharp peaks and deep valleys."""
    if isinstance(seed_or_key, int):
        key = jax.random.PRNGKey(seed_or_key)
    else:
        key = seed_or_key
    k1, k2, k3 = jax.random.split(key, 3)

    amp1 = 1800.0 + jax.random.uniform(k1, minval=-150.0, maxval=150.0)
    amp2 = 2350.0 + jax.random.uniform(k2, minval=-150.0, maxval=150.0)
    amp3 = 1550.0 + jax.random.uniform(k3, minval=-150.0, maxval=150.0)

    def terrain_fn(x, y):
        p1 = amp1 * jnp.exp(-((x + 2500.0) ** 2) / (2.0 * 750.0 ** 2))
        p2 = amp2 * jnp.exp(-(x ** 2) / (2.0 * 850.0 ** 2))
        p3 = amp3 * jnp.exp(-((x - 2500.0) ** 2) / (2.0 * 800.0 ** 2))
        return p1 + p2 + p3
    return terrain_fn


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
    parser.add_argument("--num-topographies", type=int, default=4, help="Number of training topographies via key split")
    parser.add_argument("--master-seed", type=int, default=1, help="Master PRNG seed")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (defaults: 140 for pgf_rest, 20 for acoustic_energy)")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate (defaults: 5e-3 for pgf_rest, 3e-3 for acoustic_energy)")
    parser.add_argument("--output-dir", type=str, default="output", help="Root output directory")
    args = parser.parse_args()

    n_epochs = args.epochs if args.epochs is not None else (140 if args.target == "pgf_rest" else 20)
    init_lr = args.lr if args.lr is not None else (5e-3 if args.target == "pgf_rest" else 3e-3)
    num_steps = 500 if args.target == "pgf_rest" else 100

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(args.output_dir, "weights")
    plots_dir = os.path.join(args.output_dir, "plots", "neuve")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=========================================================================")
    print(f"NEUVE Master Coordinate Training | Target: [{args.target.upper()}]")
    print("=========================================================================")
    print(f"Domain: nx={nx}, ny={ny}, nz={nz}, Lz={nz*dz:.0f} m | Topographies: {args.num_topographies}")
    if args.target == "pgf_rest":
        print("Initial state: AT REST (u=v=w=0) | Stratification: N_bv = 0.02 s^-1 | Steps: 500 (dt=0.8s)")
    else:
        print(f"Initial state: IMPINGING WIND (u=15 m/s) | Mode: {args.mode} | Stratification: N_bv = 0.01 s^-1 | Steps: 100 (dt=2.5s)")
    print(f"Hyperparameters: Epochs = {n_epochs}, Initial LR = {init_lr}, Master Seed = {args.master_seed}")

    neuve_init = NEUVECoordinate(hidden_dim=64, key_seed=42)
    master_key = jax.random.PRNGKey(args.master_seed)

    def discovery_loss(phi, key):
        terrain_fn = generate_rugged_terrain(key)
        grid, base_state, stepper, bc_fn, dt_val = make_simulation_case(args.target, phi, terrain_fn, neuve_init)

        @jax.checkpoint
        def step_fn(st_curr, _):
            next_st = stepper.step(st_curr, 0.0, None, bc_fn)
            if args.target == "pgf_rest":
                u_c = 0.5 * (next_st['u'][:-1, :, :] + next_st['u'][1:, :, :])
                v_c = 0.5 * (next_st['v'][:, :-1, :] + next_st['v'][:, 1:, :])
                w_c = 0.5 * (next_st['w'][:, :, :-1] + next_st['w'][:, :, 1:])
                step_metric = 0.5 * jnp.mean(u_c**2 + v_c**2 + w_c**2)
            else:  # acoustic_energy
                w_aloft = next_st['w'][:, :, 10:]
                step_metric = jnp.mean(w_aloft ** 2)
            return next_st, step_metric

        _, metric_series = jax.lax.scan(step_fn, base_state, jnp.arange(num_steps))
        mean_metric = jnp.mean(metric_series)

        d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)

        if args.target == "pgf_rest":
            smoothness_loss = jnp.mean(d2Z_dx2 ** 2)
            return mean_metric + 1e-6 * jnp.sqrt(smoothness_loss + 1e-12)
        else:  # acoustic_energy
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

    schedule = optax.cosine_decay_schedule(init_value=init_lr, decay_steps=n_epochs, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule)
    )

    phi = neuve_init.params
    opt_state = optimizer.init(phi)

    loss_history = []
    best_loss = float('inf')
    best_phi = phi

    metric_label = "Mean Spurious TKE (m^2/s^2)" if args.target == "pgf_rest" else "Mean Aloft Noise w^2 (m^2/s^2)"
    print(f"\nStarting optimization loop...")
    print(f"{'Epoch':<8} | {metric_label:<32} | {'Time (s)':<10}")
    print("-" * 55)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_grad = jax.tree.map(jnp.zeros_like, phi)

        epoch_keys = jax.random.split(jax.random.fold_in(master_key, epoch), args.num_topographies)
        for key in epoch_keys:
            lval, gval = val_and_grad_fn(phi, key)
            l_safe = jnp.where(jnp.isnan(lval) | (lval <= 0), 10.0 if args.target == "pgf_rest" else 0.0, lval)
            g_safe = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), gval)

            epoch_loss += float(l_safe)
            epoch_grad = jax.tree.map(lambda a, b: a + b, epoch_grad, g_safe)

        epoch_loss /= len(epoch_keys)
        epoch_grad = jax.tree.map(lambda g: g / len(epoch_keys), epoch_grad)

        updates, opt_state = optimizer.update(epoch_grad, opt_state, phi)
        phi = optax.apply_updates(phi, updates)

        loss_history.append(epoch_loss)
        dt_epoch = time.time() - t0

        print(f"{epoch:<8} | {epoch_loss:<32.6e} | {dt_epoch:<10.2f}")

        if 0 < epoch_loss < best_loss:
            best_loss = epoch_loss
            best_phi = phi

    # Save trained weights
    weights_filename = f"neuve_{args.target}_{args.mode if args.target == 'acoustic_energy' else 'default'}.npz"
    weights_path = os.path.join(weights_dir, weights_filename)
    flat_params, _ = jax.tree_util.tree_flatten(best_phi)
    np.savez(weights_path, *[np.array(p) for p in flat_params])
    print(f"\n[SUCCESS] Saved trained NEUVE weights to: {weights_path}")

    # Also save to backward-compatible names used by evaluation scripts if default options
    if args.target == "pgf_rest":
        compat_path = "output/trained_neuve_physics_pgf_rest_weights.bin.npz"
        np.savez(compat_path, *[np.array(p) for p in flat_params])
    elif args.target == "acoustic_energy":
        compat_path = f"output/trained_neuve_physics_{args.mode}_weights.bin.npz"
        np.savez(compat_path, *[np.array(p) for p in flat_params])

    # Save training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), loss_history, marker='o', linewidth=2, color='#3182CE' if args.target == 'acoustic_energy' else '#38A169')
    plt.title(f"NEUVE Training Curve: {args.target.upper()} Optimization", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel(metric_label, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = f"train_{args.target}_{args.mode if args.target == 'acoustic_energy' else 'default'}_loss.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved training loss curve to: {plot_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
