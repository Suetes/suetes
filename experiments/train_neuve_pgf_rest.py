#!/usr/bin/env python3
"""
Physics-Guided Neural Coordinate Discovery: Resting PGF Error Minimization
==========================================================================
Discovers the NEUVE vertical coordinate by directly minimizing the horizontal
pressure gradient force (PGF) discretization error at rest over rugged topography.

When an atmosphere is initialized completely at rest (u=0, v=0, w=0) with strong
vertical stratification over steep terrain, numerical truncation errors along
sloping coordinate surfaces prevent exact cancellation of the pressure gradient
terms. This generates spurious circulation and artificial kinetic energy over time.

By choosing total accumulated 3D kinetic energy from rest as the objective function:
    L(phi) = (1 / N_steps) * sum_{t=1}^{N_steps} TKE(t)
NEUVE discovers an optimal coordinate transformation that minimizes resting PGF errors.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
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
dt = 0.8
num_steps = 500

constants = {
    'g': 9.81, 'cp': 1004.0, 'cv': 717.0, 'Rd': 287.0,
    'p0': 100000.0, 'kappa': 287.0 / 1004.0, 'cvd': 717.0
}

def generate_rugged_terrain(seed_or_key):
    """Generates multi-peak rugged terrain with steep slopes."""
    if isinstance(seed_or_key, int):
        key = jax.random.PRNGKey(seed_or_key)
    else:
        key = seed_or_key
    k1, k2, k3 = jax.random.split(key, 3)

    amp1 = 1800.0 + jax.random.uniform(k1, minval=-150.0, maxval=150.0)
    amp2 = 2400.0 + jax.random.uniform(k2, minval=-150.0, maxval=150.0)
    amp3 = 1600.0 + jax.random.uniform(k3, minval=-150.0, maxval=150.0)

    def terrain_fn(x, y):
        # Sharp Western Peak at x = -2500.0 m
        p1 = amp1 * jnp.exp(-((x + 2500.0) ** 2) / (2.0 * 750.0 ** 2))
        # Massive Central Horn exactly in the middle of the domain (x = 0.0 m)
        p2 = amp2 * jnp.exp(-(x ** 2) / (2.0 * 850.0 ** 2))
        p3 = amp3 * jnp.exp(-((x - 2500.0) ** 2) / (2.0 * 800.0 ** 2))
        return p1 + p2 + p3
    return terrain_fn

def make_resting_simulation(phi_params, terrain_fn, neuve_template):
    transform_op = neuve_template.with_params(phi_params)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()

    # Strong vertical stratification N_bv = 0.02 s^{-1} to amplify resting PGF sensitivity
    physics = Euler3D(
        grid, op, constants, dt=dt, N_bv=0.02, damp_height=12000.0,
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
            'u': jnp.zeros_like(grid.Z_u),
            'v': jnp.zeros_like(grid.Z_v),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        return sponge.blend(state_in, ext_state)

    # Completely at rest: u=0, v=0, w=0 everywhere
    base_state = {
        'u': jnp.zeros_like(grid.Z_u),
        'v': jnp.zeros_like(grid.Z_v),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v']
    }

    stepper, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=op, constants=constants,
        initial_state=base_state, physics_suite=suite,
        dt=dt, ns=20, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=12000.0, max_damp=0.3, N_bv=0.02
    )
    stepper.use_checkpointing = True
    return grid, base_state, stepper, bc_fn


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/plots/neuve_physics", exist_ok=True)

    print("=========================================================================")
    print("Physics-Guided Neural Coordinate Discovery: Resting PGF Minimization")
    print("=========================================================================")
    print(f"Domain: nx={nx}, ny={ny}, nz={nz}, Lz={nz*dz:.0f} m | Initial state: AT REST (u=0, v=0, w=0)")
    print("Stratification: N_bv = 0.02 s^-1 (strong vertical stability)")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-topographies", type=int, default=4, help="Number of training topographies")
    parser.add_argument("--master-seed", type=int, default=1, help="Master PRNG seed")
    parser.add_argument("--epochs", type=int, default=140, help="Training epochs")
    args = parser.parse_args()

    neuve_init = NEUVECoordinate(hidden_dim=64, key_seed=42)

    master_key = jax.random.PRNGKey(args.master_seed)
    train_keys = jax.random.split(master_key, args.num_topographies)

    def pgf_discovery_loss(phi, key):
        terrain_fn = generate_rugged_terrain(key)
        grid, base_state, stepper, bc_fn = make_resting_simulation(phi, terrain_fn, neuve_init)

        @jax.checkpoint
        def step_fn(st_curr, _):
            next_st = stepper.step(st_curr, 0.0, None, bc_fn)
            # Average staggered velocities to cell centers to calculate total 3D kinetic energy density
            u_c = 0.5 * (next_st['u'][:-1, :, :] + next_st['u'][1:, :, :])
            v_c = 0.5 * (next_st['v'][:, :-1, :] + next_st['v'][:, 1:, :])
            w_c = 0.5 * (next_st['w'][:, :, :-1] + next_st['w'][:, :, 1:])
            
            # Spurious kinetic energy density w.r.t. resting state (u=v=w=0)
            tke_density = 0.5 * jnp.mean(u_c**2 + v_c**2 + w_c**2)
            return next_st, tke_density

        _, tke_series = jax.lax.scan(step_fn, base_state, jnp.arange(num_steps))
        mean_spurious_tke = jnp.mean(tke_series)

        # Smoothness regularization (only penalize horizontal roughness so NEUVE can discover rapid vertical decay without any penalty)
        d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)
        smoothness_loss = jnp.mean(d2Z_dx2 ** 2)

        return mean_spurious_tke + 1e-6 * jnp.sqrt(smoothness_loss + 1e-12)

    val_and_grad_fn = jax.jit(jax.value_and_grad(pgf_discovery_loss))

    n_epochs = args.epochs
    schedule = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=n_epochs, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule)
    )

    phi = neuve_init.params
    opt_state = optimizer.init(phi)

    loss_history = []
    best_loss = float('inf')
    best_phi = phi

    print("\nStarting PGF training loop across rugged topographies...")
    print(f"{'Epoch':<8} | {'Mean Spurious TKE (m^2/s^2)':<32} | {'Time (s)':<10}")
    print("-" * 55)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_grad = jax.tree.map(jnp.zeros_like, phi)

        epoch_keys = jax.random.split(jax.random.fold_in(master_key, epoch), args.num_topographies)
        for key in epoch_keys:
            lval, gval = val_and_grad_fn(phi, key)
            l_safe = jnp.where(jnp.isnan(lval) | (lval <= 0), 10.0, lval)
            g_safe = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), gval)

            epoch_loss += float(l_safe)
            epoch_grad = jax.tree.map(lambda a, b: a + b, epoch_grad, g_safe)

        epoch_loss /= len(train_keys)
        epoch_grad = jax.tree.map(lambda g: g / len(train_keys), epoch_grad)

        updates, opt_state = optimizer.update(epoch_grad, opt_state, phi)
        phi = optax.apply_updates(phi, updates)

        loss_history.append(epoch_loss)
        dt_epoch = time.time() - t0

        print(f"{epoch:<8} | {epoch_loss:<32.6e} | {dt_epoch:<10.2f}")

        if 0 < epoch_loss < best_loss:
            best_loss = epoch_loss
            best_phi = phi

    # Save trained weights
    weights_path = "output/trained_neuve_physics_pgf_rest_weights.bin.npz"
    flat_params, tree_def = jax.tree_util.tree_flatten(best_phi)
    np.savez(weights_path, *[np.array(p) for p in flat_params])
    print(f"\n[SUCCESS] Saved PDE-discovered PGF NEUVE weights to {weights_path}")

    # Save training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), loss_history, marker='o', linewidth=2, color='#D69E2E')
    plt.title("Resting PGF Error Minimization over Training Epochs", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Accumulated Spurious TKE ($m^2/s^2$)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = "output/plots/neuve_physics/neuve_training_pgf_rest_loss.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved PGF training loss curve to {plot_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
