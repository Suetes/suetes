#!/usr/bin/env python3
"""
Section 3.3.5: Physics-Guided Neural Coordinate Discovery
=========================================================
Trains the NEUVE vertical coordinate directly through the Suêtes atmospheric
dynamical core forward integration.

Instead of matching external ground truth or solving an inverse problem,
NEUVE learns to dynamically optimize vertical grid spacing by minimizing
spurious numerical acoustic/gravity wave noise aloft (z > 10 km) over steep terrain.
"""

import os
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
dt = 2.5
num_steps = 100

constants = {
    'g': 9.81, 'cp': 1004.0, 'cv': 717.0, 'Rd': 287.0,
    'p0': 100000.0, 'kappa': 287.0 / 1004.0, 'cvd': 717.0
}

def generate_rugged_terrain(seed_or_key):
    # Aggressive multi-peak mountain range with sharp peaks and deep valleys
    # Centered in the domain interior away from boundaries
    if isinstance(seed_or_key, int):
        key = jax.random.PRNGKey(seed_or_key)
    else:
        key = seed_or_key
    k1, k2, k3 = jax.random.split(key, 3)

    amp1 = 1800.0 + jax.random.uniform(k1, minval=-150.0, maxval=150.0)
    amp2 = 2300.0 + jax.random.uniform(k2, minval=-150.0, maxval=150.0)
    amp3 = 1500.0 + jax.random.uniform(k3, minval=-150.0, maxval=150.0)

    def terrain_fn(x, y):
        # Sharp Western Peak at x = -2500.0 m
        p1 = amp1 * jnp.exp(-((x + 2500.0) ** 2) / (2.0 * 750.0 ** 2))
        # Massive Central Horn exactly in the middle of the domain (x = 0.0 m)
        p2 = amp2 * jnp.exp(-(x ** 2) / (2.0 * 850.0 ** 2))
        p3 = amp3 * jnp.exp(-((x - 2500.0) ** 2) / (2.0 * 800.0 ** 2))
        return p1 + p2 + p3
    return terrain_fn

def get_wind_u(Z_u):
    # Strong impinging flow u = 15 m/s
    return jnp.full_like(Z_u, 15.0)

def get_wind_v(Z_v):
    return jnp.zeros_like(Z_v)


def make_simulation(phi_params, terrain_fn, neuve_template):
    transform_op = neuve_template.with_params(phi_params)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()

    physics = Euler3D(
        grid, op, constants, dt=dt, N_bv=0.01, damp_height=12000.0,
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
            'u': get_wind_u(grid.Z_u),
            'v': get_wind_v(grid.Z_v),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        return sponge.blend(state_in, ext_state)

    base_state = {
        'u': get_wind_u(grid.Z_u),
        'v': get_wind_v(grid.Z_v),
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
        damp_height=12000.0, max_damp=0.3, N_bv=0.01
    )
    stepper.use_checkpointing = True
    return grid, base_state, stepper, bc_fn


def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/plots/neuve_physics", exist_ok=True)

    print("=========================================================================")
    print("Physics-guided neural coordinate discovery")
    print("=========================================================================")
    print(f"Domain: nx={nx}, ny={ny}, nz={nz}, Lz={nz*dz:.0f} m | Impinging wind: u=15 m/s")

    neuve_init = NEUVECoordinate(hidden_dim=64, key_seed=42)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="unconstrained", choices=["unconstrained", "flat_aloft"])
    parser.add_argument("--num-topographies", type=int, default=4, help="Number of training topographies to generate via key split")
    parser.add_argument("--master-seed", type=int, default=1, help="Master PRNG seed for generating training topographies")
    args = parser.parse_args()

    master_key = jax.random.PRNGKey(args.master_seed)
    train_keys = jax.random.split(master_key, args.num_topographies)

    print(f"Training mode: [{args.mode}] | Num topographies: {args.num_topographies} | Master seed: {args.master_seed}")

    def physics_discovery_loss(phi, key):
        terrain_fn = generate_rugged_terrain(key)
        grid, base_state, stepper, bc_fn = make_simulation(phi, terrain_fn, neuve_init)

        @jax.checkpoint
        def step_fn(st_curr, _):
            next_st = stepper.step(st_curr, 0.0, None, bc_fn)
            w_aloft = next_st['w'][:, :, 10:]
            noise_t = jnp.mean(w_aloft ** 2)
            return next_st, noise_t

        _, noise_series = jax.lax.scan(step_fn, base_state, jnp.arange(num_steps))
        mean_acoustic_noise = jnp.mean(noise_series)

        d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)

        if args.mode == "flat_aloft":
            Z_aloft = grid.Z_m[:, :, 10:]
            dZ_dx_aloft = jnp.gradient(Z_aloft, axis=0)
            dZ_dy_aloft = jnp.gradient(Z_aloft, axis=1)
            metric_penalty_aloft = jnp.mean(dZ_dx_aloft ** 2) + jnp.mean(dZ_dy_aloft ** 2)
            return mean_acoustic_noise + 0.1 * metric_penalty_aloft + 1e-5 * jnp.sqrt(jnp.mean(d2Z_dx2 ** 2) + 1e-12)
        else:
            d2Z_dz2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=2), axis=2)
            smoothness_loss = jnp.mean(d2Z_dx2 ** 2) + jnp.mean(d2Z_dz2 ** 2)
            return mean_acoustic_noise + 5e-5 * jnp.sqrt(smoothness_loss + 1e-12)

    val_and_grad_fn = jax.jit(jax.value_and_grad(physics_discovery_loss))

    n_epochs = 20
    schedule = optax.cosine_decay_schedule(init_value=3e-3, decay_steps=n_epochs, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule)
    )

    phi = neuve_init.params
    opt_state = optimizer.init(phi)

    loss_history = []
    best_loss = float('inf')
    best_phi = phi

    print("\nStarting physics-driven training loop...")
    print(f"{'Epoch':<8} | {'Mean aloft noise w^2 (m^2/s^2)':<32} | {'Time (s)':<10}")
    print("-" * 55)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        epoch_grad = jax.tree.map(jnp.zeros_like, phi)

        for key in train_keys:
            lval, gval = val_and_grad_fn(phi, key)
            l_safe = jnp.where(jnp.isnan(lval), 0.0, lval)
            g_safe = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), gval)

            epoch_loss += float(l_safe)
            epoch_grad = jax.tree.map(lambda a, b: a + b, epoch_grad, g_safe)

        epoch_loss /= len(train_keys)
        epoch_grad = jax.tree.map(lambda g: g / len(train_keys), epoch_grad)

        updates, opt_state = optimizer.update(epoch_grad, opt_state, phi)
        phi = optax.apply_updates(phi, updates)

        loss_history.append(epoch_loss)
        dt_epoch = time.time() - t0

        print(f"{epoch:<8} | {epoch_loss:<32.6e} | {dt_epoch:<10.2f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_phi = phi

    # Save trained weights
    weights_path = f"output/trained_neuve_physics_{args.mode}_weights.bin"
    flat_params, tree_def = jax.tree_util.tree_flatten(best_phi)
    np.savez(weights_path, *[np.array(p) for p in flat_params])
    print(f"\n[SUCCESS] Saved PDE-discovered NEUVE weights to {weights_path}")

    # Save training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_epochs + 1), loss_history, marker='o', linewidth=2, color='#2B6CB0')
    plt.title("PDE numerical stability loss over training epochs", fontsize=13, fontweight='normal')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean aloft spurious kinetic energy $w^2$ ($m^2/s^2$)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = "output/plots/neuve_physics/neuve_training_physics_loss.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved training loss curve to {plot_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
