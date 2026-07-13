#!/usr/bin/env python3
"""
Zero-Shot Generalization & Evaluation of PDE-Discovered NEUVE
============================================================
Evaluates the physics-guided NEUVE coordinate zero-shot on unseen rugged topography
(SEED = 999) and compares it against standard Gal-Chen and analytical SLEVE.
"""

import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from suetes.shared.transforms import GalChenSigma, SleveSimple, NEUVECoordinate
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite

from experiments.train_neuve_physics import (
    nx, ny, nz, dx, dy, dz, dt, constants, generate_rugged_terrain,
    get_wind_u, get_wind_v, make_simulation
)

TEST_SEED = 999
eval_steps = 120


def load_trained_neuve(weights_path):
    neuve_init = NEUVECoordinate(hidden_dim=64, key_seed=42)
    if not os.path.exists(weights_path):
        print(f"[WARNING] {weights_path} not found. Using untrained NEUVE template.")
        return neuve_init
    data = np.load(weights_path)
    arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    flat_params, tree_def = jax.tree_util.tree_flatten(neuve_init.params)
    restored_params = jax.tree_util.tree_unflatten(tree_def, arrays)
    return neuve_init.with_params(restored_params)


def eval_coordinate(name, transform_op, terrain_fn):
    print(f"  -> Running forward integration for [{name}]...")
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

    def scan_fn(st_curr, _):
        next_st = stepper.step(st_curr, 0.0, None, bc_fn)
        w_aloft = next_st['w'][:, :, 10:]  # z > 10 km
        noise_t = jnp.mean(w_aloft ** 2)
        return next_st, noise_t

    _, noise_series = jax.lax.scan(scan_fn, base_state, jnp.arange(eval_steps))
    return np.array(noise_series), grid


def main():
    os.makedirs("output/plots/neuve_physics", exist_ok=True)
    print("=========================================================================")
    print("SECTION 3.3.5: ZERO-SHOT GENERALIZATION ON UNSEEN RUGGED TOPOGRAPHY")
    print("=========================================================================")
    print(f"Test Seed: {TEST_SEED}")

    terrain_fn = generate_rugged_terrain(TEST_SEED)
    neuve_unconstrained = load_trained_neuve("output/trained_neuve_physics_unconstrained_weights.bin.npz")
    neuve_flat = load_trained_neuve("output/trained_neuve_physics_flat_aloft_weights.bin.npz")

    coords = {
        'Gal-Chen': GalChenSigma(),
        'SLEVE (s=3000m)': SleveSimple(scale_s=3000.0),
        'NEUVE (unconstrained)': neuve_unconstrained,
        'NEUVE (flat-aloft)': neuve_flat
    }

    results = {}
    grids = {}

    for name, transform in coords.items():
        noise_ts, grid = eval_coordinate(name, transform, terrain_fn)
        results[name] = noise_ts
        grids[name] = grid

    print("\n=========================================================================")
    print("Zero-Shot Spurious Acoustic Kinetic Energy Aloft (Mean over t=0..300s)")
    print("=========================================================================")
    print(f"{'Coordinate':<28} | {'Mean Aloft Noise w^2 (m^2/s^2)':<32} | {'Peak w^2':<15}")
    print("-" * 79)
    for name, ts in results.items():
        mean_val = np.mean(ts)
        peak_val = np.max(ts)
        print(f"{name:<28} | {mean_val:<32.6e} | {peak_val:<15.6e}")
    print("=========================================================================\n")

    # Figure 1: 2-Panel Time-Series of Spurious Acoustic Kinetic Energy Aloft
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    time_ax = np.arange(1, eval_steps + 1) * dt
    colors = {
        'Gal-Chen': '#E53E3E',
        'SLEVE (s=3000m)': '#DD6B20',
        'NEUVE (unconstrained)': '#3182CE',
        'NEUVE (flat-aloft)': '#38A169'
    }
    styles = {
        'Gal-Chen': '--',
        'SLEVE (s=3000m)': '-.',
        'NEUVE (unconstrained)': '-',
        'NEUVE (flat-aloft)': '-'
    }

    # Left Panel: Full Time Series (Log Scale)
    for name, ts in results.items():
        ax1.plot(time_ax, ts, label=name, color=colors[name], linestyle=styles[name], linewidth=1.3)
    ax1.set_title("Full simulation horizon (log scale)", fontsize=12, fontweight='normal')
    ax1.set_xlabel("Simulation time (s)", fontsize=11)
    ax1.set_ylabel("Mean upper-level kinetic energy $w^2$ ($m^2/s^2$)", fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # Right Panel: Mature Acoustic Regime Zoom-in (t >= 60s, Linear Scale)
    mask = time_ax >= 60.0
    for name, ts in results.items():
        ax2.plot(time_ax[mask], ts[mask], label=name, color=colors[name], linestyle=styles[name], linewidth=1.3)
    ax2.set_title(r"Final stratospheric acoustic regime ($t \geq 60$ s)", fontsize=12, fontweight='normal')
    ax2.set_xlabel("Simulation time (s)", fontsize=11)
    ax2.set_ylabel("Mean upper-level kinetic energy $w^2$ ($m^2/s^2$)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    plt.suptitle("Spurious stratospheric acoustic noise ($z > 10$ km)", fontsize=14, fontweight='normal')
    plt.tight_layout()
    f1_path = "output/plots/neuve_physics/neuve_zero_shot_acoustic_noise_aloft.png"
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved acoustic noise time-series plot to {f1_path}")

    # Figure 2: Vertical Coordinate Grid Lines Comparison (1x4 subplot)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    y_mid = ny // 2
    x_km = np.arange(nx) * dx / 1000.0

    for i, (name, grid) in enumerate(grids.items()):
        ax = axes[i]
        Z_slice = grid.Z_m[:, y_mid, :] / 1000.0  # (nx, nz)
        for k in range(nz):
            ax.plot(x_km, Z_slice[:, k], color='#2D3748', linewidth=1.0, alpha=0.85)

        # Plot terrain contour
        h_slice = Z_slice[:, 0]
        ax.fill_between(x_km, 0.0, h_slice, color='#718096', alpha=0.4, label='Topography')
        ax.plot(x_km, h_slice, color='#1A202C', linewidth=2.0)

        ax.set_title(f"{name}", fontsize=12, fontweight='normal')
        ax.set_xlabel("Horizontal distance $x$ (km)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Height $z$ (km)", fontsize=11)
        ax.set_ylim(0, 16.0)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle("Vertical grid structure comparison over rugged alpine topography", fontsize=14, fontweight='normal')
    plt.tight_layout()
    f2_path = "output/plots/neuve_physics/neuve_coordinate_grid_lines_comparison.png"
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved 4-coordinate grid line comparison to {f2_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
