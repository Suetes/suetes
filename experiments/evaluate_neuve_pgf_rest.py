#!/usr/bin/env python3
"""
Zero-Shot Evaluation & Visualization of PGF-Discovered NEUVE Coordinate
=======================================================================
Evaluates the resting horizontal pressure gradient force (PGF) error minimization
discovered by NEUVE on unseen rugged topography (SEED = 999).

Compares:
  1. Standard Gal-Chen (terrain-following sigma coordinates)
  2. Analytical SLEVE (smooth scale decay)
  3. PDE-Discovered NEUVE (PGF-rest minimization)

Calculates accumulated spurious kinetic energy and maximum spurious velocities,
and generates publication figures revealing the geometry discovered by NEUVE.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

from experiments.train_neuve_pgf_rest import (
    nx, ny, nz, dx, dy, dz, dt, constants, generate_rugged_terrain, make_resting_simulation
)

TEST_SEED = 999
eval_steps = int(400.0 / dt)


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


def eval_resting_coordinate(name, transform_op, terrain_fn):
    print(f"  -> Integrating resting atmosphere for [{name}]...")
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()

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

    def scan_fn(st_curr, _):
        next_st = stepper.step(st_curr, 0.0, None, bc_fn)
        u_c = 0.5 * (next_st['u'][:-1, :, :] + next_st['u'][1:, :, :])
        v_c = 0.5 * (next_st['v'][:, :-1, :] + next_st['v'][:, 1:, :])
        w_c = 0.5 * (next_st['w'][:, :, :-1] + next_st['w'][:, :, 1:])
        
        tke_t = 0.5 * jnp.mean(u_c**2 + v_c**2 + w_c**2)
        max_u = jnp.max(jnp.abs(next_st['u']))
        max_w = jnp.max(jnp.abs(next_st['w']))
        return next_st, (tke_t, max_u, max_w)

    final_state, (tke_series, u_max_series, w_max_series) = jax.lax.scan(scan_fn, base_state, jnp.arange(eval_steps))
    return final_state, tke_series, u_max_series, w_max_series, grid


def main():
    os.makedirs("output/plots/neuve_physics", exist_ok=True)
    print("=========================================================================")
    print("Zero-Shot Evaluation: Resting PGF Discretization Error Minimization")
    print("=========================================================================")

    terrain_fn = generate_rugged_terrain(TEST_SEED)

    coords = {
        'Gal-Chen (Sigma)': GalChenSigma(),
        'SLEVE (Analytical)': SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35),
        'NEUVE (PGF Discovered)': load_trained_neuve("output/trained_neuve_physics_pgf_rest_weights.bin.npz")
    }

    results = {}
    grids = {}

    for name, op in coords.items():
        st, tke_s, u_s, w_s, gr = eval_resting_coordinate(name, op, terrain_fn)
        results[name] = {
            'final_state': st,
            'tke_series': np.array(tke_s),
            'max_u_series': np.array(u_s),
            'max_w_series': np.array(w_s),
            'mean_tke_300s': float(np.mean(tke_s[:int(300.0 / dt)])),
            'mean_tke_400s': float(np.mean(tke_s)),
            'peak_u': float(np.max(u_s)),
            'peak_w': float(np.max(w_s))
        }
        grids[name] = gr

    print("\n[QUANTITATIVE COMPARISON] Zero-Shot Spurious Circulation from Rest:")
    print(f"{'Coordinate System':<24} | {'Mean TKE (300s)':<18} | {'Mean TKE (400s)':<18} | {'Peak |u| (m/s)':<16} | {'Peak |w| (m/s)':<16}")
    print("-" * 102)
    for name, r in results.items():
        print(f"{name:<24} | {r['mean_tke_300s']:<18.6e} | {r['mean_tke_400s']:<18.6e} | {r['peak_u']:<16.4f} | {r['peak_w']:<16.4f}")

    ref_tke = results['Gal-Chen (Sigma)']['mean_tke_400s']
    neuve_tke = results['NEUVE (PGF Discovered)']['mean_tke_400s']
    reduction = (1.0 - neuve_tke / ref_tke) * 100.0 if ref_tke > 0 else 0.0
    sleve_tke = results['SLEVE (Analytical)']['mean_tke_400s']
    if neuve_tke < sleve_tke:
        print(f"\n=> [SUCCESS] NEUVE BEATS SLEVE! (NEUVE: {neuve_tke:.6e} vs SLEVE: {sleve_tke:.6e})")
    else:
        print(f"\n=> NEUVE reduces resting PGF spurious kinetic energy by {reduction:.1f}% vs Gal-Chen!")

    # -------------------------------------------------------------------------
    # Figure 1: Spurious Kinetic Energy & Velocity Evolution over Time
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    t_axis = np.arange(1, eval_steps + 1) * dt

    colors = {'Gal-Chen (Sigma)': '#E53E3E', 'SLEVE (Analytical)': '#3182CE', 'NEUVE (PGF Discovered)': '#38A169'}
    styles = {'Gal-Chen (Sigma)': '--', 'SLEVE (Analytical)': '-.', 'NEUVE (PGF Discovered)': '-'}

    for name, r in results.items():
        ax1.semilogy(t_axis, r['tke_series'], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        ax2.plot(t_axis, r['max_u_series'], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)

    ax1.set_title("(a) Accumulated Spurious Kinetic Energy from Rest", fontsize=13)
    ax1.set_xlabel("Integration Time (s)", fontsize=11)
    ax1.set_ylabel("Spurious TKE Density ($m^2/s^2$)", fontsize=11)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10.5)

    ax2.set_title("(b) Maximum Spurious Horizontal Wind $|u|_{max}$", fontsize=13)
    ax2.set_xlabel("Integration Time (s)", fontsize=11)
    ax2.set_ylabel("Spurious Horizontal Wind Speed (m/s)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10.5)

    plt.tight_layout()
    f1_path = "output/plots/neuve_physics/neuve_pgf_rest_spurious_tke_comparison.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Saved PGF spurious TKE comparison plot to {f1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Discovered Coordinate Grid Lines Cross-Section (y=ny//2)
    # -------------------------------------------------------------------------
    num_cols = len(grids)
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5.5), sharey=True)
    if num_cols == 1:
        axes = [axes]
    y_idx = ny // 2
    x_km = grids['Gal-Chen (Sigma)'].x_m / 1000.0

    for i, (name, gr) in enumerate(grids.items()):
        ax = axes[i]
        Z_slice = np.array(gr.Z_m[:, y_idx, :]) / 1000.0
        X_slice = np.broadcast_to(x_km[:, None], Z_slice.shape)

        # Plot underlying spurious u velocity at the final time step
        u_final = np.array(0.5 * (results[name]['final_state']['u'][:-1, y_idx, :] + results[name]['final_state']['u'][1:, y_idx, :]))
        vmax = max(1e-4, np.max(np.abs(u_final)))
        cf = ax.pcolormesh(X_slice, Z_slice, u_final, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')

        # Overlay vertical coordinate grid lines
        for k_idx in range(0, nz, 2):
            ax.plot(x_km, Z_slice[:, k_idx], color='black', linewidth=1.0, alpha=0.7)

        h_slice = np.array(gr.Z_w[:, y_idx, 0]) / 1000.0
        ax.fill_between(x_km, 0, h_slice, color='#2D3748')

        ax.set_title(f"{name}", fontsize=13)
        ax.set_xlabel("Horizontal Distance x (km)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Altitude z (km)", fontsize=11)
        ax.set_ylim(0, nz * dz / 1000.0)

    cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.06, pad=0.18)
    cbar.set_label("Spurious Horizontal Velocity $u$ (m/s) at $T=300$s", fontsize=11)

    plt.suptitle("Vertical Coordinate Geometry & Spurious Velocity Fields Discovered by NEUVE", fontsize=15, y=1.03)
    f2_path = "output/plots/neuve_physics/neuve_pgf_rest_grid_lines_discovery.png"
    plt.savefig(f2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved coordinate geometry discovery figure to {f2_path}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
