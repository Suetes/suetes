#!/usr/bin/env python3
"""
Master Zero-Shot Evaluation & Visualization Pipeline: NEUVE vs SLEVE vs Gal-Chen
================================================================================
Evaluates the discovered NEUVE coordinate transformation zero-shot on unseen rugged
topography (default SEED = 999) and compares performance against:
  1. Standard Gal-Chen (terrain-following sigma coordinates)
  2. Analytical SLEVE (smooth scale decay)
  3. PDE-Discovered NEUVE (PGF resting error or acoustic wave minimization)

Generates quantitative comparison tables and publication figures:
  - `output/plots/neuve/eval_{target}_comparison.png`
  - `output/plots/neuve/eval_{target}_grids.png`
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import csv
import json
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

from experiments.train_neuve import (
    nx, ny, nz, dx, dy, dz, constants, generate_rugged_terrain,
    pgf_rest_metric, coordinate_shape_penalties,
)


def load_trained_neuve(weights_path, condition_on_terrain=True,
                       legacy_behavior=False):
    neuve_init = NEUVECoordinate(
        hidden_dim=64,
        key_seed=42,
        condition_on_terrain=condition_on_terrain,
        legacy_behavior=legacy_behavior,
    )
    if not weights_path or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"NEUVE weights file '{weights_path}' was not found. Train the "
            "coordinate before evaluation; an untrained network is not a valid baseline."
        )
    data = np.load(weights_path)
    arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    flat_params, tree_def = jax.tree_util.tree_flatten(neuve_init.params)
    restored_params = jax.tree_util.tree_unflatten(tree_def, arrays)
    return neuve_init.with_params(restored_params)


def eval_coordinate_case(target, name, transform_op, terrain_fn, eval_steps, dt_val,
                         verbose=True):
    if verbose:
        print(f"  -> Integrating [{name}] over {eval_steps} steps (dt={dt_val:.1f}s)...")
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()

    if target == "pgf_rest":
        N_bv_val = 0.02
        u_init = jnp.zeros_like(grid.Z_u)
        v_init = jnp.zeros_like(grid.Z_v)
    else:  # acoustic_energy
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

    def scan_fn(st_curr, _):
        next_st = stepper.step(st_curr, 0.0, None, bc_fn)
        if target == "pgf_rest":
            tke_t = pgf_rest_metric(next_st, grid)
            max_u = jnp.max(jnp.abs(next_st['u']))
            max_w = jnp.max(jnp.abs(next_st['w']))
            return next_st, (tke_t, max_u, max_w)
        else:  # acoustic_energy
            w_aloft = next_st['w'][:, :, 10:]
            noise_t = jnp.mean(w_aloft ** 2)
            max_w_aloft = jnp.max(jnp.abs(w_aloft))
            return next_st, (noise_t, max_w_aloft)

    _, metric_data = jax.lax.scan(scan_fn, base_state, jnp.arange(eval_steps))
    return metric_data, grid


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Script for NEUVE vs SLEVE vs Gal-Chen")
    parser.add_argument("--target", type=str, default="pgf_rest", choices=["pgf_rest", "acoustic_energy"],
                        help="Evaluation target mode: pgf_rest or acoustic_energy")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Path to trained NEUVE npz weights")
    parser.add_argument("--terrain-mode", choices=["ensemble", "fixed"],
                        default="ensemble",
                        help="Training interpretation used by the checkpoint")
    parser.add_argument("--fixed-terrain-seed", type=int, default=999,
                        help="Fixed-domain seed encoded by a fixed-mode checkpoint")
    parser.add_argument(
        "--coordinate-mode", choices=["global", "spatial", "spatial-legacy"],
        default="global",
        help="NEUVE architecture used by the checkpoint",
    )
    parser.add_argument("--test-seed", type=int, default=999, help="PRNG seed for zero-shot rugged terrain")
    parser.add_argument("--num-test-topographies", type=int, default=10,
                        help="Number of consecutive, unseen terrain seeds to evaluate")
    parser.add_argument("--output-dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--sleve-scale-s", type=float, default=4000.0,
                        help="Frozen SLEVE small-scale decay height selected on validation data")
    parser.add_argument("--sleve-n", type=float, default=1.35,
                        help="Frozen SLEVE decay exponent selected on validation data")
    parser.add_argument("--sleve-config", default=None,
                        help="JSON produced by tune_sleve_pgf.py; overrides SLEVE arguments")
    args = parser.parse_args()

    if args.sleve_config:
        with open(args.sleve_config) as stream:
            sleve_config = json.load(stream)
        args.sleve_scale_s = float(sleve_config["scale_s"])
        args.sleve_n = float(sleve_config["n"])

    dt_val = 0.8 if args.target == "pgf_rest" else 2.5
    eval_steps = int(400.0 / dt_val) if args.target == "pgf_rest" else 120

    if not args.weights_path:
        if args.target == "pgf_rest":
            terrain_tag = (
                "ensemble" if args.terrain_mode == "ensemble"
                else f"fixed_seed{args.fixed_terrain_seed}"
            )
            profile_tag = (
                "spatial"
                if args.coordinate_mode in ("spatial", "spatial-legacy")
                else "global_decay_v5"
            )
            default_path = os.path.join(
                args.output_dir, "weights",
                f"neuve_pgf_rest_{profile_tag}_{terrain_tag}.npz",
            )
            compat_path = default_path
        else:
            default_path = os.path.join(args.output_dir, "weights", "neuve_acoustic_energy_unconstrained.npz")
            compat_path = "output/trained_neuve_physics_unconstrained_weights.bin.npz"
        args.weights_path = default_path if os.path.exists(default_path) else compat_path

    plots_dir = os.path.join(args.output_dir, "plots", "neuve")
    os.makedirs(plots_dir, exist_ok=True)

    evaluation_kind = (
        "Zero-Shot Evaluation"
        if args.terrain_mode == "ensemble" else "Fixed-Domain Evaluation"
    )
    print("=========================================================================")
    print(f"{evaluation_kind}: NEUVE vs SLEVE vs Gal-Chen | Target: [{args.target.upper()}]")
    print("=========================================================================")
    print(
        f"Test Seeds: {args.test_seed}..{args.test_seed + args.num_test_topographies - 1} "
        f"| Evaluation Steps: {eval_steps} (dt={dt_val:.1f}s)"
    )
    print(f"Loading NEUVE weights from: {args.weights_path}")
    print(
        f"Frozen SLEVE comparator: scale_s={args.sleve_scale_s:.1f} m, "
        f"n={args.sleve_n:.3f}"
    )

    terrain_fn = generate_rugged_terrain(args.test_seed)

    if args.target == "pgf_rest":
        sleve_op = SleveSimple(
            scale_s=args.sleve_scale_s, scale_l=15000.0, n=args.sleve_n
        )
    else:
        sleve_op = SleveSimple(scale_s=3000.0, scale_l=15000.0, n=1.35)

    coords = {
        'Gal-Chen (Sigma)': GalChenSigma(),
        'SLEVE (Analytical)': sleve_op,
        'NEUVE (PDE discovered)': load_trained_neuve(
            args.weights_path,
            condition_on_terrain=(
                args.target != "pgf_rest"
                or args.coordinate_mode in ("spatial", "spatial-legacy")
            ),
            legacy_behavior=(
                args.target == "pgf_rest"
                and args.coordinate_mode in ("spatial", "spatial-legacy")
            ),
        )
    }

    results = {}
    grids = {}
    ensemble_rows = []

    for name, op in coords.items():
        m_data, gr = eval_coordinate_case(args.target, name, op, terrain_fn, eval_steps, dt_val)
        if args.target == "pgf_rest":
            tke_s, u_s, w_s = m_data
            results[name] = {
                'series_1': np.array(tke_s),
                'series_2': np.array(u_s),
                'series_3': np.array(w_s),
                'mean_300s': float(np.mean(tke_s[:int(300.0 / dt_val)])),
                'mean_full': float(np.mean(tke_s)),
                'peak_1': float(np.max(u_s)),
                'peak_2': float(np.max(w_s))
            }
        else:  # acoustic_energy
            noise_s, max_w_s = m_data
            results[name] = {
                'series_1': np.array(noise_s),
                'series_2': np.array(max_w_s),
                'mean_300s': float(np.mean(noise_s)),
                'mean_full': float(np.mean(noise_s)),
                'peak_1': float(np.max(noise_s)),
                'peak_2': float(np.max(max_w_s))
            }
        grids[name] = gr
        flatness, curvature = coordinate_shape_penalties(gr)
        ensemble_rows.append({
            'seed': args.test_seed,
            'coordinate': name,
            'mean_metric': results[name]['mean_full'],
            'peak_u_or_metric': results[name]['peak_1'],
            'peak_w': results[name]['peak_2'],
            'minimum_layer_m': float(jnp.min(gr.dz_m_full)),
            'upper_slope_squared': float(flatness),
            'dimensionless_curvature': float(curvature),
        })

    # Evaluate additional terrain realizations.  In ensemble mode these are
    # truly unseen; in fixed mode the first seed is the optimized domain and
    # any additional seeds are explicitly out-of-domain stress tests.
    for seed in range(args.test_seed + 1,
                      args.test_seed + args.num_test_topographies):
        test_terrain = generate_rugged_terrain(seed)
        print(f"  -> Zero-shot terrain seed {seed}")
        for name, op in coords.items():
            metric_data, grid = eval_coordinate_case(
                args.target, name, op, test_terrain, eval_steps, dt_val,
                verbose=False,
            )
            if args.target == "pgf_rest":
                tke_s, u_s, w_s = metric_data
                mean_metric = float(jnp.mean(tke_s))
                peak_1 = float(jnp.max(u_s))
                peak_2 = float(jnp.max(w_s))
            else:
                noise_s, max_w_s = metric_data
                mean_metric = float(jnp.mean(noise_s))
                peak_1 = float(jnp.max(noise_s))
                peak_2 = float(jnp.max(max_w_s))
            ensemble_rows.append({
                'seed': seed,
                'coordinate': name,
                'mean_metric': mean_metric,
                'peak_u_or_metric': peak_1,
                'peak_w': peak_2,
                'minimum_layer_m': float(jnp.min(grid.dz_m_full)),
                'upper_slope_squared': float(coordinate_shape_penalties(grid)[0]),
                'dimensionless_curvature': float(coordinate_shape_penalties(grid)[1]),
            })

    print("\n=========================================================================")
    if args.target == "pgf_rest":
        print(f"{'Coordinate System':<24} | {'Mean TKE (300s)':<18} | {'Mean TKE (400s)':<18} | {'Peak |u| (m/s)':<16} | {'Peak |w| (m/s)':<16}")
    else:
        print(f"{'Coordinate System':<24} | {'Mean Aloft w^2 (m^2/s^2)':<26} | {'Peak w^2':<16} | {'Peak |w| aloft (m/s)':<20}")
    print("-" * 102)

    for name, r in results.items():
        if args.target == "pgf_rest":
            print(f"{name:<24} | {r['mean_300s']:<18.6e} | {r['mean_full']:<18.6e} | {r['peak_1']:<16.4f} | {r['peak_2']:<16.4f}")
        else:
            print(f"{name:<24} | {r['mean_full']:<26.6e} | {r['peak_1']:<16.6e} | {r['peak_2']:<20.4f}")
    print("=========================================================================\n")

    ensemble_heading = (
        "Unseen-terrain ensemble"
        if args.terrain_mode == "ensemble" else "Fixed/domain-test terrain set"
    )
    print(f"--- {ensemble_heading} (mean +/- sample standard deviation) ---")
    ensemble_values = {}
    for name in coords:
        values = np.asarray([
            row['mean_metric'] for row in ensemble_rows
            if row['coordinate'] == name
        ])
        ensemble_values[name] = values
        spread = np.std(values, ddof=1) if len(values) > 1 else 0.0
        min_layers = np.asarray([
            row['minimum_layer_m'] for row in ensemble_rows
            if row['coordinate'] == name
        ])
        upper_slopes = np.asarray([
            row['upper_slope_squared'] for row in ensemble_rows
            if row['coordinate'] == name
        ])
        curvatures = np.asarray([
            row['dimensionless_curvature'] for row in ensemble_rows
            if row['coordinate'] == name
        ])
        print(
            f"{name:<24}: {np.mean(values):.6e} +/- {spread:.2e}; "
            f"minimum layer={np.min(min_layers):.2f} m; "
            f"upper slope^2={np.mean(upper_slopes):.3e}; "
            f"curvature={np.mean(curvatures):.3e}"
        )

    ensemble_path = os.path.join(
        args.output_dir,
        f"neuve_{args.target}_{args.terrain_mode}_terrain_metrics.csv",
    )
    with open(ensemble_path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ensemble_rows[0]))
        writer.writeheader()
        writer.writerows(ensemble_rows)
    print(f"[SUCCESS] Saved terrain-set diagnostics to: {ensemble_path}\n")

    gal_ensemble = ensemble_values['Gal-Chen (Sigma)']
    sleve_ensemble = ensemble_values['SLEVE (Analytical)']
    neuve_ensemble = ensemble_values['NEUVE (PDE discovered)']
    paired_reduction = 100.0 * (1.0 - neuve_ensemble / gal_ensemble)
    paired_spread = (
        float(np.std(paired_reduction, ddof=1))
        if len(paired_reduction) > 1 else 0.0
    )
    print(
        "NEUVE paired reduction relative to Gal-Chen: "
        f"{np.mean(paired_reduction):.2f}% +/- {paired_spread:.2f} percentage points"
    )
    print(
        "NEUVE beats SLEVE on "
        f"{np.count_nonzero(neuve_ensemble < sleve_ensemble)}/"
        f"{len(neuve_ensemble)} evaluated terrains.\n"
    )

    ref_val = results['Gal-Chen (Sigma)']['mean_full']
    neuve_val = results['NEUVE (PDE discovered)']['mean_full']
    sleve_val = results['SLEVE (Analytical)']['mean_full']
    reduction = (1.0 - neuve_val / ref_val) * 100.0 if ref_val > 0 else 0.0

    print(
        "Representative plotted terrain: "
        f"NEUVE={neuve_val:.6e}, SLEVE={sleve_val:.6e}, "
        f"Gal-Chen={ref_val:.6e} ({reduction:.1f}% vs Gal-Chen)."
    )

    # -------------------------------------------------------------------------
    # Figure 1: Time Series Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    t_axis = np.arange(1, eval_steps + 1) * dt_val
    colors = {'Gal-Chen (Sigma)': '#E53E3E', 'SLEVE (Analytical)': '#3182CE', 'NEUVE (PDE discovered)': '#38A169'}
    styles = {'Gal-Chen (Sigma)': '--', 'SLEVE (Analytical)': '-.', 'NEUVE (PDE discovered)': '-'}

    if args.target == "pgf_rest":
        for name, r in results.items():
            ax1.semilogy(t_axis, r['series_1'], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
            ax2.plot(t_axis, r['series_2'], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        ax1.set_title("(a) Spurious kinetic energy from rest ($u=v=w=0$)", fontsize=13)
        ax1.set_ylabel("Volume-weighted spurious TKE ($m^2/s^2$)", fontsize=11)
        ax2.set_title("(b) Maximum spurious horizontal wind $|u|_{max}$", fontsize=13)
        ax2.set_ylabel("Spurious horizontal wind speed (m/s)", fontsize=11)
    else:
        for name, r in results.items():
            ax1.plot(t_axis, r['series_1'], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
            mask = t_axis >= 60.0
            ax2.plot(t_axis[mask], r['series_1'][mask], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        ax1.set_title("(a) Full simulation horizon (log scale)", fontsize=13)
        ax1.set_ylabel("Mean upper-level kinetic energy $w^2$ ($m^2/s^2$)", fontsize=11)
        ax1.set_yscale('log')
        ax2.set_title(r"(b) Mature stratospheric acoustic regime ($t \geq 60$ s)", fontsize=13)
        ax2.set_ylabel("Mean upper-level kinetic energy $w^2$ ($m^2/s^2$)", fontsize=11)

    for ax in (ax1, ax2):
        ax.set_xlabel("Integration time (s)", fontsize=11)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10.5)

    plt.tight_layout()
    plot_comparison = os.path.join(plots_dir, f"eval_{args.target}_comparison.png")
    plt.savefig(plot_comparison, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved multi-panel comparison figure to: {plot_comparison}")

    # -------------------------------------------------------------------------
    # Figure 2: Vertical Coordinate Grid Lines Cross-Section
    # -------------------------------------------------------------------------
    num_cols = len(grids)
    fig, axes = plt.subplots(1, num_cols, figsize=(6.5 * num_cols, 5.2), sharey=True)
    if num_cols == 1:
        axes = [axes]
    y_mid = ny // 2
    x_km = np.arange(nx) * dx / 1000.0

    for i, (name, grid_obj) in enumerate(grids.items()):
        ax = axes[i]
        Z_slice = grid_obj.Z_m[:, y_mid, :] / 1000.0
        for k in range(nz):
            ax.plot(x_km, Z_slice[:, k], color='#2D3748', linewidth=1.0, alpha=0.85)

        h_slice = Z_slice[:, 0]
        ax.fill_between(x_km, 0.0, h_slice, color='#718096', alpha=0.4, label='Topography')
        ax.plot(x_km, h_slice, color='#1A202C', linewidth=2.0)

        ax.set_title(name, fontsize=13)
        ax.set_xlabel("x (km)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Altitude z (km)", fontsize=11)
        ax.set_ylim(0, nz * dz / 1000.0)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f"Discovered vertical coordinate geometry", fontsize=14, y=0.98)
    plt.tight_layout()
    plot_grids = os.path.join(plots_dir, f"eval_{args.target}_grids.png")
    plt.savefig(plot_grids, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved coordinate geometry cross-section to: {plot_grids}")
    print("=========================================================================")


if __name__ == '__main__':
    main()
