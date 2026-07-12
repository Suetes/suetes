#!/usr/bin/env python3
"""
Evaluate NEUVE Coordinate: ZERO-SHOT GENERALIZATION on Unseen Topography
=============================================================================
Evaluates the frozen trained NEUVE coordinate (trained across TRAIN_SEEDS)
against Gal-Chen and SLEVE on a HELD-OUT test topography (TEST_SEED = 999)
that was NEVER seen during training.

Key Design:
  1. Parametric Jagged Topography Generator (shared with train_neuve_coordinate.py).
  2. Full Concentration Time-Series Observability (Breaking Snapshot Trap).
  3. Absolute Physical Altitude Sensor Array (Breaking Index Bias).
  4. Neutral nz=48 GalChenSigma ground truth (Breaking Inverse Crime).
"""

import os
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite
from suetes.shared.transforms import GalChenSigma, StretchedSleveSimple, NEUVECoordinate

output_dir = "output/plots/neuve"
os.makedirs(output_dir, exist_ok=True)

# --- 1. COMMON BENCHMARK DOMAIN & SHEARED FLOW ---
nx, ny, nz = 32, 12, 16
dx, dy, dz = 500.0, 500.0, 375.0
dt = 3.0
t_end = 900.0
num_steps = int(t_end / dt)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# --- PARAMETRIC JAGGED TOPOGRAPHY GENERATOR (shared with train_neuve_coordinate.py) ---
TRAIN_SEEDS = [101, 202, 303, 404]
TEST_SEED = 999

def generate_jagged_terrain(seed):
    """
    Returns a terrain_profile(x, y) closure for a procedurally generated
    jagged mountain barrier parameterised by an integer seed.
    """
    rng = np.random.RandomState(seed)

    h0 = rng.uniform(1500.0, 1900.0)
    cx0 = rng.uniform(-2000.0, 2000.0)
    ax0 = rng.uniform(800.0, 1400.0)
    ay0 = rng.uniform(4500.0, 7000.0)

    h1 = rng.uniform(300.0, 600.0)
    cx1 = rng.uniform(500.0, 2500.0)
    ax1 = rng.uniform(600.0, 1200.0)
    ay1 = rng.uniform(3500.0, 6000.0)

    jagged_wavelength = rng.uniform(1200.0, 3000.0)
    jagged_amp = rng.uniform(0.08, 0.18)
    jagged_phase = rng.uniform(0.0, 2.0 * np.pi)

    def terrain_profile(x, y):
        jagged = 1.0 + jagged_amp * jnp.cos(2 * jnp.pi * x / jagged_wavelength + jagged_phase)
        peak1 = h0 * jnp.exp(-((x - cx0)**2 / ax0**2 + y**2 / ay0**2)) * jagged
        peak2 = h1 * jnp.exp(-((x - cx1)**2 / ax1**2 + y**2 / ay1**2))
        return peak1 + peak2

    return terrain_profile

# Instantiate the UNSEEN test topography
terrain_profile = generate_jagged_terrain(TEST_SEED)

def get_wind_u(Z_coords, n_vert=nz, dz_val=dz):
    u_sfc, u_top = 8.0, 15.0
    H_top = n_vert * dz_val
    return u_sfc + (u_top - u_sfc) * jnp.clip(Z_coords / H_top, 0.0, 1.0)

def get_wind_v(Z_coords):
    return jnp.zeros_like(Z_coords)

# True source parameters aloft above mountain shadow zone
true_params = jnp.array([-5.0, 0.0, 3.6, 10.0])
guess_params = jnp.array([-2.5, -1.0, 4.2, 4.0])
sensor_idx_x = 24
sensor_x_km = 3.5

# Absolute physical sensor tower heights Z_obs (meters above sea level)
Z_obs = jnp.linspace(1500.0, 5500.0, 16)

def get_neuve_transform():
    weights_path = "output/trained_neuve_weights.bin"
    if os.path.exists(weights_path):
        print(f"[INFO] Loading TRAINED NEUVE coordinate from {weights_path}")
        return NEUVECoordinate.from_file(weights_path, hidden_dim=32)
    else:
        print("[WARNING] No trained NEUVE weights found. Using untrained NEUVE.")
        return NEUVECoordinate(hidden_dim=32, key_seed=101)


# =========================================================================
# HELPER: BUILD STANDARD SIMULATION & CORE INVERSION OPERATORS (nz=16)
# =========================================================================
def make_core_ops(transform_op):
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_profile,
        transform=transform_op
    )
    op = CGridOperator3D(grid)
    suite = PhysicsSuite()
    suite.register_tracer('q_tr')
    
    physics = Euler3D(
        grid, op, constants, dt=dt, N_bv=0.01, damp_height=4200.0,
        max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite
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
            'pi': bg_ref['pi'],
            'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
        }
        return sponge.blend(state_in, ext_state)
    
    base_state = {
        'u': get_wind_u(grid.Z_u),
        'v': get_wind_v(grid.Z_v),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v'],
        'q_tr': jnp.zeros((nx, ny, nz))
    }
    
    stepper, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=op, constants=constants,
        initial_state=base_state, physics_suite=suite,
        dt=dt, ns=16, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=4200.0, max_damp=0.5, N_bv=0.01
    )
    stepper.use_checkpointing = True
    return grid, base_state, stepper, bc_fn


# =========================================================================
# UNIVERSAL GROUND TRUTH SENSOR OBSERVATION TIME-SERIES (INTERPOLATED TO Z_OBS)
# =========================================================================
def compute_universal_ground_truth_timeseries():
    nz_fine = 48
    dz_fine = dz / 3.0
    fine_grid = RegionalGrid3D(
        nx, ny, nz_fine, dx, dy, dz_fine,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_profile,
        transform=GalChenSigma()
    )
    op_fine = CGridOperator3D(fine_grid)
    suite = PhysicsSuite()
    suite.register_tracer('q_tr')
    
    physics_fine = Euler3D(
        fine_grid, op_fine, constants, dt=dt, N_bv=0.01, damp_height=4200.0,
        max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite
    )
    bg_ref = {
        'rho': physics_fine.c['p0'] / (physics_fine.c['Rd'] * physics_fine.theta_bg) * \
               (physics_fine.pi_bg ** (physics_fine.c['cvd'] / physics_fine.c['Rd'])),
        'pi': physics_fine.pi_bg,
        'th_v': physics_fine.theta_bg
    }
    sponge = BenchmarkSponge(nx, ny, sponge_depth=4, axes=('x', 'y'))
    
    def bc_fine(state_in, forcing=None):
        ext_state = {
            'u': get_wind_u(fine_grid.Z_u, n_vert=nz_fine, dz_val=dz_fine),
            'v': get_wind_v(fine_grid.Z_v),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi'],
            'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
        }
        return sponge.blend(state_in, ext_state)
    
    base_state_fine = {
        'u': get_wind_u(fine_grid.Z_u, n_vert=nz_fine, dz_val=dz_fine),
        'v': get_wind_v(fine_grid.Z_v),
        'w': jnp.zeros((nx, ny, nz_fine+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz_fine+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v'],
        'q_tr': jnp.zeros((nx, ny, nz_fine))
    }
    
    stepper_fine, _ = build_dynamical_core(
        core_type="split-explicit", grid=fine_grid, operators=op_fine, constants=constants,
        initial_state=base_state_fine, physics_suite=suite,
        dt=dt, ns=24, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=4200.0, max_damp=0.5, N_bv=0.01
    )
    
    X_km = fine_grid.x_m[:, None, None] / 1000.0
    Y_km = fine_grid.y_m[None, :, None] / 1000.0
    Z_km = fine_grid.Z_m / 1000.0
    q_init = true_params[3] * jnp.exp(-((X_km - true_params[0])**2 / 2.0 + (Y_km - true_params[1])**2 / 2.0 + (Z_km - true_params[2])**2 / 1.0))
    init_s = dict(base_state_fine)
    init_s['q_tr'] = q_init
    
    def scan_gt(st, _):
        next_st = stepper_fine.step(st, 0.0, None, bc_fine)
        col_q = next_st['q_tr'][sensor_idx_x]
        col_z = fine_grid.Z_m[sensor_idx_x]
        def interp_y(z_col, q_col):
            return jnp.interp(Z_obs, z_col, q_col)
        slice_phys = jax.vmap(interp_y)(col_z, col_q)
        return next_st, slice_phys
    
    _, timeseries_gt = jax.lax.scan(scan_gt, init_s, jnp.arange(num_steps))
    return timeseries_gt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Zero-shot generalization evaluation of NEUVE coordinate.")
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps to take (default: 50)")
    args = parser.parse_args()
    steps = args.steps

    print("=========================================================================")
    print(f"NEUVE zero-shot generalization: Unseen test topography (SEED={TEST_SEED})")
    print("=========================================================================")
    print(f"Training Seeds:  {TRAIN_SEEDS}")
    print(f"Test Seed:       {TEST_SEED} ")
    
    transforms_to_test = {
        'Gal-Chen': GalChenSigma(),
        'SLEVE': StretchedSleveSimple(stretch_kappa=0.5, scale_s=5000.0),
        'NEUVE': get_neuve_transform()
    }

    print(f"\nGenerating ground truth time-series on unseen test terrain (nz=48 GalChen, t_end=900s)...")
    universal_target_ts = compute_universal_ground_truth_timeseries()
    print(f"[INFO] Universal ground truth time-series generated. Shape: {universal_target_ts.shape}, Peak signal: {float(jnp.max(universal_target_ts)):.4e}")

    # =========================================================================
    # Metric 1: Adjoint field sharpness & step 1 parameter gradients
    # =========================================================================
    print("\n-------------------------------------------------------------------------")
    print("Metric 1: Adjoint field sharpness & step 1 parameter gradients")
    print("-------------------------------------------------------------------------")

    m1_results = {}
    m1_metrics = {}
    step1_grads = {}

    for name, transform in transforms_to_test.items():
        print(f"  -> Computing time-series adjoint sensitivity & step 1 parameter gradients for [{name}]...")
        grid, base_state, stepper, bc_fn = make_core_ops(transform)
        
        X_km = grid.x_m[:, None, None] / 1000.0
        Y_km = grid.y_m[None, :, None] / 1000.0
        Z_km = grid.Z_m / 1000.0
        
        def create_tracer(p):
            return p[3] * jnp.exp(-((X_km - p[0])**2 / 2.0 + (Y_km - p[1])**2 / 2.0 + (Z_km - p[2])**2 / 1.0))
        
        def scan_step(st, _):
            next_st = stepper.step(st, 0.0, None, bc_fn)
            col_q = next_st['q_tr'][sensor_idx_x]
            col_z = grid.Z_m[sensor_idx_x]
            def interp_y(z_col, q_col):
                return jnp.interp(Z_obs, z_col, q_col)
            slice_phys = jax.vmap(interp_y)(col_z, col_q)
            return next_st, slice_phys
        
        # 1. Spatial adjoint sensitivity nabla_{q0} J (time-series discrepancy)
        def loss_from_q0(q0):
            init_st = dict(base_state)
            init_st['q_tr'] = q0
            _, sim_ts = jax.lax.scan(scan_step, init_st, jnp.arange(num_steps))
            return jnp.mean((sim_ts - universal_target_ts)**2)
        
        t0 = time.time()
        sens_field = jax.jit(jax.grad(loss_from_q0))(jnp.zeros((nx, ny, nz)))
        sens_field.block_until_ready()
        dt_m1 = time.time() - t0
        
        # 2. Step 1 Parameter Gradient evaluated over full time-series
        def inner_loss(p):
            s = dict(base_state)
            s['q_tr'] = create_tracer(p)
            _, sim_ts = jax.lax.scan(scan_step, s, jnp.arange(num_steps))
            return jnp.mean((sim_ts - universal_target_ts)**2)
        
        lval0, gval0 = jax.value_and_grad(inner_loss)(guess_params)
        
        m1_results[name] = (grid, sens_field)
        step1_grads[name] = (float(lval0), np.array(gval0))
        
        abs_s = jnp.abs(sens_field)
        tot = jnp.sum(abs_s)
        pn = abs_s / (tot + 1e-12)
        cx = jnp.sum(pn * X_km)
        cz = jnp.sum(pn * Z_km)
        sx = jnp.sqrt(jnp.sum(pn * (X_km - cx)**2))
        sz = jnp.sqrt(jnp.sum(pn * (Z_km - cz)**2))
        peak = jnp.max(abs_s)
        ent = -jnp.sum(pn * jnp.log(pn + 1e-12))
        
        m1_metrics[name] = {
            'peak': float(peak), 'sx': float(sx), 'sz': float(sz), 'entropy': float(ent), 'elapsed': float(dt_m1)
        }

    print("\n=========================================================================")
    print("Metric 1: Adjoint field sharpness & step 1 parameter gradients")
    print("=========================================================================")
    print(f"{'Coordinate':<16} | {'Peak Sens':<12} | {'Entropy':<10} | {'Step 1 Loss':<12} | {'Grad x_s':<12} | {'Grad z_s':<12}")
    print("-" * 85)
    for name, m in m1_metrics.items():
        v0, g0 = step1_grads[name]
        print(f"{name:<16} | {m['peak']:<12.4e} | {m['entropy']:<10.3f} | {v0:<12.4e} | {g0[0]:<12.4e} | {g0[2]:<12.4e}")
    print("=========================================================================")


    # =========================================================================
    # METRIC 2: TIME-SERIES LOSS LANDSCAPE BASIN SMOOTHNESS
    # =========================================================================
    print("\n-------------------------------------------------------------------------")
    print("Metric 2: Time-series loss landscape basin smoothness J(x_s, y_s)")
    print("-------------------------------------------------------------------------")

    m2_grids = {}
    x_range = jnp.linspace(-6.5, -2.5, 15)
    y_range = jnp.linspace(-2.5, 2.5, 13)

    for name, transform in transforms_to_test.items():
        print(f"  -> Computing 2D time-series loss landscape grid for [{name}]...")
        grid, base_state, stepper, bc_fn = make_core_ops(transform)
        
        X_km = grid.x_m[:, None, None] / 1000.0
        Y_km = grid.y_m[None, :, None] / 1000.0
        Z_km = grid.Z_m / 1000.0
        
        def create_tracer_xy(xs, ys):
            return true_params[3] * jnp.exp(-((X_km - xs)**2 / 2.0 + (Y_km - ys)**2 / 2.0 + (Z_km - true_params[2])**2 / 1.0))
        
        def scan_step(st, _):
            next_st = stepper.step(st, 0.0, None, bc_fn)
            col_q = next_st['q_tr'][sensor_idx_x]
            col_z = grid.Z_m[sensor_idx_x]
            def interp_y(z_col, q_col):
                return jnp.interp(Z_obs, z_col, q_col)
            slice_phys = jax.vmap(interp_y)(col_z, col_q)
            return next_st, slice_phys
        
        @jax.jit
        def loss_xy(xs, ys):
            s = dict(base_state)
            s['q_tr'] = create_tracer_xy(xs, ys)
            _, sim_ts = jax.lax.scan(scan_step, s, jnp.arange(num_steps))
            return jnp.mean((sim_ts - universal_target_ts)**2)
        
        loss_grid = np.zeros((len(y_range), len(x_range)))
        for j, ys_val in enumerate(y_range):
            for i, xs_val in enumerate(x_range):
                loss_grid[j, i] = float(loss_xy(xs_val, ys_val))
        m2_grids[name] = loss_grid


    # =========================================================================
    # Metric 3: 3D parameter trajectories & convergence speed (time-series loss)
    # =========================================================================
    print("\n-------------------------------------------------------------------------")
    print("Metric 3: Time-series inversion trajectory & ADAM convergence speed")
    print("-------------------------------------------------------------------------")

    m3_history = {}

    for name, transform in transforms_to_test.items():
        print(f"  -> Running {steps}-step source inversion against time-series for [{name}]...")
        grid, base_state, stepper, bc_fn = make_core_ops(transform)
        
        X_km = grid.x_m[:, None, None] / 1000.0
        Y_km = grid.y_m[None, :, None] / 1000.0
        Z_km = grid.Z_m / 1000.0
        
        def create_tracer(p):
            return p[3] * jnp.exp(-((X_km - p[0])**2 / 2.0 + (Y_km - p[1])**2 / 2.0 + (Z_km - p[2])**2 / 1.0))
        
        def scan_step(st, _):
            next_st = stepper.step(st, 0.0, None, bc_fn)
            col_q = next_st['q_tr'][sensor_idx_x]
            col_z = grid.Z_m[sensor_idx_x]
            def interp_y(z_col, q_col):
                return jnp.interp(Z_obs, z_col, q_col)
            slice_phys = jax.vmap(interp_y)(col_z, col_q)
            return next_st, slice_phys
        
        @jax.jit
        def loss_and_grad(p):
            def obj(params):
                s = dict(base_state)
                s['q_tr'] = create_tracer(params)
                _, sim_ts = jax.lax.scan(scan_step, s, jnp.arange(num_steps))
                return jnp.mean((sim_ts - universal_target_ts)**2)
            return jax.value_and_grad(obj)(p)
        
        inner_opt = optax.adam(0.25)
        opt_st = inner_opt.init(guess_params)
        p_curr = guess_params
        
        p_hist = [np.array(p_curr)]
        l_hist = []
        
        for k in range(steps):
            lval, gval = loss_and_grad(p_curr)
            updates, opt_st = inner_opt.update(gval, opt_st, p_curr)
            p_curr = optax.apply_updates(p_curr, updates)
            p_hist.append(np.array(p_curr))
            l_hist.append(float(lval))
        
        m3_history[name] = {'params': np.array(p_hist), 'losses': np.array(l_hist)}

    print("\n=========================================================================")
    print(f"Metric 3: Final inversion recovery accuracy after {steps} ADAM steps")
    print("=========================================================================")
    print(f"{'Coordinate':<16} | {'Final Loss MSE':<14} | {'Final xs':<10} | {'Final zs':<10} | {'3D Pos Error (km)':<18}")
    print("-" * 80)
    for name, h in m3_history.items():
        p_last = h['params'][-1]
        pos_err = np.linalg.norm(p_last[:3] - np.array(true_params[:3]))
        print(f"{name:<16} | {h['losses'][-1]:<14.4e} | {p_last[0]:<10.3f} | {p_last[2]:<10.3f} | {pos_err:<18.4f}")
    print("=========================================================================")


    # =========================================================================
    # GENERATE INVERSION TRAJECTORY & CONVERGENCE speed PLOT (Metric 3)
    # =========================================================================
    print("\nGenerating evaluation summary figures...")

    fig3, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig3.suptitle(f"Metric 3: Time-series parameter recovery trajectory & Adam convergence speed ({steps} steps)", fontsize=14, y=0.98)

    colors = {'Gal-Chen': '#e11d48', 'SLEVE': '#2563eb', 'NEUVE': '#10b981'}
    styles = {'Gal-Chen': '--', 'SLEVE': '-.', 'NEUVE': '-'}

    ax_t.plot(true_params[0], true_params[2], 'k*', markersize=16, label='True source p*')
    ax_t.plot(guess_params[0], guess_params[2], 'ko', markersize=9, label='Initial guess p(0)')

    for name, h in m3_history.items():
        p = h['params']
        ax_t.plot(p[:, 0], p[:, 2], linestyle=styles[name], color=colors[name], linewidth=1.0, marker='.', label=name)
        ax_c.semilogy(h['losses'], linestyle=styles[name], color=colors[name], linewidth=1.0, label=name)

    ax_t.set_title("Source trajectory in parameter space (x vs z)", fontsize=12)
    ax_t.set_xlabel("Source x (km)")
    ax_t.set_ylabel("Source z (km)")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_c.set_title("Adam inversion convergence curve J(p^(k))", fontsize=12)
    ax_c.set_xlabel("Adam optimization step k")
    ax_c.set_ylabel("Inversion MSE loss")
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/3_inversion_trajectories_and_convergence.png", dpi=200)
    plt.close(fig3)

    print(f"[SUCCESS] All evaluation plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
