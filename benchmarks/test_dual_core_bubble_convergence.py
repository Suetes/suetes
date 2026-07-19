import os
import gc
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

# Enable X64 for precise convergence tests
jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

output_dir = "output/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

def block_average_2d(field_2d, factor):
    nx, nz = field_2d.shape
    return jnp.mean(field_2d.reshape(nx // factor, factor, nz // factor, factor), axis=(1, 3))

def get_cell_center_slices(res_3d):
    u = res_3d['u'][:, 1, :]
    v = res_3d['v'][:, 1, :]
    w = res_3d['w'][:, 1, :]
    pi = res_3d['pi'][:, 1, :]
    th = res_3d['th_v'][:, 1, :]
    
    u_m = 0.5 * (u[:-1, :] + u[1:, :])
    w_m = 0.5 * (w[:, :-1] + w[:, 1:])
    return {
        'u': np.array(u_m),
        'v': np.array(v),
        'w': np.array(w_m),
        'pi': np.array(pi),
        'th_v': np.array(th)
    }

def run_bubble_at_resolution(core_type, dx, alpha=0.55, dt_mode="constant"):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    if dt_mode == "scaling":
        if core_type == "sisl":
            dt = (dx / 125.0) * 5.0
            core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha, "solver_tol": 1e-12, "solver_maxiter": 100, "solver_restart": 100}
        elif core_type == "split-explicit":
            dt = (dx / 125.0) * 2.5
            core_kwargs = {"dt": dt, "ns": 24, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha}
        else:
            raise ValueError(f"Unknown core: {core_type}")
    elif dt_mode == "constant":
        # Constant reference timestep to isolate spatial second-order accuracy (Option A)
        dt = 0.5
        if core_type == "sisl":
            core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha, "solver_tol": 1e-12, "solver_maxiter": 100, "solver_restart": 100}
        elif core_type == "split-explicit":
            core_kwargs = {"dt": dt, "ns": 24, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha}
        else:
            raise ValueError(f"Unknown core: {core_type}")
    else:
        raise ValueError(f"Unknown dt_mode: {dt_mode}")

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    # Force pure Cartesian geometry to eliminate pseudo-2D advection boundary artifacts
    for k in grid.m_factors: 
        grid.m_factors[k] = jnp.ones_like(grid.m_factors[k])
    grid.dm_dx_m = jnp.zeros_like(grid.dm_dx_m)
    grid.dm_dy_m = jnp.zeros_like(grid.dm_dy_m)
    grid.f_u = jnp.zeros_like(grid.f_u)
    grid.f_v = jnp.zeros_like(grid.f_v)


    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=dt)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    bubble = jnp.where((X**2 + (Z - 2000.0)**2) <= 1500.0**2, 
                       2.0 * jnp.cos(0.5 * jnp.pi * jnp.sqrt(X**2 + (Z - 2000.0)**2) / 1500.0)**2, 0.0)
    
    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    # Initialize history fields to ensure JAX carries them inside the scan loop
    state['u_prev'] = state['u']
    state['v_prev'] = state['v']
    state['w_prev'] = state['w']
    state['eta_dot_prev'] = state['eta_dot']
    state['tend_th_v_prev'] = jnp.zeros_like(state['th_v'])
    state['is_first_step'] = 1.0

    stepper, dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i*dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    raw_res = sim.run(state, t_start=0.0, t_end=200.0, chunk_steps=min(int(200.0/dt), 40))
    slices = get_cell_center_slices(raw_res)
    del raw_res
    del state
    del tmp_phys
    del stepper
    del sim
    jax.clear_caches()
    gc.collect()
    return slices

def run_study(core_type, alpha=0.55, dt_mode="constant"):
    print(f"\n========================================")
    print(f"Running self-convergence study for {core_type.upper()} (alpha={alpha}, dt_mode={dt_mode})...")
    print(f"========================================")
    dxs = [1000.0, 500.0, 250.0, 125.0, 62.5]
    slices = []
    for dx in dxs:
        s = run_bubble_at_resolution(core_type, dx, alpha, dt_mode=dt_mode)
        slices.append(s)
        print(f"Completed simulation at dx = {dx} m | VRAM caches cleared.")

    keys = ['u', 'v', 'w', 'pi', 'th_v']
    errors = {k: [] for k in keys}
    
    for i in range(len(dxs) - 1):
        coarse = slices[i]
        fine = slices[i+1]
        
        # 1500m boundary sponge/clipping zone crop
        crop_x = int(1500.0 / dxs[i])
        crop_z = int(1500.0 / dxs[i])
        
        for k in keys:
            fine_avg = block_average_2d(fine[k], 2)
            
            # Crop to interior to avoid boundary clipping/sponge artifacts
            coarse_cropped = coarse[k][crop_x:-crop_x, crop_z:-crop_z]
            fine_avg_cropped = fine_avg[crop_x:-crop_x, crop_z:-crop_z]
            
            err = float(jnp.sqrt(jnp.mean((fine_avg_cropped - coarse_cropped)**2)))
            errors[k].append(err)
            
        print(f"Res {dxs[i]:6.1f} -> {dxs[i+1]:6.1f}m | "
              f"err_u: {errors['u'][-1]:.2e} | err_v: {errors['v'][-1]:.2e} | "
              f"err_w: {errors['w'][-1]:.2e} | err_pi: {errors['pi'][-1]:.2e} | "
              f"err_th: {errors['th_v'][-1]:.2e}")

    print("\n--- Self Convergence Rates ---")
    rates = {k: [] for k in keys}
    for i in range(len(dxs) - 2):
        print(f"Res {dxs[i]:6.1f} -> {dxs[i+1]:6.1f}m:")
        for k in ['u', 'v', 'w', 'pi', 'th_v']:
            if errors[k][i+1] == 0.0 or errors[k][i] == 0.0:
                rate = np.nan
            else:
                rate = np.log2(errors[k][i] / errors[k][i+1])
            rates[k].append(rate)
            print(f"  Rate {k:2s}: {rate:.2f}")

    return errors

if __name__ == "__main__":

    errors_sisl = run_study("sisl", alpha=0.5, dt_mode="scaling")
    errors_se = run_study("split-explicit", alpha=0.5, dt_mode="scaling")

    # Plotting setup
    dx_vals = np.array([1000.0, 500.0, 250.0, 125.0])

    # --- Plot 1: Potential Temperature (theta_v) Convergence ---
    plt.figure(figsize=(10, 8))
    plt.loglog(dx_vals, errors_sisl['th_v'], 'o-', label='SISL', linewidth=2, markersize=8)
    plt.loglog(dx_vals, errors_se['th_v'], 's-', label='Split-explicit', linewidth=2, markersize=8)

    ref_start_th = max(errors_sisl['th_v'][0], errors_se['th_v'][0]) * 1.5
    ref_line_th = ref_start_th * (dx_vals / dx_vals[0])**2
    plt.loglog(dx_vals, ref_line_th, 'k--', label='Theoretical 2nd Order', alpha=0.7)

    plt.xlabel('Grid Spacing dx (m)', fontsize=12)
    plt.ylabel('L2 Error in theta_v (K)', fontsize=12)
    plt.title('Spatial convergence study: Rising bubble benchmark (theta_v potential temperature)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')

    out_path_th = f'{output_dir}/dual_core_bubble_convergence_study.png'
    out_path_th_alt = f'{output_dir}/dual_core_bubble_convergence_study_theta.png'
    os.makedirs(os.path.dirname(out_path_th), exist_ok=True)
    plt.savefig(out_path_th, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_th_alt, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bubble theta-convergence plot to {out_path_th} and {out_path_th_alt}")

    # --- Plot 2: Momentum (u) Convergence ---
    plt.figure(figsize=(10, 8))
    plt.loglog(dx_vals, errors_sisl['u'], 'o-', label='SISL', linewidth=2, markersize=8)
    plt.loglog(dx_vals, errors_se['u'], 's-', label='Split-explicit', linewidth=2, markersize=8)

    ref_start_u = max(errors_sisl['u'][0], errors_se['u'][0]) * 1.5
    ref_line_u = ref_start_u * (dx_vals / dx_vals[0])**2
    plt.loglog(dx_vals, ref_line_u, 'k--', label='Theoretical 2nd Order', alpha=0.7)

    plt.xlabel('Grid Spacing dx (m)', fontsize=12)
    plt.ylabel('L2 Error in u (m/s)', fontsize=12)
    plt.title('Spatial convergence study: Rising bubble benchmark (u momentum)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')

    out_path_u = f'{output_dir}/dual_core_bubble_convergence_study_u.png'
    plt.savefig(out_path_u, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bubble u-convergence plot to {out_path_u}")
