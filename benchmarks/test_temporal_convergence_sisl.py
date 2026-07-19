import os
import gc
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

# Keep the split-explicit method fixed throughout temporal refinement.  With a
# constant number of acoustic substeps, halving the outer timestep also halves
# the acoustic timestep.  Computing ns from dt with a lower bound changes the
# method when that bound becomes active and invalidates the final convergence
# ratio.
SPLIT_EXPLICIT_NS = 40
NUM_PROGRESS_UPDATES = 8


def release_jax_memory():
    """Release cached executables and collect arrays between independent runs."""
    jax.clear_caches()
    gc.collect()

def get_cell_center_slices(res_3d):
    """Extract cell-centered 2D slices for evaluation."""
    u = res_3d['u'][:, 1, :]
    w = res_3d['w'][:, 1, :]
    pi = res_3d['pi'][:, 1, :]
    th = res_3d['th_v'][:, 1, :]
    
    u_m = 0.5 * (u[:-1, :] + u[1:, :])
    w_m = 0.5 * (w[:, :-1] + w[:, 1:])
    return {
        'u': np.array(u_m),
        'w': np.array(w_m),
        'pi': np.array(pi),
        'th_v': np.array(th)
    }

def run_bubble_temporal(
    core_type, dt, dx=250.0, alpha=0.55,
    split_explicit_ns=SPLIT_EXPLICIT_NS,
):
    """Runs the bubble benchmark at a fixed spatial resolution and varied dt."""
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    if core_type == "sisl":
        core_kwargs = {
            "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, 
            "max_damp": 0.0, "alpha": alpha, "solver_tol": 1e-12, # <-- SET MAX_DAMP TO 0.0
            "solver_maxiter": 20, "solver_restart": 20
        }
    elif core_type == "split-explicit":
        # Use one constant refinement family: dt_acoustic = dt / ns.  In
        # particular, do not use max(4, int(dt / 0.1)); its floor activates at
        # dt=0.25 s and produces a spurious negative convergence rate.
        core_kwargs = {
            "dt": dt,
            "ns": split_explicit_ns,
            "nu_div_factor": 0.0,
            "nu_h_factor": 0.0,
            "damp_height": 7500.0,
            "max_damp": 0.05,
            "alpha": alpha,
        }
    else:
        raise ValueError(f"Unknown core: {core_type}")

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
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
    
    # Run the simulation
    t_end = 24.0
    total_steps = int(round(t_end / dt))
    chunk_steps = max(1, total_steps // NUM_PROGRESS_UPDATES)
    raw_res = sim.run(
        state, t_start=0.0, t_end=t_end, chunk_steps=chunk_steps
    )
    slices = get_cell_center_slices(raw_res)
    
    # The returned slices are host NumPy arrays, so all device-backed model
    # objects can be released before the next independent refinement run.
    del raw_res
    del state, tmp_phys, stepper, sim
    del bg_ref, bubble, X, Y, Z
    del op, grid
    release_jax_memory()
    
    return slices

def run_temporal_study(core_type, alpha=0.55):
    print(f"\n========================================")
    print(f"Running pure temporal convergence study for {core_type.upper()} (alpha={alpha})...")
    print(f"========================================")
    
    dx = 250.0
    dt_vals = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
    slices = []
    
    for dt in dt_vals:
        # Cleanup here is essential: at this point the previous invocation's
        # local frame (including its solver and compiled closures) is gone.
        release_jax_memory()
        result = run_bubble_temporal(core_type, dt, dx, alpha)
        slices.append(result)
        del result
        release_jax_memory()
        acoustic_info = (
            f" | ns={SPLIT_EXPLICIT_NS}, "
            f"dt_acoustic={dt / SPLIT_EXPLICIT_NS:.5f} s"
            if core_type == "split-explicit" else ""
        )
        print(
            f"Completed simulation at dt = {dt:4.2f} s{acoustic_info} "
            "| VRAM caches cleared."
        )

    keys = ['u', 'w', 'pi', 'th_v']
    errors = {k: [] for k in keys}
    
    for i in range(len(dt_vals) - 1):
        coarse_dt = slices[i]
        fine_dt = slices[i+1]
        
        # Crop 1500m around the boundaries to avoid Davies Sponge artifacts
        crop_xz = int(1500.0 / dx)
        
        for k in keys:
            coarse_dt_cropped = coarse_dt[k][crop_xz:-crop_xz, crop_xz:-crop_xz]
            fine_dt_cropped = fine_dt[k][crop_xz:-crop_xz, crop_xz:-crop_xz]
            
            # Since dx is identical, we can directly compute the difference between fields
            err = float(jnp.sqrt(jnp.mean((fine_dt_cropped - coarse_dt_cropped)**2)))
            errors[k].append(err)
            
        print(f"Timestep {dt_vals[i]:4.2f} -> {dt_vals[i+1]:4.2f}s | "
              f"err_u: {errors['u'][-1]:.2e} | err_w: {errors['w'][-1]:.2e} | "
              f"err_pi: {errors['pi'][-1]:.2e} | err_th: {errors['th_v'][-1]:.2e}")

    print("\n--- Temporal Self-Convergence Rates ---")
    rates = {k: [] for k in keys}
    for i in range(len(dt_vals) - 2):
        print(f"Timestep {dt_vals[i]:4.2f} -> {dt_vals[i+1]:4.2f}s:")
        for k in keys:
            if errors[k][i+1] == 0.0 or errors[k][i] == 0.0:
                rate = np.nan
            else:
                rate = np.log2(errors[k][i] / errors[k][i+1])
            rates[k].append(rate)
            print(f"  Rate {k:4s}: {rate:.2f}")

    return dt_vals[:-1], errors

if __name__ == "__main__":
    dt_vals, errors_sisl = run_temporal_study("sisl", alpha=0.5)
    _, errors_se = run_temporal_study("split-explicit", alpha=0.5)

    dt_plot = np.array(dt_vals)

    # --- Plotting Momentum (u) Temporal Convergence ---
    plt.figure(figsize=(10, 8))
    plt.loglog(dt_plot, errors_sisl['u'], 'o-', label='SISL', linewidth=2, markersize=8)
    plt.loglog(dt_plot, errors_se['u'], 's-', label='Split-explicit', linewidth=2, markersize=8)

    ref_start = max(errors_sisl['u'][0], errors_se['u'][0]) * 1.5
    ref_line_1st = ref_start * (dt_plot / dt_plot[0])**1
    ref_line_2nd = ref_start * (dt_plot / dt_plot[0])**2
    
    plt.loglog(dt_plot, ref_line_1st, 'k-.', label='Theoretical 1st Order', alpha=0.7)
    plt.loglog(dt_plot, ref_line_2nd, 'k--', label='Theoretical 2nd Order', alpha=0.7)

    plt.xlabel('Timestep dt (s)', fontsize=12)
    plt.ylabel('L2 Error in u (m/s)', fontsize=12)
    plt.title('Temporal convergence study: Rising bubble benchmark (u momentum)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')
    
    # Notice we flip the x-axis so smaller dt is on the right, matching spatial grid plotting
    plt.gca().invert_xaxis() 

    out_path_u = f'{output_dir}/temporal_convergence_study_u.png'
    plt.savefig(out_path_u, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved pure temporal convergence plot to {out_path_u}")
