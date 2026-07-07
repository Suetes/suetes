import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from types import SimpleNamespace

# Enable X64 for precise convergence tests
jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

def block_average_2d(field_2d, factor):
    nx, nz = field_2d.shape
    return jnp.mean(field_2d.reshape(nx // factor, factor, nz // factor, factor), axis=(1, 3))

def run_bubble_at_resolution(core_type, dx, alpha=0.55):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    if core_type == "sisl":
        dt = (dx / 125.0) * 5.0  # Linear scaling of timestep with grid spacing
        core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha}
    elif core_type == "split-explicit":
        dt = (dx / 125.0) * 2.5
        core_kwargs = {"dt": dt, "ns": 24, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": alpha}
    else:
        raise ValueError(f"Unknown core: {core_type}")

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    from suetes.regional3d.euler import Euler3D
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

    stepper, dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i*dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    return sim.run(state, t_start=0.0, t_end=200.0, chunk_steps=int(200.0/dt))

def run_study(core_type, alpha=0.55):
    print(f"Running study for {core_type} (alpha={alpha})...")
    res_250 = run_bubble_at_resolution(core_type, 250.0, alpha)
    res_125 = run_bubble_at_resolution(core_type, 125.0, alpha)
    res_062 = run_bubble_at_resolution(core_type, 62.5, alpha)
    res_031 = run_bubble_at_resolution(core_type, 31.25, alpha)

    err_1 = float(jnp.sqrt(jnp.mean((block_average_2d(res_125['th_v'][:, 1, :], 2) - res_250['th_v'][:, 1, :])**2)))
    err_2 = float(jnp.sqrt(jnp.mean((block_average_2d(res_062['th_v'][:, 1, :], 4) - block_average_2d(res_125['th_v'][:, 1, :], 2))**2)))
    err_3 = float(jnp.sqrt(jnp.mean((block_average_2d(res_031['th_v'][:, 1, :], 8) - block_average_2d(res_062['th_v'][:, 1, :], 4))**2)))
    
    order_12 = np.log2(err_1 / err_2)
    order_23 = np.log2(err_2 / err_3)
    avg_order = (order_12 + order_23) / 2.0
    print(f"-> Errors: {err_1:.6e}, {err_2:.6e}, {err_3:.6e} | Avg Order: {avg_order:.2f}")
    return [err_1, err_2, err_3], avg_order

# Gather data
errors_se, order_se = run_study("split-explicit")
# errors_sisl_default, order_sisl_default = run_study("sisl", alpha=0.55)
errors_sisl_symmetric, order_sisl_symmetric = run_study("sisl", alpha=0.5)

# Plotting setup
plt.figure(figsize=(10, 8))

dx_vals = np.array([250.0, 125.0, 62.5])

# Plot empirical errors
plt.loglog(dx_vals, errors_sisl_symmetric, 'o-', label=rf'SISL $\theta_v$-error (Avg Rate = {order_sisl_symmetric:.2f})', linewidth=2, markersize=8)
plt.loglog(dx_vals, errors_se, 's-', label=rf'Split-Explicit  $\theta_v$-error (Avg Rate = {order_se:.2f})', linewidth=2, markersize=8)

# Add reference 2nd order convergence slope
ref_start = errors_se[0] * 1.2
ref_line = ref_start * (dx_vals / dx_vals[0])**2
plt.loglog(dx_vals, ref_line, 'k--', label='Theoretical 2nd Order', alpha=0.7)

plt.xlabel(r'Grid Spacing $\Delta x$ (m)', fontsize=12)
plt.ylabel(r'$L_2$ Error in $\theta_v$', fontsize=12)
plt.title('Spatial convergence study: Rising bubble benchmark', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10, loc='lower right')

# Save plot to centralized output directory
out_path = 'output/plots/dual_core_bubble_convergence_study.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved convergence plot to {out_path}")
