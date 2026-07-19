import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

output_dir = "output/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "sisl"  # Toggle to "sisl" or "split-explicit"
t_end = 7200.0 
N_BV_STRATIFICATION = 0.01    # Crucial for gravity wave generation

def schaer_mountain(x, y):
    h0 = 250.0      # Maximum height of 250m
    a = 5000.0      # Envelope half-width of 5km
    lam = 4000.0    # Wavelength of 4km
    return h0 * jnp.exp(-(x / a)**2) * jnp.cos(jnp.pi * x / lam)**2

# --- SETUP GRID & CORE-SPECIFIC TIMESTEPS ---
nx, ny, nz = 200, 3, 50
dx, dy, dz = 500.0, 500.0, 400.0  

if CORE_TYPE.lower() == "sisl":
    dt = 4.0
    core_kwargs = {
        "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 12000.0, "max_damp": 0.5, "N_bv": N_BV_STRATIFICATION
    }
elif CORE_TYPE.lower() == "split-explicit":
    dt = 4.0
    core_kwargs = {
        "dt": dt, "ns": 6, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 12000.0, "max_damp": 0.5, "N_bv": N_BV_STRATIFICATION
    }

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=schaer_mountain)
op = CGridOperator3D(grid)
constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

u_bg = 10.0

# --- INITIALIZE STATE ---
# Explicitly pass N_bv so the background state is built correctly
tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=N_BV_STRATIFICATION)

bg_ref = {
    'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
           (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
    'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
}

state = {
    'u': jnp.ones((nx+1, ny, nz)) * u_bg, 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
    'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'], 'th_v': bg_ref['th_v']
}

# --- BOUNDARY CONDITION RELAXATION ---
sponge_depth = 8
mask_x = jnp.where(jnp.minimum(jnp.arange(nx), nx - jnp.arange(nx)) < sponge_depth, 
                   jnp.cos(0.5 * jnp.pi * jnp.minimum(jnp.arange(nx), nx - jnp.arange(nx)) / sponge_depth)**2, 0.0)[:, None, None]

def bc_fn(state_in, forcing):
    blended = {}
    for k in state_in.keys():
        if k in ['u', 'v', 'th_v', 'pi', 'rho']:
            m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
            
            # Explicitly define the exterior values for velocity vs thermodynamics
            if k == 'u':
                ext_val = jnp.ones_like(state_in[k]) * u_bg
            elif k == 'v':
                ext_val = jnp.zeros_like(state_in[k])
            else:
                ext_val = bg_ref[k]
                
            blended[k] = (1.0 - m) * state_in[k] + m * ext_val
        else:
            blended[k] = state_in[k]
    return blended

# --- FACTORY INSTANTIATION ---
stepper, dt = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
    initial_state=state, **core_kwargs
)

def step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
    return next_state, jnp.max(jnp.abs(next_state['w']))

sim = Simulation(step_fn=step_fn, dt=dt)

if __name__ == "__main__":
    print(f"\n[BENCHMARK] Launching {CORE_TYPE.upper()} Schär Mountain (dt={dt}s)...")
    final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=50)

    # --- VISUALIZE WAVESTRUCTURES ---
    x_start, x_end = (nx // 4), 3 * (nx // 4)
    X_plot, _ = jnp.meshgrid(grid.x_m[x_start:x_end] / 1000.0, jnp.arange(nz + 1), indexing='ij')
    Z_plot = grid.Z_w[x_start:x_end, 1, :] / 1000.0 

    plt.figure(figsize=(12, 6))
    plt.contourf(X_plot, Z_plot, final_state['w'][x_start:x_end, 1, :], levels=jnp.linspace(-2.0, 2.0, 41), cmap='RdBu_r', extend='both')
    plt.colorbar(label='Vertical Velocity W (m/s)')
    plt.title(f'{CORE_TYPE.capitalize()} Schär Mountain Waves at T = {t_end}s')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.fill_between(grid.x_m[x_start:x_end] / 1000.0, 0, schaer_mountain(grid.x_m[x_start:x_end], 0.0) / 1000.0, color='black')

    plt.savefig(f'{output_dir}/schaer_mountain_{CORE_TYPE.lower()}_{t_end}.png', dpi=150, bbox_inches='tight')
    print(f"[PLOTTING] Saved plot to '{output_dir}/schaer_mountain_{CORE_TYPE.lower()}_{t_end}.png'")

    # --- MOMENTUM FLUX CALCULATION ---
    u_m = 0.5 * (final_state['u'][1:, 1, :] + final_state['u'][:-1, 1, :])
    w_m = 0.5 * (final_state['w'][:, 1, 1:] + final_state['w'][:, 1, :-1])
    momentum_flux = jnp.sum(final_state['rho'][:, 1, :] * (u_m - u_bg) * w_m * grid.dx, axis=0)

    plt.figure(figsize=(6, 8))
    plt.plot(momentum_flux, jnp.mean(grid.Z_m[:, 1, :], axis=0) / 1000.0, color='black', linewidth=2)
    plt.axhline(12.0, color='red', linestyle='--', label='Sponge Layer Base')
    plt.axvline(0.0, color='gray', linestyle='-', linewidth=0.5)
    plt.ylim(0, 20.0)
    plt.xlabel(r"Momentum Flux $\int \rho u' w' dx$ (kg / s^2)")
    plt.ylabel("Altitude (km)")
    plt.title(f"Momentum Flux Profile ({CORE_TYPE.capitalize()})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    flux_plot_path = f'{output_dir}/schaer_mountain_momentum_flux_{CORE_TYPE.lower()}_{t_end}.png'
    plt.savefig(flux_plot_path, dpi=150, bbox_inches='tight')
    print(f"[PLOTTING] Saved flux profile to '{flux_plot_path}'")