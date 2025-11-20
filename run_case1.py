import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Precision
USE_X64 = False 
from atmos_jax.utils import setup_jax
setup_jax(USE_X64)

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.dynamics import EulerSet1

def run_case1():
    # 1. Config
    Lx = 300000.0
    Lz = 10000.0
    nx = 301
    nz = 51
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': 1.0e9 
    }
    
    t_end = 3000.0
    dt = 0.1 
    
    # 2. Grid
    def h_flat(x): return jnp.zeros_like(x)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_flat)
    
    # 3. Physics (Enable Periodic Mode)
    model = EulerSet1(grid, constants, periodic_x=True)
    
    # 4. State
    # FIX: u must be (nx+1, nz) to match grid metrics, even in periodic mode
    state = {
        'u': jnp.zeros((nx+1, nz)), 
        'w': jnp.zeros((nx, nz+1)), 
        'pi_p': jnp.zeros((nx, nz)), 
        'theta_p': jnp.zeros((nx, nz)), 
        'background': {}
    }
    
    N = 0.01
    theta0 = 300.0
    
    state['background']['theta'] = theta0 * jnp.exp((N**2/constants['g']) * grid.Z_m)
    state['background']['pi'] = 1.0 + (constants['g']**2 / (constants['cp']*theta0*N**2)) * \
                                (jnp.exp(-(N**2/constants['g']) * grid.Z_m) - 1.0)
    state['background']['dtheta_dz'] = (N**2 / constants['g']) * state['background']['theta']
    state['background']['dpi_dz'] = -constants['g'] / (constants['cp'] * state['background']['theta'])

    # Initialize u with Mean Flow (20 m/s)
    state['u'] = jnp.full_like(state['u'], 20.0)

    # Perturbation
    theta_c = 0.01
    h_c = 10000.0
    a_c = 5000.0
    x_c = 100000.0
    
    denom = 1.0 + ((grid.X_m - x_c) / a_c)**2
    num = jnp.sin(jnp.pi * grid.Z_m / h_c)
    state['theta_p'] = theta_c * (num / denom)

    # 5. Forcing & BCs
    def forcing(t):
        return {'u_ref': 20.0, 'u_acc': 0.0}
    
    def bcs(s, u_tgt=None):
        # Periodic X: We explicitly sync the redundant point u[-1] = u[0]
        # This helps numerical consistency even though operator handles wrapping
        s['u'] = s['u'].at[-1, :].set(s['u'][0, :])
        
        # Rigid Lid Z
        s['w'] = s['w'].at[:, 0].set(0.0).at[:, -1].set(0.0)
        return s

    # 6. Run
    stepper = RK4(model, dt)
    sim = Simulation(stepper, forcing, bcs)
    
    print(f"Starting Case 1 (Periodic)...")
    final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

    # 7. Plot
    print("Plotting...")
    plt.figure(figsize=(12, 5))
    
    levels = np.linspace(-0.002, 0.0035, 21)
    
    pcm = plt.contourf(grid.X_m, grid.Z_m, final['theta_p'], levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(pcm, label="Theta' (K)")
    
    plt.axvline(160000.0, color='k', linestyle='--', label='Expected Center (160km)')
    
    plt.title(f"Case 1: Periodic 4th-Order (t={t_end:.0f}s)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.legend()
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, "case1_result.png")
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")

if __name__ == "__main__":
    run_case1()