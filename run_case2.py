import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

USE_X64 = False
from suetes.utils import setup_jax
setup_jax(USE_X64)

from suetes.core import StaggeredGrid, SSPRK3, Simulation
from suetes.dynamics import EulerSet1

def run_case2():
    # 1. Config
    Lx, Lz = 1000.0, 1000.0
    nx, nz = 101, 101 
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Constants
    # nu4: Reduced to 200.0 for sharper rotors
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': 200.0 
    }
    
    t_end = 700.0
    dt = 0.005
    
    # 2. Grid
    def h_flat(x): return jnp.zeros_like(x)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_flat)
    
    # 3. Physics (No Sponge)
    class BubbleSolver(EulerSet1):
        def _init_sponge(self):
            return jnp.zeros_like(self.grid.X_m)

    model = BubbleSolver(grid, constants)
    
    # 4. State
    state = {
        'u': jnp.zeros((nx+1, nz)), 'w': jnp.zeros((nx, nz+1)), 
        'pi_p': jnp.zeros((nx, nz)), 'theta_p': jnp.zeros((nx, nz)),
        'background': {}
    }
    
    theta0 = 300.0
    state['background']['theta'] = jnp.full_like(grid.Z_m, theta0)
    state['background']['pi'] = 1.0 - (constants['g'] * grid.Z_m) / (constants['cp'] * theta0)
    state['background']['dtheta_dz'] = jnp.zeros_like(grid.Z_m)
    state['background']['dpi_dz'] = np.full_like(grid.Z_m, -constants['g']/(constants['cp']*theta0))

    # Perturbation
    xc, zc, rc, theta_c = 500.0, 350.0, 250.0, 0.5
    r = jnp.sqrt((grid.X_m - xc)**2 + (grid.Z_m - zc)**2)
    mask = r <= rc
    state['theta_p'] = jnp.where(mask, 0.5 * theta_c * (1.0 + jnp.cos(jnp.pi * r / rc)), 0.0)

    # 5. Run Setup
    
    # FIX: Define forcing as a function, not a dict
    def forcing(t):
        return {'u_ref': 0.0, 'u_acc': 0.0}
    
    def bcs(s, u_tgt=None):
        s['u'] = s['u'].at[[0,-1], :].set(0.0)
        s['w'] = s['w'].at[:, [0,-1]].set(0.0)
        return s

    # Use SSPRK3 for better shock/gradient handling
    stepper = SSPRK3(model, dt)
    sim = Simulation(stepper, forcing, bcs)
    
    print(f"Starting Case 2 (High-Res Physics)...")
    
    # Chunk steps: 5000 steps = 5 seconds per print
    final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

    # 6. Plot
    print("Plotting...")
    plt.figure(figsize=(7, 7))
    
    # Fine contours to match paper (Interval 0.025)
    # Range -0.05 to 0.525
    levels = np.arange(-0.05, 0.526, 0.025)
    
    pcm = plt.contourf(grid.X_m, grid.Z_m, final['theta_p'], levels=levels, cmap='jet')
    plt.colorbar(pcm, label="Theta' (K)")
    plt.contour(grid.X_m, grid.Z_m, final['theta_p'], levels=levels, colors='k', linewidths=0.5)
    
    plt.title(f"Case 2: Rising Bubble (t={t_end:.0f}s)\nSSPRK3, nu4={constants['nu4']:.0f}")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.axis('scaled')
    plt.xlim(0, 1000); plt.ylim(0, 1000)
    
    outfile = os.path.join(output_dir, "case2_result.png")
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")

if __name__ == "__main__":
    run_case2()