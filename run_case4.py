import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

USE_X64 = False
from suetes.utils import setup_jax
setup_jax(USE_X64)

from suetes.core import StaggeredGrid, SSPRK3, Simulation
from suetes.dynamics import EulerSet1

def run_case4():
    # ======================================================
    # 1. Configuration (Giraldo & Restelli Sec 3.4)
    # ======================================================
    # Domain: 25.6 km x 6.4 km
    Lx = 25600.0
    Lz = 6400.0
    
    # High Resolution: dx = 50m (Matches Fig 7d)
    # nx = 25600 / 50 + 1 = 513
    nx = 513
    nz = 129 
    
    # Output Directory Setup
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Constants
    # nu: 75.0 (Physical)
    # nu4: 0.0 (Disabled to prevent artificial drag)
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu': 75.0, 'nu4': 0.0 
    }
    
    t_end = 900.0
    dt = 0.05 # Reduced for 50m grid (Acoustic CFL ~ 0.3)
    
    # ======================================================
    # 2. Grid
    # ======================================================
    def h_flat(x): return jnp.zeros_like(x)
    print(f"Initializing Grid (nx={nx}, nz={nz})...")
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_flat)
    
    # ======================================================
    # 3. Physics
    # ======================================================
    class Case4Solver(EulerSet1):
        def _init_sponge(self): return jnp.zeros_like(self.grid.X_m)

    model = Case4Solver(grid, constants)
    
    # ======================================================
    # 4. State
    # ======================================================
    state = {'u': jnp.zeros((nx+1, nz)), 'w': jnp.zeros((nx, nz+1)), 
             'pi_p': jnp.zeros((nx, nz)), 'theta_p': jnp.zeros((nx, nz)), 'background': {}}
    
    theta0 = 300.0
    state['background']['theta'] = jnp.full_like(grid.Z_m, theta0)
    state['background']['pi'] = 1.0 - (constants['g'] * grid.Z_m) / (constants['cp'] * theta0)
    state['background']['dtheta_dz'] = jnp.zeros_like(grid.Z_m)
    state['background']['dpi_dz'] = jnp.full_like(grid.Z_m, -constants['g']/(constants['cp']*theta0))

    # Cold Bubble (Centered at x=0)
    xc, zc = 0.0, 3000.0
    xr, zr = 4000.0, 2000.0
    theta_c = -15.0
    
    dist = jnp.sqrt( ((grid.X_m - xc)/xr)**2 + ((grid.Z_m - zc)/zr)**2 )
    state['theta_p'] = jnp.where(dist <= 1.0, 0.5 * theta_c * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)

    # ======================================================
    # 5. Run
    # ======================================================
    def forcing(t): return {'u_ref': 0.0, 'u_acc': 0.0}
    
    def bcs(s, u_tgt=None):
        s['u'] = s['u'].at[[0,-1], :].set(0.0)
        s['w'] = s['w'].at[:, [0,-1]].set(0.0)
        return s

    stepper = SSPRK3(model, dt)
    sim = Simulation(stepper, forcing, bcs)
    
    print(f"Starting Case 4 High-Res (nu={constants['nu']})...")
    # Chunk steps: 5000 * 0.05 = 250s per print
    final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

    # ======================================================
    # 6. Plot
    # ======================================================
    print("Plotting...")
    plt.figure(figsize=(12, 4))
    
    # Levels: -16 to 0, step 1.0 (Matches Fig 7)
    levels = np.arange(-16.0, 0.1, 0.5)
    
    pcm = plt.contourf(grid.X_m, grid.Z_m, final['theta_p'], levels=levels, cmap='Blues_r')
    plt.colorbar(pcm, label="Theta' (K)")
    
    # Thin black contours for detail
    plt.contour(grid.X_m, grid.Z_m, final['theta_p'], levels=levels[::2], colors='k', linewidths=0.5)
    
    plt.title(f"Case 4: Density Current 50m (t={t_end:.0f}s)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.axis('scaled')
    plt.xlim(0, 19200)
    plt.ylim(0, 4000)
    
    outfile = os.path.join(output_dir, "case4_result.png")
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")

if __name__ == "__main__":
    run_case4()