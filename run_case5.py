import os
import numpy as np
import matplotlib.pyplot as plt
# Use numpy's emath.sqrt which handles negative numbers automatically (returns complex)
from numpy.lib.scimath import sqrt as csqrt 

import jax.numpy as jnp

USE_X64 = False 
from atmos_jax.utils import setup_jax
setup_jax(USE_X64)

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.dynamics import EulerSet1

# Import the transforms
from atmos_jax.core.transforms import GalChenSigma, HybridSigma, Sleve

def run_case5():
    # ======================================================
    # 1. Configuration
    # ======================================================
    Lx = 50000.0
    Lz = 21000.0
    nx = 201  
    nz = 65
    
    os.makedirs("figures", exist_ok=True)
    
    # Constants
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': 2.0e7 
    }
    
    # Time
    t_end = 18000.0
    dt = 0.15

    # Loop through vertical coordinates
    transforms = {
        "Gal-Chen": GalChenSigma(),
        "Hybrid": HybridSigma(scale_height=8000.0),
        "SLEVE": Sleve(scale_s=3000.0, n=1.35)
    }

    # ======================================================
    # MAIN LOOP
    # ======================================================
    for name, transform_obj in transforms.items():
        print(f"\n--- Running Case 5 with {name} Coordinates ---")

        # ======================================================
        # 2. Grid
        # ======================================================
        def h_schar(x):
            x_c = x - Lx/2.0
            hc = 250.0
            ac = 5000.0
            lam = 4000.0
            return hc * jnp.exp(-(x_c / ac)**2) * jnp.cos(jnp.pi * x_c / lam)**2

        grid = StaggeredGrid(nx, nz, Lx, Lz, h_schar, transform=transform_obj)
        model = EulerSet1(grid, constants)
        
        # ======================================================
        # 3. State
        # ======================================================
        state = {
            'u': jnp.zeros((nx+1, nz)), 'w': jnp.zeros((nx, nz+1)), 
            'pi_p': jnp.zeros((nx, nz)), 'theta_p': jnp.zeros((nx, nz)), 
            'background': {}
        }
        
        N = 0.01
        theta0 = 280.0
        Z_m = grid.Z_m
        
        state['background']['theta'] = theta0 * jnp.exp((N**2/constants['g']) * Z_m)
        state['background']['pi'] = 1.0 + (constants['g']**2 / (constants['cp']*theta0*N**2)) * \
                                    (jnp.exp(-(N**2/constants['g']) * Z_m) - 1.0)
        state['background']['dtheta_dz'] = (N**2 / constants['g']) * state['background']['theta']
        state['background']['dpi_dz'] = -constants['g'] / (constants['cp'] * state['background']['theta'])

        # ======================================================
        # 4. Forcing & BCs
        # ======================================================
        U_FINAL = 10.0
        RAMP_PERIOD = 1000.0
        
        def forcing(t):
            u = jnp.where(t < RAMP_PERIOD, U_FINAL * jnp.sin(0.5*jnp.pi*t/RAMP_PERIOD)**2, U_FINAL)
            acc = jnp.where(t < RAMP_PERIOD, U_FINAL * 2.0 * jnp.sin(0.5*jnp.pi*t/RAMP_PERIOD) * jnp.cos(0.5*jnp.pi*t/RAMP_PERIOD) * (0.5*jnp.pi/RAMP_PERIOD), 0.0)
            return {'u_ref': u, 'u_acc': acc}
        
        def bcs(s, forcing):
            u_tgt = forcing['u_ref']
            
            s['u'] = s['u'].at[0, :].set(u_tgt).at[-1, :].set(s['u'][-2, :])
            u_at_w = 0.5*(s['u'][:, 0][:-1] + s['u'][:, 0][1:])
            slope = grid.metrics['w']['z_xi'][:, 0]
            s['w'] = s['w'].at[:, 0].set(u_at_w * slope).at[:, -1].set(0.0)
            return s

        # ======================================================
        # 5. Run
        # ======================================================
        stepper = RK4(model, dt)
        sim = Simulation(stepper, forcing, bcs)
        
        print(f"Starting simulation ({name})...")
        final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

        # ======================================================
        # 6. Analytic Comparison
        # ======================================================
        print(f"Computing Exact Non-Hydrostatic Analytic Solution ({name})...")
        X = np.array(grid.X_m)
        Z = np.array(grid.Z_m)
        
        hc, ac, lam = 250.0, 5000.0, 4000.0
        X_centered = X[:, 0] - Lx/2.0
        
        dk = 2.0 * np.pi / 100000.0
        k_max = 2.0 * np.pi / 250.0
        ks = np.arange(dk, k_max, dk)
        
        k0 = 2.0 * np.pi / lam
        base_gauss = lambda k: np.exp(-(k * ac / 2.0)**2)
        factor = hc * np.sqrt(np.pi) * ac / 2.0
        H_k = factor * (base_gauss(ks) + 0.5*base_gauss(ks - k0) + 0.5*base_gauss(ks + k0))
        
        l_sq = (N / U_FINAL)**2
        ms = csqrt(l_sq - ks**2)
        m_final = np.sign(ks) * np.real(ms) + 1j * np.imag(ms)
        
        w_analytic = np.zeros_like(X)
        term_k = 1j * ks * U_FINAL * H_k * (dk / (2*np.pi))
        
        for i in range(nx):
            phase = np.exp(1j * (ks * X_centered[i] + np.outer(Z[i, :], m_final)))
            w_analytic[i, :] = 2.0 * np.real(np.sum(term_k * phase, axis=1))

        # ======================================================
        # 7. Plotting (Updated with Grid Lines)
        # ======================================================
        w_num = 0.5 * (final['w'][:, 1:] + final['w'][:, :-1])
        X_plot = X - Lx/2.0

        # 1. Cross Section
        z_target = 4000.0
        k_slice = np.abs(Z[0, :] - z_target).argmin()
        
        plt.figure(figsize=(10, 5))
        plt.plot(X_plot[:, 0], w_analytic[:, k_slice], 'k--', label='Analytic (Non-Hydro)')
        plt.plot(X_plot[:, 0], w_num[:, k_slice], 'r-', label='Numerical')
        plt.title(f"Vertical Velocity at z={Z[0, k_slice]:.0f}m ({name})")
        plt.legend()
        plt.savefig(f"figures/case5_cross_section_{name}.png")
        plt.close()
        
        # 2. Paper Match (With Grid Lines)
        plt.figure(figsize=(12, 6))
        
        # LEFT PANEL: Numerical
        plt.subplot(1, 2, 1)
        plt.pcolormesh(X_plot, Z, w_num, cmap='RdBu_r', vmin=-0.5, vmax=0.5, shading='gouraud')
        
        # --- NEW: OVERLAY GRID LINES ---
        # Convert corner coordinates to centered X for plotting
        Xc = np.array(grid.X_corner) - Lx/2.0
        Zc = np.array(grid.Z_corner)
        stride = 3 # Plot every 3rd line to avoid clutter
        
        for k in range(0, grid.nz+1, stride):
            # Fade grid lines near top for style (optional), or keep constant alpha
            alpha = 0.5 if k < grid.nz/2 else 0.3
            plt.plot(Xc[:, k], Zc[:, k], 'k-', linewidth=0.5, alpha=alpha)
        # -------------------------------

        plt.fill_between(X_plot[:,0], grid.h_func(X[:,0]), -1000, color='k')
        plt.xlim(-10000, 10000); plt.ylim(0, 10000)
        plt.title(f"Numerical w/ Grid Lines ({name})")

        # RIGHT PANEL: Analytic
        plt.subplot(1, 2, 2)
        plt.pcolormesh(X_plot, Z, w_analytic, cmap='RdBu_r', vmin=-0.5, vmax=0.5, shading='gouraud')
        plt.fill_between(X_plot[:,0], grid.h_func(X[:,0]), -1000, color='k')
        plt.xlim(-10000, 10000); plt.ylim(0, 10000)
        plt.title("Analytic (Exact Non-Hydro)")
        
        plt.tight_layout()
        plt.savefig(f"figures/case5_paper_match_{name}.png")
        plt.close()
        print(f"Saved figures for {name}.")

if __name__ == "__main__":
    run_case5()
