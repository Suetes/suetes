import os
import numpy as np
import matplotlib.pyplot as plt
# Use numpy's lib.scimath.sqrt to handle complex numbers automatically
from numpy.lib.scimath import sqrt as csqrt 

import jax.numpy as jnp

USE_X64 = False 
from atmos_jax.utils import setup_jax
setup_jax(USE_X64)

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.dynamics import EulerSet1

def run_case7():
    # ======================================================
    # 1. Configuration (Giraldo & Restelli Sec 3.7)
    # ======================================================
    
    # Output Directory Setup
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Domain: 144km wide, 30km high
    Lx = 144000.0
    Lz = 30000.0
    
    # Resolution: dx = 360m
    nx = 401  
    nz = 101   
    
    # Constants
    # nu4: Tuned for dx=360m
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': 5.0e8 
    }
    
    # Time: 5 hours
    t_end = 18000.0
    dt = 0.25
    
    # ======================================================
    # 2. Grid (Witch of Agnesi - Non-Hydrostatic)
    # ======================================================
    def h_agnesi(x):
        x_c = x - Lx/2.0
        hc = 1.0    # Linear regime height
        ac = 1000.0 # Narrow width -> Non-hydrostatic regime
        return hc / (1.0 + (x_c/ac)**2)

    print(f"Initializing Grid (nx={nx}, nz={nz}, dx={Lx/(nx-1):.1f}m)...")
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_agnesi)
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
    RAMP_PERIOD = 2000.0
    
    def forcing(t):
        u = jnp.where(t < RAMP_PERIOD, U_FINAL * jnp.sin(0.5*jnp.pi*t/RAMP_PERIOD)**2, U_FINAL)
        # acc = dU/dt
        k = 0.5 * jnp.pi / RAMP_PERIOD
        acc = jnp.where(t < RAMP_PERIOD, U_FINAL * 2.0 * jnp.sin(k*t) * jnp.cos(k*t) * k, 0.0)
        return {'u_ref': u, 'u_acc': acc}
    
    def bcs(s, u_tgt):
        # Inflow (Left)
        s['u'] = s['u'].at[0, :].set(u_tgt)
        
        # Outflow (Right) - Neumann to let waves exit
        s['u'] = s['u'].at[-1, :].set(s['u'][-2, :])
        
        # Tangent Flow Bottom
        u_bot = s['u'][:, 0]
        u_at_w = 0.5 * (u_bot[:-1] + u_bot[1:])
        slope = grid.metrics['w']['z_xi'][:, 0]
        s['w'] = s['w'].at[:, 0].set(u_at_w * slope)
        
        # Rigid Lid Top
        s['w'] = s['w'].at[:, -1].set(0.0)
        return s

    # ======================================================
    # 5. Run
    # ======================================================
    stepper = RK4(model, dt)
    sim = Simulation(stepper, forcing, bcs)
    
    print(f"Starting Case 7 (t={t_end}s)...")
    final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

    # ======================================================
    # 6. Analytic Solution
    # ======================================================
    print("Computing Analytic Solution...")
    # Use CPU/Numpy
    X = np.array(grid.X_m)
    Z = np.array(grid.Z_m)
    
    dx = grid.dx
    nk = grid.nx
    ks = 2.0 * np.pi * np.fft.fftfreq(nk, d=dx)
    
    hc, ac = 1.0, 1000.0
    X_centered = X[:, 0] - Lx/2.0
    
    # 1. FT of Terrain (Agnesi)
    h_phys = hc / (1.0 + (X_centered/ac)**2)
    h_hat = np.fft.fft(h_phys)
    
    # 2. Vertical Wavenumber (Full Non-Hydrostatic)
    # m = sqrt(Sc^2 - k^2) where Sc = N/U (Scorer Parameter)
    # If k < Sc -> m is real (Propagating)
    # If k > Sc -> m is imaginary (Evanescent)
    scorer = N / U_FINAL
    ms_squared = scorer**2 - ks**2
    ms = csqrt(ms_squared) # returns complex result
    
    # Sign Selection for Radiation Condition:
    # m_prop = sign(ks) * real(ms) (Upstream Phase Tilt)
    # m_evan = 1j * imag(ms) (Decay)
    
    m_final = np.zeros_like(ms)
    m_final += np.sign(ks) * np.real(ms)
    m_final += 1j * np.abs(np.imag(ms))
    
    w_analytic = np.zeros_like(X)
    term_k = 1j * ks * U_FINAL * h_hat * (dx / Lx) # Scaling fix
    # Standard ifft approach:
    
    for k_idx in range(grid.nz):
        z_level = Z[nk//2, k_idx]
        w_hat_z = 1j * ks * U_FINAL * h_hat * np.exp(1j * m_final * z_level)
        w_analytic[:, k_idx] = np.fft.ifft(w_hat_z).real

    # ======================================================
    # 7. Plotting (Cropped to match Paper)
    # ======================================================
    print("Plotting...")
    plt.figure(figsize=(14, 6))
    
    w_num = 0.5 * (final['w'][:, 1:] + final['w'][:, :-1])
    
    # Crop Region: x in [60km, 105km], z in [0, 12km]
    xlims = [60000, 105000]
    ylims = [0, 12000]
    
    plt.subplot(1, 2, 1)
    pcm = plt.pcolormesh(X, Z, w_num, cmap='RdBu_r', vmin=-0.005, vmax=0.005, shading='gouraud')
    plt.fill_between(X[:,0], grid.h_func(X[:,0]), -1000, color='k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title("Numerical (C-Grid)")
    plt.colorbar(label='w (m/s)')

    plt.subplot(1, 2, 2)
    pcm2 = plt.pcolormesh(X, Z, w_analytic, cmap='RdBu_r', vmin=-0.005, vmax=0.005, shading='gouraud')
    plt.fill_between(X[:,0], grid.h_func(X[:,0]), -1000, color='k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title("Analytic (Non-Hydrostatic)")
    plt.colorbar(label='w (m/s)')
    
    plt.tight_layout()
    
    outfile = os.path.join(output_dir, "case7_result.png")
    plt.savefig(outfile, dpi=150)
    print(f"[+] Saved figure to: {outfile}")

if __name__ == "__main__":
    run_case7()