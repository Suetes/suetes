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

def run_case6():
    # ======================================================
    # 1. Configuration (Giraldo & Restelli Sec 3.6)
    # ======================================================
    # Domain: 240km wide, 30km high
    Lx = 240000.0
    Lz = 30000.0
    
    # Resolution: matches paper (dx=1200m, dz=240m) roughly
    # nx = 240000/1200 + 1 = 201
    # nz = 30000/240 + 1 = 126
    nx = 201
    nz = 126 
    
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Constants
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': 2.0e9 
    }
    
    # Time: 10 Hours to reach steady state
    t_end = 36000.0 
    dt = 0.25
    
    # ======================================================
    # 2. Grid (Witch of Agnesi)
    # ======================================================
    def h_agnesi(x):
        x_c = x - Lx/2.0
        hc = 1.0      # Linear regime
        ac = 10000.0  # Hydrostatic (Na/U = 10 >> 1)
        return hc / (1.0 + (x_c/ac)**2)

    print(f"Initializing Grid (nx={nx}, nz={nz})...")
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
    
    # Isothermal Atmosphere (T = 250K)
    T0 = 250.0
    N = constants['g'] / np.sqrt(constants['cp'] * T0)
    theta0 = T0
    
    Z_m = grid.Z_m
    state['background']['theta'] = theta0 * jnp.exp((N**2/constants['g']) * Z_m)
    state['background']['pi'] = 1.0 + (constants['g']**2 / (constants['cp']*theta0*N**2)) * \
                                (jnp.exp(-(N**2/constants['g']) * Z_m) - 1.0)
    state['background']['dtheta_dz'] = (N**2 / constants['g']) * state['background']['theta']
    state['background']['dpi_dz'] = -constants['g'] / (constants['cp'] * state['background']['theta'])

    # ======================================================
    # 4. Forcing & BCs
    # ======================================================
    U_FINAL = 20.0
    RAMP_PERIOD = 2000.0
    
    def forcing(t):
        u = jnp.where(t < RAMP_PERIOD, U_FINAL * jnp.sin(0.5*jnp.pi*t/RAMP_PERIOD)**2, U_FINAL)
        acc = jnp.where(t < RAMP_PERIOD, U_FINAL * 2.0 * jnp.sin(0.5*jnp.pi*t/RAMP_PERIOD) * jnp.cos(0.5*jnp.pi*t/RAMP_PERIOD) * (0.5*jnp.pi/RAMP_PERIOD), 0.0)
        return {'u_ref': u, 'u_acc': acc}
    
    def bcs(s, forcing):
        u_tgt = forcing['u_ref']

        # Inflow / Outflow
        s['u'] = s['u'].at[0, :].set(u_tgt).at[-1, :].set(s['u'][-2, :])
        
        # Tangent Flow
        u_bot = s['u'][:, 0]
        u_at_w = 0.5 * (u_bot[:-1] + u_bot[1:])
        slope = grid.metrics['w']['z_xi'][:, 0]
        s['w'] = s['w'].at[:, 0].set(u_at_w * slope).at[:, -1].set(0.0)
        return s

    # ======================================================
    # 5. Run
    # ======================================================
    stepper = RK4(model, dt)
    sim = Simulation(stepper, forcing, bcs)
    
    print(f"Starting Case 6 (t={t_end}s)...")
    final = sim.run(state, 0.0, t_end, dt, chunk_steps=500)

    # ======================================================
    # 6. Analytic & Plotting
    # ======================================================
    print("Computing Analytic Solution...")
    X = np.array(grid.X_m)
    Z = np.array(grid.Z_m)
    X_centered = X[:, 0] - Lx/2.0
    
    dx = grid.dx
    nk = grid.nx
    ks = 2.0 * np.pi * np.fft.fftfreq(nk, d=dx)
    
    # Agnesi FT: h_hat(k) computed numerically for grid consistency
    hc, ac = 1.0, 10000.0
    h_phys = hc / (1.0 + (X_centered/ac)**2)
    h_hat = np.fft.fft(h_phys)
    
    # Hydrostatic Dispersion: m = N/U * sgn(k) (Upstream tilt -> opposite sign)
    # Note: Giraldo paper Fig 10 shows upstream tilt.
    ms = np.sign(ks) * (N / U_FINAL)
    
    w_analytic = np.zeros_like(X)
    term_k = 1j * ks * U_FINAL * h_hat / nk
    
    for k_idx in range(grid.nz):
        z_level = Z[nk//2, k_idx]
        w_hat_z = 1j * ks * U_FINAL * h_hat * np.exp(1j * ms * z_level)
        w_analytic[:, k_idx] = np.fft.ifft(w_hat_z).real

    # Plotting with Crop (Fig 10 style)
    print("Plotting...")
    plt.figure(figsize=(12, 6))
    
    w_num = 0.5 * (final['w'][:, 1:] + final['w'][:, :-1])
    
    # Crop Region: x in [80km, 160km] (Centered at 120km, +/- 40km)
    plt.subplot(1, 2, 1)
    pcm = plt.pcolormesh(X, Z, w_num, cmap='RdBu_r', vmin=-0.005, vmax=0.005, shading='gouraud')
    plt.fill_between(X[:,0], grid.h_func(X[:,0]), -1000, color='k')
    plt.xlim(80000, 160000)
    plt.ylim(0, 12000)
    plt.title("Numerical (C-Grid)")
    plt.colorbar(label='w (m/s)')

    plt.subplot(1, 2, 2)
    pcm2 = plt.pcolormesh(X, Z, w_analytic, cmap='RdBu_r', vmin=-0.005, vmax=0.005, shading='gouraud')
    plt.fill_between(X[:,0], grid.h_func(X[:,0]), -1000, color='k')
    plt.xlim(80000, 160000)
    plt.ylim(0, 12000)
    plt.title("Analytic (Hydrostatic)")
    plt.colorbar(label='w (m/s)')
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, "case6_result.png")
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")

if __name__ == "__main__":
    run_case6()