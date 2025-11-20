import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# Force CPU for easy debugging
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics import EulerSet1

def debug_physics_terms():
    print("=== DIAGNOSTIC: PHYSICS TERM CHECK ===")
    
    # 1. Setup Case 5 Geometry
    Lx, Lz = 50000.0, 21000.0
    nx, nz = 201, 65
    
    def h_schar(x):
        xc = x - Lx/2.0
        return 250.0 * jnp.exp(-(xc/5000)**2) * jnp.cos(jnp.pi*xc/4000)**2
    
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_schar)
    constants = {'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 'nu4': 0.0}
    
    model = EulerSet1(grid, constants)
    
    # 2. Initialize State with CONSTANT WIND U=10
    # This should immediately trigger vertical motion w = u * slope
    U_TEST = 10.0
    
    state = {
        'u': jnp.ones((nx+1, nz)) * U_TEST, # Flow 10 m/s everywhere
        'w': jnp.zeros((nx, nz+1)),         # W starts at 0
        'pi_p': jnp.zeros((nx, nz)),
        'theta_p': jnp.zeros((nx, nz)),
        'background': {}
    }
    
    # Enforce BCs manually for this test
    # W_surface = U * slope
    slope = grid.metrics['w']['z_xi'][:, 0]
    state['w'] = state['w'].at[:, 0].set(U_TEST * slope)
    
    # Background Stratification
    N, th0 = 0.01, 280.0
    state['background']['theta'] = th0 * jnp.exp((N**2/9.81) * grid.Z_m)
    state['background']['pi'] = 1.0 + (9.81**2/(1004.5*th0*N**2)) * (jnp.exp(-(N**2/9.81)*grid.Z_m) - 1.0)
    state['background']['dtheta_dz'] = (N**2 / 9.81) * state['background']['theta']
    state['background']['dpi_dz'] = -9.81 / (1004.5 * state['background']['theta'])
    
    # 3. Compute Tendencies (RHS)
    forcing = {'u_ref': U_TEST, 'u_acc': 0.0}
    rhs = model.compute_rhs(state, forcing)
    
    # 4. Analyze
    print("\n--- Background State Check ---")
    dth_max = np.max(state['background']['dtheta_dz'])
    print(f"Max dTheta/dz: {dth_max:.5f} (Should be ~0.003)")
    
    print("\n--- Tendency Check (t=0) ---")
    rhs_th_min = np.min(rhs['theta_p'])
    rhs_th_max = np.max(rhs['theta_p'])
    print(f"RHS Theta (Generation): Min {rhs_th_min:.5f}, Max {rhs_th_max:.5f}")
    
    if abs(rhs_th_max) < 1e-6:
        print("[FAIL] No Theta perturbation generated! Buoyancy is dead.")
    else:
        print("[PASS] Theta perturbation is active.")

    # 5. Plot Tendencies
    plt.figure(figsize=(12, 8))
    
    # Plot A: The Forcing W (Bottom Boundary)
    plt.subplot(3, 1, 1)
    plt.plot(grid.X_w[:, 0], state['w'][:, 0])
    plt.title("Initial W at Surface (Forcing)")
    plt.ylabel("w (m/s)")
    
    # Plot B: RHS Theta (Source of Buoyancy)
    # Should look like the mountain (Updraft -> Cooling -> Negative Theta source)
    plt.subplot(3, 1, 2)
    plt.pcolormesh(grid.X_m, grid.Z_m, rhs['theta_p'], cmap='RdBu_r', shading='auto')
    plt.colorbar(label="dTheta'/dt")
    plt.title("Theta Tendency (Adiabatic Cooling)")
    plt.ylim(0, 5000)
    
    # Plot C: RHS W (Response)
    # Should show reaction to pressure
    plt.subplot(3, 1, 3)
    w_tend = 0.5*(rhs['w'][:, 1:] + rhs['w'][:, :-1])
    plt.pcolormesh(grid.X_m, grid.Z_m, w_tend, cmap='RdBu_r', shading='auto')
    plt.colorbar(label="dw/dt")
    plt.title("Vertical Acceleration")
    plt.ylim(0, 5000)
    
    plt.tight_layout()
    plt.savefig('debug.png')
    plt.show()

if __name__ == "__main__":
    debug_physics_terms()