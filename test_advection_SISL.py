import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SemiLagrangianAdvector3D
from suetes.shared.transforms import SleveSimple

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def schaer_2d(x, y):
    h0, a, lam = 2000.0, 15000.0, 8000.0
    return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

def get_prescribed_flow(grid, op, u_0):
    """Generates a perfect, non-divergent terrain-following flow."""
    u = u_0 * jnp.ones_like(grid.Z_u)
    v = jnp.zeros_like(grid.Z_v)
    eta_dot = jnp.zeros_like(grid.Z_w)
    
    u_m = op.avg(u, axis=0, from_loc='u', to_loc='m')
    u_w = op.avg(u_m, axis=2, from_loc='m', to_loc='w')
    w = u_w * grid.z_xi_w 
    return {'u': u, 'v': v, 'w': w, 'eta_dot': eta_dot}

def test_elevated_and_reversible():
    print("\n=== TEST: ELEVATED BUBBLE & REVERSIBILITY ===")
    nx, ny, nz = 150, 5, 50 
    dx, dy, dz = 1000.0, 1000.0, 400.0
    
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=sleve)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.0)
    
    dt = 10.0
    advector = SemiLagrangianAdvector3D(grid, physics, dt)

    # 1. Initialize Elevated Tracer
    xc, zc = -30000.0, 12000.0
    r_bubble = 2500.0
    X_m, Z_m = grid.x_m[:, None, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    tracer_initial = jnp.where(r < r_bubble, jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)

    # 2. Forward Advection (200 steps = 2000s)
    print("Advecting forward...")
    state_fwd = get_prescribed_flow(grid, op, u_0=10.0)
    tracer = tracer_initial
    for _ in range(200):
        coords_m = advector.compute_departure_indices(state_fwd, loc='m')
        tracer = advector.advect_cubic(tracer, coords_m, use_limiter=True)
    
    tracer_midpoint = tracer

    # 3. Backward Advection (200 steps = 2000s)
    print("Advecting backward...")
    state_bwd = get_prescribed_flow(grid, op, u_0=-10.0)
    for _ in range(200):
        coords_m = advector.compute_departure_indices(state_bwd, loc='m')
        tracer = advector.advect_cubic(tracer, coords_m, use_limiter=True)
        
    tracer_final = tracer

    # 4. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    y_idx = ny // 2
    X_2d, _ = jnp.meshgrid(grid.x_m, grid.z_m, indexing='ij')

    for ax, data, title in zip(axes, 
                               [tracer_initial, tracer_midpoint, tracer_final], 
                               ["Initial (T=0s)", "Midpoint (T=2000s, over mountain)", "Final Reversed (T=4000s)"]):
        
        # Force aspect ratio to be equal so circles look like circles
        cf = ax.contourf(X_2d/1000, grid.Z_m[:, y_idx, :]/1000, data[:, y_idx, :], levels=15, cmap='Reds')
        ax.plot(grid.x_m/1000, grid.Z_w[:, y_idx, 0]/1000, color='black', linewidth=2)
        ax.set_aspect('equal') # THIS fixes the visual distortion
        ax.set_title(title)
        ax.set_ylabel("z (km)")
        
    axes[2].set_xlabel("x (km)")
    plt.tight_layout()
    plt.savefig('test_advection_purity.png', dpi=150)
    print(">>> Saved test_advection_purity.png")

if __name__ == "__main__":
    test_elevated_and_reversible()