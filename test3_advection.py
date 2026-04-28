import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SemiLagrangianAdvector3D
from suetes.shared.transforms import SleveSimple

def test_phase3_advection():
    print("=== PHASE 3: PURE ADVECTION & TRAJECTORY CHECK ===")
    
    def schaer_2d(x, y):
        h0, a, lam = 2000.0, 15000.0, 8000.0
        return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

    nx, ny, nz = 100, 5, 20
    dx, dy, dz = 1000.0, 1000.0, 400.0
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=SleveSimple())
    op = CGridOperator3D(grid)
    
    CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.01)

    dt = 10.0
    advector = SemiLagrangianAdvector3D(grid, physics, dt)

    # 1. Prescribe a perfect, steady terrain-following flow
    u_prescribed = 10.0 * jnp.ones_like(grid.Z_u)
    v_prescribed = jnp.zeros_like(grid.Z_v)
    eta_dot_prescribed = jnp.zeros_like(grid.Z_w) # Flow perfectly hugs the zeta surfaces

    # Calculate Cartesian w required to maintain eta_dot = 0
    u_m = op.avg(u_prescribed, axis=0, from_loc='u', to_loc='m')
    u_w = op.avg(u_m, axis=2, from_loc='m', to_loc='w')
    w_prescribed = u_w * grid.z_xi_w 

    state = {
        'u': u_prescribed,
        'v': v_prescribed,
        'w': w_prescribed,
        'eta_dot': eta_dot_prescribed
    }

    # 2. Initialize a sharp tracer bubble upstream
    xc, zc = 10000.0, 4000.0
    r_bubble = 3000.0
    X_m, Z_m = grid.x_m[:, None, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    tracer = jnp.where(r < r_bubble, 1.0, 0.0)

    # 3. Compute departure points and advect
    coords_m = advector.compute_departure_indices(state, loc='m')
    tracer_next = advector.advect_linear(tracer, coords_m)

    max_val = jnp.max(tracer_next)
    min_val = jnp.min(tracer_next)

    print(f"Initial Tracer Max/Min: {jnp.max(tracer):.4f} / {jnp.min(tracer):.4f}")
    print(f"Final Tracer Max/Min:   {max_val:.4f} / {min_val:.4f}")

    if max_val > 1.05 or min_val < -0.05:
        print("\n>>> FAILURE: Advection scheme is unbounded. Trajectories are blowing up.")
    else:
        print("\n>>> SUCCESS: Advection trajectories and interpolation are strictly bounded.")

if __name__ == "__main__":
    test_phase3_advection()