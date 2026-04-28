import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SemiImplicitSolver3D
from suetes.shared.transforms import SleveSimple

def test_phase2_implicit():
    print("=== PHASE 2: IMPLICIT MATRIX (LHS) CHECK ===")
    
    # 1. Setup Grid and Physics (Same as Phase 1)
    def schaer_2d(x, y):
        h0, a, lam = 2000.0, 15000.0, 8000.0
        return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

    nx, ny, nz = 50, 5, 20
    dx, dy, dz = 1000.0, 1000.0, 400.0
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=SleveSimple())
    op = CGridOperator3D(grid)
    
    CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.01)
    
    # 2. Initialize Solver
    dt = 10.0
    implicit_solver = SemiImplicitSolver3D(physics, dt)

    bg_state_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_state_ref)

    # 3. Create a perfect "Zero Forcing" RHS state
    # This represents a resting atmosphere exiting the advection step.
    rhs_prime = {
        'u': jnp.zeros_like(grid.Z_u),
        'v': jnp.zeros_like(grid.Z_v),
        'w': jnp.zeros_like(grid.Z_w),
        'pi': jnp.zeros_like(grid.Z_m),
        'eta_dot': jnp.zeros_like(grid.Z_w)
    }

    # 4. Push it through the GMRES solver
    sol = implicit_solver.solve(rhs_prime, bg_precomputed)

    max_u = jnp.max(jnp.abs(sol['u']))
    max_w = jnp.max(jnp.abs(sol['w']))
    max_pi = jnp.max(jnp.abs(sol['pi']))
    max_eta = jnp.max(jnp.abs(sol['eta_dot']))

    print(f"Solver Max U error:       {max_u:.4e} m/s")
    print(f"Solver Max W error:       {max_w:.4e} m/s")
    print(f"Solver Max Pi error:      {max_pi:.4e}")
    print(f"Solver Max Eta_dot error: {max_eta:.4e} 1/s")

    if max_u > 1e-10 or max_w > 1e-10:
        print("\n>>> FAILURE: The implicit solver is injecting spurious momentum.")
    else:
        print("\n>>> SUCCESS: The implicit solver cleanly inverts the resting state.")

if __name__ == "__main__":
    test_phase2_implicit()