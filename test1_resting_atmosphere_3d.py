import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.shared.transforms import SleveSimple

def test_resting_atmosphere():
    print("=== PHASE 1: RESTING ATMOSPHERE OVER MOUNTAIN ===")
    
    # 1. Setup Grid and Schaer Mountain
    def schaer_2d(x, y):
        h0, a, lam = 2000.0, 15000.0, 8000.0
        return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

    nx, ny, nz = 50, 5, 20
    dx, dy, dz = 1000.0, 1000.0, 400.0
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=SleveSimple())
    op = CGridOperator3D(grid)
    
    # Need constants to run physics
    CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.01)

    # 2. Setup the "State Prime"
    # To test the metric terms, we need a pi_prime that varies with height.
    # Let's introduce a uniform cold anomaly (1K) to the whole domain.
    # This should create a purely vertical hydrostatic pressure adjustment,
    # but NO horizontal winds!
    
    pi_prime_anomaly = -0.01 * jnp.ones_like(grid.Z_m) 
    
    state_prime = {
        'u': jnp.zeros_like(grid.Z_u),
        'v': jnp.zeros_like(grid.Z_v),
        'w': jnp.zeros_like(grid.Z_w),
        'pi': pi_prime_anomaly,
        'eta_dot': jnp.zeros_like(grid.Z_w)
    }

    bg_state_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_state_ref)

    # 3. Get Explicit Tendencies
    tends = physics.get_tendencies(state_prime, bg_precomputed)

    max_tend_u = jnp.max(jnp.abs(tends['u']))
    max_tend_w = jnp.max(jnp.abs(tends['w']))

    print(f"Max Horizontal Acceleration (tend_u): {max_tend_u:.6f} m/s^2")
    print(f"Max Vertical Acceleration (tend_w):   {max_tend_w:.6f} m/s^2")
    
    if max_tend_u > 1e-10:
        print("\n>>> FAILURE: Spurious horizontal winds generated. Metric terms are incorrect.")
    else:
        print("\n>>> SUCCESS: Hydrostatic balance maintained over topography.")

if __name__ == "__main__":
    test_resting_atmosphere()