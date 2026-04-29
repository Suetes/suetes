import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.shared.transforms import SleveSimple

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def schaer_2d(x, y):
    h0, a, lam = 2000.0, 15000.0, 8000.0
    return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

def setup_test_env():
    nx, ny, nz = 150, 5, 50
    dx, dy, dz = 1000.0, 1000.0, 400.0
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=sleve)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.01)
    
    bg_state_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_state_ref)
    return grid, op, physics, bg_precomputed

def test_1_metric_gradients():
    print("\n=== TEST 1: METRIC GRADIENT CONSISTENCY ===")
    grid, op, physics, bg = setup_test_env()
    
    # In a resting atmosphere with N_bv > 0, pressure is purely a function of Z_m.
    # Therefore, the true Cartesian horizontal pressure gradient MUST be zero everywhere.
    pi_prime = jnp.zeros_like(grid.Z_m)
    state_prime = {'u': jnp.zeros_like(grid.Z_u), 'v': jnp.zeros_like(grid.Z_v), 
                   'w': jnp.zeros_like(grid.Z_w), 'pi': pi_prime, 'eta_dot': jnp.zeros_like(grid.Z_w)}
    
    tends = physics.get_tendencies(state_prime, bg)
    
    max_u_accel = jnp.max(jnp.abs(tends['u']))
    max_w_accel = jnp.max(jnp.abs(tends['w']))
    
    print(f"Max spurious horizontal accel (should be ~0): {max_u_accel:.4e} m/s^2")
    print(f"Max spurious vertical accel (should be ~0):   {max_w_accel:.4e} m/s^2")
    
    if max_u_accel > 1e-6:
        print(">>> FAILED: The transformation metrics (z_xi) are failing to cancel the vertical pressure gradient on slopes.")
    else:
        print(">>> PASSED: Hydrostatic balance holds over topography.")

def test_2_kinematic_divergence():
    print("\n=== TEST 2: KINEMATIC DIVERGENCE (CONSTANT FLOW) ===")
    grid, op, physics, bg = setup_test_env()
    
    # Prescribe a perfect, uniform 10 m/s flow following the terrain
    u_0 = 10.0
    u_prescribed = u_0 * jnp.ones_like(grid.Z_u)
    v_prescribed = jnp.zeros_like(grid.Z_v)
    
    # w = u * dz/dx + v * dz/dy. Since eta_dot = 0, this should be perfectly non-divergent.
    u_m = op.avg(u_prescribed, axis=0, from_loc='u', to_loc='m')
    u_w = op.avg(u_m, axis=2, from_loc='m', to_loc='w')
    w_prescribed = u_w * grid.z_xi_w
    
    state_prime = {'u': u_prescribed, 'v': v_prescribed, 
                   'w': w_prescribed, 'pi': jnp.zeros_like(grid.Z_m), 'eta_dot': jnp.zeros_like(grid.Z_w)}
    
    tends = physics.get_tendencies(state_prime, bg)
    
    # If the flow is perfectly terrain following and non-divergent, tend_pi should be zero!
    max_div = jnp.max(jnp.abs(tends['pi']))
    print(f"Max spurious pressure tendency from divergence: {max_div:.4e} 1/s")
    
    if max_div > 1e-5:
        print(">>> FAILED: The 3D divergence operator is registering false divergence over slopes.")
    else:
        print(">>> PASSED: 3D Divergence is clean.")

def test_3_vertical_metric_smoothness():
    print("\n=== TEST 3: VERTICAL METRIC (dz) TERROR CHECK ===")
    grid, op, physics, bg = setup_test_env()
    
    dz_min = jnp.min(bg['dz_w_full'])
    dz_max = jnp.max(bg['dz_w_full'])
    
    print(f"Physical Layer Thickness (dz) - Min: {dz_min:.2f} m, Max: {dz_max:.2f} m")
    if dz_min < 10.0:
        print(">>> FAILED: Your SLEVE coordinate is compressing layers too thinly over the mountain peaks. This will shatter the CFL condition and cause GMRES to explode.")
    else:
        print(">>> PASSED: Vertical grid spacing is safe.")

if __name__ == "__main__":
    test_1_metric_gradients()
    test_2_kinematic_divergence()
    test_3_vertical_metric_smoothness()