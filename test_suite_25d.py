import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.driver import Simulation
from suetes.shared.transforms import SleveSimple

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def get_base_state_25d(grid, physics, u_0=0.0):
    """Generates the balanced background state with strictly zero V wind."""
    state = {
        'u': u_0 * jnp.ones_like(grid.Z_u),
        'v': jnp.zeros_like(grid.Z_v), # strictly zero!
        'w': jnp.zeros_like(grid.Z_w),
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros_like(grid.Z_w)
    }
    
    if u_0 > 0.0:
        state['w'] = u_0 * grid.z_xi_w
        
    return state

def terrain_bc_25d(st, physics):
    """Enforces kinematic bottom boundary and strictly zeroes out V."""
    u_m = physics.op.avg(st['u'], axis=0, from_loc='u', to_loc='m')
    u_at_w_face = physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')[:, :, 0]
    
    dh_dx = physics.grid.z_xi_w[:, :, 0]
    
    # 2D Kinematic constraint
    st['w'] = st['w'].at[:, :, 0].set(u_at_w_face * dh_dx)
    st['w'] = st['w'].at[:, :, -1].set(0.0)
    
    st['eta_dot'] = st['eta_dot'].at[:, :, 0].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, :, -1].set(0.0)
    
    # Force 2D symmetry
    st['v'] = jnp.zeros_like(st['v'])
    
    return st

# ====================================================================
# TEST A: 2.5D Cold Sinking Bubble (Tests the Implicit Solver Matrix)
# ====================================================================
def test_25d_bubble():
    print("\n" + "="*50)
    print("TEST A: 2.5D Cold Sinking Bubble (Implicit Solver Check)")
    print("="*50)
    
    nx, ny, nz = 100, 5, 80 
    dx, dy, dz = 200.0, 200.0, 100.0
    
    # Flat grid, lat_center=0 turns off Coriolis
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.01)
    
    dt = 1.0 # Very safe timestep for testing
    stepper = SISLStepper3D(physics, dt)
    state = get_base_state_25d(grid, physics, u_0=0.0)
    
    # 2D Thermal perturbation (Cylinder in Y)
    xc, zc = 0.0, 4000.0
    r_bubble = 1500.0
    
    X_m, Z_m = grid.x_m[:, None, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    
    # Negative (cold) bubble to test gravity waves
    theta_prime = jnp.where(r < r_bubble, -2.0 * jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)
    state['th_v'] += theta_prime
    state['rho'] = (physics.c['p0'] / (physics.c['Rd'] * state['th_v'])) * \
                   (state['pi'] ** (physics.c['cvd'] / physics.c['Rd']))

    sim = Simulation(stepper, forcing_fn=None, bc_fn=lambda st, f: terrain_bc_25d(st, physics))
    final_state = sim.run(state, 0.0, 600.0, dt, chunk_steps=20)
    
    th_prime_final = final_state['th_v'] - physics.theta_bg
    
    plt.figure(figsize=(8, 5))
    y_idx = ny // 2
    X_2d, _ = jnp.meshgrid(grid.x_m, grid.z_m, indexing='ij')
    
    plt.contourf(X_2d/1000, grid.Z_m[:, y_idx, :]/1000, th_prime_final[:, y_idx, :], levels=20, cmap='RdBu_r')
    plt.colorbar(label='Theta Perturbation (K)')
    plt.title("2.5D Cold Bubble (Is the solver checkerboarding?)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test_A_25d_bubble.png', dpi=150)
    print(">>> Saved test_A_25d_bubble.png")

# ====================================================================
# TEST B: 2.5D Schär Mountain Advection (Tests Trajectories & Metrics)
# ====================================================================
def test_25d_schaer_mountain():
    print("\n" + "="*50)
    print("TEST B: 2.5D Schär Mountain (Trajectory & Grid Check)")
    print("="*50)
    
    nx, ny, nz = 150, 5, 50 
    dx, dy, dz = 1000.0, 1000.0, 400.0
    
    # Classic 2D Schär Mountain Profile
    def schaer_2d(x, y):
        h0, a, lam = 2000.0, 15000.0, 8000.0
        return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)
        
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0, h_func=schaer_2d, transform=sleve)
    
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.0)
    
    dt = 10.0
    stepper = SISLStepper3D(physics, dt, use_mass_fixer=True, tracer_keys=['tracer'])
    
    base_state = get_base_state_25d(grid, physics, u_0=10.0)
    base_state['tracer'] = jnp.zeros_like(grid.Z_m) 
    sponge = DaviesSponge(grid, sponge_depth=15)
    
    state = dict(base_state) 
    xc, zc = -30000.0, 6000.0
    r_bubble = 8000.0
    X_m, Z_m = grid.x_m[:, None, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    state['tracer'] = jnp.where(r < r_bubble, jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)

    def full_bc(st, forcing):
        st = terrain_bc_25d(st, physics)
        st = sponge.blend(st, base_state)
        return st

    sim = Simulation(stepper, forcing_fn=None, bc_fn=full_bc)
    final_state = sim.run(state, 0.0, 2000.0, dt, chunk_steps=50)
    
    plt.figure(figsize=(10, 5))
    y_idx = ny // 2
    X_2d, _ = jnp.meshgrid(grid.x_m, grid.z_m, indexing='ij')
    
    cf = plt.contourf(X_2d/1000, grid.Z_m[:, y_idx, :]/1000, final_state['tracer'][:, y_idx, :], levels=15, cmap='Reds')
    plt.plot(grid.x_m/1000, grid.Z_w[:, y_idx, 0]/1000, color='black', linewidth=2)
    
    plt.colorbar(cf, label='Tracer Concentration')
    plt.title("2.5D Schär Mountain (Is the tracer hitting the ground?)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test_B_25d_schaer.png', dpi=150)
    print(">>> Saved test_B_25d_schaer.png")

if __name__ == "__main__":
    # test_25d_bubble()
    test_25d_schaer_mountain()