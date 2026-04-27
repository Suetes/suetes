import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.driver import Simulation
from suetes.shared.transforms import SleveSimple

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def get_base_state_3d(grid, physics, u_0=0.0, v_0=0.0):
    """Generates the balanced 3D background state."""
    state = {
        'u': u_0 * jnp.ones_like(grid.Z_u),
        'v': v_0 * jnp.ones_like(grid.Z_v),
        'w': jnp.zeros_like(grid.Z_w),
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros_like(grid.Z_w)
    }
    
    # Kinematic bottom boundary for w if there is background wind
    if u_0 > 0.0 or v_0 > 0.0:
        state['w'] = u_0 * grid.z_xi_w + v_0 * grid.z_eta_w
        
    return state

def boundary_conditions_3d(st, forcing):
    """Standard kinematic vertical boundary conditions for 3D."""
    # Note: A true 3D model would also handle lateral boundaries (e.g., Davies relaxation) here.
    # For these tests, we assume periodic lateral boundaries or domains large enough to avoid reflection.
    st['w'] = st['w'].at[:, :, 0].set(0.0) 
    st['w'] = st['w'].at[:, :, -1].set(0.0)
    
    st['eta_dot'] = st['eta_dot'].at[:, :, 0].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, :, -1].set(0.0)
    return st

def terrain_bc_3d(st, physics):
    """Kinematic bottom boundary condition over 3D terrain."""
    u_m = physics.op.avg(st['u'], axis=0, from_loc='u', to_loc='m')
    v_m = physics.op.avg(st['v'], axis=1, from_loc='v', to_loc='m')
    
    u_at_w_face = physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')[:, :, 0]
    v_at_w_face = physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')[:, :, 0]
    
    dh_dx = physics.grid.z_xi_w[:, :, 0]
    dh_dy = physics.grid.z_eta_w[:, :, 0]
    
    st['w'] = st['w'].at[:, :, 0].set(u_at_w_face * dh_dx + v_at_w_face * dh_dy)
    st['w'] = st['w'].at[:, :, -1].set(0.0)
    
    st['eta_dot'] = st['eta_dot'].at[:, :, 0].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, :, -1].set(0.0)
    return st

# ====================================================================
# TEST 1: 3D Pure Advection Over Topography
# ====================================================================
def test_3d_advection():
    print("\n" + "="*50)
    print("TEST 1: 3D Diagonal Advection Over Topography")
    print("="*50)
    
    nx, ny, nz = 80, 20, 40 
    dx, dy, dz = 1000.0, 1000.0, 500.0
    
    # 3D Gaussian Hill
    def hill_3d(x, y):
        hm, a = 1500.0, 15000.0
        return hm * jnp.exp(-(x**2 + y**2) / (a**2))
        
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=-52.0, h_func=hill_3d, transform=sleve)
    
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz)
    
    dt = 15.0
    # Enable the mass fixer for our tracer!
    stepper = SISLStepper3D(physics, dt, use_mass_fixer=True, tracer_keys=['tracer'])
    
    # 1. Generate the static background state for the sponge to relax towards
    base_state = get_base_state_3d(grid, physics, u_0=10.0, v_0=0.0)
    # The external state needs a tracer key so the sponge loop doesn't throw a KeyError
    base_state['tracer'] = jnp.zeros_like(grid.Z_m) 
    
    # 2. Initialize the Davies Sponge (8 grid cells thick on all lateral edges)
    sponge = DaviesSponge(grid, sponge_depth=8)
    
    # 3. Create the initial simulation state
    state = dict(base_state) 
    
    xc, yc, zc = -25000.0, 0.0, 5000.0
    r_bubble = 8000.0
    X_m, Y_m, Z_m = grid.x_m[:, None, None], grid.y_m[None, :, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Y_m - yc)**2 + (Z_m - zc)**2)
    state['tracer'] = jnp.where(r < r_bubble, jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)

    # 4. Combine the kinematic terrain condition AND the sponge layer
    def full_bc(st, forcing):
        # First, strictly enforce the flow parallel to the mountain
        st = terrain_bc_3d(st, physics)
        # Second, damp the outer 8 cells to absorb reflecting gravity waves
        st = sponge.blend(st, base_state)
        return st

    # 5. Run the simulation with the new combined boundary condition
    sim = Simulation(stepper, forcing_fn=None, bc_fn=full_bc)
    final_state = sim.run(state, 0.0, 3000.0, dt, chunk_steps=50)
    
    # Plotting a horizontal slice (xy) and a vertical slice (xz)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mid-level horizontal slice (z_idx = 5)
    z_idx = 5
    cf1 = ax1.contourf(grid.x_m/1000, grid.y_m/1000, final_state['tracer'][:, :, z_idx].T, levels=10, cmap='Reds')
    ax1.set_title(f"Horizontal Slice (z~{grid.z_m[z_idx]/1000:.1f}km)")
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("y (km)")
    fig.colorbar(cf1, ax=ax1)

    # Centerline vertical slice (y_idx = ny//2)
    y_idx = ny // 2
    
    # --- FIX: Create a 2D X-grid to match the 2D Z-grid for Matplotlib ---
    X_2d, _ = jnp.meshgrid(grid.x_m, grid.z_m, indexing='ij')
    
    cf2 = ax2.contourf(X_2d/1000, grid.Z_m[:, y_idx, :]/1000, final_state['tracer'][:, y_idx, :], levels=10, cmap='Reds')
    
    ax2.plot(grid.x_m/1000, grid.Z_w[:, y_idx, 0]/1000, color='black', linewidth=2)
    ax2.set_title(f"Vertical Slice (y=0km)")
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("z (km)")
    fig.colorbar(cf2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('test1_3d_advection.png', dpi=150)
    print(">>> Saved test1_3d_advection.png")


# ====================================================================
# TEST 2: 3D Rising Thermal Bubble
# ====================================================================
def test_3d_bubble():
    print("\n" + "="*50)
    print("TEST 2: 3D Rising Thermal Bubble")
    print("="*50)
    
    nx, ny, nz = 80, 5, 80 
    dx, dy, dz = 250.0, 250.0, 125.0
    
    # Flat grid
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=-52.0)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, CONSTANTS, damp_height=grid.Lz, N_bv=0.0) # Neutral atmosphere
    
    dt = 0.5
    stepper = SISLStepper3D(physics, dt)
    state = get_base_state_3d(grid, physics, u_0=0.0, v_0=0.0)
    
    # 3D Thermal perturbation (Modeled as a 2D cylinder in Y)
    xc, zc = 0.0, 2000.0
    r_bubble = 1500.0
    
    # Notice we drop Y_m entirely from the distance calculation!
    X_m, Z_m = grid.x_m[:, None, None], grid.Z_m
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    
    theta_prime = jnp.where(r < r_bubble, 2.0 * jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)
    
    state['th_v'] += theta_prime
    state['rho'] = (physics.c['p0'] / (physics.c['Rd'] * state['th_v'])) * \
                   (state['pi'] ** (physics.c['cvd'] / physics.c['Rd']))

    sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions_3d)
    final_state = sim.run(state, 0.0, 600.0, dt, chunk_steps=20)
    
    th_prime_final = final_state['th_v'] - physics.theta_bg
    
    plt.figure(figsize=(8, 5))
    y_idx = ny // 2
    
    # --- FIX: Create a 2D X-grid to match the 2D Z-grid for Matplotlib ---
    X_2d, _ = jnp.meshgrid(grid.x_m, grid.z_m, indexing='ij')
    
    plt.contourf(X_2d/1000, grid.Z_m[:, y_idx, :]/1000, th_prime_final[:, y_idx, :], levels=15, cmap='RdBu_r')
    plt.colorbar(label='Theta Perturbation (K)')
    plt.title("3D Rising Bubble (Cross Section y=0)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test2_3d_bubble.png', dpi=150)
    print(">>> Saved test2_3d_bubble.png")


if __name__ == "__main__":
    test_3d_advection()
    test_3d_bubble()