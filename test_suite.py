import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import SISLStepper
from suetes.core.driver import Simulation

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def get_base_state(grid, physics, u_0=0.0):
    """Helper to generate the standard balanced background state."""
    state = {
        'u': u_0 * jnp.ones_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'eta_dot': jnp.zeros_like(grid.X_w), # NEW: Added eta_dot initialization
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    
    # Initialize w to follow terrain boundary condition if u_0 > 0
    if u_0 > 0.0:
        z_xi_w = physics._get_metrics('w')['z_xi']
        state['w'] = u_0 * z_xi_w
        
    return state

def boundary_conditions(st, forcing):
    """Standard kinematic boundary conditions."""
    dh_dx = st.get('metric_z_xi_w_bottom', 0.0) # Assume flat if missing, or pass explicitly
    # For testing, we'll keep it simple. If we need exact terrain BC:
    st['w'] = st['w'].at[:, 0].set(0.0) # Will be overridden in Test 3/4
    st['w'] = st['w'].at[:, -1].set(0.0)
    return st

# ====================================================================
# TEST 1: The Null Test (Discrete Hydrostatic Balance)
# ====================================================================
def test_null_balance():
    print("\n" + "="*50)
    print("TEST 1: Null Test (Discrete Hydrostatic Balance)")
    print("="*50)
    
    nx, nz = 100, 30
    Lx, Lz = 100000.0, 30000.0
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
    grid.periodic_x = True
    
    physics = ICON2DSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 10.0
    stepper = SISLStepper(physics, dt)
    
    state = get_base_state(grid, physics, u_0=0.0)
    
    sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)
    # Run for just 1 timestep
    final_state = sim.run(state, 0.0, dt, dt, chunk_steps=1)
    
    max_w = float(jnp.max(jnp.abs(final_state['w'])))
    print(f"\n[Result] Max |w| after 1 step: {max_w:.2e} m/s")
    
    if max_w > 1e-10:
        print(">>> FAILED: Background state is NOT in discrete balance.")
        print(">>> Action: Update pi_bg and theta_bg in euler.py to use discrete integration.")
    else:
        print(">>> PASSED: Perfect discrete hydrostatic balance achieved.")

# ====================================================================
# TEST 2: Non-hydrostatic Gravity Waves
# ====================================================================
def test_gravity_waves():
    print("\n" + "="*50)
    print("TEST 2: Non-hydrostatic Gravity Waves (Flat Domain)")
    print("="*50)
    
    nx, nz = 300, 10
    Lx, Lz = 300000.0, 10000.0
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
    grid.periodic_x = True
    
    physics = ICON2DSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 12.0
    stepper = SISLStepper(physics, dt)
    
    state = get_base_state(grid, physics, u_0=20.0)
    
    # Add thermal bubble perturbation
    xc, a, H = 150000.0, 5000.0, 10000.0
    theta_prime = 1e-2 * jnp.sin(jnp.pi * grid.Z_m / H) / (1.0 + ((grid.X_m - xc) / a)**2)
    state['th_v'] += theta_prime
    
    # Must update density to maintain the equation of state!
    state['rho'] = (physics.c['p0'] / (physics.c['Rd'] * state['th_v'])) * \
                   (state['pi'] ** (physics.c['cvd'] / physics.c['Rd']))
    
    sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)
    final_state = sim.run(state, 0.0, 3000.0, dt, chunk_steps=50)
    
    plt.figure(figsize=(10, 4))
    th_prime_final = final_state['th_v'] - physics.theta_bg
    plt.contourf(grid.X_m/1000.0, grid.Z_m/1000.0, th_prime_final, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Theta Perturbation (K)')
    plt.title("Test 2: Non-hydrostatic Gravity Waves (T=3000s)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test2_gravity_waves.png', dpi=150)
    print("\n>>> Saved test2_gravity_waves.png")
    print(">>> Check: Ensure the bubble has split into symmetric, clean waves.")

# ====================================================================
# TEST 3 & 4: Linear Mountain Wave & Sponge Verification
# ====================================================================
def test_linear_mountain(run_long=False):
    test_num = 4 if run_long else 3
    print("\n" + "="*50)
    print(f"TEST {test_num}: Linear Mountain Wave {'(Sponge Verification)' if run_long else ''}")
    print("="*50)
    
    nx, nz = 360, 140 # dx=400m, dz=250m
    Lx, Lz = 144000.0, 35000.0
    
    def agnesi_hill(x):
        hm, a, xc = 1.0, 1000.0, 72000.0
        return hm * (a**2) / ((x - xc)**2 + a**2)
        
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=agnesi_hill)
    grid.periodic_x = True # Or False, depending on your domain boundaries
    
    # Sponge layer active in the top 10km (25km to 35km)
    physics = ICON2DSlice(grid, CONSTANTS, damp_height=25000.0, N_bv=0.01)
    dt = 5.0
    stepper = SISLStepper(physics, dt)
    
    state = get_base_state(grid, physics, u_0=10.0)
    
    def terrain_bc(st, forcing):
        # Enforce kinematic bottom boundary condition exactly
        u_at_w_face = physics.op.avg_u_to_m(st['u'])[:, 0]
        dh_dx = physics._get_metrics('w')['z_xi'][:, 0]
        st['w'] = st['w'].at[:, 0].set(u_at_w_face * dh_dx)
        st['w'] = st['w'].at[:, -1].set(0.0)
        return st
        
    sim = Simulation(stepper, forcing_fn=None, bc_fn=terrain_bc)
    
    t_end = 15000.0 if run_long else 4500.0
    final_state = sim.run(state, 0.0, t_end, dt, chunk_steps=100)
    
    plt.figure(figsize=(10, 4))
    plt.contourf(grid.X_w/1000.0, grid.Z_w/1000.0, final_state['w'], levels=20, cmap='coolwarm')
    plt.colorbar(label='w (m/s)')
    plt.title(f"Test {test_num}: Linear Mountain Wave (T={t_end}s)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.ylim(0, 20) # Only plot bottom 20km to match literature
    plt.savefig(f'test{test_num}_mountain_wave.png', dpi=150)
    print(f"\n>>> Saved test{test_num}_mountain_wave.png")
    
    if not run_long:
        print(">>> Check: Waves should be steady, non-distorting, and tilt downstream.")
    else:
        print(">>> Check: Look for downward reflection noise from the top of the domain.")

if __name__ == "__main__":
    test_null_balance()
    test_gravity_waves()
    test_linear_mountain(run_long=False)
    # test_linear_mountain(run_long=True)