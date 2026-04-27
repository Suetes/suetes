import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from suetes.slice2d.grids import StaggeredGrid
from suetes.slice2d.steppers import SemiLagrangianAdvector, SISLStepper
from suetes.slice2d.euler import VerticalSlice
from suetes.shared.driver import Simulation

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

def get_base_state(grid, physics, u_0=0.0):
    """Helper to generate the standard balanced background state."""
    state = {
        'u': u_0 * jnp.ones_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros_like(grid.X_w)
    }
    
    # Initialize w to follow terrain boundary condition if u_0 > 0
    if u_0 > 0.0:
        z_xi_w = physics._get_metrics('w')['z_xi']
        state['w'] = u_0 * z_xi_w
        
    return state

def boundary_conditions(st, forcing):
    """Standard kinematic boundary conditions."""
    st['w'] = st['w'].at[:, 0].set(0.0) 
    st['w'] = st['w'].at[:, -1].set(0.0)
    
    st['eta_dot'] = st['eta_dot'].at[:, 0].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, -1].set(0.0)
    return st

# ====================================================================
# TEST 1: The Null Test (Discrete Hydrostatic Balance)
# ====================================================================
def test_topographic_null_balance():
    print("\n" + "="*50)
    print("TEST 1: Topographic Null Test (Resting over Mountain)")
    print("="*50)
    
    nx, nz = 200, 40
    Lx, Lz = 100000.0, 20000.0
    
    # Use a steep mountain to stress-test the metrics
    def steep_hill(x):
        hm, a, xc = 2000.0, 5000.0, 50000.0
        return hm * jnp.exp(-((x - xc)**2) / (a**2))
        
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=steep_hill)
    grid.periodic_x = True
    
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 10.0
    stepper = SISLStepper(physics, dt)
    
    # Initialize a completely stationary atmosphere (u_0 = 0)
    state = get_base_state(grid, physics, u_0=0.0)
    
    def boundary_conditions_resting(st, forcing):
        st['w'] = st['w'].at[:, 0].set(0.0) 
        st['w'] = st['w'].at[:, -1].set(0.0)
        st['eta_dot'] = st['eta_dot'].at[:, 0].set(0.0)
        st['eta_dot'] = st['eta_dot'].at[:, -1].set(0.0)
        return st

    sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions_resting)
    
    # Run for a few hundred steps. It should remain perfectly still.
    final_state = sim.run(state, 0.0, 1000.0, dt, chunk_steps=50)
    
    max_u = float(jnp.max(jnp.abs(final_state['u'])))
    max_w = float(jnp.max(jnp.abs(final_state['w'])))
    
    print(f"\n[Result] Max |u|: {max_u:.2e} m/s")
    print(f"[Result] Max |w|: {max_w:.2e} m/s")
    
    if max_u > 1e-10 or max_w > 1e-10:
        print(">>> FAILED: Spurious topographic winds detected. The discrete metric chain rule is leaking.")
    else:
        print(">>> PASSED: Perfect discrete curvilinear balance achieved.")

# ====================================================================
# TEST 2: Advection solver
# ====================================================================

def test_pure_advection():
    print("\n" + "="*50)
    print("TEST 2: Pure Advection Over Topography (SLEVE)")
    print("="*50)
    
    nx, nz = 200, 40
    Lx, Lz = 100000.0, 20000.0
    
    def steep_hill(x):
        hm, a, xc = 2000.0, 5000.0, 50000.0
        return hm * jnp.exp(-((x - xc)**2) / (a**2))
    
    # 1. EXPLICITLY INJECT THE SLEVE TRANSFORM
    from suetes.shared.transforms import SleveSimple
    sleve_transform = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=steep_hill, transform=sleve_transform)
    grid.periodic_x = True
    
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 10.0
    
    advector = SemiLagrangianAdvector(grid, physics, dt)
    
    u_0 = 10.0
    u_phys = u_0 * jnp.ones_like(grid.X_u)
    eta_dot = jnp.zeros_like(grid.X_w)  
    
    # 2. USE A SMOOTH TRACER BUBBLE
    xc, zc = 25000.0, 5000.0
    r_bubble = 4000.0
    r = jnp.sqrt((grid.X_m - xc)**2 + (grid.Z_m - zc)**2)
    # Cosine squared bell prevents high-frequency Gibbs ringing
    tracer = jnp.where(r < r_bubble, jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)
    
    num_steps = 500  
    
    @jax.jit
    def step_fn(tr, _):
        tr_next = advector.advect(tr, u_phys, eta_dot, loc='m')
        return tr_next, None
        
    final_tracer, _ = jax.lax.scan(step_fn, tracer, jnp.arange(num_steps))
    
    plt.figure(figsize=(10, 4))
    plt.contourf(grid.X_m/1000.0, grid.Z_m/1000.0, final_tracer, levels=10, cmap='Reds')
    hx_m = steep_hill(grid.X_m[:, 0])
    plt.plot(grid.X_m[:, 0] / 1000.0, hx_m / 1000.0, color='black', linewidth=2)
    plt.fill_between(grid.X_m[:, 0] / 1000.0, 0, hx_m / 1000.0, color='gray')
    plt.title(f"Test 2: Advection (SLEVE) T={num_steps*dt}s")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test2_pure_advection.png', dpi=150)
    print("\n>>> Saved test2_pure_advection.png")

# ====================================================================
# TEST 3: Non-hydrostatic Gravity Waves
# ====================================================================
def test_gravity_waves():
    print("\n" + "="*50)
    print("TEST 2: Non-hydrostatic Gravity Waves (Flat Domain)")
    print("="*50)
    
    nx, nz = 300, 10
    Lx, Lz = 300000.0, 10000.0
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
    grid.periodic_x = True
    
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
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
    plt.title("Test 3: Non-hydrostatic Gravity Waves (T=3000s)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km)")
    plt.savefig('test3_gravity_waves.png', dpi=150)
    print("\n>>> Saved test3_gravity_waves.png")
    print(">>> Check: Ensure the bubble has split into symmetric, clean waves.")

# ====================================================================
# TEST 4/5: Linear Mountain Wave & Sponge Verification
# ====================================================================
def test_linear_mountain(run_long=False):
    test_num = 5 if run_long else 4
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
    physics = VerticalSlice(grid, CONSTANTS, damp_height=25000.0, N_bv=0.01)
    dt = 5.0
    stepper = SISLStepper(physics, dt)
    
    state = get_base_state(grid, physics, u_0=10.0)
    
    def terrain_bc(st, forcing):
        # Enforce kinematic bottom boundary condition exactly for physical w
        u_at_w_face = physics.op.avg_u_to_m(st['u'])[:, 0]
        dh_dx = physics._get_metrics('w')['z_xi'][:, 0]
        st['w'] = st['w'].at[:, 0].set(u_at_w_face * dh_dx)
        st['w'] = st['w'].at[:, -1].set(0.0)
        
        # Logical eta_dot is always zero at the boundaries, even over terrain
        st['eta_dot'] = st['eta_dot'].at[:, 0].set(0.0)
        st['eta_dot'] = st['eta_dot'].at[:, -1].set(0.0)
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


# ====================================================================
# TEST 6: Autograd verification
# ====================================================================
def test_autodiff_gradients():
    print("\n" + "="*50)
    print("TEST 6: Autodiff Gradient Verification (Reverse-Mode)")
    print("="*50)
    
    # Setup a small grid to keep compilation and execution fast
    nx, nz = 50, 10
    Lx, Lz = 50000.0, 10000.0
    
    # Use a flat domain to isolate the fluid dynamics gradients first
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
    grid.periodic_x = True
    
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 10.0
    stepper = SISLStepper(physics, dt)
    
    # 1. Define the pure, side-effect-free forward pass
    # It must take a JAX array (the parameters we want to optimize) 
    # and return a single scalar value (the loss).
    def forward_loss(u_initial_array):
        # Reconstruct the state dictionary
        state = get_base_state(grid, physics, u_0=0.0)
        state['u'] = u_initial_array
        
        # Run exactly 5 steps. We use stepper.integrate directly 
        # to avoid the print statements and python loops in the Simulation driver.
        final_state = stepper.integrate(
            state, 
            t_start=0.0, 
            num_steps=5, 
            forcing=None, 
            bc_fn=boundary_conditions
        )
        
        # Compute a mock loss: Total Kinetic Energy of the domain
        # L = 0.5 * sum(u^2 + w^2)
        u_avg = physics.op.avg_u_to_m(final_state['u'])
        w_avg = physics.op.avg_w_to_m(final_state['w'])
        ke = 0.5 * jnp.sum(u_avg**2 + w_avg**2)
        
        return ke

    # 2. Create the initial parameter state
    u_init = 10.0 * jnp.ones_like(grid.X_u)
    
    # 3. JIT compile the value and gradient function
    print("[Autodiff] Compiling forward and reverse passes...")
    loss_and_grad_fn = jax.jit(jax.value_and_grad(forward_loss))
    
    # 4. Execute
    loss_val, u_grad = loss_and_grad_fn(u_init)
    
    print(f"\n[Result] Forward Loss (Kinetic Energy): {loss_val:.4f}")
    
    # 5. Gradient diagnostics
    max_grad = float(jnp.max(jnp.abs(u_grad)))
    mean_grad = float(jnp.mean(jnp.abs(u_grad)))
    nan_count = int(jnp.isnan(u_grad).sum())
    
    print(f"[Result] Max |Gradient|:  {max_grad:.4e}")
    print(f"[Result] Mean |Gradient|: {mean_grad:.4e}")
    print(f"[Result] NaN Count:       {nan_count}")
    
    if nan_count > 0:
        print(">>> FAILED: The solver dropped gradients (NaNs detected). Check GMRES tolerance or division by zero in the metrics.")
    elif max_grad == 0.0:
        print(">>> FAILED: The gradient is strictly zero. The limiter or a detached tensor may be blocking the backward pass.")
    else:
        print(">>> PASSED: JAX successfully backpropagated through the GMRES solver and Semi-Lagrangian advection!")


# ====================================================================
# TEST 7: Mass conservation verification
# ====================================================================
def test_mass_conservation():
    print("\n" + "="*50)
    print("TEST 7: Mass Conservation Tracking (WITH FIXER)")
    print("="*50)
    
    nx, nz = 100, 20
    Lx, Lz = 100000.0, 20000.0
    
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
    grid.periodic_x = True
    
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    dt = 10.0
    
    # 1. ENABLE THE TOGGLES HERE
    stepper = SISLStepper(
        physics, 
        dt, 
        use_mass_fixer=True, 
        tracer_keys=['qv']
    )
    
    state = get_base_state(grid, physics, u_0=15.0)
    
    # 2. ADD THE TRACER TO THE DICTIONARY
    xc, zc = 25000.0, 5000.0
    r_bubble = 4000.0
    r = jnp.sqrt((grid.X_m - xc)**2 + (grid.Z_m - zc)**2)
    state['qv'] = jnp.where(r < r_bubble, jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)
    
    # Helper to calculate domain-integrated mass
    dz_m = grid.Z_w[:, 1:] - grid.Z_w[:, :-1]
    cell_volumes = grid.dx * dz_m
    
    def compute_masses(st):
        air_mass = jnp.sum(st['rho'] * cell_volumes)
        tracer_mass = jnp.sum(st['qv'] * st['rho'] * cell_volumes)
        return air_mass, tracer_mass

    init_air_mass, init_tracer_mass = compute_masses(state)
    
    # 3. RUN IT
    sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)
    final_state = sim.run(state, 0.0, 5000.0, dt, chunk_steps=100)
    
    final_air_mass, final_tracer_mass = compute_masses(final_state)
    
    air_loss_pct = (final_air_mass - init_air_mass) / init_air_mass * 100
    tracer_loss_pct = (final_tracer_mass - init_tracer_mass) / init_tracer_mass * 100
    
    print(f"\n[Result] Initial Air Mass:    {init_air_mass:.4e} kg/m")
    print(f"[Result] Air Mass Drift:      {air_loss_pct:.8f} %")
    print(f"[Result] Initial Tracer Mass: {init_tracer_mass:.4e} kg/m")
    print(f"[Result] Tracer Mass Drift:   {tracer_loss_pct:.8f} %")
    
    if abs(tracer_loss_pct) > 1e-5:
        print(">>> FAILED: Mass fixer did not strictly conserve tracer mass.")
    else:
        print(">>> PASSED: Mass fixer is working! Exact conservation achieved.")

# ====================================================================
# TEST 8: Extreme Topography ('The Brick Wall')
# ====================================================================
def test_extreme_topography():
    print("\n" + "="*50)
    print("TEST 8: Extreme Topography ('The Brick Wall')")
    print("="*50)
    
    nx, nz = 200, 40
    Lx, Lz = 100000.0, 20000.0
    
    def cliff(x):
        hm, a, xc = 3500.0, 1500.0, 50000.0
        return hm * jnp.exp(-((x - xc)**2) / (a**2))
        
    from suetes.shared.transforms import SleveSimple
    sleve_transform = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    
    try:
        # The __init__ of StaggeredGrid should now catch the tangled coordinates
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=cliff, transform=sleve_transform)
        
        # If it somehow survives initialization, run the simulation
        grid.periodic_x = True
        physics = VerticalSlice(grid, CONSTANTS, damp_height=15000.0, N_bv=0.01)
        stepper = SISLStepper(physics, 8.0)
        state = get_base_state(grid, physics, u_0=15.0)
        
        sim = Simulation(stepper, forcing_fn=None, bc_fn=lambda st, _: st)
        _ = sim.run(state, 0.0, 10.0, 8.0, chunk_steps=1)
        
        print(">>> FAILED: The grid tangling check failed to catch the extreme topography!")
        
    except ValueError as e:
        print(f"\n[Success! Grid tangling prevented execution]\nException caught: {e}")
        print("\n>>> PASSED: The diagnostic tool successfully blocked invalid topography.")

# Add to the bottom of test_suite_2d.py
if __name__ == "__main__":
    # test_topographic_null_balance()
    # test_pure_advection()
    # test_gravity_waves()
    # test_linear_mountain(run_long=False)
    # test_linear_mountain(run_long=True)
    # test_autodiff_gradients()
    test_mass_conservation()
    test_extreme_topography()
