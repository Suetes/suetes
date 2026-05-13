import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from suetes.slice2d.geometry import StaggeredGrid
from suetes.slice2d.steppers import SemiLagrangianAdvector, SISLStepper, SemiImplicitSolver, VerticalPreconditioner2D, vmap_tensor_interp_2d
from suetes.slice2d.euler import VerticalSlice
from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

CONSTANTS = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

output_dir = "suetes/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

# ====================================================================
# HELPERS
# ====================================================================
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

def setup_terrain_test_env():
    """Sets up a common test environment with topography for the diagnostics."""
    nx, nz = 150, 50
    dx, dz = 1000.0, 400.0
    Lx, Lz = nx * dx, nz * dz
    
    def agnesi_hill(x):
        return 2000.0 * (5000.0**2) / ((x - 75000.0)**2 + 5000.0**2)
        
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=agnesi_hill, transform=sleve)
    
    dt = 10.0
    physics = VerticalSlice(grid, CONSTANTS, damp_height=Lz, N_bv=0.01)
    
    return grid, physics, dt

# ====================================================================
# DIAGNOSTIC TESTS (ALIGNED WITH 3D CORE)
# ====================================================================

def test_1_geometry(grid):
    print("\n--- 1. GEOMETRY TEST ---")
    print(f"Max Mountain Height: {jnp.max(grid.Z_w[:, 0]):.2f} m") 
    print(f"Top boundary is flat: {jnp.std(grid.Z_w[:, -1]) < 1e-3}")

    min_dz = float(jnp.min(grid.Z_w[:, 1:] - grid.Z_w[:, :-1]))
    print(f"Minimum layer thickness: {min_dz:.2f} m")
    assert min_dz > 0.0, "Grid tangling detected! Transform failed."

def test_2_backtracking(grid, advector, dt):
    print("\n--- 2. BACKTRACKING TEST ---")
    # Simulate a constant 10 m/s flow
    u_phys = jnp.ones_like(grid.X_u) * 10.0
    eta_dot = jnp.zeros_like(grid.X_w)

    dep_indices = advector.compute_departure_indices(u_phys, eta_dot, loc='m', iterations=2)
    
    # In logical space, the physical X displacement should match exactly
    disp_indices = jnp.arange(grid.nx)[:, None] - dep_indices[0]
    expected_disp_indices = 10.0 * dt / grid.dx

    error = float(jnp.max(jnp.abs(disp_indices - expected_disp_indices)))
    print(f"Max trajectory error: {error:.5f} indices")
    assert error < 1e-4, "Semi-Lagrangian iterative backtracker is failing."

def test_3_mass_conservation_fixer(grid, physics, dt):
    print("\n--- 3. TRACER MASS CONSERVATION (MASS FIXER) ---")
    stepper = SISLStepper(physics, dt, use_mass_fixer=True, tracer_keys=['qv'])
    state = get_base_state(grid, physics, u_0=15.0)
    
    # Inject tracer
    xc, zc = grid.Lx / 2, 5000.0
    r = jnp.sqrt((grid.X_m - xc)**2 + (grid.Z_m - zc)**2)
    state['qv'] = jnp.where(r < 4000.0, jnp.cos(0.5 * jnp.pi * r / 4000.0)**2, 0.0)
    
    dz_m = grid.Z_w[:, 1:] - grid.Z_w[:, :-1]
    cell_volumes = grid.dx * dz_m
    
    init_mass = jnp.sum(state['qv'] * state['rho'] * cell_volumes)
    
    # Create a wrapper that matches the new Simulation driver signature
    def step_wrapper(curr_state, step_idx):
        t_curr = step_idx * dt
        # Execute the stepper
        next_state = stepper.step(curr_state, t=t_curr, forcing=None, bc_fn=boundary_conditions)
        # Return state and dummy metrics (None)
        return next_state, None
        
    # Initialize the Simulation with the wrapper and dt
    sim = Simulation(step_wrapper, dt)
    
    # Run the simulation
    final_state = sim.run(state, t_start=0.0, t_end=50.0, chunk_steps=5)
    
    final_mass = jnp.sum(final_state['qv'] * final_state['rho'] * cell_volumes)
    drift = float(abs((final_mass - init_mass) / init_mass))
    
    print(f"Initial Tracer Mass: {init_mass:.4e} kg")
    print(f"Mass Drift Percent:  {drift * 100:.8f} %")
    assert drift < 1e-12, "Mass fixer is failing to strictly conserve mass."

def test_4_advection_limiter(grid):
    print("\n--- 4. ADVECTION LIMITER TEST ---")
    blob = jnp.exp(-((grid.X_m - grid.Lx/2)**2 + (grid.Z_m - 2000.0)**2) / (1500.0**2))
    
    Xi, Zi = jnp.meshgrid(jnp.arange(grid.nx), jnp.arange(grid.nz), indexing='ij')
    coords_shifted = jnp.stack([Xi - 0.5, Zi - 0.5], axis=0) # Worst-case interpolation
    
    blob_unlim = vmap_tensor_interp_2d(blob, coords_shifted, apply_limiter=False)
    blob_lim = vmap_tensor_interp_2d(blob, coords_shifted, apply_limiter=True)

    min_unlim, min_lim = float(jnp.min(blob_unlim)), float(jnp.min(blob_lim))
    print(f"Minimum tracer value (Unlimited): {min_unlim:.6f}  <-- Notice the undershoot!")
    print(f"Minimum tracer value (Limited):   {min_lim:.6f}")
    assert min_lim >= 0.0, "Limiter failed to prevent negative tracer values!"

def test_5_hydrostatic_balance(grid, physics):
    print("\n--- 5. HYDROSTATIC BALANCE (SPURIOUS WINDS) ---")
    state = get_base_state(grid, physics, u_0=0.0)
    state_prime = {'u': state['u'], 'w': state['w'], 'pi': jnp.zeros_like(state['pi']), 'eta_dot': state['eta_dot']}
    
    bg_ref = {'rho': state['rho'], 'pi': state['pi'], 'th_v': state['th_v']}
    solver = SemiImplicitSolver(grid, physics, 10.0)
    bg = solver.precompute_bg(bg_ref)
    
    tends = physics.get_explicit_tendencies(state, state_prime, bg)

    max_u_tend = float(jnp.max(jnp.abs(tends['u'])))
    max_w_tend = float(jnp.max(jnp.abs(tends['w'])))
    print(f"Max U-velocity tendency (Spurious Wind): {max_u_tend:e} m/s^2")
    assert max_u_tend < 1e-4, "Spurious horizontal winds detected! Metric terms are misaligned."

def test_6_semi_implicit_solver(grid, physics, dt):
    print("\n--- 6. SEMI-IMPLICIT SOLVER CONVERGENCE ---")
    solver = SemiImplicitSolver(grid, physics, dt)
    state = get_base_state(grid, physics, u_0=0.0)
    
    bg_ref = {'rho': state['rho'], 'pi': state['pi'], 'th_v': state['th_v']}
    bg = solver.precompute_bg(bg_ref)

    key = jax.random.PRNGKey(42)
    rhs_dummy = {
        'u': jax.random.uniform(key, grid.X_u.shape) * 0.1,
        'w': jax.random.uniform(key, grid.X_w.shape) * 0.01,
        'pi': jax.random.uniform(key, grid.X_m.shape) * 0.001,
        'eta_dot': jax.random.uniform(key, grid.X_w.shape) * 0.001
    }

    precond = VerticalPreconditioner2D(physics, dt, bg, 0.65)
    sol = solver.solve(rhs_dummy, bg, beta=0.65, preconditioner=precond)

    is_valid = jnp.all(jnp.isfinite(sol['u'])) and jnp.all(jnp.isfinite(sol['pi']))
    print(f"Solver output contains only finite numbers: {is_valid}")
    assert is_valid, "Solver diverged and produced NaNs!"

def test_7_kinematic_bottom_boundary(grid, physics, dt):
    print("\n--- 7. KINEMATIC BOTTOM BOUNDARY (FLOW OVER MOUNTAIN) ---")
    solver = SemiImplicitSolver(grid, physics, dt)
    state = get_base_state(grid, physics, u_0=0.0)
    
    bg_ref = {'rho': state['rho'], 'pi': state['pi'], 'th_v': state['th_v']}
    bg = solver.precompute_bg(bg_ref)
    
    u_wind = 10.0
    rhs_prime = {
        'u': jnp.ones_like(grid.X_u) * u_wind,
        'w': jnp.zeros_like(grid.X_w),
        'pi': jnp.zeros_like(grid.X_m),
        'eta_dot': jnp.zeros_like(grid.X_w)
    }

    u_m = physics.op.avg_u_to_m(rhs_prime['u'])
    u_w = physics.op.avg_m_to_w(u_m)
    rhs_kinematic_bottom = u_w[:, 0] * bg['z_xi_w'][:, 0]
    rhs_prime['w'] = rhs_prime['w'].at[:, 0].set(rhs_kinematic_bottom)

    sol = solver.solve(rhs_prime, bg)

    max_w_expected = float(jnp.max(rhs_kinematic_bottom))
    max_w_actual = float(jnp.max(sol['w'][:, 0]))

    print(f"Max expected updraft from terrain slope: {max_w_expected:.4f} m/s")
    print(f"Max updraft preserved by implicit solver: {max_w_actual:.4f} m/s")
    assert jnp.isclose(max_w_expected, max_w_actual, rtol=1e-4), "Solver is not preserving the kinematic constraint!"

def test_8_metric_gradients(grid, physics):
    print("\n--- 8. METRIC GRADIENT CONSISTENCY ---")
    state_prime = {
        'u': jnp.zeros_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'pi': jnp.zeros_like(grid.X_m),
        'eta_dot': jnp.zeros_like(grid.X_w)
    }
    state = get_base_state(grid, physics)
    bg_ref = {'rho': state['rho'], 'pi': state['pi'], 'th_v': state['th_v']}
    solver = SemiImplicitSolver(grid, physics, 10.0)
    bg = solver.precompute_bg(bg_ref)
    
    tends = physics.get_explicit_tendencies(state, state_prime, bg)
    max_u = float(jnp.max(jnp.abs(tends['u'])))
    print(f"Max spurious horizontal accel (should be ~0): {max_u:.4e} m/s^2")
    assert max_u <= 1e-6, "The z_xi metrics are failing to cancel the vertical pressure gradient on slopes."

def test_9_kinematic_divergence(grid, physics):
    print("\n--- 9. KINEMATIC DIVERGENCE (CONSTANT FLOW) ---")
    state = get_base_state(grid, physics)
    bg_ref = {'rho': state['rho'], 'pi': state['pi'], 'th_v': state['th_v']}
    solver = SemiImplicitSolver(grid, physics, 10.0)
    bg = solver.precompute_bg(bg_ref)
    
    u_0 = 10.0
    u_prescribed = u_0 * jnp.ones_like(grid.X_u)
    
    flux_x = u_prescribed * bg['rho_at_u'] * bg['th_v_at_u'] * bg['dz_u']
    div_x = physics.op.diff_x_u_to_m(flux_x) / bg['dz_m_full']
    
    delta_flux_z = -div_x * bg['dz_m_full']
    flux_z = jnp.concatenate([jnp.zeros((grid.nx, 1)), jnp.cumsum(delta_flux_z, axis=1)], axis=1)
    eta_dot_balanced = flux_z / (bg['dz_w_full'] * bg['rho_at_w'] * bg['th_v_at_w'])
    
    state_prime = {
        'u': u_prescribed,
        'w': jnp.zeros_like(grid.X_w),
        'pi': jnp.zeros_like(grid.X_m),
        'eta_dot': eta_dot_balanced
    }
    
    tends = physics.get_explicit_tendencies(state, state_prime, bg)
    max_div = float(jnp.max(jnp.abs(tends['pi'][:, :-1])))
    print(f"Max spurious pressure tendency (Interior): {max_div:.4e} 1/s")
    assert max_div <= 1e-5, "The 2D divergence operator is registering false divergence over slopes."

def test_10_resting_integration(physics, dt):
    print("\n--- 10. RESTING INTEGRATION (MOUNTAIN ATMOSPHERE) ---")
    stepper = SISLStepper(physics, dt)
    state = get_base_state(physics.grid, physics, u_0=0.0)
    
    def scan_fn(curr_state, step_idx):
        return stepper.step(curr_state, t=step_idx*dt, forcing=None, bc_fn=boundary_conditions), None
        
    final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(50))
    max_w = float(jnp.max(jnp.abs(final_state['w'])))
    
    print(f"Max artificial w-wind after 50 steps: {max_w:.2e} m/s")
    assert max_w < 1e-7, "Metric terms generated artificial winds during integration!"

def test_11_extreme_topography():
    print("\n--- 11. EXTREME TOPOGRAPHY ('THE BRICK WALL') ---")
    nx, nz, Lx, Lz = 200, 40, 100000.0, 20000.0
    def cliff(x):
        return 3500.0 * jnp.exp(-((x - 50000.0)**2) / (1500.0**2))
        
    sleve_transform = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    
    try:
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=cliff, transform=sleve_transform)
        print(">>> FAILED: The grid tangling check failed to catch the extreme topography!")
    except ValueError as e:
        print(f"[Success! Grid tangling prevented execution]\nException caught: {e}")
        print(">>> PASSED: The diagnostic tool successfully blocked invalid topography.")

# ====================================================================
# SIMULATION TESTS
# ====================================================================

def test_12_autodiff_gradients(dt):
    print("\n--- 12. AUTODIFF GRADIENT VERIFICATION (REVERSE-MODE) ---")
    grid = StaggeredGrid(50, 10, 50000.0, 10000.0, h_func=lambda x: 0.0)
    physics = VerticalSlice(grid, CONSTANTS, damp_height=10000.0, N_bv=0.01)
    stepper = SISLStepper(physics, dt)
    
    def forward_loss(u_initial_array):
        state = get_base_state(grid, physics, u_0=0.0)
        state['u'] = u_initial_array
        
        final_state = stepper.integrate(state, 0.0, 5, forcing=None, bc_fn=boundary_conditions)
        
        u_avg = physics.op.avg_u_to_m(final_state['u'])
        w_avg = physics.op.avg_w_to_m(final_state['w'])
        return 0.5 * jnp.sum(u_avg**2 + w_avg**2)

    u_init = 10.0 * jnp.ones_like(grid.X_u)
    print("[Autodiff] Compiling forward and reverse passes...")
    loss_and_grad_fn = jax.jit(jax.value_and_grad(forward_loss))
    loss_val, u_grad = loss_and_grad_fn(u_init)
    
    nan_count = int(jnp.isnan(u_grad).sum())
    print(f"[Result] Forward Loss: {loss_val:.4f}")
    print(f"[Result] Max |Gradient|: {float(jnp.max(jnp.abs(u_grad))):.4e}")
    
    if nan_count > 0:
        print(">>> FAILED: The solver dropped gradients (NaNs detected).")
    else:
        print(">>> PASSED: JAX successfully backpropagated through GMRES and Advection!")


if __name__ == "__main__":
    # Setup unified test environment
    grid, physics, dt = setup_terrain_test_env()
    advector = SemiLagrangianAdvector(grid, physics, dt)
    
    test_1_geometry(grid)
    test_2_backtracking(grid, advector, dt)
    test_3_mass_conservation_fixer(grid, physics, dt)
    test_4_advection_limiter(grid)
    test_5_hydrostatic_balance(grid, physics)
    test_6_semi_implicit_solver(grid, physics, dt)
    test_7_kinematic_bottom_boundary(grid, physics, dt)
    test_8_metric_gradients(grid, physics)
    test_9_kinematic_divergence(grid, physics)
    test_10_resting_integration(physics, dt)
    test_11_extreme_topography()
    test_12_autodiff_gradients(dt)
    
    print("\n" + "="*50)
    print("All 2D Core Diagnostic Tests Completed Successfully!")
    print("="*50)