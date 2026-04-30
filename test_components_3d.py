import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SemiLagrangianAdvector3D, SISLStepper3D, SemiImplicitSolver3D, FluxFormAdvector
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.boundaries import DaviesSponge

# Dummy physics object to satisfy the advector initialization
class DummyPhysics:
    def __init__(self, grid):
        self.grid = grid
        self.op = CGridOperator3D(grid)

def test_1_geometry(grid):
    print("--- GEOMETRY TEST ---")
    print(f"Max Mountain Height: {jnp.max(grid.Z_w[:,:,0]):.2f} m") 
    print(f"Top boundary is flat: {jnp.std(grid.Z_w[:,:,-1]) < 1e-3}")

    min_dz = float(jnp.min(grid.dz_m_full))
    print(f"Minimum layer thickness: {min_dz:.2f} m")
    assert min_dz > 0.0, "Grid tangling detected! Transform failed."

def test_2_backtracking(grid, advector, nx, ny, nz, dx, dt):
    print("\n--- BACKTRACKING TEST ---")
    
    # Create a uniform wind field: 10 m/s in X, 0 in Y and Z.
    state = {
        'u': jnp.ones((nx + 1, ny, nz)) * 10.0,
        'v': jnp.zeros((nx, ny + 1, nz)),
        'eta_dot': jnp.zeros((nx, ny, nz + 1))
    }

    dep_indices = advector.compute_departure_indices(state, loc='m', iterations=2)

    idx_x, idx_y, idx_z = jnp.arange(nx), jnp.arange(ny), jnp.arange(nz)
    Xi, Yi, Zi = jnp.meshgrid(idx_x, idx_y, idx_z, indexing='ij')

    disp_x = Xi - dep_indices[0]

    map_factors_3d = grid.m_factors['m'][..., None] 
    expected_disp = (10.0 * dt / dx) * map_factors_3d

    error = jnp.max(jnp.abs(disp_x - expected_disp))

    print(f"Max trajectory error: {error:.5f} indices")
    assert error < 1e-4, "Semi-Lagrangian iterative backtracker is failing."

def test_3_mass_consistency(grid, nx, ny, nz, dx, dy, dt):
    print("\n--- MASS CONSISTENCY TEST ---")
    m_sq = grid.m_factors['m'][..., None] ** 2

    # This uses the true 3D physical depths (dz_m_full) instead of the 1D logical depth (dz)
    cell_volumes = (dx * dy / m_sq) * grid.dz_m_full

    rho_init = jnp.ones((nx, ny, nz))
    tr1_init = jnp.ones((nx, ny, nz))

    mass_initial = float(jnp.sum(tr1_init * rho_init * cell_volumes))
    print(f"Initial Mass integral: {mass_initial:e} kg")

    stepper = SISLStepper3D(DummyPhysics(grid), dt, use_mass_fixer=True, tracer_keys=['tr1'])

    state_before = {'rho': rho_init, 'tr1': tr1_init}
    state_after_loss = {'rho': rho_init * 0.95, 'tr1': tr1_init * 0.95}

    fixed_state = stepper._apply_mass_fixer(state_before, state_after_loss)

    mass_unfixed = float(jnp.sum(state_after_loss['tr1'] * state_after_loss['rho'] * cell_volumes))
    mass_fixed = float(jnp.sum(fixed_state['tr1'] * state_after_loss['rho'] * cell_volumes))

    print(f"Mass after advection loss: {mass_unfixed:e} kg")
    print(f"Mass after fixer applied:  {mass_fixed:e} kg")

    assert jnp.isclose(mass_initial, mass_fixed, rtol=1e-4), "Mass fixer is not conserving volume!"

def test_4_advection_limiter(grid, advector, nx, ny, nz):
    print("\n--- 4. ADVECTION LIMITER TEST ---")
    # Create a 3D Gaussian tracer "blob" in the middle of the domain
    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    blob = jnp.exp(-((X)**2 + (Y)**2 + (Z - 2000.0)**2) / (1500.0**2))

    # Force a displacement of 0.5 grid cells (worst-case for interpolation dispersion)
    idx_x, idx_y, idx_z = jnp.arange(nx), jnp.arange(ny), jnp.arange(nz)
    Xi, Yi, Zi = jnp.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
    coords_shifted = jnp.stack([Xi - 0.5, Yi - 0.5, Zi - 0.5], axis=0)

    # Advect with and without the quasi-monotone limiter
    blob_unlim = advector.advect_cubic(blob, coords_shifted, use_limiter=False)
    blob_lim = advector.advect_cubic(blob, coords_shifted, use_limiter=True)

    min_unlim, min_lim = float(jnp.min(blob_unlim)), float(jnp.min(blob_lim))
    print(f"Minimum tracer value (Unlimited): {min_unlim:.6f}  <-- Notice the undershoot!")
    print(f"Minimum tracer value (Limited):   {min_lim:.6f}")
    assert min_lim >= 0.0, "Limiter failed to prevent negative tracer values!"

def test_5_hydrostatic_balance(grid, physics, nx, ny, nz):
    print("\n--- 5. HYDROSTATIC BALANCE (SPURIOUS WINDS) ---")
    # Build a resting, hydrostatically balanced background state
    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_ref)

    # The prime state is entirely zero (no perturbation)
    state_prime_rest = {
        'u': jnp.zeros((nx+1, ny, nz)),
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': jnp.zeros((nx, ny, nz)),
        'eta_dot': jnp.zeros((nx, ny, nz+1))
    }

    # Calculate the tendencies. They should be effectively zero!
    tends = physics.get_tendencies(state_prime_rest, bg_precomputed)

    max_u_tend = float(jnp.max(jnp.abs(tends['u'])))
    max_w_tend = float(jnp.max(jnp.abs(tends['w'])))
    print(f"Max U-velocity tendency (Spurious Wind): {max_u_tend:e} m/s^2")
    print(f"Max W-velocity tendency:                 {max_w_tend:e} m/s^2")
    assert max_u_tend < 1e-4, "Spurious horizontal winds detected! Metric terms are misaligned."

def test_6_semi_implicit_solver(physics, dt, nx, ny, nz):
    print("\n--- 6. SEMI-IMPLICIT SOLVER CONVERGENCE ---")
    solver = SemiImplicitSolver3D(physics, dt)

    # Build a resting, hydrostatically balanced background state
    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_ref)

    # Create a random right-hand side (representing the explicit advection output)
    key = jax.random.PRNGKey(42)
    rhs_dummy = {
        'u': jax.random.uniform(key, (nx+1, ny, nz)) * 0.1,
        'v': jax.random.uniform(key, (nx, ny+1, nz)) * 0.1,
        'w': jax.random.uniform(key, (nx, ny, nz+1)) * 0.01,
        'pi': jax.random.uniform(key, (nx, ny, nz)) * 0.001,
        'eta_dot': jax.random.uniform(key, (nx, ny, nz+1)) * 0.001
    }

    # Run the solver
    sol = solver.solve(rhs_dummy, bg_precomputed)

    # Quick sanity check on outputs
    is_valid = jnp.all(jnp.isfinite(sol['u'])) and jnp.all(jnp.isfinite(sol['pi']))
    max_u_sol = float(jnp.max(jnp.abs(sol['u'])))

    print(f"Solver output contains only finite numbers: {is_valid}")
    print(f"Max U in solved state: {max_u_sol:.4f} m/s")
    assert is_valid, "Solver diverged and produced NaNs!"

def test_7_davies_sponge(grid, nx, ny, nz):
    print("\n--- 7. DAVIES SPONGE BOUNDARY TEST ---")
    # Create a sponge that is 5 grid cells deep
    sponge = DaviesSponge(grid, sponge_depth=5)

    # Simulate a model interior moving at 10 m/s, and a stationary exterior (0 m/s)
    model_state = {'u': jnp.ones((nx+1, ny, nz)) * 10.0}
    ext_state = {'u': jnp.zeros((nx+1, ny, nz))}

    blended = sponge.blend(model_state, ext_state)

    center_u = float(blended['u'][nx//2, ny//2, nz//2])
    edge_u = float(blended['u'][0, ny//2, nz//2])

    print(f"U-velocity at domain center (should be 10.0): {center_u:.2f} m/s")
    print(f"U-velocity at lateral edge (should be 0.0):   {edge_u:.2f} m/s")
    assert center_u == 10.0, "Bug: Sponge is dampening the interior of the domain!"
    assert edge_u == 0.0, "Bug: Sponge is not relaxing the lateral boundaries!"

def test_8_kinematic_bottom_boundary(grid, physics, nx, ny, nz, dt):
    print("\n--- 8. KINEMATIC BOTTOM BOUNDARY (FLOW OVER MOUNTAIN) ---")
    
    # Build a resting, hydrostatically balanced background state
    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_ref)
    
    # Force a uniform 10 m/s cross-mountain wind
    u_wind = 10.0
    state_wind = {
        'u': jnp.ones((nx+1, ny, nz)) * u_wind,
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': jnp.zeros((nx, ny, nz)),
        'eta_dot': jnp.zeros((nx, ny, nz+1))
    }

    # The linear operator applies the kinematic boundary constraint at the surface
    L_out = physics.linear_operator(state_wind, bg_precomputed, dt)

    # Calculate what the theoretical updraft should be based on the mountain slope
    u_m = physics.op.avg(state_wind['u'], axis=0, from_loc='u', to_loc='m')
    u_w = physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')

    expected_w_bottom = u_w[:,:,0] * grid.z_xi_w[:,:,0]

    max_w_expected = float(jnp.max(expected_w_bottom))
    max_w_actual = float(jnp.max(L_out['w'][:,:,0]))

    print(f"Max expected updraft from terrain slope: {max_w_expected:.4f} m/s")
    print(f"Max updraft enforced by implicit solver: {max_w_actual:.4f} m/s")

    # Ensure the solver is perfectly matching the terrain slope constraint
    assert jnp.isclose(max_w_expected, max_w_actual, rtol=1e-4), "Solver is not enforcing flow over the mountain!"

def test_9_flux_form_mass_conservation(grid, physics, dt):
    """Tests the Flux-Form Semi-Lagrangian scheme for exact mass conservation."""
    print("\n" + "="*40)
    print("TEST 9: FFSL STRICT MASS CONSERVATION (FLOAT64)")
    print("="*40)

    # Diagnose background density for the new test state
    rho_bg = physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
             (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd']))

    bg_state_ref = {
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'rho': rho_bg
    }
    bg_precomputed = physics.precompute_bg(bg_state_ref)

    state = {
        'u': jnp.ones((grid.nx + 1, grid.ny, grid.nz), dtype=jnp.float64) * 15.0,
        'v': jnp.ones((grid.nx, grid.ny + 1, grid.nz), dtype=jnp.float64) * 5.0,
        'w': jnp.zeros((grid.nx, grid.ny, grid.nz + 1), dtype=jnp.float64),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros((grid.nx, grid.ny, grid.nz + 1), dtype=jnp.float64),
        'rho': rho_bg, 
        'tracer_1': jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float64) * 2.5 
    }
    
    # 1. Calculate Initial Mass (explicitly in float64)
    m_sq = grid.m_factors['m'][..., None] ** 2
    cell_volumes = (grid.dx * grid.dy / m_sq) * grid.dz_m_full
    mass_initial = float(jnp.sum(state['rho'] * cell_volumes, dtype=jnp.float64))
    print(f"Initial Mass: {mass_initial:.6e} kg")

    # 2. Enforce a Closed Box (Solid Walls)
    state_closed = dict(state)
    state_closed['u'] = state['u'].at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    state_closed['v'] = state['v'].at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    state_closed['eta_dot'] = state['eta_dot'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

    # 3. Advect using the new FFSL scheme
    ffsl_advector = FluxFormAdvector(grid, dt)
    advect_jit = jax.jit(ffsl_advector.advect_3d_split)
    
    rho_next = advect_jit(state_closed['rho'], state_closed, bg_precomputed)

    # 4. Calculate Final Mass (explicitly in float64)
    mass_final = float(jnp.sum(rho_next * cell_volumes, dtype=jnp.float64))
    print(f"Final Mass:   {mass_final:.6e} kg")
    
    drift = mass_final - mass_initial
    print(f"Mass Drift:   {drift:.6e} kg")
    
    assert abs(drift) / mass_initial < 1e-14, f"FFSL scheme leaked mass! Drift: {drift}"
    print("STATUS: SUCCESS (Mass conserved to high precision)")


if __name__ == "__main__":
    nx, ny, nz = 32, 32, 15
    dx, dy, dz = 1000.0, 1000.0, 500.0
    lat_c, lon_c = 45.0, 0.0

    def mountain_h(x, y):
        return 1500.0 * jnp.exp(-(x**2 + y**2) / (5000.0**2))

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=mountain_h)

    dt = 100.0
    advector = SemiLagrangianAdvector3D(grid, DummyPhysics(grid), dt)

    constants = {
        'g': 9.81, 
        'cp': 1004.0, 
        'Rd': 287.0, 
        'cvd': 717.0, 
        'p0': 100000.0
    }
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, constants, N_bv=0.01)

    test_1_geometry(grid)
    test_2_backtracking(grid, advector, nx, ny, nz, dx, dt)
    test_3_mass_consistency(grid, nx, ny, nz, dx, dy, dt)
    test_4_advection_limiter(grid, advector, nx, ny, nz)
    test_5_hydrostatic_balance(grid, physics, nx, ny, nz)
    test_6_semi_implicit_solver(physics, dt, nx, ny, nz)
    test_7_davies_sponge(grid, nx, ny, nz)
    test_8_kinematic_bottom_boundary(grid, physics, nx, ny, nz, dt)
    test_9_flux_form_mass_conservation(grid, physics, dt)

    print("\nAll Boundary & Kinematic tests completed successfully!")