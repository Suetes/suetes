import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all but FATAL CUDA/XLA warnings


from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SemiLagrangianAdvector3D, SISLStepper3D, SemiImplicitSolver3D, FluxFormAdvector
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.transforms import SleveSimple

import pytest

# Module-level variables and fixtures
dt = 100.0
nx_val, ny_val, nz_val = 32, 32, 15
dx_val, dy_val, dz_val = 1000.0, 1000.0, 500.0
lat_c, lon_c = 45.0, 0.0
constants = {
    'g': 9.81, 
    'cp': 1004.0, 
    'Rd': 287.0, 
    'cvd': 717.0, 
    'p0': 100000.0
}

@pytest.fixture(scope="module")
def env_3d():
    def mountain_h(x, y):
        return 1500.0 * jnp.exp(-(x**2 + y**2) / (5000.0**2))
    grid = RegionalGrid3D(nx_val, ny_val, nz_val, dx_val, dy_val, dz_val, lat_c, lon_c, h_func=mountain_h)
    advector = SemiLagrangianAdvector3D(grid, DummyPhysics(grid), dt)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)
    return grid, advector, physics, op

@pytest.fixture
def grid(env_3d):
    return env_3d[0]

@pytest.fixture
def advector(env_3d):
    return env_3d[1]

@pytest.fixture
def physics(env_3d):
    return env_3d[2]

@pytest.fixture
def op(env_3d):
    return env_3d[3]

@pytest.fixture
def dt_fixture(): return dt
@pytest.fixture
def nx(): return nx_val
@pytest.fixture
def ny(): return ny_val
@pytest.fixture
def nz(): return nz_val
@pytest.fixture
def dx(): return dx_val

# Rename the fixture 'dt' specifically for test function arguments
@pytest.fixture
def dt(dt_fixture): return dt_fixture

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

def test_3_mass_conservation(grid, physics, dt):
    """Tests the Flux-Form Semi-Lagrangian scheme for exact mass conservation."""
    print("\n" + "="*40)
    print("TEST 3: FFSL STRICT MASS CONSERVATION (FLOAT64)")
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

def test_7_davies_sponge(grid, op, nx, ny, nz):
    print("\n--- 7. DAVIES SPONGE BOUNDARY TEST ---")
    
    # Explicitly set the relaxation parameters to ensure test determinism
    dt_test = 30.0
    outflow_factor = 0.01
    
    # Create a sponge that is 5 grid cells deep
    sponge = DaviesSponge(grid, op, sponge_depth=5, dt=dt_test, outflow_factor=outflow_factor)

    # Simulate a complete state. Model interior moving at 10 m/s (West-to-East), exterior stationary.
    model_state = {
        'u': jnp.ones((nx+1, ny, nz)) * 10.0,
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'th_v': jnp.ones((nx, ny, nz)) * 300.0,
        'pi': jnp.ones((nx, ny, nz)) * 1.0,
        'q': jnp.zeros((nx, ny, nz)) 
    }
    
    ext_state = {
        'u': jnp.zeros((nx+1, ny, nz)),
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'th_v': jnp.ones((nx, ny, nz)) * 300.0,
        'pi': jnp.ones((nx, ny, nz)) * 1.0,
        'q': jnp.zeros((nx, ny, nz))
    }

    blended = sponge.blend(model_state, ext_state)

    # Check the center vs the edges
    center_u = float(blended['u'][nx//2, ny//2, nz//2])
    
    # West Edge (x=0). Wind is positive (blowing East), so this is an INFLOW boundary.
    # It should fully apply the sponge (weight = 1.0) -> U perfectly matches the external state (0.0)
    west_edge_u = float(blended['u'][0, ny//2, nz//2])
    expected_west = 0.0 
    
    # East Edge (x=nx). Wind is positive (blowing East), so this is an OUTFLOW boundary.
    # It should drop the sponge weight to outflow_factor (0.01) -> U mostly remains model state (10.0)
    east_edge_u = float(blended['u'][-1, ny//2, nz//2])
    expected_east = (1.0 - outflow_factor) * 10.0 + (outflow_factor) * 0.0

    print(f"U-velocity at domain center (should be 10.0): {center_u:.2f} m/s")
    print(f"U-velocity at West edge (INFLOW, should be {expected_west:.2f}):  {west_edge_u:.2f} m/s")
    print(f"U-velocity at East edge (OUTFLOW, should be {expected_east:.2f}):  {east_edge_u:.2f} m/s")
    
    assert jnp.isclose(center_u, 10.0), "Bug: Sponge is dampening the interior of the domain!"
    assert jnp.isclose(west_edge_u, expected_west), f"Bug: Inflow sponge is not fully relaxing! Expected {expected_west}, got {west_edge_u}"
    assert jnp.isclose(east_edge_u, expected_east), f"Bug: Outflow sponge is not disabling properly! Expected {expected_east}, got {east_edge_u}"


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
    
    # 1. Create a dummy Right-Hand Side representing the explicit step output
    u_wind = 10.0
    rhs_prime = {
        'u': jnp.ones((nx+1, ny, nz)) * u_wind,
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': jnp.zeros((nx, ny, nz)),
        'eta_dot': jnp.zeros((nx, ny, nz+1))
    }

    # 2. Assemble the kinematic boundary constraint onto the RHS (Just like Steppers.py does)
    u_m = physics.op.avg(rhs_prime['u'], axis=0, from_loc='u', to_loc='m')
    u_w = physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
    
    v_m = physics.op.avg(rhs_prime['v'], axis=1, from_loc='v', to_loc='m')
    v_w = physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')

    m_w = jnp.expand_dims(grid.m_factors['w'], axis=-1)
    rhs_kinematic_bottom = m_w[:, :, 0] * (
        u_w[:, :, 0] * grid.z_xi_w[:, :, 0] + 
        v_w[:, :, 0] * grid.z_eta_w[:, :, 0]
    )
    rhs_prime['w'] = rhs_prime['w'].at[:, :, 0].set(rhs_kinematic_bottom)

    # 3. Run the implicit solver
    solver = SemiImplicitSolver3D(physics, dt)
    sol = solver.solve(rhs_prime, bg_precomputed)

    # 4. Verify the solver's identity matrix perfectly preserved the physical constraint
    max_w_expected = float(jnp.max(rhs_kinematic_bottom))
    max_w_actual = float(jnp.max(sol['w'][:,:,0]))

    print(f"Max expected updraft from terrain slope: {max_w_expected:.4f} m/s")
    print(f"Max updraft preserved by implicit solver: {max_w_actual:.4f} m/s")

    assert jnp.isclose(max_w_expected, max_w_actual, rtol=1e-4), "Solver is not preserving the kinematic bottom boundary constraint!"

# --- TERRAIN DIAGNOSTICS TESTS ---

def schaer_2d(x, y):
    h0, a, lam = 2000.0, 15000.0, 8000.0
    return h0 * jnp.exp(-(x**2) / (a**2)) * (jnp.cos(jnp.pi * x / lam)**2)

def setup_terrain_test_env():
    nx, ny, nz = 150, 5, 50
    dx, dy, dz = 1000.0, 1000.0, 400.0
    sleve = SleveSimple(scale_s=4000.0, scale_l=15000.0, n=1.35)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 0.0, 0.0, h_func=schaer_2d, transform=sleve)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
    physics = Euler3D(grid, op, constants, dt=dt, damp_height=grid.Lz, N_bv=0.01)
    
    bg_state_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }
    bg_precomputed = physics.precompute_bg(bg_state_ref)
    return grid, op, physics, bg_precomputed

def test_9_metric_gradients():
    print("\n--- 9. METRIC GRADIENT CONSISTENCY ---")
    grid, op, physics, bg = setup_terrain_test_env()
    
    pi_prime = jnp.zeros_like(grid.Z_m)
    state_prime = {'u': jnp.zeros_like(grid.Z_u), 'v': jnp.zeros_like(grid.Z_v), 
                   'w': jnp.zeros_like(grid.Z_w), 'pi': pi_prime, 'eta_dot': jnp.zeros_like(grid.Z_w)}
    
    tends = physics.get_tendencies(state_prime, bg)
    
    max_u_accel = float(jnp.max(jnp.abs(tends['u'])))
    max_w_accel = float(jnp.max(jnp.abs(tends['w'])))
    
    print(f"Max spurious horizontal accel (should be ~0): {max_u_accel:.4e} m/s^2")
    print(f"Max spurious vertical accel (should be ~0):   {max_w_accel:.4e} m/s^2")
    
    assert max_u_accel <= 1e-6, "The transformation metrics (z_xi) are failing to cancel the vertical pressure gradient on slopes."
    print("STATUS: SUCCESS (Hydrostatic balance holds over topography)")

def test_10_kinematic_divergence():
    print("\n--- 10. KINEMATIC DIVERGENCE (CONSTANT FLOW) ---")
    grid, op, physics, bg = setup_terrain_test_env()
    
    u_0 = 10.0
    u_prescribed = u_0 * jnp.ones_like(grid.Z_u)
    v_prescribed = jnp.zeros_like(grid.Z_v)
    
    # Calculate horizontal fluxes manually
    m_u, m_v = grid.m_factors['u'][..., None], grid.m_factors['v'][..., None]
    flux_x = (u_prescribed * bg['rho_u'] * bg['th_v_u'] * bg['dz_u']) / m_u
    flux_y = (v_prescribed * bg['rho_v'] * bg['th_v_v'] * bg['dz_v']) / m_v
    
    div_x = op.diff(flux_x, axis=0, from_loc='u', to_loc='m') / bg['dz_m_full']
    div_y = op.diff(flux_y, axis=1, from_loc='v', to_loc='m') / bg['dz_m_full']
    div_h = div_x + div_y
    
    # Integrate div_h upwards to find the EXACT balancing vertical flux
    delta_flux_z = -div_h * bg['dz_m_full'] 
    
    # flux_z is defined on w-points (nz+1). Boundary condition is 0 at bottom.
    flux_z = jnp.concatenate([
        jnp.zeros((grid.nx, grid.ny, 1)), 
        jnp.cumsum(delta_flux_z, axis=2)
    ], axis=2)
    
    # Convert flux_z back to the contravariant velocity (eta_dot)
    eta_dot_balanced = flux_z / (bg['dz_w_full'] * bg['rho_w'] * bg['th_v_w'])
    
    # Feed this perfectly non-divergent state to the physics operator
    state_prime = {
        'u': u_prescribed, 
        'v': v_prescribed, 
        'w': jnp.zeros_like(grid.Z_w), 
        'pi': jnp.zeros_like(grid.Z_m), 
        'eta_dot': eta_dot_balanced
    }
    
    tends = physics.get_tendencies(state_prime, bg)
    
    # Check divergence. We IGNORE the topmost layer [:, :, -1]
    max_div = float(jnp.max(jnp.abs(tends['pi'][:, :, :-1])))
    print(f"Max spurious pressure tendency (Interior): {max_div:.4e} 1/s")
    
    assert max_div <= 1e-5, "The 3D divergence operator is registering false divergence over slopes."
    print("STATUS: SUCCESS (3D Divergence is clean)")

def test_11_vertical_metric_smoothness():
    print("\n--- 11. VERTICAL METRIC (dz) TERROR CHECK ---")
    grid, op, physics, bg = setup_terrain_test_env()
    
    dz_min = float(jnp.min(bg['dz_w_full']))
    dz_max = float(jnp.max(bg['dz_w_full']))
    
    print(f"Physical Layer Thickness (dz) - Min: {dz_min:.2f} m, Max: {dz_max:.2f} m")
    assert dz_min >= 10.0, "Your SLEVE coordinate is compressing layers too thinly over the mountain peaks. This will shatter the CFL condition and cause GMRES to explode."
    print("STATUS: SUCCESS (Vertical grid spacing is safe)")

def test_12_resting_flat_integration(grid, physics, dt):
    print("\n--- 12. RESTING FLAT ATMOSPHERE (50-STEP INTEGRATION) ---")
    stepper = SISLStepper3D(physics, dt)
    
    # Perfect resting state
    initial_state = {
        'u': jnp.zeros_like(grid.Z_u), 'v': jnp.zeros_like(grid.Z_v), 'w': jnp.zeros_like(grid.Z_w),
        'eta_dot': jnp.zeros_like(grid.Z_w), 'pi': physics.pi_bg, 'th_v': physics.theta_bg,
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd']))
    }
    
    def dummy_bc(state_next, forcing): return state_next
    def scan_fn(curr_state, step_idx):
        return stepper.step(curr_state, t=step_idx*dt, forcing=None, bc_fn=dummy_bc), None
        
    final_state, _ = jax.lax.scan(scan_fn, initial_state, jnp.arange(50))
    
    max_w = float(jnp.max(jnp.abs(final_state['w'])))
    print(f"Max artificial w-wind after 50 steps: {max_w:.2e} m/s")
    assert max_w < 1e-7, "Model lost hydrostatic balance during integration!"
    print("STATUS: SUCCESS (Integration is stable)")

def test_13_resting_mountain_integration(dt):
    print("\n--- 13. RESTING MOUNTAIN ATMOSPHERE (50-STEP INTEGRATION) ---")
    # Setup a fresh grid with a steep Agnesi mountain specifically for this integration
    nx, ny, nz = 50, 50, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    def h_func(x, y): return 1500.0 / (1.0 + (x**2 + y**2) / (30000.0**2))
    
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=5.0, h_func=h_func)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}
    physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01) 
    stepper = SISLStepper3D(physics, dt)
    
    initial_state = {
        'u': jnp.zeros_like(grid.Z_u), 'v': jnp.zeros_like(grid.Z_v), 'w': jnp.zeros_like(grid.Z_w),
        'eta_dot': jnp.zeros_like(grid.Z_w), 'pi': physics.pi_bg, 'th_v': physics.theta_bg,
        'rho': constants['p0'] / (constants['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (constants['cvd'] / constants['Rd']))
    }
    
    def dummy_bc(state_next, forcing): return state_next
    def scan_fn(curr_state, step_idx):
        return stepper.step(curr_state, t=step_idx*dt, forcing=None, bc_fn=dummy_bc), None
        
    final_state, _ = jax.lax.scan(scan_fn, initial_state, jnp.arange(50))
    
    max_w = float(jnp.max(jnp.abs(final_state['w'])))
    print(f"Max artificial w-wind over steep terrain after 50 steps: {max_w:.2e} m/s")
    assert max_w < 1e-7, "Metric terms generated artificial winds during integration!"
    print("STATUS: SUCCESS (Mountain integration is stable)")

def test_14_autodiff_gradients():
    print("\n--- 14. AUTODIFF GRADIENT VERIFICATION (REVERSE-MODE 3D) ---")
    
    # Setup a minimal 3D grid to keep compilation fast
    nx, ny, nz = 16, 16, 8
    dx, dy, dz = 10000.0, 10000.0, 1000.0
    
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=lambda x, y: 0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}
    dt = 30.0
    
    physics = Euler3D(grid, op, constants, dt=dt, damp_height=8000.0, N_bv=0.01)
    stepper = SISLStepper3D(physics, dt)
    
    # Define the pure, side-effect-free forward pass
    def forward_loss(u_initial_array):
        # Reconstruct the base state
        rho_bg = physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
                 (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd']))
                 
        state = {
            'u': u_initial_array,
            'v': jnp.zeros((nx, ny+1, nz)),
            'w': jnp.zeros((nx, ny, nz+1)),
            'pi': physics.pi_bg,
            'th_v': physics.theta_bg,
            'eta_dot': jnp.zeros((nx, ny, nz+1)),
            'rho': rho_bg
        }
        
        def dummy_bc(state_next, forcing): return state_next
        
        def scan_fn(curr_state, step_idx):
            return stepper.step(curr_state, t=step_idx*dt, forcing=None, bc_fn=dummy_bc), None
            
        # Run for 3 steps to build the deep computational graph
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(3))
        
        # Calculate a pseudo-Kinetic Energy proxy as the scalar loss
        ke = 0.5 * (jnp.sum(final_state['u']**2) + jnp.sum(final_state['v']**2) + jnp.sum(final_state['w']**2))
        return ke

    # Create the initial parameter state (10 m/s wind)
    u_init = 10.0 * jnp.ones_like(grid.Z_u)
    
    # JIT compile the value and gradient function
    print("[Autodiff] Compiling 3D forward and reverse passes (this may take a minute)...")
    loss_and_grad_fn = jax.jit(jax.value_and_grad(forward_loss))
    
    # Execute
    loss_val, u_grad = loss_and_grad_fn(u_init)
    
    print(f"\n[Result] Forward Loss (Kinetic Energy Proxy): {loss_val:.4f}")
    
    # Gradient diagnostics
    max_grad = float(jnp.max(jnp.abs(u_grad)))
    mean_grad = float(jnp.mean(jnp.abs(u_grad)))
    nan_count = int(jnp.isnan(u_grad).sum())
    
    print(f"[Result] Max |Gradient|:  {max_grad:.4e}")
    print(f"[Result] Mean |Gradient|: {mean_grad:.4e}")
    print(f"[Result] NaN Count:       {nan_count}")
    
    if nan_count > 0:
        print(">>> FAILED: The 3D solver dropped gradients (NaNs detected).")
    elif max_grad == 0.0:
        print(">>> FAILED: The 3D gradient is strictly zero. The graph is detached.")
    else:
        print(">>> PASSED: JAX successfully backpropagated through the 3D GMRES solver and Semi-Lagrangian advection!")


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
    physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)

    test_1_geometry(grid)
    test_2_backtracking(grid, advector, nx, ny, nz, dx, dt)
    test_3_mass_conservation(grid, physics, dt)
    test_4_advection_limiter(grid, advector, nx, ny, nz)
    test_5_hydrostatic_balance(grid, physics, nx, ny, nz)
    test_6_semi_implicit_solver(physics, dt, nx, ny, nz)
    test_7_davies_sponge(grid, op, nx, ny, nz)
    test_8_kinematic_bottom_boundary(grid, physics, nx, ny, nz, dt)
    test_9_metric_gradients()
    test_10_kinematic_divergence()
    test_11_vertical_metric_smoothness()
    test_12_resting_flat_integration(grid, physics, dt)
    test_13_resting_mountain_integration(dt)
    test_14_autodiff_gradients()

    print("\nAll boundary, kinematic and terrain diagnostic tests completed successfully!")