"""
End-to-end Jablonowski-Williamson perfect-model pipeline in one script.

Stage 1 (dinosaur):  Run dinosaur's J-W baroclinic instability test and write a
                     netCDF in the format expected by DinosaurProcessor
                     (schema: suetes/preprocessing/dinosaur2suetes.py).
Stage 2 (Suetes):    Drive the Suetes regional core with that netCDF and produce
                     diagnostics.

The two stages communicate through a single netCDF on disk (--output). This is
the same handoff the two original scripts used; merging just chains them.

Notes:
  - jax_enable_x64 is required by the dinosaur stage and is set process-wide, so
    it is also active during the Suetes stage. If that precision change is a
    problem, run the stages as two separate processes instead.
  - The dinosaur --sim_hours controls how many hourly frames the driver holds.
    The Suetes stage uses its own internal start_hour=215 and sim_hours=24 to
    index into those frames, so the driver must contain >= ~240 frames. The
    Suetes stage self-truncates if the driver is shorter.

  Make sure you use DFI!!
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from dinosaur import (
    coordinate_systems,
    primitive_equations,
    primitive_equations_states,
    sigma_coordinates,
    spherical_harmonic,
    time_integration,
    xarray_utils,
    scales,
)

from suetes.preprocessing.dinosaur2suetes import DinosaurProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager

from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.physics import (
    PhysicsSuite, NewtonianRelaxation,
    McFarlaneVerticalDiffusion, McFarlaneSurfaceDrag,
    SmagorinskyLillySGS,
)

from suetes.vis.visualizer import Visualizer

units = scales.units


# Standard pressure levels (hPa) for output. Matches a common ERA5 selection.
# 50 hPa is well below the dinosaur top with 24 equidistant sigma layers
# (top is ~2 hPa), so no extrapolation is needed.
PRESSURE_LEVELS_HPA = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dtype=np.float64,
)


# ===========================================================================
#                          STAGE 1:  DINOSAUR DRIVER
# ===========================================================================

# ---------------------------------------------------------------------------
#                              SETUP
# ---------------------------------------------------------------------------

def setup(resolution='T42', n_layers=24):
    """Build dinosaur coordinates, physics, equation, initial state."""
    grid = getattr(spherical_harmonic.Grid, resolution)()
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(n_layers)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()

    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    steady_state = initial_state_fn()

    # Overlay the J-W Gaussian-hill u perturbation. State is a
    # @tree_math.struct dataclass; addition combines them element-wise.
    perturbation = primitive_equations_states.baroclinic_perturbation_jw(
        coords, physics_specs
    )
    state = steady_state + perturbation

    ref_temperature = aux_features[xarray_utils.REF_TEMP_KEY]
    orography_modal = primitive_equations.truncated_modal_orography(
        aux_features[xarray_utils.OROGRAPHY], coords,
    )

    eq = primitive_equations.PrimitiveEquations(
        reference_temperature=ref_temperature,
        orography=orography_modal,
        coords=coords,
        physics_specs=physics_specs,
    )
    return coords, physics_specs, eq, state, ref_temperature


def build_step_fn(eq, coords, dt_seconds, physics_specs):
    """Build an IMEX-RK SIL3 step with an exponential mode-attenuation filter."""
    dt_nondim = physics_specs.nondimensionalize(dt_seconds * units.s)
    step_fn = time_integration.imex_rk_sil3(eq, dt_nondim)
    filters = [time_integration.exponential_step_filter(
        coords.horizontal, dt_nondim,
        tau=0.0087504, order=1.5, cutoff=0.8,
    )]
    return time_integration.step_with_filters(step_fn, filters), dt_nondim


# ---------------------------------------------------------------------------
#                       SPECTRAL  ->  NODAL  CONVERSION
# ---------------------------------------------------------------------------

def to_nodal_si(state, coords, physics_specs, ref_temperature):
    """Convert one spectral state to nodal SI fields in (level, lat, lon) order.

    dinosaur stores nodal data in (level, lon, lat); we transpose to the
    (level, lat, lon) layout that ERA5 and Suetes use.
    """
    horizontal = coords.horizontal

    u_nd, v_nd = spherical_harmonic.vor_div_to_uv_nodal(
        horizontal, state.vorticity, state.divergence,
    )
    T_var_nd = horizontal.to_nodal(state.temperature_variation)
    T_nd = T_var_nd + ref_temperature[:, None, None]
    log_ps_nd = horizontal.to_nodal(state.log_surface_pressure)
    ps_nd = jnp.exp(log_ps_nd)

    # physics_specs.dimensionalize returns a pint Quantity; .magnitude strips units.
    u  = np.asarray(physics_specs.dimensionalize(u_nd,  units.m / units.s).magnitude)
    v  = np.asarray(physics_specs.dimensionalize(v_nd,  units.m / units.s).magnitude)
    T  = np.asarray(physics_specs.dimensionalize(T_nd,  units.kelvin).magnitude)
    ps = np.asarray(physics_specs.dimensionalize(ps_nd, units.pascal).magnitude)

    # (level, lon, lat) -> (level, lat, lon)
    u = np.transpose(u, (0, 2, 1))
    v = np.transpose(v, (0, 2, 1))
    T = np.transpose(T, (0, 2, 1))
    # ps has shape (1, lon, lat) -> drop level dim and transpose to (lat, lon)
    ps = np.transpose(ps[0], (1, 0))
    return u, v, T, ps


# ---------------------------------------------------------------------------
#                  SIGMA  ->  PRESSURE  INTERPOLATION
# ---------------------------------------------------------------------------

def sigma_to_pressure(field_sigma, sigma_values, ps, target_pressures_pa):
    """
    Vertically interpolate a sigma-level field onto fixed pressure levels.

    For each (lat, lon) column, p_sigma_k = sigma_k * ps. Interpolation is
    linear in log(p).

    Args:
        field_sigma:         (L_sigma, Y, X) field on sigma levels.
        sigma_values:        (L_sigma,) sigma at layer centers (top to surf).
        ps:                  (Y, X) surface pressure [Pa].
        target_pressures_pa: (L_p,) target pressure levels [Pa], ascending.

    Returns:
        (L_p, Y, X) field on target pressure levels.
    """
    L_p = target_pressures_pa.shape[0]
    _, ny, nx = field_sigma.shape
    out = np.empty((L_p, ny, nx), dtype=field_sigma.dtype)
    log_target = np.log(target_pressures_pa)

    for j in range(ny):
        for i in range(nx):
            p_col = sigma_values * ps[j, i]
            log_p_col = np.log(p_col)
            out[:, j, i] = np.interp(log_target, log_p_col, field_sigma[:, j, i])
    return out


def compute_geopotential_sigma(T_sigma, sigma_values, g=9.80616, R=287.04,
                               z_surf=0.0):
    """
    Hydrostatic integration on dinosaur's native sigma levels.

    Args:
        T_sigma:      (L_sigma, Y, X) temperature on sigma levels (top to surf).
        sigma_values: (L_sigma,) sigma at layer centers, ascending top to surf.
        z_surf:       scalar or (Y, X) surface elevation [m]. 0 for flat earth.

    Returns:
        (L_sigma, Y, X) geopotential = g*z [m**2 s-2] at the sigma centers.
    """
    L = sigma_values.shape[0]
    z = np.empty_like(T_sigma)

    # Bottom layer: integrate from surface (sigma=1) to lowest model sigma.
    # ln(p_surf / p_bottom) = ln(1 / sigma_{L-1}) since p = sigma * p_s.
    z[L - 1] = z_surf + R * T_sigma[L - 1] / g * np.log(1.0 / sigma_values[L - 1])

    # Layer-by-layer upward integration with trapezoidal-mean temperature.
    for k in range(L - 2, -1, -1):
        T_mid = 0.5 * (T_sigma[k] + T_sigma[k + 1])
        z[k] = z[k + 1] + R * T_mid / g * np.log(
            sigma_values[k + 1] / sigma_values[k]
        )
    return g * z


# ---------------------------------------------------------------------------
#               HYDROSTATIC  OMEGA  FROM  HORIZONTAL  DIVERGENCE
# ---------------------------------------------------------------------------

def compute_hydrostatic_omega(u_pl, v_pl, p_pa, lats_deg, lons_deg,
                              R_earth=6371229.0):
    """
    Diagnose omega = dp/dt from u, v on pressure levels.

    On pressure levels in a hydrostatic system the continuity equation is

        d(omega)/dp = -div_h(u, v)

    Args:
        u_pl:     (n_out, Lp, ny, nx) zonal wind on pressure levels [m/s].
        v_pl:     (n_out, Lp, ny, nx) meridional wind on pressure levels [m/s].
        p_pa:     (Lp,) pressure levels in Pa, ascending top -> surface.
        lats_deg: (ny,) latitudes in degrees.
        lons_deg: (nx,) longitudes in degrees.
        R_earth:  Earth radius in meters.

    Returns:
        (n_out, Lp, ny, nx) omega in Pa/s, with omega = 0 at the top level.
    """
    n_out, Lp, ny, nx = u_pl.shape

    deg = np.pi / 180.0
    cos_lat = np.cos(lats_deg * deg)               # (ny,)
    dlon_rad = (lons_deg[1] - lons_deg[0]) * deg   # uniform spacing
    dlat_rad = (lats_deg[1] - lats_deg[0]) * deg

    omega_pl = np.zeros((n_out, Lp, ny, nx), dtype=np.float32)

    for t in range(n_out):
        for k in range(Lp):
            u_k = u_pl[t, k].astype(np.float64)
            v_k = v_pl[t, k].astype(np.float64)

            # d(u)/d(lon), periodic in longitude (dinosaur covers the globe).
            du_dlon = np.empty_like(u_k)
            du_dlon[:, 1:-1] = (u_k[:, 2:] - u_k[:, :-2]) / (2.0 * dlon_rad)
            du_dlon[:, 0]    = (u_k[:, 1]  - u_k[:, -1]) / (2.0 * dlon_rad)
            du_dlon[:, -1]   = (u_k[:, 0]  - u_k[:, -2]) / (2.0 * dlon_rad)

            # d(v cos(lat))/d(lat), one-sided at the meridional edges.
            vc = v_k * cos_lat[:, None]
            dvc_dlat = np.empty_like(vc)
            dvc_dlat[1:-1, :] = (vc[2:, :] - vc[:-2, :]) / (2.0 * dlat_rad)
            dvc_dlat[0, :]    = (vc[1, :]  - vc[0, :])    / dlat_rad
            dvc_dlat[-1, :]   = (vc[-1, :] - vc[-2, :])   / dlat_rad

            div_h = (du_dlon / (R_earth * cos_lat[:, None])) \
                  + (dvc_dlat / (R_earth * cos_lat[:, None]))

            if k == 0:
                # omega = 0 at the top of the atmosphere.
                omega_pl[t, k] = 0.0
            else:
                dp = float(p_pa[k] - p_pa[k - 1])   # positive, p increases down
                omega_pl[t, k] = omega_pl[t, k - 1] - (div_h * dp).astype(np.float32)

        if t % 24 == 0:
            print(f'  omega for snapshot {t}/{n_out}, '
                  f'max|omega|={float(np.abs(omega_pl[t]).max()):.3e} Pa/s')

    return omega_pl


def run_dinosaur(output_path, resolution='T42', layers=24,
                 sim_hours=240, save_every_h=1.0, dt_seconds=200.0):
    """Run the dinosaur J-W test and write the driver netCDF to output_path."""
    coords, physics_specs, eq, state, ref_temperature = setup(
        resolution, layers,
    )
    step_fn, dt_nondim = build_step_fn(eq, coords, dt_seconds, physics_specs)

    n_steps_total  = int(round(sim_hours * 3600.0 / dt_seconds))
    steps_per_save = int(round(save_every_h * 3600.0 / dt_seconds))
    # n_saves+1 frames so the output covers hours [0, 1, ..., sim_hours]
    # inclusive (e.g. sim_hours 240 gives 241 hourly frames). With
    # start_with_input=True the saves happen at the START of each outer
    # iteration, so we capture t=sim_hours before the (discarded) final step.
    n_saves        = n_steps_total // steps_per_save + 1
    print(f'[DINOSAUR] {resolution}, {layers} layers, '
          f'dt={dt_seconds:.0f}s, {n_steps_total} steps, '
          f'save every {steps_per_save} steps (n_saves={n_saves}).')

    # trajectory_from_step compiles one outer step (inner_steps dynamics steps
    # = one save) and accumulates outer_steps frames via jax.lax.scan.
    # start_with_input=True includes the initial state at output index 0.
    traj_fn = time_integration.trajectory_from_step(
        step_fn,
        outer_steps=n_saves,
        inner_steps=steps_per_save,
        start_with_input=True,
    )
    traj_fn = jax.jit(traj_fn)

    print('[DINOSAUR] compiling and running forward integration...')
    t_wall = time.time()
    _final_state, trajectory = traj_fn(state)
    trajectory.vorticity.block_until_ready()
    print(f'[DINOSAUR] forward integration done in {time.time() - t_wall:.1f}s')

    # Coordinates: dinosaur stores lat, lon in radians; convert to degrees.
    sigma_values = np.asarray(coords.vertical.centers)
    lats_deg = np.degrees(np.asarray(coords.horizontal.latitudes))
    lons_deg = np.degrees(np.asarray(coords.horizontal.longitudes))
    target_pa = PRESSURE_LEVELS_HPA * 100.0

    ny, nx = lats_deg.size, lons_deg.size
    n_out = trajectory.vorticity.shape[0]
    Lp = PRESSURE_LEVELS_HPA.size

    print(f'[DINOSAUR] converting {n_out} snapshots to pressure levels '
          f'(ny={ny}, nx={nx}, levels={Lp})')

    u_out   = np.empty((n_out, Lp, ny, nx), dtype=np.float32)
    v_out   = np.empty((n_out, Lp, ny, nx), dtype=np.float32)
    T_out   = np.empty((n_out, Lp, ny, nx), dtype=np.float32)
    q_out   = np.zeros((n_out, Lp, ny, nx), dtype=np.float32)
    z_out   = np.empty((n_out, Lp, ny, nx), dtype=np.float32)
    sp_out  = np.empty((n_out, ny, nx),     dtype=np.float32)
    skt_out = np.empty((n_out, ny, nx),     dtype=np.float32)

    for t in range(n_out):
        snap_state = jax.tree_util.tree_map(lambda x: x[t], trajectory)
        u_s, v_s, T_s, ps_s = to_nodal_si(
            snap_state, coords, physics_specs, ref_temperature,
        )
        # Compute geopotential on the sigma grid first, then interpolate
        # to pressure levels. This preserves the hydrostatic balance between
        # T and z that dinosaur enforced at 24-layer resolution, instead of
        # reconstructing geopotential from only 13 down-sampled levels.
        phi_s = compute_geopotential_sigma(T_s, sigma_values)
        u_pl = sigma_to_pressure(u_s,   sigma_values, ps_s, target_pa)
        v_pl = sigma_to_pressure(v_s,   sigma_values, ps_s, target_pa)
        T_pl = sigma_to_pressure(T_s,   sigma_values, ps_s, target_pa)
        z_pl = sigma_to_pressure(phi_s, sigma_values, ps_s, target_pa)
        u_out[t]   = u_pl.astype(np.float32)
        v_out[t]   = v_pl.astype(np.float32)
        T_out[t]   = T_pl.astype(np.float32)
        z_out[t]   = z_pl.astype(np.float32)
        sp_out[t]  = ps_s.astype(np.float32)
        skt_out[t] = T_s[-1].astype(np.float32)
        if t % 24 == 0:
            print(f'  snapshot {t}/{n_out}')

    # ---------------------------------------------------------------------
    # Compute omega from horizontal divergence of (u, v) by integrating
    # d(omega)/dp = -div_h downward from p_top (50 hPa). This replaces the
    # zero w field that previous versions wrote, which forced any LAM driven
    # by this netCDF to start with w=0 despite strong divergent flow in the
    # baroclinic wave.
    # ---------------------------------------------------------------------
    print('[DINOSAUR] diagnosing hydrostatic omega from u, v divergence...')
    w_out = compute_hydrostatic_omega(
        u_out, v_out, target_pa, lats_deg, lons_deg,
    )
    print(f'[DINOSAUR] omega range: '
          f'[{float(w_out.min()):.3e}, {float(w_out.max()):.3e}] Pa/s')

    times_hours = np.arange(n_out, dtype=np.float32) * save_every_h

    ds_out = xr.Dataset(
        data_vars={
            'u':   (('time', 'level', 'latitude', 'longitude'), u_out,
                    {'units': 'm s-1',    'long_name': 'zonal wind'}),
            'v':   (('time', 'level', 'latitude', 'longitude'), v_out,
                    {'units': 'm s-1',    'long_name': 'meridional wind'}),
            't':   (('time', 'level', 'latitude', 'longitude'), T_out,
                    {'units': 'K',        'long_name': 'air temperature'}),
            'q':   (('time', 'level', 'latitude', 'longitude'), q_out,
                    {'units': 'kg kg-1',  'long_name': 'specific humidity'}),
            'z':   (('time', 'level', 'latitude', 'longitude'), z_out,
                    {'units': 'm**2 s-2', 'long_name': 'geopotential'}),
            'w':   (('time', 'level', 'latitude', 'longitude'), w_out,
                    {'units': 'Pa s-1',   'long_name': 'omega (vertical velocity)'}),
            'sp':  (('time', 'latitude', 'longitude'), sp_out,
                    {'units': 'Pa',       'long_name': 'surface pressure'}),
            'skt': (('time', 'latitude', 'longitude'), skt_out,
                    {'units': 'K',        'long_name': 'skin temperature'}),
            'lsm': (('latitude', 'longitude'),
                    np.zeros((ny, nx), dtype=np.float32),
                    {'units': '1',        'long_name': 'land-sea mask (all ocean)'}),
        },
        coords={
            'time':      ('time', times_hours, {'units': 'hour'}),
            'level':     ('level', PRESSURE_LEVELS_HPA.astype(np.float32),
                          {'units': 'hPa'}),
            'latitude':  ('latitude', lats_deg.astype(np.float32),
                          {'units': 'degrees_north'}),
            'longitude': ('longitude', lons_deg.astype(np.float32),
                          {'units': 'degrees_east'}),
        },
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ds_out.to_netcdf(output_path)
    print(f'[DINOSAUR] wrote {output_path}')


# ===========================================================================
#                          STAGE 2:  SUETES REGIONAL
# ===========================================================================

class SimulationBlowupError(RuntimeError):
    """Raised by the debug callback when a NaN/Inf or runaway value is detected."""
    pass


def run_suetes(driver_nc):
    # ==========================================
    # 1. CONFIG
    # ==========================================
    lat_c, lon_c = 45.0, 195.0
    RUN_NAME = "jw_pm_test_diffusive"
    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 160, 120, 30
    dx, dy, dz = 50000.0, 50000.0, 500.0
    sponge_depth = 10

    dt = 100.0
    start_hour = 215
    sim_hours = 24   # stop just before the known blowup window (~step 908)
    chunk_steps = 60

    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }

    USE_MOISTURE = False

    DRIVER_NC = driver_nc

    print(f"[CONFIG] Perfect-model J-W test, sub-domain center "
          f"({lat_c}N, {lon_c}E)")
    print(f"[CONFIG] Davies relaxation: Newtonian blend toward BC "
          f"(diffusive term not used; DaviesSponge.blend only)")

    y_half_km = (ny / 2.0) * (dy / 1000.0)
    x_half_km = (nx / 2.0) * (dx / 1000.0)
    R_earth_km = 6371.229
    deg_north = np.degrees(2.0 * np.arctan(y_half_km / (2.0 * R_earth_km)))
    print(f"[GEOM] Half-widths: x={x_half_km:.0f} km, y={y_half_km:.0f} km")
    print(f"[GEOM] North edge ~{lat_c + deg_north:.1f} N, "
          f"south edge ~{lat_c - deg_north:.1f} N")
    print(f"[GEOM] Model top at z = {nz * dz / 1000.0:.1f} km")

    # ==========================================
    # 2. DRIVER (dinosaur reference)
    # ==========================================
    print(f"[DRIVER] Loading dinosaur reference from {DRIVER_NC}")
    era5_proc = DinosaurProcessor(DRIVER_NC)

    # ==========================================
    # 3. GEOMETRY (flat earth)
    # ==========================================
    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} mesh "
          f"(dx={dx/1000:.0f} km), FLAT earth")
    flat_h_func = lambda x, y: jnp.zeros_like(x)
    sleve_transform = SleveSimple(scale_s=10000.0, n=1.0)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_c, lon_c,
        h_func=flat_h_func, transform=sleve_transform,
    )

    # ==========================================
    # 4. BOUNDARY
    # ==========================================
    n_available = era5_proc.ds.sizes['time']
    if start_hour + sim_hours + 1 > n_available:
        sim_hours = n_available - 1 - start_hour
        print(f"[BOUNDARY] Driver has {n_available} hourly states; "
              f"truncating sim_hours to {sim_hours} given start_hour={start_hour}")
    num_states = sim_hours + 1
    print(f"[BOUNDARY] Building {num_states} hourly boundary states "
          f"covering dinosaur T={start_hour}-{start_hour + sim_hours}h")

    raw_t0 = era5_proc.get_stitched_state(time_idx=start_hour)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'],
                               constants)

    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    z0_ocean, z0_land = 1e-4, 0.1
    epsilon_ocean, epsilon_land = 0.3, 0.0
    z_0_field = land_fraction * z0_land + (1.0 - land_fraction) * z0_ocean
    epsilon_field = (
        land_fraction * epsilon_land
        + (1.0 - land_fraction) * epsilon_ocean
    )

    suetes_bc_states = []
    times_sec = []
    for i in range(num_states):
        if i % 24 == 0:
            print(f"  -> Regridding dinosaur state for "
                  f"sim_T={i}h (dinosaur T={start_hour + i}h)")
        raw_state = era5_proc.get_stitched_state(time_idx=start_hour + i)
        bc_state_jax = bridge.process(raw_state)
        bc_state = jax.tree_util.tree_map(np.asarray, bc_state_jax)
        suetes_bc_states.append(bc_state)
        times_sec.append(float(i * 3600.0))
        del bc_state_jax

    time_manager = TimeManager(suetes_bc_states, times_sec, grid)
    initial_state = {k: jnp.asarray(v) for k, v in suetes_bc_states[0].items()}

    if USE_MOISTURE:
        initial_state['q_c'] = jnp.zeros_like(initial_state['q'])
    else:
        initial_state.pop('q', None)
        initial_state.pop('q_c', None)

    initial_state['theta_surf'] = (
        land_fraction * initial_state['theta_skt']
        + (1.0 - land_fraction) * initial_state['th_v'][:, :, 0]
    )
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state.pop('theta_skt', None)

    # ==========================================
    # 5. PHYSICS / STEPPER / SPONGE
    # ==========================================
    print(f"[DYNAMICS] Initializing core")
    operators = CGridOperator3D(grid)
    physics_suite = PhysicsSuite()

    sgs_scheme = SmagorinskyLillySGS(
        grid, operators, constants,
        Cs=0.20, Pr_t=1.0, critical_Ri=0.25,
    )
    #physics_suite.add_tendency_scheme(sgs_scheme)

    pbl_scheme = McFarlaneSurfaceDrag(
        grid, operators, constants,
        z_0=z_0_field, epsilon=epsilon_field,
        theta_surf=initial_state['theta_surf'],
    )
    #physics_suite.add_tendency_scheme(pbl_scheme)

    vert_diff_scheme = McFarlaneVerticalDiffusion(
        grid, operators, constants,
        epsilon=epsilon_field[..., None],
    )
    #physics_suite.add_tendency_scheme(vert_diff_scheme)

    nudging_scheme = NewtonianRelaxation(tau_relax_hours=6.0)
    #physics_suite.add_tendency_scheme(nudging_scheme)

    physics = Euler3D(
        grid, operators, constants, dt=dt,
        initial_era5_state=initial_state,
        damp_height=10000.0, max_damp=3.0,
        nu_div_factor=0.1, nu_h_factor=0.1,
        physics_suite=physics_suite,
    )
    stepper = SISLStepper3D(physics, dt)
    sponge = DaviesSponge(
        grid, operators, sponge_depth=sponge_depth, dt=dt,
        tau_bndy_factor=10.0,
    )

    # ==========================================
    # 6. PREFLIGHT DIAGNOSTICS
    # ==========================================
    print("-" * 60)

    def _state_stats(name, state):
        print(f"[PREFLIGHT] {name}:")
        for key in sorted(state.keys()):
            val = state[key]
            if hasattr(val, "shape") and getattr(val, "size", 0) > 1:
                v = np.asarray(val)
                nan_count = int(np.isnan(v).sum())
                inf_count = int(np.isinf(v).sum())
                flag = ""
                if nan_count > 0:
                    flag = f"  !!! {nan_count} NaN"
                elif inf_count > 0:
                    flag = f"  !!! {inf_count} Inf"
                print(f"  {key:18s} shape={str(v.shape):16s} "
                      f"min={float(np.nanmin(v)):12.4g}  "
                      f"max={float(np.nanmax(v)):12.4g}  "
                      f"mean={float(np.nanmean(v)):12.4g}{flag}")

    _state_stats("Initial state", initial_state)

    print("[PREFLIGHT] Running one diagnostic step (no JIT)...")
    bc0 = {k: jnp.asarray(v) for k, v in suetes_bc_states[0].items()}
    def _preflight_bc_fn(state_next, _):
        return sponge.blend(state_next, bc0)
    trial_state = stepper.step(initial_state, 0.0, forcing=None,
                               bc_fn=_preflight_bc_fn)
    _state_stats("State after 1 step", trial_state)
    if any(bool(jnp.any(jnp.isnan(jnp.asarray(v))))
           for v in trial_state.values()
           if hasattr(v, "shape") and getattr(v, "size", 0) > 1):
        print("[PREFLIGHT] !!! NaN appeared after one step. ABORTING.")
        return
    del trial_state, bc0
    print("-" * 60)

    # ==========================================
    # 7. INTEGRATION
    # ==========================================
    DBG_EVERY = 10
    RUNAWAY_THRESHOLD = 1.0e6
    _u_shape = initial_state['u'].shape

    def _dbg_print(step, t, mu, mv, mw, mth, mpi,
                   u_argmax_flat, u_at_max, bc_u_at_max):
        maxes = {
            '|u|': float(mu),  '|v|':  float(mv),
            '|w|': float(mw),  '|th|': float(mth),
            '|pi|': float(mpi),
        }
        bad = [k for k, val in maxes.items()
               if (not np.isfinite(val)) or abs(val) > RUNAWAY_THRESHOLD]
        if bad:
            i, j, k = np.unravel_index(int(u_argmax_flat), _u_shape)
            print(f"\n[KILL] step={int(step)} t={float(t)/60.0:.1f}min: "
                  f"non-finite or runaway values in {bad}")
            print(f"[KILL] all maxes: " +
                  ", ".join(f"{kk}={vv:.4g}" for kk, vv in maxes.items()))
            print(f"[KILL] u argmax at (i={i}, j={j}, k={k})")
            raise SimulationBlowupError(
                f"Blowup at step {int(step)} "
                f"(t={float(t)/60.0:.1f} min): {bad}"
            )

        if int(step) % DBG_EVERY != 0:
            return
        i, j, k = np.unravel_index(int(u_argmax_flat), _u_shape)
        in_x = (i < sponge_depth) or (i >= _u_shape[0] - sponge_depth)
        in_y = (j < sponge_depth) or (j >= _u_shape[1] - sponge_depth)
        if   in_x and in_y: where = "CORNER  "
        elif in_x:          where = "X-SPNG  "
        elif in_y:          where = "Y-SPNG  "
        else:               where = "INTERIOR"
        print(f"    [DBG step={int(step):5d} t={float(t)/60.0:7.1f}min] "
              f"|u|={float(mu):7.2f}@(i={i:3d},j={j:3d},k={k:3d})/{where} "
              f"u={float(u_at_max):+7.2f} bc_u={float(bc_u_at_max):+7.2f} | "
              f"|v|={float(mv):6.2f} |w|={float(mw):.4f} "
              f"|th|={float(mth):6.1f} |pi|={float(mpi):.6f}")

    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        bc_state_t = time_manager.get_forcing(t_curr)
        curr_state['theta_surf'] = (
            land_fraction * bc_state_t['theta_skt']
            + (1.0 - land_fraction) * bc_state_t['th_v'][:, :, 0]
        )
        curr_state['target_th_v'] = bc_state_t['th_v']

        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)

        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        next_state['theta_surf'] = curr_state['theta_surf']
        next_state['target_th_v'] = curr_state['target_th_v']

        abs_u = jnp.abs(next_state['u'])
        max_u = jnp.max(abs_u)
        u_argmax = jnp.argmax(abs_u)
        u_at_max = next_state['u'].ravel()[u_argmax]
        bc_u_at_max = bc_state_t['u'].ravel()[u_argmax]

        max_v  = jnp.max(jnp.abs(next_state['v']))
        max_w  = jnp.max(jnp.abs(next_state['w']))
        max_th = jnp.max(jnp.abs(next_state['th_v']))
        max_pi = jnp.max(jnp.abs(next_state['pi']))

        jax.debug.callback(_dbg_print, step_idx, t_curr,
                           max_u, max_v, max_w, max_th, max_pi,
                           u_argmax, u_at_max, bc_u_at_max)
        return next_state, max_w

    sim = Simulation(step_fn=step_fn, dt=dt)
    t0 = time.time()
    try:
        final_state = sim.run(
            initial_state, t_start=0.0, t_end=sim_hours * 3600.0,
            chunk_steps=chunk_steps,
        )
    except SimulationBlowupError as e:
        print(f"[SIMULATION] aborted after {time.time() - t0:.1f}s: {e}")
        return
    print(f"[SIMULATION] done in {time.time() - t0:.1f}s")
    _state_stats("Final state", final_state)

    # ==========================================
    # 8. DIAGNOSTICS
    # ==========================================
    print("-" * 60)
    print("[PLOT] Producing dashboards, spectrum, comparison")
    visualizer = Visualizer()
    final_bc = {k: jnp.asarray(v) for k, v in suetes_bc_states[-1].items()}

    # Model dashboards at multiple levels. Level 0 is the surface; levels 5,
    # 15, 25 are progressively higher in the column (k=5 ~ 2.5 km, k=15 ~
    # 7.5 km, k=25 ~ 12.5 km).
    for z in [0, 5, 15, 25]:
        visualizer.plot_dashboard(
            grid, final_state, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_dash_z{z}_{sim_hours}h.png"
            ),
        )

    # BC dashboards at the same levels for direct comparison. Level 0 will
    # show zero vertical velocity by construction (BoundaryProcessor zeros
    # the kinematic bottom over flat terrain). Levels 5, 15, 25 should show
    # the new omega-diagnosed w field.
    for z in [0, 5, 15, 25]:
        visualizer.plot_dashboard(
            grid, final_bc, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_BC_dash_z{z}_{sim_hours}h.png"
            ),
        )

    visualizer.plot_energy_spectrum(
        grid, final_state, 'w', z_idx=5, sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_energy_{sim_hours}h.png"
        ),
    )
    visualizer.plot_comparison(
        grid, final_state, final_bc, 'th_v', z_idx=0,
        sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_compare_th_v_{sim_hours}h.png"
        ),
    )


# ===========================================================================
#                            ORCHESTRATION
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the dinosaur J-W global driver, then the Suetes "
                    "regional perfect-model test that consumes its netCDF.",
    )
    parser.add_argument('--output', default='suetes/data/test.nc',
                        help='Driver netCDF path. Reused as a cache if it '
                             'already exists; otherwise written by the dinosaur '
                             'stage. Read by the Suetes stage either way.')
    parser.add_argument('--resolution', default='T42',
                        help='Spherical-harmonic truncation (T42, T85, T170, ...).')
    parser.add_argument('--layers', type=int, default=24,
                        help='Number of equidistant sigma layers.')
    parser.add_argument('--sim_hours', type=int, default=240,
                        help='Dinosaur run duration [hours]. 240 = 10 days. '
                             'Must cover the Suetes start_hour (215) + window.')
    parser.add_argument('--save_every_h', type=float, default=1.0,
                        help='Driver output sampling cadence [hours].')
    parser.add_argument('--dt_seconds', type=float, default=200.0,
                        help='Dinosaur dynamics timestep [s]. 200 is conservative at T42.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run the dinosaur stage and overwrite the '
                             'driver netCDF even if a cached one already exists.')
    args = parser.parse_args()

    cache_exists = os.path.exists(args.output)
    if cache_exists and not args.force:
        print(f"[PIPELINE] Found cached driver at {args.output}, "
              f"skipping dinosaur stage (pass --force to regenerate)")
    else:
        if cache_exists:
            print(f"[PIPELINE] --force set, regenerating driver at {args.output}")
        # The dinosaur stage needs float64; the Suetes stage must run in float32.
        jax.config.update('jax_enable_x64', True)
        run_dinosaur(
            args.output, args.resolution, args.layers,
            args.sim_hours, args.save_every_h, args.dt_seconds,
        )
        jax.config.update('jax_enable_x64', False)

    run_suetes(args.output)


if __name__ == '__main__':
    main()
