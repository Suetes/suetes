"""
Run dinosaur's Jablonowski-Williamson baroclinic instability test case and
save the output as a netCDF in the format expected by ``DinosaurProcessor``
(see ``suetes/preprocessing/dinosaur2suetes.py`` for the schema).

Make sure you use DFI!!
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

jax.config.update('jax_enable_x64', True)

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

units = scales.units


# Standard pressure levels (hPa) for output. Matches a common ERA5 selection.
# 50 hPa is well below the dinosaur top with 24 equidistant sigma layers
# (top is ~2 hPa), so no extrapolation is needed.
PRESSURE_LEVELS_HPA = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dtype=np.float64,
)


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


# ---------------------------------------------------------------------------
#                                 MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='Output netCDF path.')
    parser.add_argument('--resolution', default='T42',
                        help='Spherical-harmonic truncation (T42, T85, T170, ...).')
    parser.add_argument('--layers', type=int, default=24,
                        help='Number of equidistant sigma layers.')
    parser.add_argument('--sim_hours', type=int, default=240,
                        help='Simulation duration [hours]. 240 = 10 days.')
    parser.add_argument('--save_every_h', type=float, default=1.0,
                        help='Output sampling cadence [hours].')
    parser.add_argument('--dt_seconds', type=float, default=200.0,
                        help='Dynamics timestep [s]. 200 is conservative at T42.')
    args = parser.parse_args()

    coords, physics_specs, eq, state, ref_temperature = setup(
        args.resolution, args.layers,
    )
    step_fn, dt_nondim = build_step_fn(eq, coords, args.dt_seconds, physics_specs)

    n_steps_total  = int(round(args.sim_hours * 3600.0 / args.dt_seconds))
    steps_per_save = int(round(args.save_every_h * 3600.0 / args.dt_seconds))
    # n_saves+1 frames so the output covers hours [0, 1, ..., sim_hours]
    # inclusive (e.g. --sim_hours 240 gives 241 hourly frames). With
    # start_with_input=True the saves happen at the START of each outer
    # iteration, so we capture t=sim_hours before the (discarded) final step.
    n_saves        = n_steps_total // steps_per_save + 1
    print(f'[DINOSAUR] {args.resolution}, {args.layers} layers, '
          f'dt={args.dt_seconds:.0f}s, {n_steps_total} steps, '
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

    times_hours = np.arange(n_out, dtype=np.float32) * args.save_every_h

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
    ds_out.to_netcdf(args.output)
    print(f'[DINOSAUR] wrote {args.output}')


if __name__ == '__main__':
    main()
