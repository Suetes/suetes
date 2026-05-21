"""
Test for something like CORDEX NAM-22 downscaling.

Drives the CORDEX North America domain at 22 km horizontal resolution
with ERA5 boundary conditions, covering the February 18-20 2025 Arctic
outbreak.

ERA5 is box-averaged to ~84 km effective resolution before driving
Suetes, giving a 3.8:1 downscaling ratio rather than the
near-identity 28 km to 22 km of raw ERA5.

Comparison plots use native ERA5 (not the coarsened driver) so that
the downscaling skill is visible: Suetes should recover structure that
is in native ERA5 but absent in the coarsened driver.

Physics is masked off in the Davies sponge zone via the `interior_mask`
mechanism: drag, vertical diffusion, and Newtonian nudging act only on
interior cells, leaving the sponge to enforce LBC consistency without
fighting the physics tendencies.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from scipy.ndimage import uniform_filter

from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager

from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.physics import (
    PhysicsSuite,
    McFarlaneVerticalDiffusion, McFarlaneSurfaceDrag,
    NewtonianRelaxation,
)

from suetes.vis.visualizer import Visualizer


DATA_DIR = "data"


def coarsen_state(state, window=3):
    """Box-average a raw ERA5 stitched state in the horizontal (lat, lon) plane."""
    new_state = {}
    for key, val in state.items():
        if key in ('latitude', 'longitude'):
            new_state[key] = val
            continue
        arr = np.asarray(val)
        if arr.ndim == 2:
            arr = uniform_filter(arr, size=window, mode='nearest')
        elif arr.ndim == 3:
            arr = uniform_filter(arr, size=(1, window, window), mode='nearest')
        new_state[key] = jnp.asarray(arr)
    return new_state


def build_interior_mask(sponge):
    """Build interior masks for u, v, w, th_v from a DaviesSponge."""
    mask = {}
    for loc in ('u', 'v', 'm'):
        masks = sponge.masks[loc]
        combined = jnp.maximum(
            jnp.maximum(masks['west'], masks['east']),
            jnp.maximum(masks['south'], masks['north']),
        )
        interior = 1.0 - combined
        if loc == 'u':
            mask['u'] = interior
        elif loc == 'v':
            mask['v'] = interior
        else:
            mask['th_v'] = interior
            mask['w'] = interior
    return mask


def _bc_cache_path(cache_dir, prefix, num_states, coarsen_window, sponge_depth,
                   smooth_sigma, kind):
    tag = (f"{prefix}_n{num_states}_cw{coarsen_window}"
           f"_sd{sponge_depth}_ss{smooth_sigma:.1f}_{kind}")
    return os.path.join(cache_dir, f"{tag}_bc_states.pkl")


def save_bc_cache(path, suetes_bc_states):
    payload = [
        {k: np.asarray(v) for k, v in state.items()}
        for state in suetes_bc_states
    ]
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bc_cache(path):
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return [{k: jnp.asarray(v) for k, v in state.items()} for state in payload]


def main():
    ACTIVE_DOMAIN = "nam22"
    RUN_NAME = f"{ACTIVE_DOMAIN}_NAM22_july2025"

    lat_c, lon_c = 47.5, -97.0

    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 310, 260, 32
    dx, dy, dz = 22000.0, 22000.0, 500.0
    sponge_depth = 15
    smooth_sigma = 2.0
    coarsen_window = 3

    dt = 120.0
    sim_hours = 42

    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = int(sim_hours) + 1

    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }
    USE_MOISTURE = False

    YEAR = "2025"
    MONTH = "07"
    DAYS = [str(d).zfill(2) for d in range(18, 21)]

    dynamic_bbox = ERA5Manager.calculate_required_bbox(
        lat_c, lon_c, nx, ny, dx, dy, buffer_deg=3.0,
    )

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()} (CORDEX NAM-22)")
    print(f"[CONFIG] Center: ({lat_c}N, {lon_c}E)")
    print(f"[CONFIG] Grid: {nx}x{ny}x{nz} at dx={dx/1000:.0f} km, "
          f"dz={dz:.0f} m -> {nx*dx/1000:.0f} x {ny*dy/1000:.0f} km")
    print(f"[CONFIG] Period: {YEAR}-{MONTH}-{DAYS[0]} to "
          f"{YEAR}-{MONTH}-{DAYS[-1]} ({sim_hours} h)")

    print(f"[DATA] Validating ERA5 forcing files...")
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')

    cache_prefix = f"{ACTIVE_DOMAIN}_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year=YEAR, month=MONTH, days=DAYS,
        area=dynamic_bbox, prefix=cache_prefix,
    )

    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} terrain-following mesh "
          f"(dx={dx/1000} km)")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)

    topo_proc = TopographyProcessor(
        era5_sl_path=sl_file,
        gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"),
    )
    h_func = topo_proc.process_and_blend(
        base_grid, sponge_depth=sponge_depth, smooth_sigma=smooth_sigma,
    )
    sleve_transform = SleveSimple(scale_s=10000.0, n=1.0)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_c, lon_c,
        h_func=h_func, transform=sleve_transform,
    )

    bc_cache_path_coarse = _bc_cache_path(
        DATA_DIR, cache_prefix, num_era5_states, coarsen_window,
        sponge_depth, smooth_sigma, kind='coarse',
    )
    bc_cache_path_native = _bc_cache_path(
        DATA_DIR, cache_prefix, num_era5_states, coarsen_window,
        sponge_depth, smooth_sigma, kind='native',
    )

    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)

    bridge = BoundaryProcessor(
        grid, raw_t0['latitude'], raw_t0['longitude'], constants,
    )
    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    if os.path.exists(bc_cache_path_coarse):
        print(f"[BOUNDARY] Loading cached coarsened BC states from "
              f"{bc_cache_path_coarse}")
        suetes_bc_states = load_bc_cache(bc_cache_path_coarse)
    else:
        print(f"[BOUNDARY] Building coarsened-driver BC states "
              f"({sim_hours} h)...")
        suetes_bc_states = []
        for i in range(num_era5_states):
            if i % 6 == 0:
                print(f"[BOUNDARY] -> Regridding coarsened ERA5 for T={i}h")
            raw_state = era5_proc.get_stitched_state(time_idx=i)
            raw_state = coarsen_state(raw_state, window=coarsen_window)
            bc_state = bridge.process(raw_state)
            suetes_bc_states.append(bc_state)

        print(f"[BOUNDARY] Caching coarsened BC states to "
              f"{bc_cache_path_coarse}")
        save_bc_cache(bc_cache_path_coarse, suetes_bc_states)

    if os.path.exists(bc_cache_path_native):
        print(f"[BOUNDARY] Loading cached native BC states from "
              f"{bc_cache_path_native}")
        suetes_bc_states_native = load_bc_cache(bc_cache_path_native)
    else:
        print(f"[BOUNDARY] Building native-resolution reference BC states "
              f"({sim_hours} h)...")
        suetes_bc_states_native = []
        for i in range(num_era5_states):
            if i % 6 == 0:
                print(f"[BOUNDARY] -> Regridding native ERA5 for T={i}h")
            raw_state = era5_proc.get_stitched_state(time_idx=i)
            bc_state = bridge.process(raw_state)
            suetes_bc_states_native.append(bc_state)

        print(f"[BOUNDARY] Caching native BC states to "
              f"{bc_cache_path_native}")
        save_bc_cache(bc_cache_path_native, suetes_bc_states_native)

    times_sec = [float(i * 3600.0) for i in range(num_era5_states)]
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)

    initial_state = dict(suetes_bc_states[0])

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

    z0_ocean, z0_land = 1e-4, 0.1
    epsilon_ocean, epsilon_land = 0.3, 0.0
    z_0_field = land_fraction * z0_land + (1.0 - land_fraction) * z0_ocean
    epsilon_field = (
        land_fraction * epsilon_land
        + (1.0 - land_fraction) * epsilon_ocean
    )

    print(f"[DYNAMICS] Initializing dynamical core")
    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(
        grid, operators, sponge_depth=sponge_depth, dt=dt,
        tau_bndy_factor=10.0,
    )
    interior_mask = build_interior_mask(sponge)
    print(f"[DYNAMICS] Interior mask built: "
          f"physics scaled to zero in {sponge_depth}-cell sponge zone")

    physics_suite = PhysicsSuite()

    pbl_scheme = McFarlaneSurfaceDrag(
        grid, operators, constants,
        z_0=z_0_field, epsilon=epsilon_field,
        theta_surf=initial_state['theta_surf'],
    )
    vert_diff_scheme = McFarlaneVerticalDiffusion(
        grid, operators, constants,
        epsilon=epsilon_field[..., None],
    )
    nudging_scheme = NewtonianRelaxation(tau_relax_hours=6.0)

    physics_suite.add_tendency_scheme(pbl_scheme)
    physics_suite.add_tendency_scheme(vert_diff_scheme)
    physics_suite.add_tendency_scheme(nudging_scheme)

    physics = Euler3D(
        grid, operators, constants, dt=dt,
        initial_era5_state=initial_state,
        damp_height=9000.0, max_damp=3.0,
        nu_div_factor=0.1, nu_h_factor=0.1,
        physics_suite=physics_suite,
        interior_mask=interior_mask,
    )
    stepper = SISLStepper3D(physics, dt)

    print("-" * 60)

    chunk_steps = int(3600.0 / dt)

    hov_times = [0.0]
    initial_anom = initial_state['th_v'][:, :, 0] - initial_state['th_v'][:, :, 0]
    hov_data = [np.array(jnp.mean(initial_anom, axis=1))]

    def save_hovmoller_data(hour, anom_array):
        hov_times.append(float(hour))
        hov_data.append(np.array(anom_array))

    _SAVE_KEYS = ('u', 'v', 'w', 'th_v', 'pi', 'rho', 'eta_dot')
    snapshots = [
        {k: np.asarray(v) for k, v in initial_state.items() if k in _SAVE_KEYS}
    ]

    def save_state_snapshot(state_sub):
        snapshots.append({k: np.asarray(v) for k, v in state_sub.items()})

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

        max_w = jnp.max(jnp.abs(next_state['w']))

        anom = next_state['th_v'][:, :, 0] - bc_state_t['th_v'][:, :, 0]
        y_avg_anom = jnp.mean(anom, axis=1)
        current_hour = (step_idx + 1) * dt / 3600.0

        is_hourly = ((step_idx + 1) % int(3600.0 / dt)) == 0

        jax.lax.cond(
            is_hourly,
            lambda: jax.debug.callback(save_hovmoller_data,
                                       current_hour, y_avg_anom),
            lambda: None,
        )

        state_to_save = {k: next_state[k] for k in _SAVE_KEYS if k in next_state}
        jax.lax.cond(
            is_hourly,
            lambda: jax.debug.callback(save_state_snapshot, state_to_save,
                                       ordered=True),
            lambda: None,
        )

        return next_state, max_w

    sim = Simulation(step_fn=step_fn, dt=dt)

    start_time = time.time()
    final_state = sim.run(
        initial_state,
        t_start=0.0, t_end=sim_time_seconds,
        chunk_steps=chunk_steps,
    )
    print(f"[SIMULATION] Done in {time.time() - start_time:.1f}s")

    print("-" * 60)
    print("[PLOT] Generating diagnostic plots...")
    visualizer = Visualizer()

    final_era5_state = suetes_bc_states_native[-1]
    
    for z in [0, 5, 15, 30]:
        visualizer.plot_dashboard(
            grid, final_state, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_dash_z{z}_{sim_hours}h.png"
            ),
        )

    for z in [500.0, 3000.0, 5000.0, 10000.0]:
        visualizer.plot_dashboard(
            grid, final_state, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_dash_z{int(z)}m_{sim_hours}h.png"
            ),
        )

    visualizer.plot_energy_spectrum(
        grid, final_state, 'w', z_idx=5, sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_energy_{sim_hours}h.png"
        ),
    )

    visualizer.plot_comparison(
        grid, final_state, final_era5_state, 'th_v',
        z_idx=0, sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_compare_th_v_surf_{sim_hours}h.png"
        ),
    )

    visualizer.plot_level_strip(
        grid, final_state, 'th_v', z_indices=[0, 5, 15, 30],
        sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_levels_th_v_{sim_hours}h.png"
        ),
    )

    visualizer.plot_hovmoller(
        grid, hov_times, np.array(hov_data),
        variable='Surface th_v Anomaly [K]',
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_hovmoller_th_v_{sim_hours}h.png"
        ),
    )

    visualizer.plot_slice_locator_dashboard(
        grid, final_state, map_var='th_v', slice_var='w', map_z=5,
        sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_slices_w_{sim_hours}h.png"
        ),
    )

    # Per-point near-surface temperature time series.
    POINTS = [
        ("Denver",      39.74, -104.99),
        ("Kansas City", 39.10,  -94.58),
        ("Minneapolis", 44.98,  -93.27),
        ("Winnipeg",    49.90,  -97.14),
        ("Atlanta",     33.75,  -84.39),
    ]

    Xi_m, Yi_m = jnp.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
    times_hours_pt = list(range(sim_hours + 1))

    print("-" * 60)
    print("[DEBUG] Surface th_v and pi at each sample point (t=0 hour):")
    print(f"{'Location':<14}{'i':>5}{'j':>5}{'Z_m[0] [m]':>13}"
          f"{'th_v [K]':>11}{'pi':>9}{'T [K]':>9}{'T [C]':>9}")
    for name, p_lat, p_lon in POINTS:
        dist2 = (lat_m - p_lat) ** 2 + (lon_m - p_lon) ** 2
        i_p, j_p = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))
        z0 = float(grid.Z_m[i_p, j_p, 0])
        th_v = float(snapshots[0]['th_v'][i_p, j_p, 0])
        pi = float(snapshots[0]['pi'][i_p, j_p, 0])
        T = th_v * pi
        print(f"{name:<14}{i_p:>5d}{j_p:>5d}{z0:>13.1f}"
              f"{th_v:>11.2f}{pi:>9.4f}{T:>9.2f}{T-273.15:>9.2f}")
    print("-" * 60)

    for name, p_lat, p_lon in POINTS:
        dist2 = (lat_m - p_lat) ** 2 + (lon_m - p_lon) ** 2
        i_p, j_p = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))

        suetes_T_C = [
            float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15
            for s in snapshots
        ]
        era5_T_C = [
            float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15
            for s in suetes_bc_states_native
        ]

        ew = 'W' if p_lon < 0 else 'E'
        ns = 'N' if p_lat > 0 else 'S'
        location_str = (f"{name} ({abs(p_lat):.2f} deg {ns}, "
                        f"{abs(p_lon):.2f} deg {ew})")
        safe_name = name.lower().replace(' ', '_')

        visualizer.plot_point_timeseries(
            times_hours_pt, suetes_T_C, era5_T_C,
            location_name=location_str,
            units='deg C',
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_T_{safe_name}_{sim_hours}h.png"
            ),
        )


if __name__ == "__main__":
    main()
