"""
Test for something like CORDEX NAM-22 downscaling.

Drives the CORDEX North America domain at 22 km horizontal resolution
with ERA5 boundary conditions, covering the February 18-20 2025 Arctic
outbreak.

ERA5 is box-averaged to ~84 km effective resolution before driving
Suetes, giving a 3.8:1 downscaling ratio rather than the
near-identity 28 km to 22 km of raw ERA5.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import jax
import jax.numpy as jnp
import numpy as np

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

from suetes.physics.base import PhysicsSuite
from suetes.physics.forcing import NewtonianRelaxation
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.turbulence import McFarlaneVerticalDiffusion

from suetes.vis.visualizer import Visualizer

DATA_DIR = "suetes/data"

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
    print(f"[CONFIG] Grid: {nx}x{ny}x{nz} at dx={dx/1000:.0f} km")

    print(f"[DATA] Validating ERA5 forcing files...")
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')
    cache_prefix = f"{ACTIVE_DOMAIN}_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year=YEAR, month=MONTH, days=DAYS,
        area=dynamic_bbox, prefix=cache_prefix,
    )

    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} terrain-following mesh")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)

    topo_proc = TopographyProcessor(
        era5_sl_path=sl_file,
        gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"),
    )
    h_func = topo_proc.process_and_blend(
        base_grid, sponge_depth=sponge_depth, smooth_sigma=smooth_sigma,
    )
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_c, lon_c,
        h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0),
    )

    # ---------------------------------------------------------
    # 1. SETUP PROCESSORS & STATIC FIELDS
    # ---------------------------------------------------------
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    # ---------------------------------------------------------
    # 2. AUTOMATED BOUNDARY CACHING & LOAD
    # ---------------------------------------------------------
    bc_cache_path_coarse = os.path.join(DATA_DIR, f"{cache_prefix}_cw{coarsen_window}_coarse.pkl")
    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")

    print(f"[BOUNDARY] Preparing coarsened driver BC states...")
    suetes_bc_states = bridge.build_or_load_timeseries(
        era5_proc=era5_proc,
        num_states=num_era5_states,
        cache_path=bc_cache_path_coarse,
        coarsen_window=coarsen_window
    )

    print(f"[BOUNDARY] Preparing native reference BC states...")
    suetes_bc_states_native = bridge.build_or_load_timeseries(
        era5_proc=era5_proc,
        num_states=num_era5_states,
        cache_path=bc_cache_path_native,
        coarsen_window=None
    )

    times_sec = [float(i * 3600.0) for i in range(num_era5_states)]
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)

    # ---------------------------------------------------------
    # 3. INITIAL STATE SETUP
    # ---------------------------------------------------------
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
    epsilon_field = land_fraction * epsilon_land + (1.0 - land_fraction) * epsilon_ocean

    # ---------------------------------------------------------
    # 4. DYNAMICS & SPONGE
    # ---------------------------------------------------------
    print(f"[DYNAMICS] Initializing dynamical core")
    operators = CGridOperator3D(grid)
    
    sponge = DaviesSponge(
        grid, operators, sponge_depth=sponge_depth, dt=dt,
        tau_bndy_factor=10.0,
    )
    interior_mask = sponge.get_interior_mask()
    print(f"[DYNAMICS] Interior mask built: physics masked in {sponge_depth}-cell sponge")

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
