"""
Wreckhouse Windstorm Downscaling Experiment.

Drives a high-resolution (2 km) domain over southwestern Newfoundland
with ERA5 boundary conditions, covering the February 14-16 2025 
severe downslope wind event.
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
    ACTIVE_DOMAIN = "wreckhouse"
    RUN_NAME = f"{ACTIVE_DOMAIN}_2km_feb2025"

    # Centered over Wreckhouse, NL
    lat_c, lon_c = 47.71, -59.31
    output_dir = "suetes/plots/wreckhouse"
    os.makedirs(output_dir, exist_ok=True)

    # 400x400 km domain to capture upstream Gulf flow and terrain
    nx, ny, nz = 200, 200, 40
    dx, dy, dz = 2000.0, 2000.0, 350.0
    sponge_depth = 15
    smooth_sigma = 0.1  # Less smoothing to preserve steep Long Range Mountains
    
    # Reduced dt for high winds and steep terrain at 2km resolution
    dt = 20.0
    sim_hours = 13  # Feb 14 00z to Feb 16 00z
    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = int(sim_hours) + 1

    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }
    USE_MOISTURE = False

    YEAR = "2025"
    MONTH = "02"
    DAYS = [str(d).zfill(2) for d in range(14, 17)]

    dynamic_bbox = ERA5Manager.calculate_required_bbox(
        lat_c, lon_c, nx, ny, dx, dy, buffer_deg=2.0,
    )

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()}")
    print(f"[CONFIG] Center: ({lat_c}N, {lon_c}E)")
    print(f"[CONFIG] Grid: {nx}x{ny}x{nz} at dx={dx/1000:.1f} km")

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
    # No coarsening window used here to retain native ERA5 gradients
    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")

    print(f"[BOUNDARY] Preparing native reference BC states...")
    suetes_bc_states = bridge.build_or_load_timeseries(
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

    if not USE_MOISTURE:
        initial_state.pop('q', None)
        initial_state.pop('q_c', None)

    initial_state['theta_surf'] = initial_state['theta_skt']
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
        nu_div_factor=0.1, nu_h_factor=0.2,
        physics_suite=physics_suite, 
        interior_mask=interior_mask,
    )
    stepper = SISLStepper3D(physics, dt)

    print("-" * 60)

    chunk_steps = int(3600.0 / dt)

    _SAVE_KEYS = ('u', 'v', 'w', 'th_v', 'pi', 'rho', 'eta_dot')
    snapshots = [
        {k: np.asarray(v) for k, v in initial_state.items() if k in _SAVE_KEYS}
    ]

    def save_state_snapshot(state_sub):
        snapshots.append({k: np.asarray(v) for k, v in state_sub.items()})

    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        bc_state_t = time_manager.get_forcing(t_curr)

        curr_state['theta_surf'] = bc_state_t['theta_skt']
        curr_state['target_th_v'] = bc_state_t['th_v']

        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)

        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        next_state['theta_surf'] = curr_state['theta_surf']
        next_state['target_th_v'] = curr_state['target_th_v']

        max_w = jnp.max(jnp.abs(next_state['w']))
        is_hourly = ((step_idx + 1) % int(3600.0 / dt)) == 0

        state_to_save = {k: next_state[k] for k in _SAVE_KEYS if k in next_state}
        jax.lax.cond(
            is_hourly,
            lambda: jax.debug.callback(save_state_snapshot, state_to_save, ordered=True),
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
    final_era5_state = suetes_bc_states[-1]
    
    # 1. Targeted Slice Locator Dashboard
    # Wreckhouse is at the center (index 100). Cluster tightly (-8km, 0, +8km).
    y_targets = [96, 100, 104] 
    x_targets = [96, 100, 104] 
    
    # The grid coordinates (grid.x_m) are centered at 0.0. 
    # Grab a 100km window around the center point.
    x_center_km = 0.0
    zoom_x_km = [x_center_km - 50.0, x_center_km + 50.0]
    zoom_z_m = [0, 5000] # Focus on the lower 5km of the troposphere

    visualizer.plot_slice_locator_dashboard(
        grid, final_state, map_var='th_v', slice_var='w', map_z=2,
        sponge_depth=sponge_depth,
        x_indices=x_targets, y_indices=y_targets, 
        slice_xlim=zoom_x_km, slice_ylim=zoom_z_m,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_zoomed_slices_w_{sim_hours}h.png"),
    )

    # 2. Zoomed Horizontal Dashboard
    lon_min, lon_max = lon_c - 0.75, lon_c + 0.75
    lat_min, lat_max = lat_c - 0.75, lat_c + 0.75
    wreckhouse_extent = [lon_min, lon_max, lat_min, lat_max]

    target_hours = [6, 8, 10, 12]
    
    print("[PLOT] Generating temporal evolution dashboards...")
    for h in target_hours:
        # Ensure the hour exists in our snapshots list
        if h < len(snapshots):
            state_at_h = snapshots[h]
            
            visualizer.plot_dashboard(
                grid, state_at_h, z_idx=2, sponge_depth=sponge_depth,
                time_hours=h, extent=wreckhouse_extent,
                quiver_stride=6, 
                save_path=os.path.join(output_dir, f"{RUN_NAME}_zoomed_dash_{h}h.png"),
            )
            print(f"  -> Saved dashboard for T={h}h")

    # ---------------------------------------------------------
    # 5. TIME SERIES EXTRACTION FOR LOCAL POINTS
    # ---------------------------------------------------------
    POINTS = [
        ("Wreckhouse", 47.71, -59.31),
        ("Port aux Basques", 47.57, -59.13),
        ("Cape Ray", 47.62, -59.30),
        ("Stephenville", 48.55, -58.57),
    ]

    Xi_m, Yi_m = jnp.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
    times_hours_pt = list(range(sim_hours + 1))

    print("-" * 60)
    print("[DEBUG] Surface wind speeds at each sample point (t=0 hour):")
    for name, p_lat, p_lon in POINTS:
        dist2 = (lat_m - p_lat) ** 2 + (lon_m - p_lon) ** 2
        i_p, j_p = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))
        
        # Approximate staggered U/V to mass points for magnitude
        u_p = 0.5 * (snapshots[0]['u'][i_p, j_p, 0] + snapshots[0]['u'][i_p+1, j_p, 0])
        v_p = 0.5 * (snapshots[0]['v'][i_p, j_p, 0] + snapshots[0]['v'][i_p, j_p+1, 0])
        spd = float(jnp.sqrt(u_p**2 + v_p**2))
        
        print(f"{name:<18} (i:{i_p:>3d}, j:{j_p:>3d}) - Initial Wind Speed: {spd:>5.1f} m/s")
    print("-" * 60)

    for name, p_lat, p_lon in POINTS:
        dist2 = (lat_m - p_lat) ** 2 + (lon_m - p_lon) ** 2
        i_p, j_p = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))

        suetes_wind = []
        for s in snapshots:
            u_p = 0.5 * (s['u'][i_p, j_p, 0] + s['u'][i_p+1, j_p, 0])
            v_p = 0.5 * (s['v'][i_p, j_p, 0] + s['v'][i_p, j_p+1, 0])
            suetes_wind.append(float(jnp.sqrt(u_p**2 + v_p**2)) * 3.6) # Convert to km/h

        era5_wind = []
        for s in suetes_bc_states[:sim_hours + 1]:
            u_p = 0.5 * (s['u'][i_p, j_p, 0] + s['u'][i_p+1, j_p, 0])
            v_p = 0.5 * (s['v'][i_p, j_p, 0] + s['v'][i_p, j_p+1, 0])
            era5_wind.append(float(jnp.sqrt(u_p**2 + v_p**2)) * 3.6)

        safe_name = name.lower().replace(' ', '_')

        visualizer.plot_point_timeseries(
            times_hours_pt, suetes_wind, era5_wind,
            location_name=f"{name}",
            units='km/h',
            save_path=os.path.join(output_dir, f"{RUN_NAME}_Wind_{safe_name}_{sim_hours}h.png"),
        )

if __name__ == "__main__":
    main()