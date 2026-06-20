"""
Test for something like CORDEX NAM-22 downscaling.
Drives the CORDEX North America domain at 22 km horizontal resolution
with ERA5 boundary conditions. Compares Standard Physics against ML-Corrected Physics.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import pickle
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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
from suetes.physics.turbulence import McFarlaneVerticalDiffusion
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.forcing import NewtonianRelaxation
from suetes.physics.ml import MLPhysicsClosure
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
    sim_hours = 3
    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = int(sim_hours) + 1

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}

    YEAR, MONTH, DAYS = "2025", "07", [str(d).zfill(2) for d in range(18, 21)]

    dynamic_bbox = ERA5Manager.calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=3.0)

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()} (CORDEX NAM-22)")
    print(f"[DATA] Validating ERA5 forcing files...")
    
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')
    cache_prefix = f"{ACTIVE_DOMAIN}_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"
    sl_file, pl_file = manager.download_regional_subset(year=YEAR, month=MONTH, days=DAYS, area=dynamic_bbox, prefix=cache_prefix)

    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} terrain-following mesh")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"))
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=smooth_sigma)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0))

    # SETUP PROCESSORS & STATIC FIELDS
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    static_fields = bridge.process_static(raw_t0)
    land_fraction = jnp.asarray(static_fields['land_fraction'], dtype=float)

    # LOAD BOUNDARIES
    bc_cache_path_coarse = os.path.join(DATA_DIR, f"{cache_prefix}_cw{coarsen_window}_coarse.pkl")
    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")

    suetes_bc_states = bridge.build_or_load_timeseries(era5_proc, num_states=num_era5_states, cache_path=bc_cache_path_coarse, coarsen_window=coarsen_window)
    suetes_bc_states_native = bridge.build_or_load_timeseries(era5_proc, num_states=num_era5_states, cache_path=bc_cache_path_native, coarsen_window=None)
    time_manager = TimeManager(suetes_bc_states, [i * 3600.0 for i in range(num_era5_states)], grid)

    # INITIAL STATE SETUP
    initial_state = dict(suetes_bc_states[0])
    initial_state.pop('q', None)
    initial_state.pop('q_c', None)
    
    initial_state['theta_surf'] = (land_fraction * initial_state['theta_skt'] + (1.0 - land_fraction) * initial_state['th_v'][:, :, 0])
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state['land_fraction'] = land_fraction
    initial_state.pop('theta_skt', None)
    initial_state = jax.tree.map(lambda x: jnp.asarray(x, dtype=float), initial_state)

    z_0_field = land_fraction * 0.1 + (1.0 - land_fraction) * 1e-4
    epsilon_field = land_fraction * 0.0 + (1.0 - land_fraction) * 0.3

    # LOAD UNIVERSAL ML WEIGHTS
    weights_path = "suetes/weights/universal_ml_closure.pkl"
    print(f"[ML CLOSURE] Loading universal weights from {weights_path}...")
    with open(weights_path, "rb") as f:
        checkpoint = pickle.load(f)
    saved_ml_params, saved_norm_stats = checkpoint['params'], checkpoint['norm_stats']

    # DYNAMICS & SPONGE SETUP
    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    interior_mask = sponge.get_interior_mask()

    # MODEL 1: BASELINE SETUP
    physics_suite_base = PhysicsSuite()
    physics_suite_base.add_tendency_scheme(McFarlaneSurfaceDrag(grid, operators, constants, z_0=z_0_field, epsilon=epsilon_field, theta_surf=initial_state['theta_surf']))
    physics_suite_base.add_tendency_scheme(McFarlaneVerticalDiffusion(grid, operators, constants, epsilon=epsilon_field[..., None]))
    physics_suite_base.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))

    core_base = Euler3D(grid, operators, constants, dt=dt, initial_era5_state=initial_state, damp_height=9000.0, max_damp=3.0, nu_div_factor=0.1, nu_h_factor=0.1, physics_suite=physics_suite_base, interior_mask=interior_mask)
    stepper_base = SISLStepper3D(core_base, dt)

    # MODEL 2: ML CORRECTED SETUP
    physics_suite_ml = PhysicsSuite()
    physics_suite_ml.add_tendency_scheme(McFarlaneSurfaceDrag(grid, operators, constants, z_0=z_0_field, epsilon=epsilon_field, theta_surf=initial_state['theta_surf']))
    physics_suite_ml.add_tendency_scheme(McFarlaneVerticalDiffusion(grid, operators, constants, epsilon=epsilon_field[..., None]))
    physics_suite_ml.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))
    physics_suite_ml.add_tendency_scheme(MLPhysicsClosure(operators, saved_norm_stats))

    core_ml = Euler3D(grid, operators, constants, dt=dt, initial_era5_state=initial_state, damp_height=9000.0, max_damp=3.0, nu_div_factor=0.1, nu_h_factor=0.1, physics_suite=physics_suite_ml, interior_mask=interior_mask)
    stepper_ml = SISLStepper3D(core_ml, dt)

    # SIMULATION RUNNERS
    _SAVE_KEYS = ('u', 'v', 'w', 'th_v', 'pi')
    chunk_steps = int(3600.0 / dt)

    def run_variant(stepper, use_ml=False):
        snapshots = [{k: np.asarray(v) for k, v in initial_state.items() if k in _SAVE_KEYS}]
        
        def save_state_snapshot(state_sub):
            snapshots.append({k: np.asarray(v) for k, v in state_sub.items()})

        def step_fn(curr_state, step_idx):
            t_curr = step_idx * dt
            bc_state_t = time_manager.get_forcing(t_curr)

            dynamic_theta_surf = (land_fraction * jnp.asarray(bc_state_t['theta_skt'], dtype=float) + 
                                  (1.0 - land_fraction) * jnp.asarray(bc_state_t['th_v'][:, :, 0], dtype=float))

            def bc_fn(state_next, _):
                return sponge.blend(state_next, bc_state_t)

            if use_ml:
                augmented_ml_params = {
                    'nn_params': saved_ml_params 
                }
                next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn, ml_params=augmented_ml_params)
            else:
                next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)

            next_state['theta_surf'] = dynamic_theta_surf
            next_state['target_th_v'] = jnp.asarray(bc_state_t['th_v'], dtype=float)
            next_state['land_fraction'] = land_fraction

            is_hourly = ((step_idx + 1) % chunk_steps) == 0
            state_to_save = {k: next_state[k] for k in _SAVE_KEYS if k in next_state}
            jax.lax.cond(is_hourly, lambda: jax.debug.callback(save_state_snapshot, state_to_save, ordered=True), lambda: None)

            return next_state, jnp.max(jnp.abs(next_state['w']))

        sim = Simulation(step_fn=step_fn, dt=dt)
        t0 = time.time()
        final_state = sim.run(initial_state, t_start=0.0, t_end=sim_time_seconds, chunk_steps=chunk_steps)
        print(f"    -> Completed in {time.time() - t0:.1f}s")
        return final_state, snapshots

    print("-" * 60)
    print(f"[SIMULATION 1] Running Baseline (McFarlane Only)...")
    final_state_base, snapshots_base = run_variant(stepper_base, use_ml=False)

    print(f"\n[SIMULATION 2] Running ML-Corrected (McFarlane + Neural Net)...")
    final_state_ml, snapshots_ml = run_variant(stepper_ml, use_ml=True)

    # EVALUATION AND PLOTTING
    print("-" * 60)
    print("[PLOT] Generating diagnostic maps and time series...")
    visualizer = Visualizer()
    final_era5_state = suetes_bc_states_native[-1]

    # Baseline Anomaly Map
    visualizer.plot_comparison(
        grid, final_state_base, final_era5_state, 'th_v', z_idx=0, sponge_depth=sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_compare_th_v_surf_{sim_hours}h_BASELINE.jpg")
    )

    # ML-Corrected Anomaly Map
    visualizer.plot_comparison(
        grid, final_state_ml, final_era5_state, 'th_v', z_idx=0, sponge_depth=sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_compare_th_v_surf_{sim_hours}h_ML_CORRECTED.jpg")
    )

    # Generate Time Series
    POINTS = [
        ("Denver",      39.74, -104.99),
        ("Kansas City", 39.10,  -94.58),
        ("Minneapolis", 44.98,  -93.27),
        ("Winnipeg",    49.90,  -97.14),
        ("Atlanta",     33.75,  -84.39),
        ("Phoenix",     33.45, -112.07)
    ]

    Xi_m, Yi_m = jnp.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
    
    num_captured_hours = len(snapshots_base)
    times_hours_pt = [float(i) for i in range(num_captured_hours)]

    for name, p_lat, p_lon in POINTS:
        dist2 = (lat_m - p_lat) ** 2 + (lon_m - p_lon) ** 2
        i_p, j_p = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))

        # Enforce that the target array is sliced to match the exact length of the simulation run
        era5_T_C = [float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15 for s in suetes_bc_states_native[:num_captured_hours]]
        base_T_C = [float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15 for s in snapshots_base]
        ml_T_C   = [float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15 for s in snapshots_ml]

        ew = 'W' if p_lon < 0 else 'E'
        ns = 'N' if p_lat > 0 else 'S'
        location_str = f"{name} ({abs(p_lat):.2f} deg {ns}, {abs(p_lon):.2f} deg {ew})"
        safe_name = name.lower().replace(' ', '_')

        visualizer.plot_point_timeseries(
            times_hours_pt, base_T_C, era5_T_C,
            location_name=location_str, units='deg C',
            save_path=os.path.join(output_dir, f"{RUN_NAME}_T_{safe_name}_{sim_hours}h_COMPARISON.png"),
            ml_values=ml_T_C
        )

    print("[SUCCESS] All comparisons generated successfully.")

if __name__ == "__main__":
    main()