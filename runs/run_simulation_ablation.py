"""
Test for CORDEX NAM-22 downscaling via Ablation Study.

Drives the CORDEX North America domain at 22 km horizontal resolution
with ERA5 boundary conditions. Sequentially runs multiple physics 
configurations to isolate the impact of explicit parameterizations.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import copy
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
from suetes.physics.surface import McFarlaneSurfaceDrag, BucketLSM
from suetes.physics.turbulence import McFarlaneVerticalDiffusion, TKE15Closure
from suetes.physics.microphysics import KesslerWarmRain, SimplifiedBettsMiller
from suetes.physics.gravity_waves import UpperRayleighDamping

from suetes.vis.comparisons import (
        plot_ablation_spatial_matrix, 
        plot_ablation_cross_section_matrix,
        plot_ablation_hovmoller_matrix,
        plot_ablation_spectrum
    )

DATA_DIR = "inputs"

def main():
    ACTIVE_DOMAIN = "nam22"
    RUN_NAME = f"{ACTIVE_DOMAIN}_NAM22_july2025"

    lat_c, lon_c = 47.5, -97.0
    output_dir = "output/plots"
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

    YEAR, MONTH, DAYS = "2025", "07", [str(d).zfill(2) for d in range(18, 21)]

    # ---------------------------------------------------------
    # 0. DEFINE ABLATION EXPERIMENTS
    # ---------------------------------------------------------
    ablation_runs = [
        {"name": "1_Dry_Core", "pbl": False, "tke": False, "micro": False, "sponge": True},
        {"name": "2_Add_PBL",  "pbl": True,  "tke": False, "micro": False, "sponge": True},
        {"name": "3_Add_TKE",  "pbl": True,  "tke": True,  "micro": False, "sponge": True},
        {"name": "4_Add_Rain", "pbl": True,  "tke": True,  "micro": True,  "sponge": True},
    ]

    # ---------------------------------------------------------
    # 1. SETUP ENVIRONMENT & GEOMETRY (RUN ONCE)
    # ---------------------------------------------------------
    dynamic_bbox = ERA5Manager.calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=3.0)

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()} (CORDEX NAM-22)")
    print(f"[DATA] Validating ERA5 forcing files...")
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')
    cache_prefix = f"{ACTIVE_DOMAIN}_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year=YEAR, month=MONTH, days=DAYS, area=dynamic_bbox, prefix=cache_prefix
    )

    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} terrain-following mesh")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"))
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=smooth_sigma)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0))

    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    # ---------------------------------------------------------
    # 2. BOUNDARY CACHING & LOAD (RUN ONCE)
    # ---------------------------------------------------------
    bc_cache_path_coarse = os.path.join(DATA_DIR, f"{cache_prefix}_cw{coarsen_window}_coarse.pkl")
    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")

    print(f"[BOUNDARY] Preparing driver BC states...")
    suetes_bc_states = bridge.build_or_load_timeseries(era5_proc, num_era5_states, bc_cache_path_coarse, coarsen_window)
    suetes_bc_states_native = bridge.build_or_load_timeseries(era5_proc, num_era5_states, bc_cache_path_native, None)

    time_manager = TimeManager(suetes_bc_states, [float(i * 3600.0) for i in range(num_era5_states)], grid)

    initial_state = dict(suetes_bc_states[0])
    initial_state['theta_surf'] = (land_fraction * initial_state['theta_skt'] + (1.0 - land_fraction) * initial_state['th_v'][:, :, 0])
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state.pop('theta_skt', None)

    z_0_field = land_fraction * 0.1 + (1.0 - land_fraction) * 1e-4
    epsilon_field = land_fraction * 0.0 + (1.0 - land_fraction) * 0.3

    # ---------------------------------------------------------
    # 3. ABLATION SIMULATION LOOP
    # ---------------------------------------------------------
    chunk_steps = int(3600.0 / dt)
    final_states = {}
    hov_data_all_runs = {}
    
    # 1. Create a persistent mutable context object OUTSIDE the loop.
    # The compiled JAX graph will always point to this exact memory address.
    active_tracker = {"times": [], "data": []}

    def save_hovmoller_data(hour, anom_array):
        # Always appends to whatever list is currently active in the tracker
        active_tracker["times"].append(float(hour))
        active_tracker["data"].append(np.array(anom_array))

    for run in ablation_runs:
        run_name = run["name"]
        print(f"\n{'='*60}")
        print(f"[EXPERIMENT] Starting {run_name}")
        
        # Deep copy to ensure previous runs don't mutate the start state
        run_initial_state = copy.deepcopy(initial_state)

        # Add land fraction to initial state
        run_initial_state['land_fraction'] = land_fraction
        run_initial_state['dt'] = dt

        # 2. Reset the tracker's internal lists for THIS specific run
        initial_anom = run_initial_state['th_v'][:, :, 0] - run_initial_state['target_th_v'][:, :, 0]
        active_tracker["times"] = [0.0]
        active_tracker["data"] = [np.array(jnp.mean(initial_anom, axis=1))]

        # Dynamically build the Physics Suite
        physics_suite = PhysicsSuite()
        operators = CGridOperator3D(grid)

        # Initialize explicit moisture arrays if microphysics is active
        if run["micro"]:
            if 'q' not in run_initial_state:
                run_initial_state['q'] = jnp.zeros_like(run_initial_state['th_v'])
            run_initial_state['q_c'] = jnp.zeros_like(run_initial_state['th_v'])
            run_initial_state['q_r'] = jnp.zeros_like(run_initial_state['th_v'])
            run_initial_state['precip_conv'] = jnp.zeros_like(run_initial_state['th_v'][:, :, 0])
            
            physics_suite.register_tracer('q')
            physics_suite.register_tracer('q_c')
            physics_suite.register_tracer('q_r')
        else:
            run_initial_state.pop('q', None)
            run_initial_state.pop('q_c', None)
            run_initial_state.pop('q_r', None)
            run_initial_state.pop('precip_conv', None)

        # Initialize TKE array
        if run["tke"]:
            run_initial_state['tke'] = jnp.full_like(run_initial_state['th_v'], 1e-4)
            physics_suite.register_tracer('tke')
        else:
            run_initial_state.pop('tke', None)
        
        if run["sponge"]:
            physics_suite.add_tendency_scheme(UpperRayleighDamping(grid, operators))
            
        if run["pbl"]:
            physics_suite.add_tendency_scheme(BucketLSM(grid, operators, constants))
            
        if run["tke"]:
            physics_suite.add_tendency_scheme(TKE15Closure(grid, operators, constants))
        elif run["pbl"]: 
            # Fallback to local diffusion if PBL is on but prognostic TKE is off
            physics_suite.add_tendency_scheme(McFarlaneVerticalDiffusion(grid, operators, constants, epsilon=epsilon_field[..., None]))
            
        if run["micro"]:
            physics_suite.add_update_scheme(SimplifiedBettsMiller(constants))
            physics_suite.add_update_scheme(KesslerWarmRain(constants))

        # Always keep baseline boundary relaxation
        physics_suite.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))

        sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
        
        physics = Euler3D(
            grid, operators, constants, dt=dt, initial_era5_state=run_initial_state,
            damp_height=9000.0, max_damp=3.0, nu_div_factor=0.1, nu_h_factor=0.1,
            physics_suite=physics_suite, interior_mask=sponge.get_interior_mask(),
        )
        stepper = SISLStepper3D(physics, dt)

        def step_fn(curr_state, step_idx):
            t_curr = step_idx * dt
            bc_state_t = time_manager.get_forcing(t_curr)

            curr_state['theta_surf'] = (land_fraction * bc_state_t['theta_skt'] + (1.0 - land_fraction) * bc_state_t['th_v'][:, :, 0])
            curr_state['target_th_v'] = bc_state_t['th_v']

            def bc_fn(state_next, _):
                return sponge.blend(state_next, bc_state_t)

            next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
            next_state['theta_surf'] = curr_state['theta_surf']
            next_state['target_th_v'] = curr_state['target_th_v']
            next_state['land_fraction'] = curr_state['land_fraction']
            next_state['dt'] = curr_state['dt']

            anom = next_state['th_v'][:, :, 0] - bc_state_t['th_v'][:, :, 0]
            y_avg_anom = jnp.mean(anom, axis=1)
            current_hour = (step_idx + 1) * dt / 3600.0

            is_hourly = ((step_idx + 1) % chunk_steps) == 0
            
            # The callback now executes safely regardless of JAX caching
            jax.lax.cond(
                is_hourly,
                lambda: jax.debug.callback(save_hovmoller_data, current_hour, y_avg_anom),
                lambda: None
            )

            return next_state, jnp.max(jnp.abs(next_state['w']))

        sim = Simulation(step_fn=step_fn, dt=dt)
        start_time = time.time()
        
        final_state = sim.run(run_initial_state, t_start=0.0, t_end=sim_time_seconds, chunk_steps=chunk_steps)
        print(f"[SIMULATION] {run_name} completed in {time.time() - start_time:.1f}s")
        
        # 3. Store the cleanly collected arrays and copy the times for the plotter
        final_states[run_name] = final_state
        hov_data_all_runs[run_name] = np.array(active_tracker["data"])
        hov_times = list(active_tracker["times"])

    # ---------------------------------------------------------
    # 4. ABLATION VISUALIZATION DASHBOARDS
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("[PLOT] Generating dynamic ablation comparison dashboards...")
    final_era5_state = suetes_bc_states_native[-1]

    # 1. Surface and Mid-Level Spatial Maps
    plot_ablation_spatial_matrix(grid, final_states, final_era5_state, 'th_v', 0, sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_th_v_surf_{sim_hours}h.png"))
    plot_ablation_spatial_matrix(grid, final_states, final_era5_state, 'w', 5, sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_w_z5_{sim_hours}h.png"))

    # 2. X-Z Cross Sections
    plot_ablation_cross_section_matrix(grid, final_states, final_era5_state, 'w', y_idx=grid.ny//2, sponge_depth=sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_cross_w_{sim_hours}h.png"))
    plot_ablation_cross_section_matrix(grid, final_states, final_era5_state, 'th_v', y_idx=grid.ny//2, sponge_depth=sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_cross_th_v_{sim_hours}h.png"))

    # 3. Hovmöller Matrix
    plot_ablation_hovmoller_matrix(grid, hov_times, hov_data_all_runs, variable='Surface th_v Anomaly', 
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_hovmoller_{sim_hours}h.png"))

    # 4. Power Spectra
    plot_ablation_spectrum(grid, final_states, 'w', 5, sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_ablation_spectrum_{sim_hours}h.png"))

    print("[SUCCESS] All simulations and visualizations complete.")

if __name__ == "__main__":
    main()