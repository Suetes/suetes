"""
Wreckhouse Adjoint Sensitivity Experiment.

Computes the exact gradient of the peak Wreckhouse wind gust with respect 
to the initial upstream atmospheric state (u, v, th_v) using reverse-mode 
automatic differentiation (JAX).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import math
import jax
import jax.numpy as jnp
import numpy as np

from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager


from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

from suetes.regional3d.steppers import build_dynamical_core
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge

from suetes.physics.base import PhysicsSuite
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.turbulence import SmagorinskyLillySGS
from suetes.physics.gravity_waves import McFarlaneGWD
from suetes.physics.forcing import NewtonianRelaxation

from suetes.vis.visualizer import Visualizer

DATA_DIR = "suetes/data"

def main():
    ACTIVE_DOMAIN = "wreckhouse_adjoint"
    RUN_NAME = f"{ACTIVE_DOMAIN}_2km_feb2025"

    lat_c, lon_c = 47.71, -59.31
    wreckhouse_lat, wreckhouse_lon = 47.71, -59.31
    
    output_dir = "suetes/plots/wreckhouse_adjoint"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 200, 200, 40
    dx, dy, dz = 2000.0, 2000.0, 350.0
    sponge_depth = 15
    smooth_sigma = 0.2  
    
    dt = 5.0
    sim_hours = 1.0
    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = math.ceil(sim_hours) + 1
    
    CHUNK_STEPS = 90 

    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }

    YEAR, MONTH = "2025", "02"
    DAYS = [str(d).zfill(2) for d in range(14, 17)]

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()}")
    print(f"[CONFIG] Grid: {nx}x{ny}x{nz} at dx={dx/1000:.1f} km")
    print(f"[CONFIG] Adjoint Memory Chunking: {CHUNK_STEPS} steps/chunk")

    # ---------------------------------------------------------
    # 1. GEOMETRY, TOPOGRAPHY & BOUNDARIES
    # ---------------------------------------------------------
    dynamic_bbox = ERA5Manager.calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=2.0)
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')
    cache_prefix = f"wreckhouse_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year=YEAR, month=MONTH, days=DAYS, area=dynamic_bbox, prefix=cache_prefix,
    )

    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"))
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=smooth_sigma)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0))

    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")
    suetes_bc_states = bridge.build_or_load_timeseries(
        era5_proc=era5_proc, num_states=num_era5_states, cache_path=bc_cache_path_native, coarsen_window=None
    )
    time_manager = TimeManager(suetes_bc_states, [float(i * 3600.0) for i in range(num_era5_states)], grid)

    # ---------------------------------------------------------
    # 2. INITIAL STATE & TARGET INDICES
    # ---------------------------------------------------------
    initial_state = dict(suetes_bc_states[0])
    initial_state.pop('q', None)
    initial_state.pop('q_c', None)
    initial_state['theta_surf'] = initial_state['theta_skt']
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state.pop('theta_skt', None)

    z_0_field = land_fraction * 1.0 + (1.0 - land_fraction) * 1e-4
    epsilon_field = land_fraction * 0.0 + (1.0 - land_fraction) * 0.3

    # Locate Wreckhouse in grid space
    Xi_m, Yi_m = jnp.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
    dist2 = (lat_m - wreckhouse_lat) ** 2 + (lon_m - wreckhouse_lon) ** 2
    i_w, j_w = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))

    # ---------------------------------------------------------
    # 3. DYNAMICS & PHYSICS SETUP
    # ---------------------------------------------------------
    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    
    physics_suite = PhysicsSuite()
    # physics_suite.add_tendency_scheme(McFarlaneSurfaceDrag(grid, operators, constants, z_0=z_0_field, epsilon=epsilon_field, theta_surf=initial_state['theta_surf']))
    physics_suite.add_tendency_scheme(SmagorinskyLillySGS(grid, operators, constants, dt=dt, Cs=0.15, Pr_t=1.0, critical_Ri=0.25))
    physics_suite.add_tendency_scheme(McFarlaneGWD(grid, operators, constants, h_variance=jnp.where(land_fraction > 0.5, 2500.0, 0.0), F_c=0.7, mu=1.5e-5))
    physics_suite.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))

    # At 2km resolution, a 5.0s timestep with 6 acoustic substeps should maintain CFL stability
    core_kwargs = {
        "dt": dt, 
        "ns": 4, 
        "nu_div_factor": 0.05, 
        "nu_h_factor": 0.05, 
        "damp_height": 9000.0, 
        "max_damp": 3.0
    }

    stepper, dt = build_dynamical_core(
        core_type="split-explicit", 
        grid=grid, 
        operators=operators, 
        constants=constants, 
        initial_state=initial_state, 
        physics_suite=physics_suite, 
        interior_mask=sponge.get_interior_mask(),
        **core_kwargs
    )

    # ---------------------------------------------------------
    # 4. ADJOINT OBJECTIVE DEFINITION
    # ---------------------------------------------------------
    def wreckhouse_sensitivity_objective(init_u, init_v, init_th_v):
        state = dict(initial_state)
        state['u'] = init_u
        state['v'] = init_v
        state['th_v'] = init_th_v
        state['max_wh_wind_exp'] = 0.0  

        # Discard intra-step physics activations
        @jax.checkpoint
        def diff_step_fn(curr_state, step_idx, forcing=None, bc_fn=None):
            # Explicitly calculate time in seconds
            t_curr = step_idx * dt
            
            # Fetch and freeze the boundary forcing for this timestep
            bc_state_t = jax.tree_util.tree_map(
                jax.lax.stop_gradient, 
                time_manager.get_forcing(t_curr)
            )
            
            # Update the state before the physics step, this ensures Newtonian Relaxation and Surface Drag have the correct targets.
            updated_curr_state = dict(curr_state)
            updated_curr_state['theta_surf'] = bc_state_t['theta_skt']
            updated_curr_state['target_th_v'] = bc_state_t['th_v']

            def dynamic_bc_fn(state_next, _):
                return sponge.blend(state_next, bc_state_t)

            # Step the core using the updated state
            next_state = stepper.step(
                updated_curr_state, t_curr, forcing=None, bc_fn=dynamic_bc_fn
            )
            
            # Propagate the non-prognostic keys to satisfy jax.lax.scan rules
            next_state['theta_surf'] = updated_curr_state['theta_surf']
            next_state['target_th_v'] = updated_curr_state['target_th_v']
            
            # Track the objective
            u_p = 0.5 * (next_state['u'][i_w, j_w, 0] + next_state['u'][i_w+1, j_w, 0])
            v_p = 0.5 * (next_state['v'][i_w, j_w, 0] + next_state['v'][i_w, j_w+1, 0])
            current_wind = jnp.sqrt(u_p**2 + v_p**2 + 1e-4)
            
            # Exponentially smoothed maximal wind (to make sure we don't get any spikes which would happen if we tracked only the max wind itself)
            alpha = 0.5 
            next_state['max_wh_wind_exp'] = curr_state['max_wh_wind_exp'] + jnp.exp(alpha * current_wind)
            
            return next_state

        sim = Simulation(step_fn=diff_step_fn, dt=dt)

        final_state = sim.run_differentiable(
            state, t_start=0.0, t_end=sim_time_seconds, chunk_steps=CHUNK_STEPS
        )
        
        return final_state['max_wh_wind_exp']

    # ---------------------------------------------------------
    # 5. EXECUTE FORWARD & BACKWARD PASS
    # ---------------------------------------------------------
    print("-" * 60)
    print("[ADJOINT] Compiling forward and backward passes...")
    
    # Use jax.jit here so XLA respects memory chunking
    grad_fn = jax.jit(jax.value_and_grad(wreckhouse_sensitivity_objective, argnums=(0, 1, 2)))

    start_time = time.time()
    
    # We only need to unpack the wind and gradients
    max_wind, (grad_u, grad_v, grad_th_v) = grad_fn(
        initial_state['u'], 
        initial_state['v'], 
        initial_state['th_v']
    )
    
    # Force Python to wait for the GPU to finish the math!
    max_wind.block_until_ready()
    
    wall_time = time.time() - start_time
    print("-" * 60)
    print(f"[ADJOINT] Complete in {wall_time:.1f}s")
    
    # ---------------------------------------------------------
    # 6. SAVE & VISUALIZE ADJOINT SENSITIVITIES
    # ---------------------------------------------------------
    print("-" * 60)
    print("[ADJOINT] Saving sensitivity arrays...")
    np.save(os.path.join(output_dir, f"{RUN_NAME}_grad_u.npy"), np.array(grad_u))
    np.save(os.path.join(output_dir, f"{RUN_NAME}_grad_v.npy"), np.array(grad_v))
    np.save(os.path.join(output_dir, f"{RUN_NAME}_grad_th_v.npy"), np.array(grad_th_v))

    print("[PLOT] Generating Adjoint Dashboards...")
    visualizer = Visualizer()
    
    # Spatial Bounds from the forward script
    y_targets = [96, 100, 104] 
    x_targets = [96, 100, 104] 
    zoom_x_km = [-50.0, 50.0]
    zoom_z_m = [0, 5000]
    
    lon_min, lon_max = lon_c - 0.75, lon_c + 0.75
    lat_min, lat_max = lat_c - 0.75, lat_c + 0.75
    wreckhouse_extent = [lon_min, lon_max, lat_min, lat_max]

    try:
        # Pre-calculate mass-centered momentum gradients to prevent C-grid staggering errors
        grad_u_m = 0.5 * (grad_u[:-1, :, :] + grad_u[1:, :, :])
        grad_v_m = 0.5 * (grad_v[:, :-1, :] + grad_v[:, 1:, :])
        grad_wind_mag = jnp.sqrt(grad_u_m**2 + grad_v_m**2 + 1e-8)

        # --- PLOT 1: ADJOINT THERMODYNAMIC SLICES (Sensitivities) ---
        sens_state_th = dict(initial_state)
        sens_state_th['th_v'] = grad_th_v
        
        visualizer.plot_slice_locator_dashboard(
            grid, sens_state_th, map_var='th_v', slice_var='th_v', map_z=2,
            sponge_depth=sponge_depth,
            x_indices=x_targets, y_indices=y_targets, 
            slice_xlim=zoom_x_km, slice_ylim=zoom_z_m,
            save_path=os.path.join(output_dir, f"{RUN_NAME}_ADJOINT_slices_th_v.png")
        )
        print("  -> Saved Adjoint Thermodynamic Slices (grad_th_v)")

        # --- PLOT 2: ADJOINT ZONAL WIND (u) SLICES ---
        sens_state_u = dict(initial_state)
        
        # Inject the mass-centered u-gradient into the th_v carrier key
        sens_state_u['th_v'] = grad_u_m 
        
        visualizer.plot_slice_locator_dashboard(
            grid, sens_state_u, map_var='th_v', slice_var='th_v', map_z=2,
            sponge_depth=sponge_depth,
            x_indices=x_targets, y_indices=y_targets, 
            slice_xlim=zoom_x_km, slice_ylim=zoom_z_m,
            save_path=os.path.join(output_dir, f"{RUN_NAME}_ADJOINT_slices_u.png")
        )
        print("  -> Saved Adjoint Zonal Wind Slices (grad_u_m)")

        # --- PLOT 3: ADJOINT KINEMATIC DASHBOARD (Optimal Perturbations) ---
        sens_state_wind = dict(initial_state)
        sens_state_wind['th_v'] = grad_wind_mag
        sens_state_wind['u'] = grad_u
        sens_state_wind['v'] = grad_v

        visualizer.plot_dashboard(
            grid, sens_state_wind, z_idx=2, sponge_depth=sponge_depth,
            time_hours=0, extent=wreckhouse_extent,
            quiver_stride=6, 
            save_path=os.path.join(output_dir, f"{RUN_NAME}_ADJOINT_wind_dash.png")
        )
        print("  -> Saved Adjoint Kinematic Dashboard (grad_u, grad_v)")

    except Exception as e:
        print(f"[PLOT] Could not generate auto-plots: {e}")

if __name__ == "__main__":
    main()