"""
Wreckhouse Worst-Case Scenario Experiment.

1. Spins up the atmosphere from t=0 to t=5h.
2. Uses the Adjoint model to find optimal perturbations at t=5h that 
   maximize Wreckhouse wind speeds at t=7h.
3. Runs the perturbed state forward to simulate the "worst-case" storm.
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

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.steppers import build_dynamical_core

from suetes.physics.base import PhysicsSuite
from suetes.physics.turbulence import SmagorinskyLillySGS
from suetes.physics.gravity_waves import McFarlaneGWD

from suetes.vis.visualizer import Visualizer

DATA_DIR = "output/data"


def create_wind_tracker(target_i, target_j):
        ts = []
        def callback(u, v):
            # Calculate magnitude at target point
            u_p = 0.5 * (u[target_i, target_j, 0] + u[target_i+1, target_j, 0])
            v_p = 0.5 * (v[target_i, target_j, 0] + v[target_i, target_j+1, 0])
            speed = float(np.sqrt(u_p**2 + v_p**2)) * 3.6
            ts.append(speed)
        return ts, callback

def main():
    # =====================================================================
    # 0. CONFIGURATION & GEOMETRY
    # =====================================================================
    ACTIVE_DOMAIN = "wreckhouse"
    RUN_NAME = f"{ACTIVE_DOMAIN}_worst_case_feb2025"
    output_dir = "output/plots/wreckhouse_worst_case"
    os.makedirs(output_dir, exist_ok=True)

    lat_c, lon_c = 47.71, -59.31
    wreckhouse_lat, wreckhouse_lon = 47.71, -59.31
    
    nx, ny, nz = 200, 200, 40
    dx, dy, dz = 2000.0, 2000.0, 350.0
    sponge_depth = 15
    smooth_sigma = 0.2

    # Timings
    dt = 5.0
    t_spinup_hours = 6.0
    t_peak_hours = 7.0
    t_total_hours = 9.0
    
    t_spinup_end = t_spinup_hours * 3600.0
    t_peak = t_peak_hours * 3600.0
    t_total_end = t_total_hours * 3600.0
    
    CHUNK_STEPS = 90
    sim_hours = 12 # This is our maximal runtime, but we don't go this far
    num_era5_states = sim_hours + 1
    
    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }

    YEAR, MONTH = "2025", "02"
    DAYS = [str(d).zfill(2) for d in range(14, 17)]

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()} Worst-Case")
    print(f"[CONFIG] Adjoint Window: T={t_spinup_hours}h to T={t_peak_hours}h")

    # =====================================================================
    # 1. DATA LOADING & BOUNDARIES
    # =====================================================================
    print("[SETUP] Loading Cached ERA5 Boundary Conditions...")
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels='buffered')
    dynamic_bbox = ERA5Manager.calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=2.0)
    cache_prefix = f"{ACTIVE_DOMAIN}_{YEAR}{MONTH}{DAYS[0]}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year=YEAR, month=MONTH, days=DAYS, area=dynamic_bbox, prefix=cache_prefix
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
    time_manager = TimeManager(suetes_bc_states, [i * 3600.0 for i in range(num_era5_states)], grid)

    initial_state = dict(suetes_bc_states[0])
    initial_state.pop('q', None)
    initial_state.pop('q_c', None)
    initial_state['theta_surf'] = initial_state.pop('theta_skt')
    initial_state['target_th_v'] = initial_state['th_v']

    # Locate Wreckhouse target indices
    Xi_m, Yi_m = jnp.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
    dist2 = (lat_m - wreckhouse_lat) ** 2 + (lon_m - wreckhouse_lon) ** 2
    i_w, j_w = np.unravel_index(int(jnp.argmin(dist2)), (grid.nx, grid.ny))

    # =====================================================================
    # 2. DUAL DYNAMICAL CORES (Forward vs. Adjoint)
    # =====================================================================
    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    
    # FORWARD STEPPER (High Fidelity Physics)
    physics_forward = PhysicsSuite()
    physics_forward.add_tendency_scheme(SmagorinskyLillySGS(grid, operators, constants, dt=dt, Cs=0.15, Pr_t=1.0, critical_Ri=0.25))
    physics_forward.add_tendency_scheme(McFarlaneGWD(grid, operators, constants, h_variance=jnp.where(land_fraction > 0.5, 2500.0, 0.0), F_c=0.7, mu=1.5e-5))

    stepper_forward, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=operators, 
        constants=constants, initial_state=initial_state, 
        physics_suite=physics_forward, interior_mask=sponge.get_interior_mask(), 
        dt=dt, ns=4, nu_div_factor=0.05, nu_h_factor=0.05, damp_height=9000.0, max_damp=3.0
    )

    # ADJOINT STEPPER (High Diffusion, Smooth Physics)
    # We drop the chaotic SGS and GWD schemes to prevent sqrt(0) AD traps
    # and multiply diffusion by 4x to absorb grid-scale adjoint noise.
    physics_adjoint = PhysicsSuite()

    stepper_adjoint, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=operators, 
        constants=constants, initial_state=initial_state, 
        physics_suite=physics_adjoint, interior_mask=sponge.get_interior_mask(), 
        dt=dt, ns=4, nu_div_factor=0.20, nu_h_factor=0.20, damp_height=9000.0, max_damp=3.0
    )

    # Core physics steps
    @jax.checkpoint
    def forward_step_fn(curr_state, step_idx, forcing=None, bc_fn=None):
        t_curr = step_idx * dt
        bc_state_t = jax.tree_util.tree_map(jax.lax.stop_gradient, time_manager.get_forcing(t_curr))
        
        updated_curr_state = dict(curr_state)
        updated_curr_state['theta_surf'] = bc_state_t['theta_skt']
        updated_curr_state['target_th_v'] = bc_state_t['th_v']
        
        def dynamic_bc_fn(state_next, _): return sponge.blend(state_next, bc_state_t)
        
        # Executes using the high-fidelity forward engine
        next_state = stepper_forward.step(updated_curr_state, t_curr, forcing=None, bc_fn=dynamic_bc_fn)
        next_state['theta_surf'] = updated_curr_state['theta_surf']
        next_state['target_th_v'] = updated_curr_state['target_th_v']
        
        max_w = jnp.max(jnp.abs(next_state['w']))
        return next_state, max_w

    @jax.checkpoint
    def adjoint_step_fn(curr_state, step_idx, forcing=None, bc_fn=None):
        t_curr = step_idx * dt
        bc_state_t = jax.tree_util.tree_map(jax.lax.stop_gradient, time_manager.get_forcing(t_curr))
        
        updated_curr_state = dict(curr_state)
        updated_curr_state['theta_surf'] = bc_state_t['theta_skt']
        updated_curr_state['target_th_v'] = bc_state_t['th_v']
        
        def dynamic_bc_fn(state_next, _): return sponge.blend(state_next, bc_state_t)
        
        # Executes using the highly-diffused, smooth adjoint engine
        next_state = stepper_adjoint.step(updated_curr_state, t_curr, forcing=None, bc_fn=dynamic_bc_fn)
        next_state['theta_surf'] = updated_curr_state['theta_surf']
        next_state['target_th_v'] = updated_curr_state['target_th_v']
        
        return next_state

    # Initialize two separate simulation drivers
    sim_forward = Simulation(step_fn=forward_step_fn, dt=dt)
    sim_adjoint = Simulation(step_fn=adjoint_step_fn, dt=dt)

    # =====================================================================
    # PHASE 1: FORWARD SPIN-UP (t=0 to t=5h)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 1] Spinning up forward model from t=0 to t={t_spinup_hours}h...")
    
    ts_baseline, track_baseline = create_wind_tracker(i_w, j_w)
    
    @jax.checkpoint
    def forward_step_fn_baseline(curr_state, step_idx, forcing=None, bc_fn=None):
        next_state, max_w = forward_step_fn(curr_state, step_idx, forcing, bc_fn)
        # Extract and save just the single float value on the fly
        jax.debug.callback(track_baseline, next_state['u'], next_state['v'], ordered=True)
        return next_state, max_w

    sim_baseline = Simulation(step_fn=forward_step_fn_baseline, dt=dt)
    
    # Capture the t=0 initial point
    track_baseline(initial_state['u'], initial_state['v'])
    
    start_time = time.time()
    state_t5 = sim_baseline.run(
        initial_state, t_start=0.0, t_end=t_spinup_end, chunk_steps=int(3600.0 / dt)
    )
    print(f"  -> Spin-up complete in {time.time() - start_time:.1f}s")

    # =====================================================================
    # PHASE 2: SHORT-WINDOW ADJOINT (t=5 to t=6h)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 2] Running Adjoint from t={t_peak_hours}h back to t={t_spinup_hours}h...")

    def peak_wind_objective(init_u, init_v, init_th_v):
        state = dict(state_t5)
        state['u'] = init_u
        state['v'] = init_v
        state['th_v'] = init_th_v

        final_state = sim_adjoint.run_differentiable(
            state, t_start=t_spinup_end, t_end=t_peak, chunk_steps=CHUNK_STEPS
        )
        
        # 3x3 footprint objective at Wreckhouse, mapped to mass points
        u_p = 0.5 * (final_state['u'][i_w-1:i_w+2, j_w-1:j_w+2, 0] + final_state['u'][i_w:i_w+3, j_w-1:j_w+2, 0])
        v_p = 0.5 * (final_state['v'][i_w-1:i_w+2, j_w-1:j_w+2, 0] + final_state['v'][i_w-1:i_w+2, j_w:j_w+3, 0])
        
        wind_patch = jnp.sqrt(u_p**2 + v_p**2 + 1e-8)
        return jnp.mean(wind_patch)

    print("  -> Compiling adjoint graph (this may take a while)...")
    grad_fn = jax.jit(jax.value_and_grad(peak_wind_objective, argnums=(0, 1, 2)))
    
    start_time = time.time()
    baseline_peak_wind, (grad_u, grad_v, grad_th_v) = grad_fn(state_t5['u'], state_t5['v'], state_t5['th_v'])
    baseline_peak_wind.block_until_ready()
    print(f"  -> Baseline peak wind over Wreckhouse (T={t_peak_hours}h): {float(baseline_peak_wind) * 3.6:.1f} km/h")
    print(f"  -> Adjoint completed in {time.time() - start_time:.1f}s")

    # =====================================================================
    # PHASE 3: CONSTRUCT OPTIMAL PERTURBATION
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 3] Normalizing and applying the worst-case perturbations...")

    grad_u_safe = jnp.nan_to_num(grad_u, nan=0.0, posinf=0.0, neginf=0.0)
    grad_v_safe = jnp.nan_to_num(grad_v, nan=0.0, posinf=0.0, neginf=0.0)
    grad_th_safe = jnp.nan_to_num(grad_th_v, nan=0.0, posinf=0.0, neginf=0.0)

    interior = sponge.get_interior_mask()
    
    # Apply a basic spatial smoother to prevent 2dx grid-scale acoustic shocks
    # Using a simple 3x3 uniform convolution footprint via JAX
    def smooth_field(field):
        kernel = jnp.ones((3, 3, 1)) / 9.0
        return jax.scipy.signal.convolve(field, kernel, mode='same')
        
    grad_u_smooth = smooth_field(grad_u_safe) * interior['u']
    grad_v_smooth = smooth_field(grad_v_safe) * interior['v']
    grad_th_smooth = smooth_field(grad_th_safe) * interior['th_v']

    # Safely compute scaling caps
    max_u_grad = jnp.max(jnp.abs(grad_u_smooth))
    print(f"  -> Max Smoothed Adjoint Zonal Gradient: {max_u_grad:.3e}")
    if max_u_grad == 0.0:
        print("  [WARNING] Adjoint returned pure NaNs/Zeros.")

    # To make this more extreme, we could tune this (but then the core might blow up)
    MAX_WIND_PERT = 2.0  
    MAX_TEMP_PERT = 1.0  

    scale_u = MAX_WIND_PERT / (max_u_grad + 1e-12)
    scale_v = MAX_WIND_PERT / (jnp.max(jnp.abs(grad_v_smooth)) + 1e-12)
    scale_th = MAX_TEMP_PERT / (jnp.max(jnp.abs(grad_th_smooth)) + 1e-12)

    pert_u = grad_u_smooth * scale_u
    pert_v = grad_v_smooth * scale_v
    pert_th_v = grad_th_smooth * scale_th

    # Apply perturbations
    worst_case_state_t5 = dict(state_t5)
    worst_case_state_t5['u'] = state_t5['u'] + pert_u
    worst_case_state_t5['v'] = state_t5['v'] + pert_v
    worst_case_state_t5['th_v'] = state_t5['th_v'] + pert_th_v

    # =====================================================================
    # PHASE 4: THE WORST CASE SIMULATION (9-HOUR RUN)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 4] Running baseline and worst-case simulations to T={t_total_hours}h...")
    
    from suetes.vis.comparisons import plot_worst_case_dashboard

    # Initialize worst-case tracker and pad with the Phase 1 spin-up history
    ts_worst_case, track_worst_case = create_wind_tracker(i_w, j_w)
    ts_worst_case.extend(ts_baseline) 
    
    @jax.checkpoint
    def forward_step_fn_worst_case(curr_state, step_idx, forcing=None, bc_fn=None):
        next_state, max_w = forward_step_fn(curr_state, step_idx, forcing, bc_fn)
        jax.debug.callback(track_worst_case, next_state['u'], next_state['v'], ordered=True)
        return next_state, max_w

    sim_worst_case = Simulation(step_fn=forward_step_fn_worst_case, dt=dt)

    # --- PART A: Run to the peak (T=5h to T=7h) ---
    print(f"  -> Advancing to peak target (T={t_peak_hours}h)...")
    
    state_baseline_t7 = sim_baseline.run(
        state_t5, t_start=t_spinup_end, t_end=t_peak, chunk_steps=int(3600.0 / dt)
    )
    
    state_worst_case_t7 = sim_worst_case.run(
        worst_case_state_t5, t_start=t_spinup_end, t_end=t_peak, chunk_steps=int(3600.0 / dt)
    )

    # --- PART B: Run the rest of the way (T=7h to T=9h) ---
    print(f"  -> Advancing to end of simulation (T={t_total_hours}h)...")
    
    _ = sim_baseline.run(
        state_baseline_t7, t_start=t_peak, t_end=t_total_end, chunk_steps=int(3600.0 / dt)
    )
    
    _ = sim_worst_case.run(
        state_worst_case_t7, t_start=t_peak, t_end=t_total_end, chunk_steps=int(3600.0 / dt)
    )

    # Extract ERA5 driver time series up to T=9h
    ts_era5 = []
    time_axis_mins = np.linspace(0, t_total_hours * 60, len(ts_baseline))
    
    for mins in time_axis_mins:
        t_sec = mins * 60.0
        bc_state = time_manager.get_forcing(t_sec) 
        u_p = 0.5 * (bc_state['u'][i_w, j_w, 0] + bc_state['u'][i_w+1, j_w, 0])
        v_p = 0.5 * (bc_state['v'][i_w, j_w, 0] + bc_state['v'][i_w, j_w+1, 0])
        ts_era5.append(float(np.sqrt(u_p**2 + v_p**2)) * 3.6)

    # Calculate exact index for the T=7h peak to print metrics
    peak_idx = int((t_peak_hours * 3600.0) / dt)
    
    print("-" * 60)
    print("=== FINAL IMPACT RESULTS ===")
    print(f" ERA5 coarse wind (T={t_peak_hours}h):      {ts_era5[peak_idx]:.1f} km/h")
    print(f" Baseline wind speed (T={t_peak_hours}h):   {ts_baseline[peak_idx]:.1f} km/h")
    print(f" Worst-case wind speed (T={t_peak_hours}h): {ts_worst_case[peak_idx]:.1f} km/h")
    print(f" Net amplification at peak:       +{ts_worst_case[peak_idx] - ts_baseline[peak_idx]:.1f} km/h")

    # Generate dashboard using the T=7h states for the cross-sections
    print("[PLOT] Generating adjoint impact dashboard...")
    plot_worst_case_dashboard(
        grid=grid,
        time_axis_mins=time_axis_mins,
        ts_era5=ts_era5,
        ts_baseline=ts_baseline,
        ts_worst_case=ts_worst_case,
        pert_th_v_2d=pert_th_v[:, :, 0], 
        state_baseline=state_baseline_t7,       # Uses the T=7h slice
        state_worst_case=state_worst_case_t7,   # Uses the T=7h slice
        loc_idx=(i_w, j_w),
        sponge_depth=sponge_depth,
        save_path=os.path.join(output_dir, f"{RUN_NAME}_dashboard.png")
    )

if __name__ == "__main__":
    main()