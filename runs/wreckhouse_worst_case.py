"""
Wreckhouse Worst-Case Scenario Experiment.

1. Spins up the atmosphere from t=0 to t=6h.
2. Computes a differentiable control-model gradient of the 7h Wreckhouse
   target wind with respect to a pressure-balanced thermal perturbation.
3. Transfers that perturbation to the full-physics forecast and runs to 9h.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import time
import math
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager

from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation
from suetes.shared.experiment import save_plot_dataset

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.steppers import build_dynamical_core

from suetes.physics.base import PhysicsSuite
from suetes.physics.turbulence import SmagorinskyLillySGS
from suetes.physics.gravity_waves import McFarlaneGWD

from suetes.vis.visualizer import Visualizer

DATA_DIR = "inputs"


def create_wind_tracker():
    values = []

    def callback(speed_kmh):
        values.append(float(speed_kmh))

    return values, callback

def main():
    # =====================================================================
    # 0. CONFIGURATION & GEOMETRY
    # =====================================================================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="output/wreckhouse_worst_case",
        help="Flat directory for diagnostics, plot data, metadata, and figures",
    )
    args = parser.parse_args()

    ACTIVE_DOMAIN = "wreckhouse"
    RUN_NAME = f"{ACTIVE_DOMAIN}_worst_case_feb2025"
    output_dir = args.output_dir
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
    # 2. FULL FORECAST AND DIFFERENTIABLE CONTROL MODELS
    # =====================================================================
    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    
    physics_forward = PhysicsSuite()
    physics_forward.add_tendency_scheme(SmagorinskyLillySGS(grid, operators, constants, dt=dt, Cs=0.15, Pr_t=1.0, critical_Ri=0.25))
    physics_forward.add_tendency_scheme(McFarlaneGWD(grid, operators, constants, h_variance=jnp.where(land_fraction > 0.5, 2500.0, 0.0), F_c=0.7, mu=1.5e-5))

    stepper_forward, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=operators, 
        constants=constants, initial_state=initial_state, 
        physics_suite=physics_forward, interior_mask=sponge.get_interior_mask(), 
        dt=dt, ns=4, nu_div_factor=0.05, nu_h_factor=0.05, damp_height=9000.0, max_damp=3.0
    )

    # SGS stability switches and GWD saturation/critical-level switches do not
    # define a robust one-hour reverse trajectory.  The control model retains
    # the same resolved dynamics and boundary forcing, but omits those schemes
    # and damps grid-scale adjoint noise.  Its perturbation is subsequently
    # evaluated in the independent full-physics model above.
    physics_control = PhysicsSuite()
    stepper_control, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=operators,
        constants=constants, initial_state=initial_state,
        physics_suite=physics_control, interior_mask=sponge.get_interior_mask(),
        dt=dt, ns=4, nu_div_factor=0.20, nu_h_factor=0.20,
        damp_height=9000.0, max_damp=3.0,
    )

    def advance_state(curr_state, t_curr, stepper):
        bc_state_t = jax.tree_util.tree_map(jax.lax.stop_gradient, time_manager.get_forcing(t_curr))
        
        updated_curr_state = dict(curr_state)
        updated_curr_state['theta_surf'] = bc_state_t['theta_skt']
        updated_curr_state['target_th_v'] = bc_state_t['th_v']
        
        def dynamic_bc_fn(state_next, _): return sponge.blend(state_next, bc_state_t)
        
        next_state = stepper.step(updated_curr_state, t_curr, forcing=None, bc_fn=dynamic_bc_fn)
        next_state['theta_surf'] = updated_curr_state['theta_surf']
        next_state['target_th_v'] = updated_curr_state['target_th_v']
        
        return next_state

    @jax.checkpoint
    def differentiable_step_fn(curr_state, t_curr, forcing=None, bc_fn=None):
        del forcing, bc_fn
        return advance_state(curr_state, t_curr, stepper_control)

    def forward_step_fn(curr_state, step_idx, forcing=None, bc_fn=None):
        del forcing, bc_fn
        next_state = advance_state(curr_state, step_idx * dt, stepper_forward)
        return next_state, jnp.max(jnp.abs(next_state['w']))

    sim_adjoint = Simulation(step_fn=differentiable_step_fn, dt=dt)

    # =====================================================================
    # PHASE 1: FORWARD SPIN-UP (t=0 to t=6h)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 1] Spinning up forward model from t=0 to t={t_spinup_hours}h...")
    
    def point_wind_kmh(state):
        u_point = 0.5 * (state['u'][i_w, j_w, 0] + state['u'][i_w + 1, j_w, 0])
        v_point = 0.5 * (state['v'][i_w, j_w, 0] + state['v'][i_w, j_w + 1, 0])
        return 3.6 * jnp.sqrt(u_point**2 + v_point**2 + 1.0e-12)

    def target_wind_ms(state):
        """Smooth 3x3 near-surface wind diagnostic at Wreckhouse."""
        u_point = 0.5 * (
            state['u'][i_w-1:i_w+2, j_w-1:j_w+2, 0]
            + state['u'][i_w:i_w+3, j_w-1:j_w+2, 0]
        )
        v_point = 0.5 * (
            state['v'][i_w-1:i_w+2, j_w-1:j_w+2, 0]
            + state['v'][i_w-1:i_w+2, j_w:j_w+3, 0]
        )
        return jnp.mean(jnp.sqrt(u_point**2 + v_point**2 + 1.0e-8))

    ts_baseline, track_baseline = create_wind_tracker()
    
    @jax.checkpoint
    def forward_step_fn_baseline(curr_state, step_idx, forcing=None, bc_fn=None):
        next_state, max_w = forward_step_fn(curr_state, step_idx, forcing, bc_fn)
        jax.debug.callback(track_baseline, point_wind_kmh(next_state), ordered=True)
        return next_state, max_w

    sim_baseline = Simulation(step_fn=forward_step_fn_baseline, dt=dt)
    
    # Capture the t=0 initial point
    track_baseline(point_wind_kmh(initial_state))
    
    start_time = time.time()
    state_t6 = sim_baseline.run(
        initial_state, t_start=0.0, t_end=t_spinup_end, chunk_steps=int(3600.0 / dt)
    )
    print(f"  -> Spin-up complete in {time.time() - start_time:.1f}s")

    # =====================================================================
    # PHASE 2: SHORT-WINDOW ADJOINT (t=6 to t=7h)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 2] Running control-model adjoint from t={t_peak_hours}h back to t={t_spinup_hours}h...")

    def apply_thermal_control(delta_th_v):
        """Apply a pressure-balanced theta control and enforce the EOS."""
        state = dict(state_t6)
        state['th_v'] = state_t6['th_v'] + delta_th_v
        state['rho'] = (
            constants['p0'] / (constants['Rd'] * state['th_v'])
            * state['pi'] ** (constants['cvd'] / constants['Rd'])
        )
        return state

    def target_wind_objective(delta_th_v):
        state = apply_thermal_control(delta_th_v)

        final_state = sim_adjoint.run_differentiable(
            state, t_start=t_spinup_end, t_end=t_peak, chunk_steps=CHUNK_STEPS
        )
        return target_wind_ms(final_state)

    print("  -> Compiling adjoint graph (this may take a while)...")
    grad_fn = jax.jit(jax.value_and_grad(target_wind_objective))
    
    start_time = time.time()
    zero_control = jnp.zeros_like(state_t6['th_v'])
    control_baseline_wind, grad_th_v = grad_fn(zero_control)
    control_baseline_wind.block_until_ready()
    grad_th_v.block_until_ready()
    print(f"  -> Control-model target wind (T={t_peak_hours}h): {float(control_baseline_wind) * 3.6:.1f} km/h")
    print(f"  -> Adjoint completed in {time.time() - start_time:.1f}s")

    # =====================================================================
    # PHASE 3: CONSTRUCT OPTIMAL PERTURBATION
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 3] Constructing the norm-constrained adverse perturbation...")

    if not bool(jnp.all(jnp.isfinite(grad_th_v))):
        raise FloatingPointError(
            "The differentiable control-model thermal gradient contains "
            "non-finite values."
        )

    interior = sponge.get_interior_mask()
    
    # Apply a compact 3-D correlation operator to suppress grid-scale controls.
    # At this resolution the footprint spans 6 km x 6 km x 1.05 km.
    def smooth_field(field):
        kernel = jnp.ones((3, 3, 3)) / 27.0
        return jax.scipy.signal.convolve(field, kernel, mode='same')
        
    grad_th_smooth = smooth_field(grad_th_v) * interior['th_v']

    # Safely compute scaling caps
    max_th_grad = jnp.max(jnp.abs(grad_th_smooth))
    print(f"  -> Max smoothed thermal gradient: {max_th_grad:.3e}")
    if float(max_th_grad) == 0.0:
        raise FloatingPointError("The thermal gradient is identically zero")

    # To make this more extreme, we could tune this (but then the core might blow up)
    MAX_TEMP_PERT = 1.0
    control_direction = grad_th_smooth / max_th_grad
    pert_th_v = MAX_TEMP_PERT * control_direction

    # One-sided Taylor test of the exact differentiated objective.  The small
    # amplitude is in kelvin because max|control_direction| = 1.
    taylor_epsilon = 1.0e-2
    directional_derivative = jnp.vdot(grad_th_v, control_direction)
    objective_epsilon = jax.jit(target_wind_objective)(
        taylor_epsilon * control_direction
    )
    objective_epsilon.block_until_ready()
    actual_increment = objective_epsilon - control_baseline_wind
    linear_increment = taylor_epsilon * directional_derivative
    taylor_relative_error = jnp.abs(actual_increment - linear_increment) / jnp.maximum(
        jnp.abs(linear_increment), 1.0e-14
    )
    print(
        "  -> Control-model Taylor check at 0.01 K: "
        f"actual={float(actual_increment):.6e} m/s, "
        f"linear={float(linear_increment):.6e} m/s, "
        f"relative error={float(taylor_relative_error):.3e}"
    )
    if float(taylor_relative_error) > 5.0e-2:
        raise AssertionError("Thermal-control gradient failed the Taylor check")

    # Apply perturbations
    adverse_state_t6 = apply_thermal_control(pert_th_v)

    # =====================================================================
    # PHASE 4: THE WORST CASE SIMULATION (9-HOUR RUN)
    # =====================================================================
    print("-" * 60)
    print(f"[PHASE 4] Running baseline and adverse-perturbation simulations to T={t_total_hours}h...")
    
    from suetes.vis.comparisons import plot_worst_case_dashboard

    # Initialize worst-case tracker and pad with the Phase 1 spin-up history
    ts_worst_case, track_worst_case = create_wind_tracker()
    ts_worst_case.extend(ts_baseline) 
    
    @jax.checkpoint
    def forward_step_fn_worst_case(curr_state, step_idx, forcing=None, bc_fn=None):
        next_state, max_w = forward_step_fn(curr_state, step_idx, forcing, bc_fn)
        jax.debug.callback(track_worst_case, point_wind_kmh(next_state), ordered=True)
        return next_state, max_w

    sim_worst_case = Simulation(step_fn=forward_step_fn_worst_case, dt=dt)

    # --- PART A: Run from the control time to the target time (T=6h to T=7h) ---
    print(f"  -> Advancing to peak target (T={t_peak_hours}h)...")
    
    state_baseline_t7 = sim_baseline.run(
        state_t6, t_start=t_spinup_end, t_end=t_peak, chunk_steps=int(3600.0 / dt)
    )
    
    state_worst_case_t7 = sim_worst_case.run(
        adverse_state_t6, t_start=t_spinup_end, t_end=t_peak, chunk_steps=int(3600.0 / dt)
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
    expected_samples = int(t_total_end / dt) + 1
    if len(ts_baseline) != expected_samples or len(ts_worst_case) != expected_samples:
        raise RuntimeError(
            "Wind tracker length mismatch: "
            f"baseline={len(ts_baseline)}, adverse={len(ts_worst_case)}, "
            f"expected={expected_samples}"
        )
    time_axis_mins = np.arange(expected_samples) * dt / 60.0
    
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
    baseline_target = float(target_wind_ms(state_baseline_t7)) * 3.6
    adverse_target = float(target_wind_ms(state_worst_case_t7)) * 3.6
    print(f" Baseline point wind (T={t_peak_hours}h):   {ts_baseline[peak_idx]:.1f} km/h")
    print(f" Adverse point wind (T={t_peak_hours}h):    {ts_worst_case[peak_idx]:.1f} km/h")
    print(f" Baseline 3x3 target wind:                  {baseline_target:.1f} km/h")
    print(f" Adverse 3x3 target wind:                   {adverse_target:.1f} km/h")
    print(f" Net target amplification:                 +{adverse_target - baseline_target:.1f} km/h")

    result_path = os.path.join(output_dir, f"{RUN_NAME}_diagnostics.npz")
    np.savez_compressed(
        result_path,
        time_minutes=time_axis_mins,
        era5_wind_kmh=np.asarray(ts_era5),
        baseline_wind_kmh=np.asarray(ts_baseline),
        adverse_wind_kmh=np.asarray(ts_worst_case),
        delta_th_v=np.asarray(pert_th_v),
        baseline_target_wind_kmh=baseline_target,
        adverse_target_wind_kmh=adverse_target,
        control_model_baseline_objective_kmh=float(control_baseline_wind) * 3.6,
        control_model="dry resolved dynamics; nu_h=nu_div=0.20",
        taylor_relative_error=float(taylor_relative_error),
        control_time_seconds=t_spinup_end,
        target_time_seconds=t_peak,
        end_time_seconds=t_total_end,
        dt_seconds=dt,
        maximum_temperature_perturbation_K=MAX_TEMP_PERT,
    )
    print(f"[OUTPUT] Saved numerical diagnostics to {result_path}")

    # Store every field consumed by the publication dashboard.  This artifact
    # is deliberately independent of live model/grid objects so the figure can
    # be regenerated without ERA5 access or another nine-hour integration.
    u_baseline_m = 0.5 * (
        state_baseline_t7["u"][:-1] + state_baseline_t7["u"][1:]
    )
    u_adverse_m = 0.5 * (
        state_worst_case_t7["u"][:-1] + state_worst_case_t7["u"][1:]
    )
    v_baseline_m = 0.5 * (
        state_baseline_t7["v"][:, :-1] + state_baseline_t7["v"][:, 1:]
    )
    v_adverse_m = 0.5 * (
        state_worst_case_t7["v"][:, :-1] + state_worst_case_t7["v"][:, 1:]
    )
    baseline_speed_m = np.sqrt(
        np.asarray(u_baseline_m) ** 2 + np.asarray(v_baseline_m) ** 2
    )
    adverse_speed_m = np.sqrt(
        np.asarray(u_adverse_m) ** 2 + np.asarray(v_adverse_m) ** 2
    )
    x_mesh, y_mesh = np.meshgrid(grid.x_m, grid.y_m, indexing="ij")
    latitude, longitude = grid.proj.get_lat_lon(x_mesh, y_mesh)
    artifact = xr.Dataset(
        data_vars={
            "era5_wind": ("time", np.asarray(ts_era5)),
            "baseline_wind": ("time", np.asarray(ts_baseline)),
            "adverse_wind": ("time", np.asarray(ts_worst_case)),
            "thermal_perturbation_surface": (
                ("x", "y"), np.asarray(pert_th_v[:, :, 0])
            ),
            "latitude": (("x", "y"), np.asarray(latitude)),
            "longitude": (("x", "y"), np.asarray(longitude)),
            "terrain_height_map": (
                ("x", "y"), np.asarray(grid.Z_w[:, :, 0])
            ),
            "baseline_surface_u": (
                ("x", "y"), np.asarray(u_baseline_m[:, :, 0])
            ),
            "baseline_surface_v": (
                ("x", "y"), np.asarray(v_baseline_m[:, :, 0])
            ),
            "physical_height": (
                ("x", "z"), np.asarray(grid.Z_m[:, j_w, :])
            ),
            "terrain_height": ("x", np.asarray(grid.Z_w[:, j_w, 0])),
            "baseline_zonal_wind": (
                ("x", "z"), np.asarray(u_baseline_m[:, j_w, :]) * 3.6
            ),
            "zonal_wind_anomaly": (
                ("x", "z"),
                np.asarray(u_adverse_m[:, j_w, :] - u_baseline_m[:, j_w, :])
                * 3.6,
            ),
            "baseline_wind_speed": (
                ("x", "z"), baseline_speed_m[:, j_w, :] * 3.6
            ),
            "wind_speed_anomaly": (
                ("x", "z"),
                (adverse_speed_m[:, j_w, :] - baseline_speed_m[:, j_w, :])
                * 3.6,
            ),
            "baseline_virtual_potential_temperature": (
                ("x", "z"), np.asarray(state_baseline_t7["th_v"][:, j_w, :])
            ),
        },
        coords={
            "time": time_axis_mins * 60.0,
            "x": np.asarray(grid.x_m),
            "y": np.asarray(grid.y_m),
            "z": np.arange(grid.nz),
        },
        attrs={
            "target_i": i_w, "target_j": j_w,
            "sponge_depth": sponge_depth,
            "control_time_seconds": t_spinup_end,
            "target_time_seconds": t_peak,
            "end_time_seconds": t_total_end,
            "dt_seconds": dt,
            "baseline_target_wind_kmh": baseline_target,
            "adverse_target_wind_kmh": adverse_target,
            "taylor_relative_error": float(taylor_relative_error),
        },
    )
    artifact_path = save_plot_dataset(
        artifact, os.path.join(output_dir, f"{RUN_NAME}_plot_data.nc"),
        experiment="wreckhouse_worst_case",
        metadata={"control_model": "dry resolved dynamics; nu_h=nu_div=0.20"},
    )
    print(f"[OUTPUT] Saved plot-ready dashboard artifact to {artifact_path}")

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
    # Re-render from the portable artifact using the publication layout.  This
    # also verifies that the saved NetCDF is sufficient to reproduce the
    # figure without rerunning the forecast.
    subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "plot_wreckhouse_worst_case.py"),
            str(artifact_path),
            "--output",
            os.path.join(output_dir, f"{RUN_NAME}_dashboard.png"),
        ],
        check=True,
    )

if __name__ == "__main__":
    main()
