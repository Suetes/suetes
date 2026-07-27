# Case runner.
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Suetes Imports
from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager
from suetes.shared.transforms import SleveSimple
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.physics.base import PhysicsSuite
from suetes.physics.surface import BulkAerodynamicPBL
from suetes.physics.turbulence import FastVerticalDiffusion
from suetes.physics.forcing import NewtonianRelaxation
from suetes.vis.visualizer import Visualizer
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset

# ==========================================
# 0. DOMAIN PRESETS
# ==========================================
DOMAINS = {
    "labrador_sea": {"lat_c": 48.0, "lon_c": -60.0},
    "alps": {"lat_c": 45.0, "lon_c": 5.0},
    "nz_south_island": {"lat_c": -43.5, "lon_c": 170.5},
    "western_canada": {"lat_c": 50.0, "lon_c": -120.0},
}


def main():
    parser = argparse.ArgumentParser(description="Run the ERA5-coupled adjoint experiment")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="experiments", case="run_simulation_adjoint", execution=args.name, output_root=args.output_root
        ).create()
        data_dir, figure_dir = layout.data, layout.figures
    else:
        data_dir = figure_dir = args.output_dir
        data_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. PIPELINE ORCHESTRATION & SETUP
    # ==========================================
    ACTIVE_DOMAIN = "alps"
    cfg = DOMAINS[ACTIVE_DOMAIN]

    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    sponge_depth = 30

    dt = 30.0
    sim_hours = 1.0

    sim_time_seconds = sim_hours * 3600.0
    num_steps = int(sim_time_seconds / dt)
    num_era5_states = int(sim_hours) + 1

    constants = {"g": 9.81, "Rd": 287.0, "cp": 1004.0, "cvd": 717.0, "p0": 100000.0, "epsilon": 0.622}

    USE_MOISTURE = False

    dynamic_bbox = ERA5Manager.calculate_required_bbox(cfg["lat_c"], cfg["lon_c"], nx, ny, dx, dy, buffer_deg=2.0)

    print("-" * 60)
    print(f"[ADJOINT] Initializing Sensitivity Analysis for {ACTIVE_DOMAIN.upper()}...")
    print(f"[CONFIG] ERA5 Bounding Box [N, W, S, E]: {[round(x, 2) for x in dynamic_bbox]}")

    # ==========================================
    # 2. DATA ACQUISITION
    # ==========================================
    manager = ERA5Manager(data_dir="inputs")
    days_to_run = [str(i).zfill(2) for i in range(1, 5)]
    cache_prefix = f"{ACTIVE_DOMAIN}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year="2026", month="05", days=days_to_run, area=dynamic_bbox, prefix=cache_prefix
    )

    # ==========================================
    # 3. GEOMETRY & TOPOGRAPHY
    # ==========================================
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, cfg["lat_c"], cfg["lon_c"])
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path="inputs/gebco_data.nc")
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=2.0)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, cfg["lat_c"], cfg["lon_c"], h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0)
    )

    # ==========================================
    # 4. ERA5 BOUNDARY PROCESSING
    # ==========================================
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0["latitude"], raw_t0["longitude"], constants)

    suetes_bc_states = [bridge.process(era5_proc.get_stitched_state(time_idx=i)) for i in range(num_era5_states)]
    times_sec = [float(i * 3600.0) for i in range(num_era5_states)]
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)

    initial_state = suetes_bc_states[0].copy()

    if USE_MOISTURE:
        initial_state["q_c"] = jnp.zeros_like(initial_state["q"])
    else:
        initial_state.pop("q", None)
        initial_state.pop("q_c", None)

    # Initialize the forcing variables so the PyTree structure matches
    initial_state["theta_surf"] = initial_state["th_v"][:, :, 0]
    initial_state["target_th_v"] = initial_state["th_v"]

    # ==========================================
    # 5. PHYSICS, STEPPER & SPONGE INITIALIZATION
    # ==========================================
    operators = CGridOperator3D(grid)
    physics_suite = PhysicsSuite()

    pbl_scheme = BulkAerodynamicPBL(
        grid, operators, theta_surf=initial_state["th_v"][:, :, 0], Cd_ocean=0.001, Ch_ocean=0.0
    )
    vert_diff_scheme = FastVerticalDiffusion(grid, operators)
    nudging_scheme = NewtonianRelaxation(tau_relax_hours=6.0)

    physics_suite.add_tendency_scheme(pbl_scheme)
    physics_suite.add_tendency_scheme(vert_diff_scheme)
    physics_suite.add_tendency_scheme(nudging_scheme)

    physics = Euler3D(
        grid,
        operators,
        constants,
        dt=dt,
        initial_era5_state=initial_state,
        damp_height=9000.0,
        max_damp=3.0,
        nu_div_factor=0.1,
        nu_h_factor=0.1,
        physics_suite=physics_suite,
    )

    # VRAM-saving mode enabled for adjoint
    stepper = SISLStepper3D(physics, dt, use_checkpointing=True)

    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    # ==========================================
    # 6. ADJOINT SENSITIVITY FORMULATION
    # ==========================================
    def step_wrapper(curr_state, t_curr, bc_state_t):
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)

        return stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)

    checkpointed_step = jax.checkpoint(step_wrapper)

    def compute_forecast_metric(x0_state):
        # Optimal Square Root Chunking (24 chunks of 30 steps = 720 steps)
        chunk_size = 30
        num_chunks = num_steps // chunk_size

        @jax.checkpoint
        def scan_chunk(curr_state, chunk_idx):
            def inner_scan_fn(state, step_offset):
                step_idx = chunk_idx * chunk_size + step_offset
                t_curr = step_idx * dt

                # Sever the Boundary Graph!
                raw_bc = time_manager.get_forcing(t_curr)
                bc_state_t = jax.tree_util.tree_map(jax.lax.stop_gradient, raw_bc)

                # Inject boundaries safely into the PyTree
                state["theta_surf"] = bc_state_t["th_v"][:, :, 0]
                state["target_th_v"] = bc_state_t["th_v"]

                next_state = checkpointed_step(state, t_curr, bc_state_t)

                # Carry the PyTree structure forward to the next step
                next_state["theta_surf"] = state["theta_surf"]
                next_state["target_th_v"] = state["target_th_v"]

                return next_state, None

            chunk_final_state, _ = jax.lax.scan(inner_scan_fn, curr_state, jnp.arange(chunk_size))
            return chunk_final_state, None

        final_state, _ = jax.lax.scan(scan_chunk, x0_state, jnp.arange(num_chunks))

        w_final = final_state["w"]
        mid_x, mid_y = grid.nx // 2, grid.ny // 2
        w_energy = w_final[mid_x - 30 : mid_x + 30, mid_y - 30 : mid_y + 30, 20] ** 2
        return jnp.sum(w_energy)

    print("-" * 60)
    print("[ADJOINT] Compiling and running forward/backward trajectories...")
    start_time = time.time()

    # Run the adjoint
    adjoint_fn = jax.value_and_grad(compute_forecast_metric)
    J_0, sensitivities = adjoint_fn(initial_state)

    # 1. Force JAX to complete BOTH passes before moving on
    J_0 = J_0.block_until_ready()
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), sensitivities)

    print(f"[ADJOINT] Completed in {(time.time() - start_time) / 60:.2f} minutes.")

    # 2. Pull J_0 to a standard Python float
    J_0_cpu = float(J_0)
    print(f"[ADJOINT] Baseline Target Metric (J_0): {J_0_cpu:.4e}")

    # 3. Offload the massive gradient PyTree from GPU to CPU RAM
    sensitivities_cpu = jax.device_get(sensitivities)

    # ==========================================
    # 7. VISUALIZE RESULTS
    # ==========================================
    print("-" * 60)
    print("[PLOT] Generating Adjoint Diagnostic plots...")
    visualizer = Visualizer()
    mid_y = grid.ny // 2

    # Now use sensitivities_cpu instead of sensitivities for all plotting logic!
    delta_th_v = 1.0
    impact_absolute = sensitivities_cpu["th_v"][:, :, 0] * delta_th_v
    impact_percentage = (impact_absolute / J_0_cpu) * 100.0

    # --- 7B. Kinetic Sensitivity Magnitude (Wind Blocking) ---
    # Extracts how sensitive the target is to upstream horizontal winds at the top of the PBL
    du = sensitivities_cpu["u"][:, :, 2]
    dv = sensitivities_cpu["v"][:, :, 2]

    # Average to mass points to compute magnitude safely
    du_m = operators.avg(du, axis=0, from_loc="u", to_loc="m")
    dv_m = operators.avg(dv, axis=1, from_loc="v", to_loc="m")
    wind_sens_mag = jnp.sqrt(du_m**2 + dv_m**2)
    level_indices = np.asarray([2, 10, 18, 26])
    artifact = xr.Dataset(
        data_vars={
            "thermal_impact_percentage": (("x", "y"), np.asarray(impact_percentage)),
            "wind_sensitivity_magnitude": (("x", "y"), np.asarray(wind_sens_mag)),
            "vertical_velocity_sensitivity_levels": (
                ("level", "x", "y"),
                np.stack([np.asarray(sensitivities_cpu["w"][:, :, index]) for index in level_indices]),
            ),
            "thermal_sensitivity_cross_section": (("x", "z"), np.asarray(sensitivities_cpu["th_v"][:, mid_y, :])),
            "vertical_velocity_cross_section": (("x", "z_interface"), np.asarray(sensitivities_cpu["w"][:, mid_y, :])),
        },
        coords={
            "x": np.asarray(grid.x_m),
            "y": np.asarray(grid.y_m),
            "z": np.arange(grid.nz),
            "z_interface": np.arange(grid.nz + 1),
            "level": level_indices,
        },
        attrs={"domain": ACTIVE_DOMAIN, "forecast_hours": sim_hours, "target_metric": J_0_cpu},
    )
    artifact_path = save_plot_dataset(artifact, data_dir / "artifact.nc", experiment="run_simulation_adjoint")
    print(f"[OUTPUT] Saved {artifact_path}")
    if args.no_render:
        return

    visualizer.plot_adjoint_overlay(
        grid,
        initial_state,
        impact_percentage,
        z_idx=5,
        sponge_depth=sponge_depth,
        stride=10,
        save_path=str(figure_dir / "adjoint_impact_percentage.png"),
    )

    visualizer.plot_2d_field(
        grid,
        initial_state,
        variable="pi",
        plot_data=wind_sens_mag,
        z_idx=2,
        sponge_depth=sponge_depth,
        cmap="hot",
        scale="linear",
        title=f"Kinetic Sensitivity Magnitude (Top of PBL)\n{ACTIVE_DOMAIN.upper()} - {sim_hours}h",
        save_path=str(figure_dir / "adjoint_kinetic_sensitivity.png"),
    )

    # --- 7C. Adjoint Level Strips ---
    # Traces the sensitivity 'bullseye' spreading horizontally as it moves downward
    visualizer.plot_level_strip(
        grid,
        sensitivities,
        "w",
        z_indices=[2, 10, 18, 26],
        sponge_depth=sponge_depth,
        save_path=str(figure_dir / "adjoint_vertical_velocity_levels.png"),
    )

    # --- 7D. 3D Cross-Sections (Ray Tracing) ---
    visualizer.plot_cross_section(
        grid,
        sensitivities,
        "th_v",
        y_idx=mid_y,
        sponge_depth=sponge_depth,
        overlay_isentropes=False,
        save_path=str(figure_dir / "adjoint_thermal_cross_section.png"),
    )

    visualizer.plot_cross_section(
        grid,
        sensitivities,
        "w",
        y_idx=mid_y,
        sponge_depth=sponge_depth,
        overlay_isentropes=False,
        save_path=str(figure_dir / "adjoint_vertical_velocity_cross_section.png"),
    )

    print("[PLOT] Adjoint visualizations saved.")


if __name__ == "__main__":
    main()
