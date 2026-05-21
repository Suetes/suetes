"""
Perfect-model lateral-boundary-condition test for Suetes.

Drives the Suetes regional core with output from a dinosaur global J-W run
(see ``run_dinosaur_jw.py``). 

Configuration:
  - Flat earth (no orography). J-W is a flat-bottom test.
  - All-ocean (land-sea mask is zero everywhere in the dinosaur output).
  - Dry (USE_MOISTURE = False). J-W is a dry test.
  - Dinosaur output replaces the ERA5 download/processing path entirely.

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time

import jax
import jax.numpy as jnp
import numpy as np

from suetes.preprocessing.dinosaur2suetes import DinosaurProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager

from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.physics import (
    PhysicsSuite, NewtonianRelaxation,
    McFarlaneVerticalDiffusion, McFarlaneSurfaceDrag,
    SmagorinskyLillySGS,
)

from suetes.vis.visualizer import Visualizer


class SimulationBlowupError(RuntimeError):
    """Raised by the debug callback when a NaN/Inf or runaway value is detected."""
    pass


def main():
    # ==========================================
    # 1. CONFIG
    # ==========================================
    lat_c, lon_c = 45.0, 195.0
    RUN_NAME = "jw_pm_test_diffusive"
    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 160, 120, 30
    dx, dy, dz = 50000.0, 50000.0, 500.0
    sponge_depth = 10

    dt = 100.0
    start_hour = 215
    sim_hours = 24   # stop just before the known blowup window (~step 908)
    chunk_steps = 60

    SPONGE_NU_MAX = 0.15

    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
        'p0': 100000.0, 'epsilon': 0.622,
    }

    USE_MOISTURE = False

    DRIVER_NC = "data/test.nc"

    print(f"[CONFIG] Perfect-model J-W test, sub-domain center "
          f"({lat_c}N, {lon_c}E)")
    print(f"[CONFIG] Davies two-term relaxation: Newtonian blend + "
          f"diffusive perturbation (nu_max={SPONGE_NU_MAX})")

    y_half_km = (ny / 2.0) * (dy / 1000.0)
    x_half_km = (nx / 2.0) * (dx / 1000.0)
    R_earth_km = 6371.229
    deg_north = np.degrees(2.0 * np.arctan(y_half_km / (2.0 * R_earth_km)))
    print(f"[GEOM] Half-widths: x={x_half_km:.0f} km, y={y_half_km:.0f} km")
    print(f"[GEOM] North edge ~{lat_c + deg_north:.1f} N, "
          f"south edge ~{lat_c - deg_north:.1f} N")
    print(f"[GEOM] Model top at z = {nz * dz / 1000.0:.1f} km")

    # ==========================================
    # 2. DRIVER (dinosaur reference)
    # ==========================================
    print(f"[DRIVER] Loading dinosaur reference from {DRIVER_NC}")
    era5_proc = DinosaurProcessor(DRIVER_NC)

    # ==========================================
    # 3. GEOMETRY (flat earth)
    # ==========================================
    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} mesh "
          f"(dx={dx/1000:.0f} km), FLAT earth")
    flat_h_func = lambda x, y: jnp.zeros_like(x)
    sleve_transform = SleveSimple(scale_s=10000.0, n=1.0)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_c, lon_c,
        h_func=flat_h_func, transform=sleve_transform,
    )

    # ==========================================
    # 4. BOUNDARY
    # ==========================================
    n_available = era5_proc.ds.sizes['time']
    if start_hour + sim_hours + 1 > n_available:
        sim_hours = n_available - 1 - start_hour
        print(f"[BOUNDARY] Driver has {n_available} hourly states; "
              f"truncating sim_hours to {sim_hours} given start_hour={start_hour}")
    num_states = sim_hours + 1
    print(f"[BOUNDARY] Building {num_states} hourly boundary states "
          f"covering dinosaur T={start_hour}-{start_hour + sim_hours}h")

    raw_t0 = era5_proc.get_stitched_state(time_idx=start_hour)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'],
                               constants)

    static_fields = bridge.process_static(raw_t0)
    land_fraction = static_fields['land_fraction']

    z0_ocean, z0_land = 1e-4, 0.1
    epsilon_ocean, epsilon_land = 0.3, 0.0
    z_0_field = land_fraction * z0_land + (1.0 - land_fraction) * z0_ocean
    epsilon_field = (
        land_fraction * epsilon_land
        + (1.0 - land_fraction) * epsilon_ocean
    )

    suetes_bc_states = []
    times_sec = []
    for i in range(num_states):
        if i % 24 == 0:
            print(f"  -> Regridding dinosaur state for "
                  f"sim_T={i}h (dinosaur T={start_hour + i}h)")
        raw_state = era5_proc.get_stitched_state(time_idx=start_hour + i)
        bc_state_jax = bridge.process(raw_state)
        bc_state = jax.tree_util.tree_map(np.asarray, bc_state_jax)
        suetes_bc_states.append(bc_state)
        times_sec.append(float(i * 3600.0))
        del bc_state_jax

    time_manager = TimeManager(suetes_bc_states, times_sec, grid)
    initial_state = {k: jnp.asarray(v) for k, v in suetes_bc_states[0].items()}

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

    # ==========================================
    # 5. PHYSICS / STEPPER / SPONGE
    # ==========================================
    print(f"[DYNAMICS] Initializing core")
    operators = CGridOperator3D(grid)
    physics_suite = PhysicsSuite()

    sgs_scheme = SmagorinskyLillySGS(
        grid, operators, constants,
        Cs=0.20, Pr_t=1.0, critical_Ri=0.25,
    )
    #physics_suite.add_tendency_scheme(sgs_scheme)

    pbl_scheme = McFarlaneSurfaceDrag(
        grid, operators, constants,
        z_0=z_0_field, epsilon=epsilon_field,
        theta_surf=initial_state['theta_surf'],
    )
    #physics_suite.add_tendency_scheme(pbl_scheme)

    vert_diff_scheme = McFarlaneVerticalDiffusion(
        grid, operators, constants,
        epsilon=epsilon_field[..., None],
    )
    #physics_suite.add_tendency_scheme(vert_diff_scheme)

    nudging_scheme = NewtonianRelaxation(tau_relax_hours=6.0)
    #physics_suite.add_tendency_scheme(nudging_scheme)

    physics = Euler3D(
        grid, operators, constants, dt=dt,
        initial_era5_state=initial_state,
        damp_height=10000.0, max_damp=3.0,
        nu_div_factor=0.1, nu_h_factor=0.1,
        physics_suite=physics_suite,
    )
    stepper = SISLStepper3D(physics, dt)
    sponge = DaviesSponge(
        grid, operators, sponge_depth=sponge_depth, dt=dt,
        tau_bndy_factor=10.0,
    )

    # ==========================================
    # 6. PREFLIGHT DIAGNOSTICS
    # ==========================================
    print("-" * 60)

    def _state_stats(name, state):
        print(f"[PREFLIGHT] {name}:")
        for key in sorted(state.keys()):
            val = state[key]
            if hasattr(val, "shape") and getattr(val, "size", 0) > 1:
                v = np.asarray(val)
                nan_count = int(np.isnan(v).sum())
                inf_count = int(np.isinf(v).sum())
                flag = ""
                if nan_count > 0:
                    flag = f"  !!! {nan_count} NaN"
                elif inf_count > 0:
                    flag = f"  !!! {inf_count} Inf"
                print(f"  {key:18s} shape={str(v.shape):16s} "
                      f"min={float(np.nanmin(v)):12.4g}  "
                      f"max={float(np.nanmax(v)):12.4g}  "
                      f"mean={float(np.nanmean(v)):12.4g}{flag}")

    _state_stats("Initial state", initial_state)

    print("[PREFLIGHT] Running one diagnostic step (no JIT)...")
    bc0 = {k: jnp.asarray(v) for k, v in suetes_bc_states[0].items()}
    def _preflight_bc_fn(state_next, _):
        diffused = sponge.diffuse_perturbation(state_next, bc0, nu_max=SPONGE_NU_MAX)
        return sponge.blend(diffused, bc0)
    trial_state = stepper.step(initial_state, 0.0, forcing=None,
                               bc_fn=_preflight_bc_fn)
    _state_stats("State after 1 step", trial_state)
    if any(bool(jnp.any(jnp.isnan(jnp.asarray(v))))
           for v in trial_state.values()
           if hasattr(v, "shape") and getattr(v, "size", 0) > 1):
        print("[PREFLIGHT] !!! NaN appeared after one step. ABORTING.")
        return
    del trial_state, bc0
    print("-" * 60)

    # ==========================================
    # 7. INTEGRATION
    # ==========================================
    DBG_EVERY = 10
    RUNAWAY_THRESHOLD = 1.0e6
    _u_shape = initial_state['u'].shape

    def _dbg_print(step, t, mu, mv, mw, mth, mpi,
                   u_argmax_flat, u_at_max, bc_u_at_max):
        maxes = {
            '|u|': float(mu),  '|v|':  float(mv),
            '|w|': float(mw),  '|th|': float(mth),
            '|pi|': float(mpi),
        }
        bad = [k for k, val in maxes.items()
               if (not np.isfinite(val)) or abs(val) > RUNAWAY_THRESHOLD]
        if bad:
            i, j, k = np.unravel_index(int(u_argmax_flat), _u_shape)
            print(f"\n[KILL] step={int(step)} t={float(t)/60.0:.1f}min: "
                  f"non-finite or runaway values in {bad}")
            print(f"[KILL] all maxes: " +
                  ", ".join(f"{kk}={vv:.4g}" for kk, vv in maxes.items()))
            print(f"[KILL] u argmax at (i={i}, j={j}, k={k})")
            raise SimulationBlowupError(
                f"Blowup at step {int(step)} "
                f"(t={float(t)/60.0:.1f} min): {bad}"
            )

        if int(step) % DBG_EVERY != 0:
            return
        i, j, k = np.unravel_index(int(u_argmax_flat), _u_shape)
        in_x = (i < sponge_depth) or (i >= _u_shape[0] - sponge_depth)
        in_y = (j < sponge_depth) or (j >= _u_shape[1] - sponge_depth)
        if   in_x and in_y: where = "CORNER  "
        elif in_x:          where = "X-SPNG  "
        elif in_y:          where = "Y-SPNG  "
        else:               where = "INTERIOR"
        print(f"    [DBG step={int(step):5d} t={float(t)/60.0:7.1f}min] "
              f"|u|={float(mu):7.2f}@(i={i:3d},j={j:3d},k={k:3d})/{where} "
              f"u={float(u_at_max):+7.2f} bc_u={float(bc_u_at_max):+7.2f} | "
              f"|v|={float(mv):6.2f} |w|={float(mw):.4f} "
              f"|th|={float(mth):6.1f} |pi|={float(mpi):.6f}")

    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        bc_state_t = time_manager.get_forcing(t_curr)
        curr_state['theta_surf'] = (
            land_fraction * bc_state_t['theta_skt']
            + (1.0 - land_fraction) * bc_state_t['th_v'][:, :, 0]
        )
        curr_state['target_th_v'] = bc_state_t['th_v']

        def bc_fn(state_next, _):
            diffused = sponge.diffuse_perturbation(
                state_next, bc_state_t, nu_max=SPONGE_NU_MAX,
            )
            return sponge.blend(diffused, bc_state_t)

        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        next_state['theta_surf'] = curr_state['theta_surf']
        next_state['target_th_v'] = curr_state['target_th_v']

        abs_u = jnp.abs(next_state['u'])
        max_u = jnp.max(abs_u)
        u_argmax = jnp.argmax(abs_u)
        u_at_max = next_state['u'].ravel()[u_argmax]
        bc_u_at_max = bc_state_t['u'].ravel()[u_argmax]

        max_v  = jnp.max(jnp.abs(next_state['v']))
        max_w  = jnp.max(jnp.abs(next_state['w']))
        max_th = jnp.max(jnp.abs(next_state['th_v']))
        max_pi = jnp.max(jnp.abs(next_state['pi']))

        jax.debug.callback(_dbg_print, step_idx, t_curr,
                           max_u, max_v, max_w, max_th, max_pi,
                           u_argmax, u_at_max, bc_u_at_max)
        return next_state, max_w

    sim = Simulation(step_fn=step_fn, dt=dt)
    t0 = time.time()
    try:
        final_state = sim.run(
            initial_state, t_start=0.0, t_end=sim_hours * 3600.0,
            chunk_steps=chunk_steps,
        )
    except SimulationBlowupError as e:
        print(f"[SIMULATION] aborted after {time.time() - t0:.1f}s: {e}")
        return
    print(f"[SIMULATION] done in {time.time() - t0:.1f}s")
    _state_stats("Final state", final_state)

    # ==========================================
    # 8. DIAGNOSTICS
    # ==========================================
    print("-" * 60)
    print("[PLOT] Producing dashboards, spectrum, comparison")
    visualizer = Visualizer()
    final_bc = {k: jnp.asarray(v) for k, v in suetes_bc_states[-1].items()}

    # Model dashboards at multiple levels. Level 0 is the surface; levels 5,
    # 15, 25 are progressively higher in the column (k=5 ~ 2.5 km, k=15 ~
    # 7.5 km, k=25 ~ 12.5 km).
    for z in [0, 5, 15, 25]:
        visualizer.plot_dashboard(
            grid, final_state, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_dash_z{z}_{sim_hours}h.png"
            ),
        )

    # BC dashboards at the same levels for direct comparison. Level 0 will
    # show zero vertical velocity by construction (BoundaryProcessor zeros
    # the kinematic bottom over flat terrain). Levels 5, 15, 25 should show
    # the new omega-diagnosed w field.
    for z in [0, 5, 15, 25]:
        visualizer.plot_dashboard(
            grid, final_bc, z_idx=z, sponge_depth=sponge_depth,
            time_hours=sim_hours,
            save_path=os.path.join(
                output_dir, f"{RUN_NAME}_BC_dash_z{z}_{sim_hours}h.png"
            ),
        )

    visualizer.plot_energy_spectrum(
        grid, final_state, 'w', z_idx=5, sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_energy_{sim_hours}h.png"
        ),
    )
    visualizer.plot_comparison(
        grid, final_state, final_bc, 'th_v', z_idx=0,
        sponge_depth=sponge_depth,
        save_path=os.path.join(
            output_dir, f"{RUN_NAME}_compare_th_v_{sim_hours}h.png"
        ),
    )


if __name__ == "__main__":
    main()
