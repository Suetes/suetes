"""CORDEX NAM-22 forward run -- DRY baseline (no moisture, no radiation).

PURE EXECUTION ENGINE: this script ONLY runs the simulation and writes the output
NetCDF files. It does NOT download ERA5, does NOT preprocess, and does NOT produce
figures. The boundary/initial inputs are the chunked Zarr store built separately by
`runs/preprocess_NAM22.py`; render figures separately from the output with
`runs/render_nam22.py`. All three stages are driven by the SAME shared config.

This is the non-radiation sibling of `run_simulation_NAM22_radiation.py`: identical
config -> run-only -> daily-NetCDF-output architecture, but the dry standard physics
set (McFarlane surface drag + McFarlane vertical diffusion + Newtonian relaxation)
and NO dependency on the RRTMGP radiation modules. Water vapour `q` from ERA5 is
dropped; the model integrates the dry dynamical core only.

Configuration: --config configs/nam22_config.yaml (see suetes/shared/config.py).
Env vars override individual knobs (resolve_params): SIM_HOURS, KAPPA, START_DAY,
DT, CORE_TYPE, NO_NUDGE, X64, NU_H, NS, ALPHA, TAG.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false') # Sometimes having this be true messes with the compilation.
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import argparse
import functools as _ft
import time
import numpy as np
import jax
# run in F32 here
jax.config.update("jax_enable_x64", os.environ.get("X64", "1") == "0")
import jax.numpy as jnp

from suetes.preprocessing.bc_store import LazyZarrTimeManager, read_state, read_static
from suetes.shared.output import SimulationOutputWriter, DailyGroupedWriter
from suetes.shared.config import load_config, resolve_params, build_grid_from_static
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.regional3d.boundaries import DaviesSponge
from suetes.physics.base import PhysicsSuite
from suetes.physics.forcing import NewtonianRelaxation
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.turbulence import McFarlaneVerticalDiffusion

# Prognostics saved each hour (dry: no q/q_c/q_r, no physics-group fields). These
# fall in the writer's "prog" group; rho/eta_dot are reconstructed on RESUME so they
# need not be stored.
_SAVE = ('u', 'v', 'w', 'th_v', 'pi')


def main():
    # Single shared config (the same file the preprocessing CLI uses); env vars
    # still override individual knobs via resolve_params.
    ap = argparse.ArgumentParser(description="NAM22 dry forward run (config-driven).")
    ap.add_argument('--config', default='configs/nam22_config.yaml')
    cli_args, _ = ap.parse_known_args()
    cfg = load_config(cli_args.config)
    p = resolve_params(cfg)

    ACTIVE_DOMAIN = p.domain
    lat_c, lon_c = p.lat_c, p.lon_c
    nx, ny, nz = p.nx, p.ny, p.nz
    dx, dy, dz = p.dx, p.dy, p.dz
    sponge_depth, coarsen_window = p.sponge_depth, p.coarsen_window

    core_type = p.core_type
    dt = p.dt
    sim_hours = p.sim_hours
    use_nudge = p.use_nudge
    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = p.num_states
    kappa = p.kappa

    RUN_NAME = (f"{ACTIVE_DOMAIN}_dry_{core_type}_n{nx}x{ny}_dt{int(dt)}_{sim_hours}h"
                f"_{'nudge' if use_nudge else 'nonudge'}")
    if os.environ.get('TAG'):                    # extra experiment label, avoids overwriting
        RUN_NAME += f"_{os.environ['TAG']}"
    out_base = os.environ.get("SUETES_OUT_DIR") or p.output_dir   # env overrides config
    out_run_dir = os.path.join(out_base, RUN_NAME)
    os.makedirs(out_run_dir, exist_ok=True)

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
                 'p0': 100000.0, 'epsilon': 0.622}
    # Day window (auto-sized to run length) resolved from config + env.
    YEAR, MONTH = p.year, p.month
    start_day = p.start_day
    DAYS = p.days

    print(f"[CONFIG] {RUN_NAME} | dry | nudge={'ON' if use_nudge else 'OFF'} "
          f"| core={core_type} dt={dt:g}")

    times_sec = [i * 3600.0 for i in range(num_era5_states)]
    coarse_store = os.path.join(p.store_dir, f"{p.cache_prefix}_cw{coarsen_window}_coarse.zarr")
    if not os.path.exists(coarse_store):
        raise SystemExit(
            f"[ERROR] boundary store not found:\n    {coarse_store}\n"
            f"This script ONLY runs the simulation. Build the inputs first with:\n"
            f"    python runs/preprocess_NAM22.py --config {cli_args.config}")

    print("[GEOMETRY] rebuilding grid from stored topography")
    static = read_static(coarse_store)
    if 'h' not in static or 'land_fraction' not in static:
        raise SystemExit(
            f"[ERROR] store {coarse_store} lacks static fields (h/land_fraction).\n"
            f"It predates the decoupled preprocessing -- rebuild it:\n"
            f"    python runs/preprocess_NAM22.py --config {cli_args.config}")
    grid = build_grid_from_static(p, static['h'])
    land_fraction = jnp.asarray(static['land_fraction'], dtype=jnp.float64)
    _dzc = np.asarray(grid.dz_m_full[nx // 2, ny // 2, :])
    print(f"[GRID] kappa={kappa} | layer thickness bottom {_dzc[0]:.0f} m -> top {_dzc[-1]:.0f} m | "
          f"min in domain {float(jnp.min(grid.dz_m_full)):.0f} m")

    time_manager = LazyZarrTimeManager(coarse_store, times_sec, grid)

    # ---- DRY initial state: keep the skin temperature for surface fluxes, drop all
    #      moisture/cloud fields coming from ERA5 (this is a dry dynamical run). ----
    initial_state = dict(read_state(coarse_store, 0))
    initial_state['theta_surf'] = initial_state['theta_skt']   # skin temp drives the fluxes
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state.pop('theta_skt', None)
    for _k in ('q', 'q_c', 'q_r', 'cc', 'clwc', 'ciwc', 'tcc', 'land_fraction'):
        initial_state.pop(_k, None)
    initial_state = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float64), initial_state)

    z0 = land_fraction * 0.1 + (1.0 - land_fraction) * 1e-4
    eps = land_fraction * 0.0 + (1.0 - land_fraction) * 0.3

    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    interior_mask = sponge.get_interior_mask()

    # ---- physics: surface drag + vertical diffusion + (optional) nudging ----
    suite = PhysicsSuite()
    suite.add_tendency_scheme(McFarlaneSurfaceDrag(grid, operators, constants, z_0=z0, epsilon=eps,
                                                   theta_surf=initial_state['theta_surf']))
    suite.add_tendency_scheme(McFarlaneVerticalDiffusion(grid, operators, constants,
                                                         epsilon=eps[..., None]))
    if use_nudge:
        suite.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))

    nu_h = p.nu_h
    core_kwargs = dict(dt=dt, nu_div_factor=nu_h, nu_h_factor=nu_h)
    if core_type == 'split-explicit':
        core_kwargs['ns'] = p.ns
        core_info = f"ns={core_kwargs['ns']} acoustic substeps"
    else:
        core_kwargs['alpha'] = p.alpha
        core_info = f"alpha={core_kwargs['alpha']:g}"
    stepper, dt = build_dynamical_core(core_type, grid, operators, constants, initial_state,
                                       physics_suite=suite, interior_mask=interior_mask,
                                       **core_kwargs)
    print(f"[CORE] {core_type} | dt={dt:g}s | {core_info}")

    chunk_steps = int(round(3600.0 / dt))

    _out_attrs = {'run_name': RUN_NAME, 'dt_seconds': float(dt),
                  'sim_hours': int(sim_hours), 'kappa': float(kappa),
                  'nx': int(nx), 'ny': int(ny), 'nz': int(nz),
                  'dx': float(dx), 'dy': float(dy), 'dz': float(dz),
                  'lat_c': float(lat_c), 'lon_c': float(lon_c),
                  'coarsen_window': int(coarsen_window), 'sponge_depth': int(sponge_depth),
                  'domain': ACTIVE_DOMAIN, 'year': YEAR, 'month': MONTH, 'day0': DAYS[0]}

    def _make_writer(path, keys, date):
        return SimulationOutputWriter(path, grid, keys, ref_date=date, attrs=_out_attrs)

    writer = DailyGroupedWriter(out_run_dir, RUN_NAME, _make_writer,
                                year=YEAR, month=MONTH, start_day=start_day)
    _hour = [0]        # snapshot index -> day/hour routing

    def _save(d):
        writer.write(_hour[0], d)
        _hour[0] += 1

    def save_snap(sub):
        _save({k: np.asarray(v) for k, v in sub.items()})

    # ---- start fresh, or RESUME from the last written daily output (RESUME=1) ----
    start_hour = 0
    if os.environ.get("RESUME", "0") == "1":
        import glob as _glob, datetime as _dt, netCDF4 as _nc
        progs = sorted(_glob.glob(os.path.join(out_run_dir, f"{RUN_NAME}_prog_*.nc")))
        if not progs:
            raise SystemExit(f"[RESUME] no prog files to resume from in {out_run_dir}")
        last = progs[-1]
        ymd = last.rsplit("_prog_", 1)[1].split(".")[0]
        dpr = _nc.Dataset(last, "r")
        kt = dpr.variables["time"].shape[0] - 1                       # last saved hour-in-day
        seed = {}
        for k in _SAVE:                                               # stored (z,y,x); model wants (x,y,z)
            if k in dpr.variables:
                a = np.asarray(dpr.variables[k][kt])
                seed[k] = jnp.asarray(np.transpose(a, (2, 1, 0)) if a.ndim == 3 else np.transpose(a, (1, 0)))
        dpr.close()
        seed["rho"] = constants["p0"] / (constants["Rd"] * seed["th_v"]) * seed["pi"] ** (constants["cvd"] / constants["Rd"])
        seed["eta_dot"] = jnp.zeros((nx, ny, nz + 1))
        initial_state = {**initial_state, **seed}                     # keep theta_surf/target_th_v
        d0 = _dt.date(int(YEAR), int(MONTH), int(start_day))
        dd = _dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
        start_hour = (dd - d0).days * 24 + kt
        _hour[0] = start_hour + 1
        print(f"[RESUME] seeded from {os.path.basename(last)} idx {kt} = global hour {start_hour} ({ymd}) -> {sim_hours}h")
    else:
        save_snap({k: v for k, v in initial_state.items() if k in _SAVE})

    @_ft.partial(jax.jit, static_argnames=["n_steps"])
    def run_hour(st_in, start_step, n_steps, bc0, bc1, ta, tb):
        def body(s, off):
            t = (start_step + off) * dt
            a = jnp.clip((t - ta) / (tb - ta), 0.0, 1.0)
            bc = {k: (1.0 - a) * bc0[k] + a * bc1[k] for k in bc0}   # on-device interp
            s["theta_surf"] = bc["theta_skt"]                        # skin temp drives fluxes
            s["target_th_v"] = bc["th_v"]
            nxt = stepper.step(s, t, forcing=None, bc_fn=lambda x, _: sponge.blend(x, bc))
            nxt["theta_surf"] = bc["theta_skt"]
            nxt["target_th_v"] = bc["th_v"]
            return nxt, jnp.max(jnp.abs(nxt["w"]))
        return jax.lax.scan(body, st_in, jnp.arange(n_steps))

    print(f"[RUN] {sim_hours} h forward from hour {start_hour} "
          f"({int(sim_time_seconds/dt)} steps @ dt={dt:g}s)...")
    state = initial_state
    t0 = time.time()
    for h in range(start_hour, sim_hours):
        s0, s1, ta, tb = time_manager.bounding_pair(h * 3600.0)       # ONE host fetch / hour
        bc0 = {k: jnp.asarray(v) for k, v in s0.items()}
        bc1 = {k: jnp.asarray(v) for k, v in s1.items()}
        tc = time.time()
        state, metrics = run_hour(state, h * chunk_steps, chunk_steps, bc0, bc1, ta, tb)
        jax.block_until_ready(state["w"])
        save_snap({k: state[k] for k in _SAVE if k in state})
        print(f"    Progress: {(h+1)*3600.0:.0f}s / {sim_time_seconds:.0f}s | "
              f"Max W: {float(jnp.max(jnp.abs(metrics))):.4f} m/s | Chunk Wall: {time.time()-tc:.2f}s",
              flush=True)
    print(f"[RUN] done in {time.time()-t0:.1f}s")

    # Finalize the daily output files (one prog file per simulated day).
    writer.close()
    print(f"[DONE] daily prognostic NetCDF output in {out_run_dir}")


if __name__ == "__main__":
    main()
