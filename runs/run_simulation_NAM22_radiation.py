"""CORDEX NAM-22 forward run WITH RRTMGP radiation + Sundqvist diagnostic clouds.

PURE EXECUTION ENGINE: this script ONLY runs the simulation and writes the output
NetCDF files (native projected `_native.nc` + regular-lat/lon `_output.nc`, plus a
final-state npz). It does NOT produce figures. Render those separately from the
output with `runs/render_nam22.py`. Preprocessing/ERA5 ingestion is likewise a
separate step (`runs/preprocess_NAM22.py`); both are driven by the same config.

When radiation is on, FORWARD MODEL ONLY (no gradients through the RRTMPG module yet). Fully moist: water-vapour `q` is kept
from ERA5, advected as a tracer, and nudged at the lateral boundary; a Sundqvist
diagnostic cloud feeds RRTMGP longwave+shortwave heating, applied as a th_v
tendency alongside the standard McFarlane surface drag + Newtonian relaxation.

Configuration: --config configs/nam22_config.yaml (see suetes/shared/config.py).
Env vars override individual knobs (resolve_params): SIM_HOURS, KAPPA, START_DAY,
DT, CORE_TYPE, RH_CRIT, RAD_COARSE, NO_RAD/NO_NUDGE/NO_MICRO/NO_CONV, X64,
RAD_REDUCED, RAD_SCAN, RAD_EVERY_H, NU_H, NS, ALPHA, AFGL, TAG, DIAG_RAD.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
# Disable CUDA command-buffers: across the many per-step calls the cached CUDA graphs
# accumulate and OOM at CUBIN load ("Failed to load in-memory CUBIN").
# Must be set before any JAX import.
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import argparse
import time
import numpy as np
import jax
# Precision env-knob: X64=1 (default) -> float64; X64=0 -> float32 (forward runs,
# halves GPU memory). Must be set before any jnp array is created.
jax.config.update("jax_enable_x64", os.environ.get("X64", "1") == "0")
import jax.numpy as jnp

from suetes.preprocessing.bc_store import LazyZarrTimeManager, read_state, read_static
from suetes.shared.driver import Simulation
from suetes.shared.output import SimulationOutputWriter, DailyGroupedWriter
from suetes.shared.config import load_config, resolve_params, build_grid_from_static
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.regional3d.boundaries import DaviesSponge
from suetes.physics.base import PhysicsSuite
from suetes.physics.forcing import NewtonianRelaxation
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.microphysics import KesslerWarmRain, SimplifiedBettsMiller
from suetes.physics.radiation import SundqvistCloud, RRTMGPRadiation, CachedRadiation
from suetes.physics.afgl import AFGLColumnExtension

DATA_DIR = "inputs"          # default; overridden per-run by the config's data_dir
OUTDIR = os.environ.get("SUETES_OUT_DIR", "output/simulations")   # NetCDF output dir (redirectable to another disk)


def main():
    # Single shared config (same file the preprocessing CLI uses); env vars still
    # override individual knobs via resolve_params.
    ap = argparse.ArgumentParser(description="NAM22 radiation forward run (config-driven).")
    ap.add_argument('--config', default='configs/nam22_config.yaml')
    cli_args, _ = ap.parse_known_args()
    cfg = load_config(cli_args.config)
    p = resolve_params(cfg)

    ACTIVE_DOMAIN = p.domain
    lat_c, lon_c = p.lat_c, p.lon_c
    # ERA5 is downloaded on the full domain; DL_* == grid here.
    DL_NX, DL_NY = p.nx, p.ny
    nx, ny, nz = p.nx, p.ny, p.nz
    dx, dy, dz = p.dx, p.dy, p.dz
    sponge_depth, smooth_sigma, coarsen_window = p.sponge_depth, p.smooth_sigma, p.coarsen_window

    core_type = p.core_type
    dt = p.dt
    sim_hours = p.sim_hours
    rh_crit = p.rh_crit
    reduced = os.environ.get('RAD_REDUCED', '1') == '1'
    use_rad, use_nudge, use_micro = p.use_rad, p.use_nudge, p.use_micro
    use_conv = use_micro and p.use_conv     # Betts-Miller only with microphysics
    sim_time_seconds = sim_hours * 3600.0
    num_era5_states = p.num_states
    RUN_NAME = (f"{ACTIVE_DOMAIN}_radiation_{core_type}_n{nx}x{ny}_dt{int(dt)}_{sim_hours}h"
                f"_{'rad' if use_rad else 'norad'}_{'nudge' if use_nudge else 'nonudge'}"
                f"_{'kessler' if use_micro else 'sundq'}")
    if os.environ.get('TAG'):                    # extra experiment label, avoids overwriting
        RUN_NAME += f"_{os.environ['TAG']}"
    # Each run gets its own output subfolder (NetCDF + final-state npz). Figures
    # are produced separately by runs/render_nam22.py.
    out_base = os.environ.get("SUETES_OUT_DIR") or p.output_dir   # env overrides config
    out_run_dir = os.path.join(out_base, RUN_NAME)
    os.makedirs(out_run_dir, exist_ok=True)

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0,
                 'epsilon': 0.622, 'rh_crit': rh_crit}
    # Day window (auto-sized to run length) resolved from config + env.
    YEAR, MONTH = p.year, p.month
    start_day = p.start_day
    DAYS = p.days
    doy0 = float((np.datetime64(f"{YEAR}-{MONTH}-{DAYS[0]}") - np.datetime64(f"{YEAR}-01-01")).astype(int) + 1)

    print(f"[CONFIG] {RUN_NAME} | moist | radiation={'ON' if use_rad else 'OFF'} "
          f"| reduced_tables={reduced} | core={core_type} dt={dt:g} doy0={doy0:.0f}")

    kappa = p.kappa
    times_sec = [i * 3600.0 for i in range(num_era5_states)]
    coarse_store = os.path.join(p.store_dir, f"{p.cache_prefix}_cw{coarsen_window}_coarse.zarr")
    if not os.path.exists(coarse_store):
        raise SystemExit(
            f"[ERROR] boundary store not found:\n    {coarse_store}\n"
            f"This script ONLY runs the simulation. Build the inputs first with:\n"
            f"    python runs/preprocess_NAM22.py --config {cli_args.config}")

    # PURE EXECUTION ENGINE: rebuild the grid + surface fields from the PRE-BUILT
    # store (topography 'h' + 'land_fraction' were saved by preprocessing). No
    # ERA5/GEBCO, no download, no regridding -- just load and run.
    print("[GEOMETRY] rebuilding grid from stored topography")
    static = read_static(coarse_store)
    if 'h' not in static or 'land_fraction' not in static:
        raise SystemExit(
            f"[ERROR] store {coarse_store} lacks static fields (h/land_fraction).\n"
            f"It predates the decoupled preprocessing -- rebuild it:\n"
            f"    python runs/preprocess_NAM22.py --config {cli_args.config}")
    grid = build_grid_from_static(p, static['h'])
    land_fraction = jnp.asarray(static['land_fraction'], dtype=float)
    _dzc = np.asarray(grid.dz_m_full[nx // 2, ny // 2, :])
    print(f"[GRID] kappa={kappa} | layer thickness bottom {_dzc[0]:.0f} m -> top {_dzc[-1]:.0f} m | "
          f"min in domain {float(jnp.min(grid.dz_m_full)):.0f} m")

    time_manager = LazyZarrTimeManager(coarse_store, times_sec, grid)
    assert 'q' in time_manager.keys, "BC store lacks humidity 'q' -- ERA5 ingestion problem"

    # ---- MOIST initial state: KEEP q (advected tracer), init cloud water to 0 ----
    initial_state = dict(read_state(coarse_store, 0))
    initial_state['theta_surf'] = land_fraction * initial_state['theta_skt'] + (1.0 - land_fraction) * initial_state['th_v'][:, :, 0]
    initial_state['target_th_v'] = initial_state['th_v']
    initial_state['q_c'] = jnp.zeros_like(initial_state['q'])
    if use_micro:
        # Prognostic Kessler rain + per-step surface precip + run accumulation [mm]
        initial_state['q_r'] = jnp.zeros_like(initial_state['q'])
        initial_state['precip_step'] = jnp.zeros(initial_state['q'].shape[:2])
        initial_state['rain_acc'] = jnp.zeros(initial_state['q'].shape[:2])
    if use_conv:
        initial_state['precip_conv_step'] = jnp.zeros(initial_state['q'].shape[:2])
        initial_state['conv_acc'] = jnp.zeros(initial_state['q'].shape[:2])
    initial_state.pop('theta_skt', None)
    for _k in ('cc', 'clwc', 'ciwc', 'tcc', 'land_fraction'):  # ERA5 cloud is diagnostic-only (read from BC), not prognostic
        initial_state.pop(_k, None)
    initial_state = jax.tree.map(lambda x: jnp.asarray(x, dtype=float), initial_state)

    z0 = land_fraction * 0.1 + (1.0 - land_fraction) * 1e-4
    eps = land_fraction * 0.0 + (1.0 - land_fraction) * 0.3

    operators = CGridOperator3D(grid)
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)
    interior_mask = sponge.get_interior_mask()

    # ---- physics: surface drag + nudging + (optionally) radiation ----
    sundqvist = SundqvistCloud(constants, rh_crit=rh_crit)
    suite = PhysicsSuite()
    suite.add_tendency_scheme(McFarlaneSurfaceDrag(grid, operators, constants, z_0=z0, epsilon=eps, theta_surf=initial_state['theta_surf']))
    # McFarlaneVerticalDiffusion REMOVED for the stretched grid: it is EXPLICIT and its
    # K*dt/dz^2 CFL is violated by the fine near-surface dz (would go unstable). Surface
    # drag + radiation + nudging remain. (A dz-aware K cap could restore it later.)
    if use_nudge:
        suite.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))
    rad = None
    if use_rad:
        rad_coarse = p.rad_coarse
        print(f"[RAD] spatial coarsening: RRTMGP on every {rad_coarse} column "
              f"(~{rad_coarse**2}x fewer columns), heating interpolated back")
        # AFGL climatological extension above the lid (AFGL=0 disables). p_lid sits
        # 10% below the lowest top-level pressure so the seam stays p-monotonic in
        # every column; July run -> mid-latitude summer atmosphere.
        afgl = None
        if p.afgl:                       # config physics.afgl (env AFGL overrides)
            p_top_min = float(jnp.min(constants['p0'] * initial_state['pi'][:, :, -1]
                                      ** (constants['cp'] / constants['Rd'])))
            afgl = AFGLColumnExtension(model='midlatitude_summer', n_ext=12,
                                       p_lid_hPa=0.9 * p_top_min / 100.0, p_top_hPa=1.0)
            print(f"[RAD] AFGL extension: 12 layers, lid p_min={p_top_min/100:.1f} hPa "
                  f"-> ext {0.9*p_top_min/100:.1f}..1.0 hPa (midlatitude_summer)")
        # RAD_SCAN=1 -> lax.scan over g-points (fewer compiled kernels / less driver
        # memory, slightly slower) — useful at the full domain where CUBIN loads OOM.
        # 16x16 coarse-grid tiles: per-tile sun angle resolves the E-W hour angle
        # and N-S latitude to ~tile scale, and true night tiles get exactly zero
        # SW via the stock jax-rrtmgp night guard (no library modifications).
        rad = RRTMGPRadiation(grid, constants, sundqvist, doy0=doy0, reduced=reduced,
                              use_scan=os.environ.get('RAD_SCAN', '0') == '1',
                              save_lw_sw=True, rad_coarse=rad_coarse,
                              tile_nx=int(os.environ.get('RAD_TILE_NX', 16)),
                              tile_ny=int(os.environ.get('RAD_TILE_NY', 16)),
                              afgl=afgl, use_state_condensate=use_micro)
        # Radiation is sub-cycled at the configured hourly cadence in the run
        # loop; this cheap scheme applies cached heating each model step.
        suite.add_tendency_scheme(CachedRadiation())
        initial_state['rad_th_tend'] = jnp.zeros_like(initial_state['th_v'])
        initial_state['sw_sfc'] = jnp.zeros_like(initial_state['th_v'][:, :, 0])
    if use_conv:
        # Betts-Miller convective adjustment FIRST in the update sequence: it
        # vents subgrid moisture (condenses below grid-box saturation toward
        # RH_ref) before the grid-scale Kessler path — the missing vapor sink
        # behind the run-long RH/cloud build-up.
        suite.add_update_scheme(SimplifiedBettsMiller(constants, dt=dt, grid=grid,
                                                      tau_adj=7200.0, rh_ref=0.85))
        print("[CONV] Betts-Miller adjustment ON (tau=2h, rh_ref=0.85, before Kessler)")
    if use_micro:
        # Kessler OWNS condensation and the prognostic q_c/q_r (saturation
        # adjustment + warm rain + sedimentation + rain evaporation, with latent
        # heating). Sundqvist must NOT also run as an update scheme: it would
        # overwrite the prognostic q_c every step (last-writer-wins in
        # apply_state_updates) and double-count condensation. It survives only
        # as the cloud-FRACTION diagnostic inside RRTMGPRadiation/figures.
        kessler = KesslerWarmRain(constants, dt=dt, grid=grid)
        suite.add_update_scheme(kessler)
        suite.register_tracer('q_c')        # prognostic cloud water: advected
        suite.register_tracer('q_r')        # prognostic rain water: advected
        print(f"[MICRO] Kessler warm rain ON (prognostic q_c/q_r, sedimentation, "
              f"nfall={kessler.nfall} substeps)")
    else:
        suite.add_update_scheme(sundqvist)  # diagnoses q_c each step (diagnostic mode)
    suite.register_tracer('q')              # advect water vapour

    if use_rad and os.environ.get('DIAG_RAD'):
        import sys
        c = constants; pi = initial_state['pi']; q = initial_state['q']
        p = c['p0'] * pi ** (c['cp'] / c['Rd'])
        T = (initial_state['th_v'] * pi) / (1.0 + (1.0 / c['epsilon'] - 1.0) * q)
        print(f"[DIAG] pi[{float(jnp.min(pi)):.4f},{float(jnp.max(pi)):.4f}] "
              f"p[Pa][{float(jnp.min(p)):.0f},{float(jnp.max(p)):.0f}] "
              f"T[K][{float(jnp.min(T)):.1f},{float(jnp.max(T)):.1f}] "
              f"q[g/kg][{float(jnp.min(q))*1e3:.2f},{float(jnp.max(q))*1e3:.2f}] "
              f"finite: pi={bool(jnp.all(jnp.isfinite(pi)))} p={bool(jnp.all(jnp.isfinite(p)))} T={bool(jnp.all(jnp.isfinite(T)))}")
        Q, diags = rad.heating_and_diagnostics(initial_state, ml_params={'t_curr': 0.0})
        Qd = np.asarray(Q)
        ok = np.all(np.isfinite(Qd))
        print(f"[DIAG] radiation heating finite={ok} | n_nonfinite={int(np.sum(~np.isfinite(Qd)))} "
              f"| K/day[{np.nanmin(Qd)*86400:.2f},{np.nanmax(Qd)*86400:.2f}]")
        for nm in ('rad_heat_lw_3d', 'rad_heat_sw_3d'):
            if nm in diags:
                a = np.asarray(diags[nm])
                print(f"[DIAG]   {nm}: finite={np.all(np.isfinite(a))} K/day[{np.nanmin(a)*86400:.2f},{np.nanmax(a)*86400:.2f}]")
        if not ok:
            b = np.argwhere(~np.isfinite(Qd))[0]
            print(f"[DIAG]   first non-finite at (i,j,k)={tuple(int(x) for x in b)}, T-col={np.asarray(T)[b[0],b[1]].round(1)}")
        sys.exit(0)

    # Horizontal hyperdiffusion/divergence damping stay ON: with the explicit vertical
    # diffusion removed the model needs its scale-selective dissipation (dx-based CFL,
    # so NOT constrained by the stretched vertical grid). Rayleigh upper sponge stays.
    # Defaults per core: 0.1 (SISL @ dt=120) / 0.03 (split-explicit @ dt=40).
    nu_h = p.nu_h          # config core.nu_h (env NU_H overrides)
    core_kwargs = dict(dt=dt, nu_div_factor=nu_h, nu_h_factor=nu_h)
    if core_type == 'split-explicit':
        core_kwargs['ns'] = p.ns
        core_info = f"ns={core_kwargs['ns']} acoustic substeps"
    else:
        # Off-centering (SISL only): config core.alpha (env ALPHA overrides).
        core_kwargs['alpha'] = p.alpha
        core_info = f"alpha={core_kwargs['alpha']:g}"
    stepper, dt = build_dynamical_core(core_type, grid, operators, constants, initial_state,
                                       physics_suite=suite, interior_mask=interior_mask,
                                       **core_kwargs)
    print(f"[CORE] {core_type} | dt={dt:g}s | {core_info}")

    chunk_steps = int(round(3600.0 / dt))
    _SAVE = ('u', 'v', 'w', 'th_v', 'pi', 'q', 'q_c', 'q_r', 'sw_sfc', 'rain_acc', 'conv_acc')
    # Diagnostic cloud fields are derived per hour and saved alongside the
    # prognostics: cloud_fraction (3D, cloud at each model level) + total cloud
    # cover (2D, max-random overlap).
    # Plus hourly precip RATE (mm/hour): the per-snapshot increment of the run
    # accumulators (rain_acc/conv_acc are kept for run totals). 'precip' is the
    # grid-scale (Kessler) hourly amount, 'precip_conv' the convective (Betts-Miller).
    _SAVE_OUT = _SAVE + ('cloud_fraction', 'total_cloud_cover', 'precip', 'precip_conv', 'rad_th_tend')
    _SAVE_RUN = _SAVE + ('rad_th_tend',)     # fields pulled from the live state each hour
    # SINGLE lossless output on the NATIVE curvilinear grid (CF-1.7, 2D lat/lon +
    # grid_mapping), written as DAILY files split into prognostic vs physics groups:
    #   <run>_prog_<YYYYMMDD>.nc  (u,v,w,th_v,pi,q,q_c,q_r)
    #   <run>_phys_<YYYYMMDD>.nc  (sw_sfc,rain_acc,conv_acc,precip,precip_conv,clouds,rad_th_tend)
    # No regridding (lossless). View with cartopy/Panoply (or ncview).
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
    _prev_acc = {}     # last-saved run accumulators, for the hourly increment
    _hour = [0]        # snapshot index -> day/hour routing

    def _save_with_clouds(d):
        cf, tcc = _diagnose_clouds(d, constants, rh_crit)
        d['cloud_fraction'] = cf
        d['total_cloud_cover'] = tcc
        # Hourly precip = increment of the run accumulator since the last save.
        for acc_key, rate_key in (('rain_acc', 'precip'), ('conv_acc', 'precip_conv')):
            if acc_key in d:
                cur = np.asarray(d[acc_key])
                prev = _prev_acc.get(acc_key)
                d[rate_key] = cur - prev if prev is not None else np.zeros_like(cur)
                _prev_acc[acc_key] = cur
        writer.write(_hour[0], d)
        _hour[0] += 1

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
        for k in _SAVE:                                               # prog + sw_sfc/rain_acc/conv_acc
            if k in dpr.variables:
                a = np.asarray(dpr.variables[k][kt])                  # stored (z,y,x); model wants (x,y,z)
                seed[k] = jnp.asarray(np.transpose(a, (2, 1, 0)) if a.ndim == 3 else np.transpose(a, (1, 0)))
        dpr.close()
        physf = last.replace("_prog_", "_phys_")
        if os.path.exists(physf):
            dph = _nc.Dataset(physf, "r")
            for k in ("rain_acc", "conv_acc", "sw_sfc"):
                if k in dph.variables:
                    seed[k] = jnp.asarray(np.transpose(np.asarray(dph.variables[k][kt]), (1, 0)))
            dph.close()
        seed["rho"] = constants["p0"] / (constants["Rd"] * seed["th_v"]) * seed["pi"] ** (constants["cvd"] / constants["Rd"])
        seed["eta_dot"] = jnp.zeros((nx, ny, nz + 1))
        initial_state = {**initial_state, **seed}                     # keep rad_th_tend/theta_surf/etc.
        d0 = _dt.date(int(YEAR), int(MONTH), int(start_day))
        dd = _dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
        start_hour = (dd - d0).days * 24 + kt
        _hour[0] = start_hour + 1
        _prev_acc["rain_acc"] = np.asarray(initial_state.get("rain_acc", np.zeros((nx, ny))))
        _prev_acc["conv_acc"] = np.asarray(initial_state.get("conv_acc", np.zeros((nx, ny))))
        print(f"[RESUME] seeded from {os.path.basename(last)} idx {kt} = global hour {start_hour} ({ymd}) -> {sim_hours}h")
    else:
        _save_with_clouds({k: np.asarray(v) for k, v in initial_state.items() if k in _SAVE_RUN})

    # ---- per-CHUNK forcing: fetch the bounding ERA5 pair ONCE per hour (host) and
    #      interpolate on-device every step. Removes the per-step pure_callback that
    #      capped GPU utilization; radiation + hourly save are also hoisted out of the
    #      jitted scan. ----
    import functools as _ft
    rad_every_h = max(1, int(round(p.rad_every_h)))   # hours between full RRTMGP calls
    if use_rad:
        print(f"[RAD] sub-cycling: full RRTMGP every {rad_every_h} h; "
              "cached heating applied each model step")

    @_ft.partial(jax.jit, static_argnames=["n_steps"])
    def run_hour(st_in, start_step, n_steps, bc0, bc1, ta, tb):
        def body(s, off):
            t = (start_step + off) * dt
            a = jnp.clip((t - ta) / (tb - ta), 0.0, 1.0)
            bc = {k: (1.0 - a) * bc0[k] + a * bc1[k] for k in bc0}   # on-device interp
            thsurf = land_fraction * bc["theta_skt"] + (1.0 - land_fraction) * bc["th_v"][:, :, 0]
            s["theta_surf"] = thsurf
            s["target_th_v"] = bc["th_v"]
            nxt = stepper.step(s, t, forcing=None, bc_fn=lambda x, _: sponge.blend(x, bc),
                               ml_params={"t_curr": t})
            nxt["theta_surf"] = thsurf
            nxt["target_th_v"] = bc["th_v"]
            if use_rad:
                nxt["rad_th_tend"] = s["rad_th_tend"]; nxt["sw_sfc"] = s["sw_sfc"]
            if use_micro:
                ps = nxt.get("precip_step", jnp.zeros_like(s["rain_acc"]))
                nxt["precip_step"] = ps; nxt["rain_acc"] = s["rain_acc"] + ps
            if use_conv:
                pc = nxt.get("precip_conv_step", jnp.zeros_like(s["conv_acc"]))
                nxt["precip_conv_step"] = pc; nxt["conv_acc"] = s["conv_acc"] + pc
            return nxt, jnp.max(jnp.abs(nxt["w"]))
        return jax.lax.scan(body, st_in, jnp.arange(n_steps))

    print(f"[RUN] {sim_hours} h forward from hour {start_hour} "
          f"({int(sim_time_seconds/dt)} steps @ dt={dt:g}s)...")
    _timing = os.environ.get("TIMING", "0") == "1"
    state = initial_state
    t0 = time.time()
    for h in range(start_hour, sim_hours):
        ta0 = time.time()
        s0, s1, ta, tb = time_manager.bounding_pair(h * 3600.0)       # ONE host fetch / hour
        bc0 = {k: jnp.asarray(v) for k, v in s0.items()}
        bc1 = {k: jnp.asarray(v) for k, v in s1.items()}
        if use_rad and (h % rad_every_h == 0):
            state["rad_th_tend"], state["sw_sfc"] = rad.tend_and_sw(state, ml_params={"t_curr": h * 3600.0})
        tc = time.time()
        state, metrics = run_hour(state, h * chunk_steps, chunk_steps, bc0, bc1, ta, tb)
        jax.block_until_ready(state["w"])
        ts = time.time()
        _save_with_clouds({k: np.asarray(state[k]) for k in _SAVE_RUN if k in state})
        if _timing:
            print(f"    Progress: {(h+1)*3600.0:.0f}s | Max W: {float(jnp.max(jnp.abs(metrics))):.4f} | "
                  f"fetch+rad {tc-ta0:.2f}s | gpu {ts-tc:.2f}s | save {time.time()-ts:.2f}s | "
                  f"total {time.time()-ta0:.2f}s", flush=True)
        else:
            print(f"    Progress: {(h+1)*3600.0:.0f}s / {sim_time_seconds:.0f}s | "
                  f"Max W: {float(jnp.max(jnp.abs(metrics))):.4f} m/s | Chunk Wall: {time.time()-tc:.2f}s", flush=True)
    print(f"[RUN] done in {time.time()-t0:.1f}s")

    # Finalize the daily output files (one prog + one phys per simulated day).
    writer.close()
    print(f"[DONE] daily prognostic/physics NetCDF output in {out_run_dir}")
    print(f"[NEXT] render figures: python runs/render_nam22.py --rundir {out_run_dir}")



def _tcc_maxrandom(C):
    """Total cloud cover under maximum-random overlap (ECMWF convention):
    adjacent cloudy layers overlap maximally (same cloud), separated layers
    randomly. Random overlap across many thin levels badly inflates tcc."""
    # float64 + a float32-safe cap: float32(1-1e-9)==1.0, which makes den=1-Cc=0
    # for fully-cloudy columns -> NaN. 1e-6 is representable in float32.
    Cc = np.clip(np.asarray(C, dtype=np.float64), 0.0, 1.0 - 1e-6)
    num = 1.0 - np.maximum(Cc[:, :, 1:], Cc[:, :, :-1])
    den = 1.0 - Cc[:, :, :-1]
    return 1.0 - (1.0 - Cc[:, :, 0]) * np.prod(num / den, axis=2)


def _diagnose_clouds(s, constants, rh_crit):
    """NumPy Sundqvist diagnosis for output (matches physics.radiation.SundqvistCloud).

    Returns (cloud_fraction [nx,ny,nz], total_cloud_cover [nx,ny]) as float32.
    cloud_fraction is the cloud at each model level; total cover is max-random
    overlap. Computed host-side from the saved (th_v, pi, q, q_c, q_r) so it adds
    no work to the GPU step.
    """
    c = constants
    pi = np.clip(np.asarray(s['pi']), 1e-3, None)
    q = np.asarray(s['q'])
    qc = np.asarray(s.get('q_c', np.zeros_like(q)))
    qr = np.asarray(s.get('q_r', np.zeros_like(q)))
    p = c['p0'] * pi ** (c['cp'] / c['Rd'])
    T = (np.asarray(s['th_v']) * pi) / (1.0 + (1.0 / c['epsilon'] - 1.0) * q - qc - qr)
    eps = c['epsilon']
    es = np.minimum(610.94 * np.exp(17.625 * (T - 273.15) / (T - 30.11)), 0.9 * p)   # Tetens
    qs = eps * es / (p - (1.0 - eps) * es)
    rh = q / np.clip(qs, 1e-9, None)
    b = np.clip((rh - rh_crit) / (1.0 - rh_crit), 0.0, 1.0)
    C = (1.0 - np.sqrt(1.0 - b)) * (p > 1.5e4)        # tropospheric clouds only
    return C.astype(np.float32), _tcc_maxrandom(C).astype(np.float32)


if __name__ == "__main__":
    main()
