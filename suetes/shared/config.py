"""
Shared, single-source configuration for suetes.

A small dataclass layer over a YAML file so preprocessing and simulation read the
*same* grid, time, vertical-coordinate, physics, and I/O settings instead of each
script hardcoding its own constants. Load with :func:`load_config`; build the
common derived objects (vertical transform, constants dict, cache prefix) with
the helpers below so the CLI and the runner stay consistent.
"""

from dataclasses import dataclass, field, asdict
import jax
import yaml


@dataclass
class DomainConfig:
    name: str = "nam22"
    lat_c: float = 47.5
    lon_c: float = -97.0
    nx: int = 310
    ny: int = 260
    nz: int = 32
    dx: float = 22000.0
    dy: float = 22000.0
    dz: float = 500.0


@dataclass
class TimeConfig:
    year: str = "2025"
    month: str = "07"
    days: list = field(default_factory=lambda: ["18", "19", "20"])
    sim_hours: int = 42

    @property
    def num_states(self) -> int:
        return self.sim_hours + 1


@dataclass
class VerticalConfig:
    # kappa > 1 -> StretchedSleveSimple (near-surface stretching); else SleveSimple.
    kappa: float = 1.0
    scale_s: float = 10000.0
    n: float = 1.0


@dataclass
class CoreConfig:
    type: str = "split-explicit"   # "split-explicit" | "sisl"
    dt: float = 40.0
    nu_h: float = 0.03
    ns: int = 3                     # acoustic substeps (split-explicit)
    alpha: float = 0.7             # off-centering (sisl); auto 0.7 on stretched grids
    precision: str = "float32"      # "float32" | "float64"



@dataclass
class PhysicsConfig:
    radiation: bool = True
    nudge: bool = True
    microphysics: bool = True      # Kessler warm rain
    convection: bool = True        # Betts-Miller adjustment
    rh_crit: float = 0.80
    rad_coarse: int = 8
    rad_every_h: float = 1.0
    afgl: bool = True
    drag: bool = True              # McFarlane surface drag
    diffusion: bool = True         # McFarlane vertical diffusion
    sgs: bool = False              # Smagorinsky-Lilly SGS turbulence
    gwd: bool = False              # McFarlane gravity wave drag



@dataclass
class ConstantsConfig:
    g: float = 9.81
    Rd: float = 287.0
    cp: float = 1004.0
    cvd: float = 717.0
    p0: float = 100000.0
    epsilon: float = 0.622


@dataclass
class IOConfig:
    data_dir: str = "output/data"    # ERA5 .nc + GEBCO (download output == preprocess input)
    store_dir: str = ""              # Zarr stores (preprocess output == runner input); "" -> data_dir
    output_dir: str = ""             # simulation NetCDF output; "" -> "output/simulations"
    fig_dir: str = "output/plots/radiation"
    store_format: str = "zarr"     # "zarr" (chunked, lazy) | "pkl" (legacy monolithic)
    coarsen_window: int = 3
    sponge_depth: int = 15
    smooth_sigma: float = 2.0
    pressure_levels: str = "buffered"
    buffer_deg: float = 3.0
    download_workers: int = 3        # parallel per-day CDS requests (CDS caps concurrency; keep small)


@dataclass
class RenderConfig:
    levels_z: list = field(default_factory=lambda: [0, 5, 15, 30])
    levels_m: list = field(default_factory=lambda: [500.0, 3000.0, 5000.0, 10000.0])
    energy_levels: list = field(default_factory=lambda: [5])
    energy_var: str = "w"
    compare_var: str = "th_v"
    compare_levels_z: list = field(default_factory=lambda: [0])
    strip_var: str = "th_v"
    strip_levels_z: list = field(default_factory=lambda: [0, 5, 15, 30])
    hovmoller_var: str = "th_v"
    points: list = field(default_factory=list)
    slices: list = field(default_factory=list)
    target_hours: list = field(default_factory=list)
    zoom_extent: list = field(default_factory=list)
    quiver_stride: int = 10
    dashboard_hours: list = field(default_factory=list)
    compare_hours: list = field(default_factory=list)
    energy_hours: list = field(default_factory=list)
    strip_hours: list = field(default_factory=list)
    hovmoller_hours: list = field(default_factory=list)



@dataclass
class SuetesConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    vertical: VerticalConfig = field(default_factory=VerticalConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    constants: ConstantsConfig = field(default_factory=ConstantsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def as_dict(self):
        return asdict(self)


def load_config(path) -> SuetesConfig:
    """Load a SuetesConfig from a YAML file (missing sections/keys use defaults)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    def section(cls, key):
        data = raw.get(key) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown keys in '{key}': {sorted(unknown)}")
        return cls(**data)

    cfg = SuetesConfig(
        domain=section(DomainConfig, "domain"),
        time=section(TimeConfig, "time"),
        vertical=section(VerticalConfig, "vertical"),
        core=section(CoreConfig, "core"),
        physics=section(PhysicsConfig, "physics"),
        constants=section(ConstantsConfig, "constants"),
        io=section(IOConfig, "io"),
        render=section(RenderConfig, "render"),
    )

    jax.config.update("jax_enable_x64", cfg.core.precision == "float64")

    return cfg


# --- derived-object helpers (shared by the preprocessing CLI and the runner) ---

def build_constants(cfg: SuetesConfig) -> dict:
    """The physical-constants dict consumed by BoundaryProcessor / physics."""
    c = asdict(cfg.constants)
    c["rh_crit"] = cfg.physics.rh_crit
    
    # Precision-dependent safety factors to prevent division by zero or negative square root arguments
    if cfg.core.precision == "float64":
        c["eps"] = 1e-15
        c["eps_l"] = 1e-12
        c["eps_s"] = 1e-8
    else:
        c["eps"] = 1e-7
        c["eps_l"] = 1e-5
        c["eps_s"] = 1e-5
        
    return c


def build_transform(cfg: SuetesConfig):
    """Vertical coordinate transform (Stretched if kappa>1, else plain SLEVE)."""
    from suetes.shared.transforms import SleveSimple, StretchedSleveSimple
    v = cfg.vertical
    if v.kappa > 1.0:
        return StretchedSleveSimple(stretch_kappa=v.kappa, scale_s=v.scale_s, n=v.n)
    return SleveSimple(scale_s=v.scale_s, n=v.n)


def coord_tag(cfg: SuetesConfig) -> str:
    """Cache-key suffix encoding the vertical coordinate (matches the runner)."""
    return f"_k{cfg.vertical.kappa:g}" if cfg.vertical.kappa > 1.0 else ""


def cache_prefix(cfg: SuetesConfig) -> str:
    """Store/cache filename prefix, e.g. nam22_20250718_Nx310_Ny260_dx22000_n43."""
    d, t = cfg.domain, cfg.time
    day0 = str(t.days[0]).zfill(2)
    return (f"{d.name}_{t.year}{t.month}{day0}_Nx{d.nx}_Ny{d.ny}"
            f"_dx{int(d.dx)}_n{t.num_states}{coord_tag(cfg)}")


def resolve_params(cfg: SuetesConfig):
    """Effective run parameters: config values with environment-variable overrides.

    The preprocessing script and the simulation runner BOTH call this so they
    derive identical grid/time/store keys from one config (and honor the same
    env knobs, e.g. SIM_HOURS / KAPPA / START_DAY), guaranteeing the preprocessed
    Zarr stores are exactly what the runner streams.
    """
    import os
    import math
    from types import SimpleNamespace

    d, t, v, io, ph, co = cfg.domain, cfg.time, cfg.vertical, cfg.io, cfg.physics, cfg.core

    def _flag(env, default):
        # env var (when present) overrides the config boolean; "0" -> False.
        return (os.environ[env] != "1") if env in os.environ else default

    sim_hours = int(os.environ["SIM_HOURS"]) if "SIM_HOURS" in os.environ else t.sim_hours
    num_states = sim_hours + 1
    kappa = float(os.environ.get("KAPPA", v.kappa))
    start_day = int(os.environ.get("START_DAY", str(t.days[0])))
    ndays = max(1, math.ceil(num_states / 24))
    import datetime as _dt
    _start = _dt.date(int(t.year), int(t.month), start_day)
    dates = [(_start + _dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(ndays)]
    days = [ds[6:8] for ds in dates]
    ctag = f"_k{kappa:g}" if kappa > 1.0 else ""
    core_type = os.environ.get("CORE_TYPE", co.type)

    p = SimpleNamespace(
        domain=d.name, lat_c=d.lat_c, lon_c=d.lon_c,
        nx=d.nx, ny=d.ny, nz=d.nz, dx=d.dx, dy=d.dy, dz=d.dz,
        sponge_depth=io.sponge_depth, smooth_sigma=io.smooth_sigma,
        coarsen_window=io.coarsen_window, pressure_levels=io.pressure_levels,
        buffer_deg=io.buffer_deg, data_dir=io.data_dir,
        store_dir=(io.store_dir or io.data_dir),
        output_dir=(io.output_dir or "output/simulations"),
        download_workers=int(os.environ["DOWNLOAD_WORKERS"]) if "DOWNLOAD_WORKERS" in os.environ else io.download_workers,
        year=t.year, month=t.month, start_day=start_day, days=days, dates=dates,
        sim_hours=sim_hours, num_states=num_states,
        kappa=kappa, coord_tag=ctag,
        core_type=core_type, dt=float(os.environ.get("DT", co.dt)),
        nu_h=float(os.environ.get("NU_H", co.nu_h)),
        ns=int(os.environ["NS"]) if "NS" in os.environ else co.ns,
        alpha=float(os.environ.get("ALPHA", co.alpha)),
        scale_s=float(v.scale_s), n_sleve=float(v.n),
        rh_crit=float(os.environ.get("RH_CRIT", ph.rh_crit)),
        rad_coarse=int(os.environ["RAD_COARSE"]) if "RAD_COARSE" in os.environ else ph.rad_coarse,
        rad_every_h=float(os.environ.get("RAD_EVERY_H", ph.rad_every_h)),
        afgl=((os.environ["AFGL"] == "1") if "AFGL" in os.environ else ph.afgl),
        use_rad=_flag("NO_RAD", ph.radiation), use_nudge=_flag("NO_NUDGE", ph.nudge),
        use_micro=_flag("NO_MICRO", ph.microphysics), use_conv=_flag("NO_CONV", ph.convection),
        use_drag=_flag("NO_DRAG", ph.drag), use_diffusion=_flag("NO_DIFFUSION", ph.diffusion),
        use_sgs=_flag("NO_SGS", ph.sgs) if ph.sgs else ((os.environ.get("USE_SGS") == "1") if "USE_SGS" in os.environ else ph.sgs),
        use_gwd=_flag("NO_GWD", ph.gwd) if ph.gwd else ((os.environ.get("USE_GWD") == "1") if "USE_GWD" in os.environ else ph.gwd),
        render=(lambda: [
            copy_rc := __import__('copy').deepcopy(cfg.render),
            setattr(copy_rc, 'levels_z', [min(d.nz - 1, z) for z in copy_rc.levels_z] if copy_rc.levels_z else []),
            setattr(copy_rc, 'compare_levels_z', [min(d.nz - 1, z) for z in copy_rc.compare_levels_z] if copy_rc.compare_levels_z else []),
            setattr(copy_rc, 'strip_levels_z', [min(d.nz - 1, z) for z in copy_rc.strip_levels_z] if copy_rc.strip_levels_z else []),
            setattr(copy_rc, 'energy_levels', [min(d.nz - 1, z) for z in copy_rc.energy_levels] if copy_rc.energy_levels else []),
        ][-1] or copy_rc)(),
        eps=(1e-15 if co.precision == "float64" else 1e-7),
    )
    # ERA5 download filename keyed by the GEOGRAPHIC domain ONLY (centre + physical
    # extent Lx=nx*dx, Ly=ny*dy + buffer + pressure preset) -- NOT the model
    # resolution and NOT the config/domain name. The raw ERA5 subset is identical for
    # any nx/dx split (e.g. nam22 vs nam11) or any config name over the same geographic
    # box, so every such run reuses the one download instead of re-fetching. The neutral
    # "era5_" prefix makes that sharing explicit (no domain token to diverge on).
    p.dl_prefix = (f"era5_{p.year}{p.month}_c{p.lat_c:g}_{p.lon_c:g}"
                   f"_Lx{int(round(p.nx * p.dx / 1000))}_Ly{int(round(p.ny * p.dy / 1000))}"
                   f"_b{p.buffer_deg:g}_{p.pressure_levels}")
    p.cache_prefix = (f"{p.domain}_{p.year}{p.month}{p.days[0]}_Nx{p.nx}_Ny{p.ny}"
                      f"_dx{int(p.dx)}_n{p.num_states}{p.coord_tag}")
    return p


def build_grid(p, sl_file):
    """Build the terrain-following RegionalGrid3D from resolved params + ERA5 orography.

    ``p`` is the namespace returned by :func:`resolve_params`. ``sl_file`` is any
    ERA5 single-levels file (orography is time-invariant). Used by both the
    preprocessing CLI and (optionally) the runner so the grid is identical.
    """
    import os
    from suetes.regional3d.geometry import RegionalGrid3D
    from suetes.preprocessing.topography import TopographyProcessor
    from suetes.shared.transforms import SleveSimple, StretchedSleveSimple

    base = RegionalGrid3D(p.nx, p.ny, p.nz, p.dx, p.dy, p.dz, p.lat_c, p.lon_c, eps=p.eps)
    topo = TopographyProcessor(era5_sl_path=sl_file,
                               gebco_path=os.path.join(p.data_dir, "gebco_data.nc"))
    h_func = topo.process_and_blend(base, sponge_depth=p.sponge_depth, smooth_sigma=p.smooth_sigma)
    return RegionalGrid3D(p.nx, p.ny, p.nz, p.dx, p.dy, p.dz, p.lat_c, p.lon_c,
                          h_func=h_func, transform=_transform_from_params(p), eps=p.eps)


def _transform_from_params(p):
    from suetes.shared.transforms import SleveSimple, StretchedSleveSimple
    if p.kappa > 1.0:
        return StretchedSleveSimple(stretch_kappa=p.kappa, scale_s=p.scale_s, n=p.n_sleve)
    return SleveSimple(scale_s=p.scale_s, n=p.n_sleve)


def topography_array(grid):
    """Blended topography height h(x,y) sampled on the grid mass points (nx,ny).

    Saved by preprocessing so the simulation can rebuild the grid without ERA5/GEBCO.
    """
    import jax.numpy as jnp
    Xi, Yi = jnp.meshgrid(jnp.asarray(grid.x_m), jnp.asarray(grid.y_m), indexing="ij")
    return jnp.asarray(grid.h_func(Xi, Yi))


def build_grid_from_static(p, h_array):
    """Rebuild the terrain-following grid from resolved params + a SAVED topography
    array -- no ERA5/GEBCO, no regridding (used by the pure-execution runner)."""
    import jax.numpy as jnp
    import jax.scipy.ndimage as jnd
    from suetes.regional3d.geometry import RegionalGrid3D

    base = RegionalGrid3D(p.nx, p.ny, p.nz, p.dx, p.dy, p.dz, p.lat_c, p.lon_c, eps=p.eps)
    H = jnp.asarray(h_array)
    x0, y0, dx, dy = float(base.x_m[0]), float(base.y_m[0]), base.dx, base.dy

    def h_func(x, y):   # same form as TopographyProcessor's continuous h_func
        return jnd.map_coordinates(H, [(x - x0) / dx, (y - y0) / dy], order=1, mode="nearest")

    return RegionalGrid3D(p.nx, p.ny, p.nz, p.dx, p.dy, p.dz, p.lat_c, p.lon_c,
                          h_func=h_func, transform=_transform_from_params(p), eps=p.eps)
