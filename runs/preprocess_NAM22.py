"""
NAM22 preprocessing -- sibling of run_simulation_NAM22_radiation.py.

Downloads ALL the inputs the run needs IN PARALLEL -- the per-day ERA5 files
(one .nc per day, integrity-checked and resumable) and the ~8 GB GEBCO
topography, fetched concurrently with `io.download_workers` threads (env
DOWNLOAD_WORKERS overrides) -- then preprocesses them into the chunked,
lazily-streamed Zarr boundary stores that the NAM22 simulation consumes. This is
the single download+build step (no separate downloader). Driven by the SAME
shared config
(configs/nam22_config.yaml) and the same env overrides (SIM_HOURS / KAPPA /
START_DAY / ...) the runner honors, so the stores it writes are exactly what the
runner streams (the runner's own build step then just finds them complete).

GPU-free: ingestion + regridding are light and pinned to CPU.

Usage:
    python runs/preprocess_NAM22.py --config configs/nam22_config.yaml
    SIM_HOURS=168 KAPPA=1.0 python runs/preprocess_NAM22.py --config configs/nam22_config.yaml
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # preprocessing needs no GPU

import cdsapi

from suetes.shared.config import (load_config, resolve_params, build_grid, build_constants,
                                   topography_array)
from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.era2suetes import BoundaryProcessor
from suetes.preprocessing.bc_store import write_timeseries_zarr


def main():
    ap = argparse.ArgumentParser(
        description="Download + preprocess ERA5 into chunked Zarr BC stores for the NAM22 run.")
    ap.add_argument("--config", default="configs/nam22_config.yaml",
                    help="shared config (the same file the runner uses)")
    ap.add_argument("--coarse-only", action="store_true", help="build only the coarse driver store")
    ap.add_argument("--native-only", action="store_true", help="build only the native reference store")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = resolve_params(cfg)
    os.makedirs(p.data_dir, exist_ok=True)     # ERA5 .nc + GEBCO (read)
    os.makedirs(p.store_dir, exist_ok=True)    # Zarr stores (write)
    print(f"[PREPROCESS] {p.cache_prefix} | N={p.num_states} states "
          f"| days {p.days[0]}..{p.days[-1]} | kappa={p.kappa:g}")
    print(f"[PREPROCESS] ERA5 in: {p.data_dir} | stores out: {p.store_dir}")

    # 1) Download the per-DATE ERA5 pairs in parralel (independent CDS requests,
    #    calendar-correct so multi-week/month windows cross month boundaries), and
    #    build the terrain-following grid which downloads the ~8 GB GEBCO
    #    topography and blends it with the ERA5 orography concurrently on its own
    #    thread. The grid build only needs day-0 ERA5, so the big GEBCO download
    #    overlaps the remaining ERA5 day downloads. Days already on disk are skipped
    #    (download_regional_subset integrity-checks partial files), so it's resumable.
    mgr = ERA5Manager(data_dir=p.data_dir, pressure_levels=p.pressure_levels)
    bbox = ERA5Manager.calculate_required_bbox(p.lat_c, p.lon_c, p.nx, p.ny, p.dx, p.dy,
                                               buffer_deg=p.buffer_deg)
    workers = int(os.environ.get("DOWNLOAD_WORKERS", p.download_workers))

    def _dl_era5(ds):                                    # ds = 'YYYYMMDD'
        day_prefix = f"{p.dl_prefix}_{ds}"
        sl = os.path.join(p.data_dir, f"{day_prefix}_single_levels.nc")
        pl = os.path.join(p.data_dir, f"{day_prefix}_pressure_levels.nc")
        if os.path.exists(sl) and os.path.exists(pl):
            return sl, pl
        # per-thread cdsapi client (clients are not safe to share across retrieves)
        cl = cdsapi.Client() if workers > 1 else mgr.client
        return mgr.download_regional_subset(year=ds[:4], month=ds[4:6], days=[ds[6:8]],
                                            area=bbox, prefix=day_prefix, client=cl)

    print(f"[PREPROCESS] downloading {len(p.dates)} ERA5 days + building grid/topo "
          f"with {workers} parallel workers")
    sl_files, pl_files = [None] * len(p.dates), [None] * len(p.dates)
    with ThreadPoolExecutor(max_workers=workers + 1) as ex:
        era5_futs = {ex.submit(_dl_era5, ds): i for i, ds in enumerate(p.dates)}
        day0_fut = next(f for f, i in era5_futs.items() if i == 0)

        def _build_grid_when_ready():
            sl0, _ = day0_fut.result()        # topo blend needs day-0 ERA5 orography
            return build_grid(p, sl0)         # constructs TopographyProcessor: GEBCO + blend

        grid_fut = ex.submit(_build_grid_when_ready)
        for fut in as_completed(era5_futs):
            i = era5_futs[fut]
            sl_files[i], pl_files[i] = fut.result()                 # order-preserving
        grid = grid_fut.result()
    print(f"[PREPROCESS] {len(p.dates)} ERA5 days {p.dates[0]}..{p.dates[-1]} + grid/topo ready")

    # 2) Boundary processor (lazy multi-file ERA5 read).
    era5 = ERA5Processor(pl_path=pl_files, sl_path=sl_files)
    raw0 = era5.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw0["latitude"], raw0["longitude"], build_constants(cfg))

    # Static inputs saved into the coarse store so the runner rebuilds the grid +
    # surface fields WITHOUT any ERA5/GEBCO access or regridding.
    import numpy as np
    static = {"h": np.asarray(topography_array(grid)),
              "land_fraction": np.asarray(bridge.process_static(raw0)["land_fraction"])}

    # 3) Write the chunked Zarr stores, exactly what the runner streams.
    times = [i * 3600.0 for i in range(p.num_states)]
    prefix = os.path.join(p.store_dir, p.cache_prefix)
    coarse_path = f"{prefix}_cw{p.coarsen_window}_coarse.zarr"
    native_path = f"{prefix}_native.zarr"
    if not args.native_only:
        write_timeseries_zarr(coarse_path, era5, bridge, p.num_states, times,
                              coarsen_window=p.coarsen_window, static=static)
    if not args.coarse_only:
        write_timeseries_zarr(native_path, era5, bridge, p.num_states, times,
                              coarsen_window=None, static=static)

    print("[PREPROCESS] done. Stores ready for the NAM22 run:")
    if not args.native_only:
        print(f"  coarse: {coarse_path}")
    if not args.coarse_only:
        print(f"  native: {native_path}")


if __name__ == "__main__":
    main()
