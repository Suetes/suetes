#!/usr/bin/env python3
"""Unified, configuration-driven preprocessing script.

Downloads all required inputs (ERA5 + GEBCO) and builds chunked Zarr stores
for boundary and initial conditions based on the provided configuration.

Usage:
    python runs/preprocess.py --config configs/nam22_config.yaml
    python runs/preprocess.py --config configs/wreckhouse25_config.yaml
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
        description="Download + preprocess ERA5 into chunked Zarr BC stores for a simulation run.")
    ap.add_argument("--config", required=True,
                    help="Path to configuration YAML file.")
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

    # 1) Download the per-DATE ERA5 pairs in parallel and build the terrain-following grid
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
        # per-thread cdsapi client
        cl = cdsapi.Client() if workers > 1 else mgr.client
        return mgr.download_regional_subset(year=ds[:4], month=ds[4:6], days=[ds[6:8]],
                                             area=bbox, prefix=day_prefix, client=cl)

    print(f"[PREPROCESS] downloading {len(p.dates)} ERA5 days with {workers} parallel workers")
    sl_files, pl_files = [None] * len(p.dates), [None] * len(p.dates)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        era5_futs = {ex.submit(_dl_era5, ds): i for i, ds in enumerate(p.dates)}
        for fut in as_completed(era5_futs):
            i = era5_futs[fut]
            sl_files[i], pl_files[i] = fut.result()                 # order-preserving

    print("[PREPROCESS] building grid/topo...")
    grid = build_grid(p, sl_files[0])
    print(f"[PREPROCESS] {len(p.dates)} ERA5 days {p.dates[0]}..{p.dates[-1]} + grid/topo ready")

    # 2) Boundary processor (lazy multi-file ERA5 read).
    era5 = ERA5Processor(pl_path=pl_files, sl_path=sl_files)
    raw0 = era5.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw0["latitude"], raw0["longitude"], build_constants(cfg))

    # Static inputs saved into the coarse store
    import numpy as np
    static = {"h": np.asarray(topography_array(grid)),
              "land_fraction": np.asarray(bridge.process_static(raw0)["land_fraction"])}

    # 3) Write the chunked Zarr stores
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

    print(f"[PREPROCESS] done. Stores ready for config {args.config}:")
    if not args.native_only:
        print(f"  coarse: {coarse_path}")
    if not args.coarse_only:
        print(f"  native: {native_path}")


if __name__ == "__main__":
    main()
