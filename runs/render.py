#!/usr/bin/env python3
"""Unified, configuration-driven rendering/plotting script.

Reads simulation NetCDF outputs and generates diagnostic plots according to
the `render:` configuration block in the YAML file.

Usage:
    python runs/render.py --config configs/nam22_config.yaml
    python runs/render.py --config configs/wreckhouse25_config.yaml
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Plotting does not need GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use('Agg')


import argparse
import numpy as np

from suetes.preprocessing.bc_store import read_static, read_state, read_columns
from suetes.shared.output import DailyNetcdfSnapshots
from suetes.shared.config import load_config, resolve_params, build_grid_from_static
from suetes.vis.visualizer import Visualizer


def main():
    ap = argparse.ArgumentParser(description="Unified rendering script (config-driven).")
    ap.add_argument('--config', required=True, help='Path to configuration YAML file.')
    ap.add_argument('--tag', default='', help='Optional experiment tag suffix.')
    args, _ = ap.parse_known_args()

    # Load configuration
    cfg = load_config(args.config)
    p = resolve_params(cfg)

    # Determine run name and directories
    CORE_TYPE = p.core_type
    sim_hours = p.sim_hours
    nx, ny = p.nx, p.ny
    dt = p.dt
    use_nudge = p.use_nudge
    sponge_depth = p.sponge_depth
    coarsen_window = p.coarsen_window
    rc = p.render

    RUN_NAME = f"{p.domain}_dry_{CORE_TYPE}_n{nx}x{ny}_dt{int(dt)}_{sim_hours}h_{'nudge' if use_nudge else 'nonudge'}"
    if args.tag or os.environ.get('TAG'):
        tag = args.tag or os.environ.get('TAG')
        RUN_NAME += f"_{tag}"

    out_base = os.environ.get("SUETES_OUT_DIR") or p.output_dir
    run_dir = os.path.join(out_base, RUN_NAME)
    
    # We write plots to output_dir/plots or output_dir/plots/domain
    plot_dir = os.path.join(out_base, "plots", p.domain)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[RENDER] Loading simulation output from: {run_dir}")
    print(f"[RENDER] Run name: {RUN_NAME}")

    # Rebuild the grid
    coarse_store = os.path.join(p.store_dir, f"{p.cache_prefix}_cw{coarsen_window}_coarse.zarr")
    native_store = os.path.join(p.store_dir, f"{p.cache_prefix}_native.zarr")
    
    if not os.path.exists(coarse_store):
        raise SystemExit(f"[ERROR] coarse store not found: {coarse_store}")
        
    static = read_static(coarse_store)
    grid = build_grid_from_static(p, static['h'])

    # Load daily NetCDF snapshots
    snapshots = DailyNetcdfSnapshots(run_dir, RUN_NAME)
    sim_hours_actual = len(snapshots) - 1

    print(f"[RENDER] Loaded {len(snapshots)} states (T=0h to T={sim_hours_actual}h)")

    visualizer = Visualizer()
    final_state = snapshots[-1]

    # Convert zoom extent list if present
    extent = rc.zoom_extent if rc.zoom_extent else None

    # 1. Dashboards (final state)
    if rc.levels_z:
        print(f"[RENDER] Generating final-state dashboards at logical levels: {rc.levels_z}...")
        for z in rc.levels_z:
            visualizer.plot_dashboard(
                grid, final_state, z_idx=z, sponge_depth=sponge_depth, time_hours=sim_hours_actual,
                extent=None, quiver_stride=rc.quiver_stride,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_dash_z{z}_{sim_hours_actual}h.png"),
            )

    if rc.levels_m:
        print(f"[RENDER] Generating final-state dashboards at heights: {rc.levels_m} m...")
        for z in rc.levels_m:
            visualizer.plot_dashboard(
                grid, final_state, z_idx=z, sponge_depth=sponge_depth, time_hours=sim_hours_actual,
                extent=None, quiver_stride=rc.quiver_stride,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_dash_z{int(z)}m_{sim_hours_actual}h.png"),
            )

    # 2. Zoomed/evolution dashboards for specific target hours
    if rc.target_hours:
        print(f"[RENDER] Generating dashboards for target hours: {rc.target_hours}...")
        for h in rc.target_hours:
            if h < len(snapshots):
                visualizer.plot_dashboard(
                    grid, snapshots[h], z_idx=rc.levels_z[0] if rc.levels_z else 2,
                    sponge_depth=sponge_depth, time_hours=h, extent=None,
                    quiver_stride=rc.quiver_stride,
                    save_path=os.path.join(plot_dir, f"{RUN_NAME}_dash_{h}h.png"),
                )

    # 3. Energy spectrum
    if rc.energy_levels:
        print(f"[RENDER] Generating energy spectrum for '{rc.energy_var}' at levels {rc.energy_levels}...")
        for ez in rc.energy_levels:
            visualizer.plot_energy_spectrum(
                grid, final_state, rc.energy_var, z_idx=ez, sponge_depth=sponge_depth,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_energy_z{ez}_{sim_hours_actual}h.png"),
            )

    # 4. Comparison maps with native reference
    if os.path.exists(native_store) and rc.compare_levels_z:
        print(f"[RENDER] Generating comparison maps with native reference at levels {rc.compare_levels_z}...")
        final_era5_state = read_state(native_store, -1)
        for cz in rc.compare_levels_z:
            visualizer.plot_comparison(
                grid, final_state, final_era5_state, rc.compare_var, z_idx=cz, sponge_depth=sponge_depth,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_compare_{rc.compare_var}_z{cz}_{sim_hours_actual}h.png"),
            )

    # 5. Level strips
    if rc.strip_levels_z:
        print(f"[RENDER] Generating level strips for '{rc.strip_var}'...")
        visualizer.plot_level_strip(
            grid, final_state, rc.strip_var, z_indices=rc.strip_levels_z, sponge_depth=sponge_depth,
            save_path=os.path.join(plot_dir, f"{RUN_NAME}_levels_{rc.strip_var}_{sim_hours_actual}h.png"),
        )

    # 6. Hovmöller surface anomaly
    if rc.hovmoller_var:
        print(f"[RENDER] Calculating Hovmöller anomaly data for '{rc.hovmoller_var}'...")
        hov_times = list(range(sim_hours_actual + 1))
        hov_data = []
        for h in hov_times:
            sim_state = snapshots[h]
            bc_state_t = read_state(coarse_store, h)
            anom = sim_state[rc.hovmoller_var][:, :, 0] - bc_state_t[rc.hovmoller_var][:, :, 0]
            y_avg_anom = np.mean(anom, axis=1)
            hov_data.append(y_avg_anom)

        visualizer.plot_hovmoller(
            grid, hov_times, np.array(hov_data), variable=f'Surface {rc.hovmoller_var} Anomaly [K]',
            save_path=os.path.join(plot_dir, f"{RUN_NAME}_hovmoller_{rc.hovmoller_var}_{sim_hours_actual}h.png"),
        )

    # 7. Slices / Cross sections
    if rc.slices:
        print(f"[RENDER] Generating vertical slices locator dashboards...")
        for s in rc.slices:
            name = s.get('name', 'slices')
            map_var = s.get('map_var', 'th_v')
            slice_var = s.get('slice_var', 'w')
            map_z = s.get('map_z', 5)
            x_indices = s.get('x_indices', None)
            y_indices = s.get('y_indices', None)
            xlim = s.get('xlim', None)
            ylim = s.get('ylim', None)

            # Final state slices
            visualizer.plot_slice_locator_dashboard(
                grid, final_state, map_var=map_var, slice_var=slice_var, map_z=map_z,
                sponge_depth=sponge_depth, x_indices=x_indices, y_indices=y_indices,
                slice_xlim=xlim, slice_ylim=ylim,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_{name}_{sim_hours_actual}h.png"),
            )

            # Evolution slices
            if rc.target_hours:
                for h in rc.target_hours:
                    if h < len(snapshots):
                        visualizer.plot_slice_locator_dashboard(
                            grid, snapshots[h], map_var=map_var, slice_var=slice_var, map_z=map_z,
                            sponge_depth=sponge_depth, x_indices=x_indices, y_indices=y_indices,
                            slice_xlim=xlim, slice_ylim=ylim,
                            save_path=os.path.join(plot_dir, f"{RUN_NAME}_{name}_{h}h.png"),
                        )

    # 8. Point time series
    if rc.points:
        compare_store = native_store if os.path.exists(native_store) else coarse_store
        print(f"[RENDER] Generating point time-series plots using data from: {compare_store}")
        
        Xi_m, Yi_m = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lat_m, lon_m = grid.proj.get_lat_lon(Xi_m, Yi_m)
        times_hours_pt = list(range(sim_hours_actual + 1))

        point_coords = []
        for pt in rc.points:
            dist2 = (lat_m - pt['lat']) ** 2 + (lon_m - pt['lon']) ** 2
            i_p, j_p = np.unravel_index(int(np.argmin(dist2)), (grid.nx, grid.ny))
            point_coords.append((i_p, j_p))

        # We must support 'temperature' and 'wind' variables
        for idx, pt in enumerate(rc.points):
            name = pt['name']
            p_lat = pt['lat']
            p_lon = pt['lon']
            var_type = pt['variable']
            units = pt['units']
            i_p, j_p = point_coords[idx]
            safe_name = name.lower().replace(' ', '_')

            if var_type == 'temperature':
                native_columns = read_columns(compare_store, times_hours_pt, point_coords, keys=("th_v", "pi"), z=0)
                suetes_vals = [float(s['th_v'][i_p, j_p, 0] * s['pi'][i_p, j_p, 0]) - 273.15 for s in snapshots]
                era5_vals = [float(native_columns['th_v'][t, idx] * native_columns['pi'][t, idx]) - 273.15 for t in times_hours_pt]
            elif var_type == 'wind':
                native_columns = read_columns(compare_store, times_hours_pt, point_coords, keys=("u", "v"), z=0)
                suetes_vals = []
                for s in snapshots:
                    u_p = 0.5 * (s['u'][i_p, j_p, 0] + s['u'][i_p+1, j_p, 0])
                    v_p = 0.5 * (s['v'][i_p, j_p, 0] + s['v'][i_p, j_p+1, 0])
                    suetes_vals.append(float(np.sqrt(u_p**2 + v_p**2)) * 3.6)
                era5_vals = []
                for t in times_hours_pt:
                    u_p = 0.5 * (native_columns['u'][t, idx] + native_columns['u'][t, idx])
                    v_p = 0.5 * (native_columns['v'][t, idx] + native_columns['v'][t, idx])
                    era5_vals.append(float(np.sqrt(u_p**2 + v_p**2)) * 3.6)
            else:
                print(f"[WARNING] Unknown variable type '{var_type}' for point {name}. Skipping.")
                continue

            ew = 'W' if p_lon < 0 else 'E'
            ns = 'N' if p_lat > 0 else 'S'
            location_str = f"{name} ({abs(p_lat):.2f} deg {ns}, {abs(p_lon):.2f} deg {ew})"

            visualizer.plot_point_timeseries(
                times_hours_pt, suetes_vals, era5_vals,
                location_name=location_str, units=units,
                save_path=os.path.join(plot_dir, f"{RUN_NAME}_{var_type.capitalize()}_{safe_name}_{sim_hours_actual}h.png"),
            )

    snapshots.close()
    print(f"[RENDER] Success! Diagnostic plots written to {plot_dir}")


if __name__ == "__main__":
    main()
