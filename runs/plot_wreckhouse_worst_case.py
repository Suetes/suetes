#!/usr/bin/env python3
"""Regenerate the Wreckhouse dashboard from saved plot data."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--output")
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if not artifact.exists():
        parser.error(
            f"artifact does not exist: {artifact}. The existing diagnostics "
            "NPZ lacks the cross-section fields, so rerun "
            "runs/wreckhouse_worst_case.py once with the updated code."
        )
    output = Path(args.output) if args.output else artifact.with_name(
        artifact.stem.replace("_plot_data", "_dashboard") + ".png"
    )
    with xr.open_dataset(artifact) as source:
        data = source.load()

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(2, 2)
    axis = fig.add_subplot(grid[0, 0])
    hours = data.time.values / 3600.0
    axis.plot(hours, data.era5_wind, "r--", linewidth=2, label="ERA5 driver")
    axis.plot(hours, data.baseline_wind, "b-", linewidth=2, alpha=.7,
              label="Baseline Suêtes")
    axis.plot(hours, data.adverse_wind, "k-", linewidth=2,
              label="Adjoint-directed adverse run")
    axis.set(title="Surface wind speed evolution at target",
             xlabel="Simulation time [hours]", ylabel="Wind speed [km/h]")
    axis.grid(True, linestyle="--", alpha=.6)
    axis.legend()

    axis = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    perturbation = data.thermal_perturbation_surface.values
    limit = max(float(np.max(np.abs(perturbation))), .1)
    image = axis.pcolormesh(
        data.longitude, data.latitude, perturbation, shading="auto",
        cmap="RdBu_r", vmin=-limit, vmax=limit, transform=ccrs.PlateCarree(),
    )
    i, j = int(data.attrs["target_i"]), int(data.attrs["target_j"])
    axis.plot(data.longitude.values[i, j], data.latitude.values[i, j],
              "k*", markersize=16, transform=ccrs.PlateCarree(), label="Target")
    axis.coastlines()
    axis.set_title(r"Adjoint-directed thermal perturbation ($\Delta\theta_v$)")
    axis.legend(loc="lower right")
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.08,
                 label="Thermal shift [K]")

    x = data.x.values / 1000.0
    z = data.physical_height.values / 1000.0
    x2d = np.broadcast_to(x[:, None], z.shape)
    axis = fig.add_subplot(grid[1, 0])
    image = axis.pcolormesh(
        x2d, z, data.baseline_zonal_wind, shading="auto",
        cmap="RdBu_r", vmin=-50, vmax=150,
    )
    axis.contour(
        x2d, z, data.baseline_virtual_potential_temperature,
        colors="black", linewidths=.7, alpha=.7,
    )
    axis.fill_between(x, 0, data.terrain_height.values / 1000.0, color="0.35")
    axis.set(title="Baseline zonal wind [km/h] across terrain",
             xlabel="Distance [km]", ylabel="Height [km]", ylim=(0, 4))
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.15,
                 label="Baseline zonal wind [km/h]")

    axis = fig.add_subplot(grid[1, 1])
    anomaly = data.zonal_wind_anomaly.values
    depth = int(data.attrs["sponge_depth"])
    interior = anomaly[depth:-depth] if depth else anomaly
    limit = max(float(np.max(np.abs(interior))), 1.0)
    image = axis.pcolormesh(
        x2d, z, anomaly, shading="auto", cmap="seismic",
        vmin=-limit, vmax=limit,
    )
    axis.fill_between(x, 0, data.terrain_height.values / 1000.0, color="0.35")
    axis.set(title=r"Zonal wind anomaly induced by adjoint",
             xlabel="Distance [km]", ylabel="Height [km]", ylim=(0, 4))
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.15,
                 label="Wind difference [km/h]")
    fig.suptitle("Suêtes downscaling: adjoint-directed adverse perturbation",
                 fontsize=18, y=.98)
    fig.subplots_adjust(top=.90, bottom=.08)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
