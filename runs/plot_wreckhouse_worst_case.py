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
    parser.add_argument(
        "--full-timeseries",
        action="store_true",
        help="Plot the full time series from beginning to end instead of zooming in on the target window.",
    )
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

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, hspace=.28, wspace=.16)
    axis = fig.add_subplot(grid[0, 0])
    hours = data.time.values / 3600.0
    baseline = data.baseline_wind.values
    adverse = data.adverse_wind.values
    control_hour = float(data.attrs["control_time_seconds"]) / 3600.0
    target_hour = float(data.attrs["target_time_seconds"]) / 3600.0
    end_hour = min(float(data.attrs["end_time_seconds"]) / 3600.0,
                   target_hour + 1.0)
    axis.plot(hours, baseline, color="#3973b7", linewidth=2.2,
              label="Baseline Suêtes")
    axis.plot(hours, adverse, color="black", linewidth=2.2,
              label="Adjoint-directed run")
    axis.axvline(control_hour, color=".35", linestyle=":", linewidth=1.4)
    axis.axvline(target_hour, color=".35", linestyle="--", linewidth=1.4)
    if args.full_timeseries:
        axis.set_xlim(hours[0], hours[-1])
        margin = 5.0
        axis.set_ylim(
            min(baseline.min(), adverse.min()) - margin,
            max(baseline.max(), adverse.max()) + margin,
        )
    else:
        axis.set_xlim(control_hour, end_hour)
        visible = (hours >= control_hour) & (hours <= end_hour)
        margin = 1.5
        axis.set_ylim(
            min(baseline[visible].min(), adverse[visible].min()) - margin,
            max(baseline[visible].max(), adverse[visible].max()) + margin,
        )
    axis.set(title="Target-footprint wind response",
             xlabel="Simulation time [hours]", ylabel="Wind speed [km/h]")
    axis.grid(True, linestyle="--", alpha=.6)
    axis.legend(loc="lower left")

    axis = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
    perturbation = data.thermal_perturbation_surface.values
    limit = max(float(np.max(np.abs(perturbation))), .1)
    image = axis.pcolormesh(
        data.longitude, data.latitude, perturbation, shading="auto",
        cmap="RdBu_r", vmin=-limit, vmax=limit, transform=ccrs.PlateCarree(),
    )
    i, j = int(data.attrs["target_i"]), int(data.attrs["target_j"])
    # The lower panels use the model-x section at the target y index.
    axis.plot(
        data.longitude.values[:, j], data.latitude.values[:, j],
        color=".15", linestyle="--", linewidth=1.3,
        transform=ccrs.PlateCarree(), label="Cross-section",
    )
    axis.plot(data.longitude.values[i, j], data.latitude.values[i, j],
              "k*", markersize=16, transform=ccrs.PlateCarree(), label="Target")
    if "terrain_height_map" in data:
        terrain_map = data.terrain_height_map.values / 1000.0
        maximum = float(np.nanmax(terrain_map))
        if maximum > .25:
            axis.contour(
                data.longitude, data.latitude, terrain_map,
                levels=np.arange(.25, maximum + .25, .25),
                colors=".35", linewidths=.6, alpha=.65,
                transform=ccrs.PlateCarree(),
            )
    if {"baseline_surface_u", "baseline_surface_v"} <= set(data.data_vars):
        stride = 12
        axis.quiver(
            data.longitude.values[::stride, ::stride],
            data.latitude.values[::stride, ::stride],
            data.baseline_surface_u.values[::stride, ::stride],
            data.baseline_surface_v.values[::stride, ::stride],
            color=".2", alpha=.65, scale=320, width=.0018,
            transform=ccrs.PlateCarree(),
        )
    axis.coastlines()
    axis.set_title(r"Adjoint-directed thermal perturbation ($\Delta\theta_v$)")
    axis.legend(loc="lower right")
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.08,
                 label="Thermal shift [K]")

    x = data.x.values / 1000.0
    terrain = data.terrain_height.values / 1000.0
    z_mass = data.physical_height.values / 1000.0

    def padded_mass_section(field):
        """Match ``plot_cross_section`` mass-level plotting geometry."""
        z_plot = np.concatenate([terrain[:, None], z_mass], axis=1)
        field_plot = np.concatenate([field[:, :1], field], axis=1)
        x_plot = np.broadcast_to(x[:, None], z_plot.shape)
        return x_plot, z_plot, field_plot

    axis = fig.add_subplot(grid[1, 0])
    baseline_name = (
        "baseline_wind_speed"
        if "baseline_wind_speed" in data
        else "baseline_zonal_wind"
    )
    anomaly_name = (
        "wind_speed_anomaly"
        if "wind_speed_anomaly" in data
        else "zonal_wind_anomaly"
    )
    using_speed = baseline_name == "baseline_wind_speed"
    x_plot, z_plot, baseline_plot = padded_mass_section(data[baseline_name].values)
    # Determine the colour range from the portion of the atmosphere that is
    # actually displayed.  Otherwise upper-level jet speeds above the 4-km
    # axis limit wash out the downslope-flow structure.
    displayed = baseline_plot[z_plot <= 4.0]
    base_min = max(
        0.0, np.floor(float(np.nanpercentile(displayed, 1)) / 10.0) * 10.0
    )
    base_max = (
        np.ceil(float(np.nanpercentile(displayed, 99.5)) / 10.0) * 10.0
    )
    image = axis.contourf(
        x_plot, z_plot, baseline_plot,
        levels=np.linspace(base_min, base_max, 31),
        cmap="viridis", vmin=base_min, vmax=base_max, extend="both",
    )
    theta = data.baseline_virtual_potential_temperature.values
    _, _, theta_plot = padded_mass_section(theta)
    theta_levels = np.arange(np.floor(theta.min()), np.ceil(theta.max()), 2.0)
    axis.contour(
        x_plot, z_plot, theta_plot, levels=theta_levels,
        colors="black", linewidths=1.0, alpha=.7,
    )
    axis.fill_between(x, 0, terrain, color="dimgray")
    baseline_title = (
        "Baseline wind speed and isentropes at verification time"
        if using_speed else
        "Baseline zonal wind and isentropes at verification time"
    )
    baseline_label = "Wind speed [km/h]" if using_speed else "Zonal wind [km/h]"
    axis.set(title=baseline_title,
             xlabel="Distance [km]", ylabel="Height [km]", ylim=(0, 4))
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.15,
                 label=baseline_label)

    axis = fig.add_subplot(grid[1, 1])
    anomaly = data[anomaly_name].values
    depth = int(data.attrs["sponge_depth"])
    interior = anomaly[depth:-depth] if depth else anomaly
    limit = max(float(np.max(np.abs(interior))), 1.0)
    x_plot, z_plot, anomaly_plot = padded_mass_section(anomaly)
    image = axis.contourf(
        x_plot, z_plot, anomaly_plot,
        levels=np.linspace(-limit, limit, 31), cmap="seismic",
        vmin=-limit, vmax=limit, extend="both",
    )
    axis.fill_between(x, 0, terrain, color="dimgray")
    _, _, theta_plot = padded_mass_section(theta)
    axis.contour(
        x_plot, z_plot, theta_plot, levels=theta_levels,
        colors=".25", linewidths=.7, alpha=.45,
    )
    response_title = (
        "Adjoint-directed wind-speed response at verification time"
        if using_speed else
        "Adjoint-directed zonal-wind response at verification time"
    )
    response_label = (
        r"$\Delta$ wind speed [km/h]"
        if using_speed else r"$\Delta$ zonal wind [km/h]"
    )
    axis.set(title=response_title,
             xlabel="Distance [km]", ylabel="Height [km]", ylim=(0, 4))
    fig.colorbar(image, ax=axis, orientation="horizontal", pad=.15,
                 label=response_label)
    fig.subplots_adjust(top=.96, bottom=.08)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
