#!/usr/bin/env python3
"""Render two 3-D storm snapshots and accumulated rain as one paper figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

from render_3d import CLOUD_COLOR, QC_THRESHOLD, QR_THRESHOLD, add_isosurface, select_volume


def draw_storm(axis, dataset, requested_time, norm, cmap, mesh_step, elevation, azimuth, cloud_alpha):
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    z = dataset.z.values / 1000.0
    dx = float(dataset.attrs["dx_m"]) / 1000.0
    dz = float(dataset.attrs["dz_m"]) / 1000.0
    selected_time, cloud, rain = select_volume(dataset, requested_time)
    time_index = int(np.argmin(np.abs(np.asarray(dataset.time.values) - selected_time)))
    theta_xz = (
        dataset.dry_potential_temperature_xz.values[time_index] - dataset.environment_dry_potential_temperature.values
    )
    theta_surface = (
        dataset.surface_dry_potential_temperature.values[time_index]
        - dataset.surface_dry_potential_temperature.values[0]
    )

    xx, yy = np.meshgrid(x, y, indexing="ij")
    surface_colors = np.asarray(cmap(norm(np.clip(theta_surface, norm.vmin, norm.vmax))), dtype=np.float64)
    axis.plot_surface(
        xx,
        yy,
        np.zeros_like(xx),
        facecolors=surface_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.88,
        zorder=0,
    )

    xx_vertical, zz_vertical = np.meshgrid(x, z, indexing="ij")
    y_vertical = np.full_like(xx_vertical, y[len(y) // 2])
    vertical_colors = np.asarray(cmap(norm(np.clip(theta_xz, norm.vmin, norm.vmax))), dtype=np.float64)
    axis.plot_surface(
        xx_vertical,
        y_vertical,
        zz_vertical,
        facecolors=vertical_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        alpha=0.72,
        zorder=1,
    )

    origin = (float(x[0]), float(y[0]), float(z[0]))
    spacing = (dx, dx, dz)
    add_isosurface(axis, rain, QR_THRESHOLD, origin, spacing, color="#b2182b", alpha=0.70, step_size=mesh_step)
    add_isosurface(
        axis, cloud, QC_THRESHOLD, origin, spacing, color=CLOUD_COLOR, alpha=cloud_alpha, step_size=mesh_step
    )

    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_zlim(0.0, float(z.max() + 0.5 * dz))
    axis.set_xlabel("$x$ (km)", labelpad=4)
    axis.set_ylabel("$y$ (km)", labelpad=4)
    axis.set_zlabel("$z$ (km)", labelpad=3)
    axis.set_box_aspect((150.0, 100.0, 48.0), zoom=1.30)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#bbbbbb")
    axis.text2D(
        0.50, 0.91, f"$t={selected_time / 3600.0:g}$ h", transform=axis.transAxes, ha="center", va="top", fontsize=11
    )


def draw_rain(axis, dataset):
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    rain = dataset.accumulated_rain.values[-1]
    cloud = dataset.column_maximum_cloud_water.values[-1]
    rain_water = dataset.column_maximum_rain_water.values[-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    levels = np.linspace(0.0, max(1.0, float(np.nanmax(rain))), 21)
    image = axis.contourf(xx, yy, rain, levels=levels, cmap="Blues", extend="max")
    if np.nanmax(cloud) >= QC_THRESHOLD:
        axis.contour(xx, yy, cloud, levels=[QC_THRESHOLD], colors="#777777", linewidths=0.8)
    if np.nanmax(rain_water) >= QR_THRESHOLD:
        axis.contour(xx, yy, rain_water, levels=[QR_THRESHOLD], colors="#8c2d04", linewidths=0.9)
    axis.set_xlabel("$x$ (km)")
    axis.set_ylabel("$y$ (km)")
    axis.set_aspect("equal")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_title(f"Surface accumulation at $t={float(dataset.time.values[-1]) / 3600.0:g}$ h")
    return image


def render(artifact, output, mesh_step, elevation, azimuth, cloud_alpha):
    with xr.open_dataset(artifact) as source:
        dataset = source.load()

    fig = plt.figure(figsize=(12.8, 7.7))
    grid = fig.add_gridspec(2, 2)
    axes_3d = (
        fig.add_subplot(grid[0, 0], projection="3d", computed_zorder=False),
        fig.add_subplot(grid[0, 1], projection="3d", computed_zorder=False),
    )
    rain_axis = fig.add_subplot(grid[1, :])
    axes_3d[0].set_position((0.055, 0.46, 0.40, 0.50))
    axes_3d[1].set_position((0.465, 0.46, 0.40, 0.50))
    rain_axis.set_position((0.18, 0.055, 0.64, 0.39))

    theta_limit = 3.0
    norm = colors.TwoSlopeNorm(vmin=-theta_limit, vcenter=0.0, vmax=theta_limit)
    cmap = plt.get_cmap("RdBu_r")
    for axis, time_s in zip(axes_3d, (1800.0, 7200.0)):
        draw_storm(axis, dataset, time_s, norm, cmap, mesh_step, elevation, azimuth, cloud_alpha)

    theta_bar_axis = fig.add_axes((0.885, 0.575, 0.014, 0.265))
    theta_bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=theta_bar_axis)
    theta_bar.set_label(r"$\theta_d'$ (K)")
    fig.legend(
        handles=[
            Patch(facecolor=CLOUD_COLOR, edgecolor="#777777", alpha=cloud_alpha, label=r"$q_c=10^{-5}$ kg kg$^{-1}$"),
            Patch(facecolor="#b2182b", alpha=0.70, label=r"$q_r=10^{-4}$ kg kg$^{-1}$"),
        ],
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.012, 0.255),
    )

    rain_image = draw_rain(rain_axis, dataset)
    rain_bar_axis = fig.add_axes((0.835, 0.09, 0.014, 0.31))
    rain_bar = fig.colorbar(rain_image, cax=rain_bar_axis)
    rain_bar.set_label("Accumulated rain (mm)")

    axes_3d[0].text2D(0.03, 0.88, "(a)", transform=axes_3d[0].transAxes, fontweight="bold")
    axes_3d[1].text2D(0.03, 0.88, "(b)", transform=axes_3d[1].transAxes, fontweight="bold")
    rain_axis.text(0.01, 0.97, "(c)", transform=rain_axis.transAxes, va="top", fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-step", type=int, default=1)
    parser.add_argument("--elevation", type=float, default=14.0)
    parser.add_argument("--azimuth", type=float, default=-108.0)
    parser.add_argument("--cloud-alpha", type=float, default=0.32)
    args = parser.parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    if args.mesh_step < 1:
        raise ValueError("--mesh-step must be positive")
    if not 0.0 <= args.cloud_alpha <= 1.0:
        raise ValueError("--cloud-alpha must lie in [0, 1]")
    render(args.artifact, args.output, args.mesh_step, args.elevation, args.azimuth, args.cloud_alpha)


if __name__ == "__main__":
    main()
