#!/usr/bin/env python3
"""Render the saved coarse ERF three-dimensional supercell artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


QC_THRESHOLD = 1.0e-5  # kg kg-1, threshold used in the ERF paper
QR_THRESHOLD = 1.0e-4  # kg kg-1, threshold used in the ERF paper


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


def panel_label(axis, label):
    axis.text(
        0.015, 0.97, label, transform=axis.transAxes,
        ha="left", va="top", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )


def draw_cross_section(axis, dataset, index, colorbar=False):
    x = dataset.x.values / 1000.0
    z = dataset.z.values / 1000.0
    theta = dataset.dry_potential_temperature_xz.values[index]
    environment = dataset.environment_dry_potential_temperature.values
    cloud = 1000.0 * dataset.cloud_water_xz.values[index]
    rain = 1000.0 * dataset.rain_water_xz.values[index]
    xx, zz = np.meshgrid(x, z, indexing="ij")

    theta_prime = np.ma.masked_where(
        np.abs(theta - environment) < 0.5, theta - environment
    )
    axis.contourf(
        xx, zz, theta_prime,
        levels=np.linspace(-8.0, 8.0, 33),
        cmap="Greys", extend="both", alpha=0.55,
    )
    rain_levels = np.linspace(
        1000.0 * QR_THRESHOLD,
        max(0.2, float(np.nanmax(rain))),
        16,
    )
    rain_plot = axis.contourf(
        xx, zz, rain, levels=rain_levels, cmap="turbo", extend="max"
    )
    if np.nanmax(cloud) >= 1000.0 * QC_THRESHOLD:
        axis.contour(
            xx, zz, cloud,
            levels=[1000.0 * QC_THRESHOLD],
            colors="#ef8a00",
            linewidths=1.15,
        )
    axis.set_xlim(x.min(), x.max())
    axis.set_ylim(0.0, z.max() + 0.25)
    axis.set_title(f"$t={dataset.time.values[index] / 3600.0:g}$ h")
    axis.set_xlabel("$x$ (km)")
    if colorbar:
        bar = plt.colorbar(rain_plot, ax=axis, pad=0.015)
        bar.set_label(r"$q_r$ (g kg$^{-1}$)")


def render_evolution(dataset, output):
    indices = range(1, dataset.sizes["time"])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.5), sharex=True, sharey=True)
    for number, (axis, index) in enumerate(zip(axes.flat, indices)):
        draw_cross_section(axis, dataset, index, colorbar=number == 3)
        panel_label(axis, f"({chr(97 + number)})")
    axes[0, 0].set_ylabel("$z$ (km)")
    axes[1, 0].set_ylabel("$z$ (km)")
    fig.subplots_adjust(wspace=0.10, hspace=0.28)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_final(dataset, output):
    index = -1
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rain = dataset.accumulated_rain.values[index]
    surface_theta = dataset.surface_dry_potential_temperature.values[index]
    theta_initial = dataset.surface_dry_potential_temperature.values[0]
    cloud_envelope = dataset.column_maximum_cloud_water.values[index]
    rain_envelope = dataset.column_maximum_rain_water.values[index]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5))
    draw_cross_section(axes[0], dataset, index, colorbar=True)
    axes[0].set_ylabel("$z$ (km)")
    panel_label(axes[0], "(a)")

    levels = np.linspace(1.0, max(2.0, float(np.nanmax(rain))), 20)
    filled = axes[1].contourf(
        xx, yy, rain, levels=levels, cmap="turbo", extend="max"
    )
    cold_pool = surface_theta - theta_initial
    cold_levels = [
        level for level in (-6.0, -4.0, -2.0, -1.0)
        if np.nanmin(cold_pool) <= level <= np.nanmax(cold_pool)
    ]
    if cold_levels:
        cold_contours = axes[1].contour(
            xx, yy, cold_pool, levels=cold_levels,
            colors="white", linewidths=0.75, linestyles="dashed",
        )
        axes[1].clabel(
            cold_contours, fmt=lambda value: f"{value:g} K",
            fontsize=7, inline_spacing=2,
        )
    if np.nanmax(cloud_envelope) >= QC_THRESHOLD:
        axes[1].contour(
            xx, yy, cloud_envelope,
            levels=[QC_THRESHOLD],
            colors="#444444", linewidths=1.15,
        )
    if np.nanmax(rain_envelope) >= QR_THRESHOLD:
        axes[1].contour(
            xx, yy, rain_envelope,
            levels=[QR_THRESHOLD],
            colors="#d7301f", linewidths=1.3,
        )
    bar = fig.colorbar(filled, ax=axes[1], pad=0.015)
    bar.set_label("Accumulated rain (mm)")
    axes[1].set_xlim(x.min(), x.max())
    axes[1].set_ylim(y.min(), y.max())
    axes[1].set_xlabel("$x$ (km)")
    axes[1].set_ylabel("$y$ (km)")
    axes[1].set_title(
        f"Surface accumulation at $t={dataset.time.values[index] / 3600.0:g}$ h"
    )
    panel_label(axes[1], "(b)")
    axes[1].plot(
        [], [], color="#444444", linewidth=1.15,
        label=r"$q_c=10^{-5}$ kg kg$^{-1}$",
    )
    axes[1].plot(
        [], [], color="#d7301f", linewidth=1.3,
        label=r"$q_r=10^{-4}$ kg kg$^{-1}$",
    )
    axes[1].plot(
        [], [], color="white", linestyle="dashed", linewidth=0.9,
        label=r"surface $\theta_d'$",
    )
    legend = axes[1].legend(
        loc="upper right", frameon=True, framealpha=0.9, fontsize=8
    )
    # The white cold-pool key needs a dark legend background to remain visible.
    legend.get_frame().set_facecolor("#eeeeee")
    fig.subplots_adjust(hspace=0.32)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_erf_diagnostics(dataset):
    """Print quantities described explicitly in the ERF validation paper."""
    if "cloud_water_3d" in dataset:
        cloud = dataset.cloud_water_3d.values[-1]
        rain = dataset.rain_water_3d.values[-1]
    else:
        cloud = dataset.final_cloud_water.values
        rain = dataset.final_rain_water.values
    accumulation = dataset.accumulated_rain.values[-1]
    dx = float(dataset.attrs["dx_m"])
    dz = float(dataset.attrs["dz_m"])

    def extents(field, threshold):
        mask = field >= threshold
        if not np.any(mask):
            return np.nan, np.nan, np.nan
        ix, iy, iz = np.where(mask)
        x_span = (np.ptp(dataset.x.values[ix]) + dx) / 1000.0
        y_span = (np.ptp(dataset.y.values[iy]) + dx) / 1000.0
        cloud_top = (np.max(dataset.z.values[iz]) + 0.5 * dz) / 1000.0
        return x_span, y_span, cloud_top

    cloud_x, cloud_y, cloud_top = extents(cloud, QC_THRESHOLD)
    rain_x, rain_y, rain_top = extents(rain, QR_THRESHOLD)
    print("\nERF-style diagnostics at the final time")
    print(
        f"  q_c >= 1e-5 kg/kg: {cloud_x:.1f} x {cloud_y:.1f} km, "
        f"top={cloud_top:.1f} km"
    )
    print(
        f"  q_r >= 1e-4 kg/kg: {rain_x:.1f} x {rain_y:.1f} km, "
        f"top={rain_top:.1f} km"
    )
    print(f"  peak accumulated rain: {np.nanmax(accumulation):.1f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", type=Path,
        default=Path("output/squall_line_3d/artifact.nc"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/squall_line_3d")
    )
    args = parser.parse_args()
    if not args.artifact.exists():
        raise FileNotFoundError(
            f"{args.artifact} does not exist; run run.py first"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.artifact) as source:
        dataset = source.load()
    print_erf_diagnostics(dataset)
    outputs = (
        args.output_dir / "squall_line_3d_evolution.png",
        args.output_dir / "squall_line_3d.png",
    )
    render_evolution(dataset, outputs[0])
    render_final(dataset, outputs[1])
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
