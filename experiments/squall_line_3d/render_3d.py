#!/usr/bin/env python3
"""Render ERF-style 3-D hydrometeor isosurfaces from a saved artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes
import xarray as xr


QC_THRESHOLD = 1.0e-5
QR_THRESHOLD = 1.0e-4
VISUAL_QC_THRESHOLD = QC_THRESHOLD
CLOUD_COLOR = "#b8b8b8"


def add_isosurface(axis, field, level, origin, spacing, color, alpha, step_size):
    if not np.nanmin(field) <= level <= np.nanmax(field):
        return None
    vertices, faces, _, _ = marching_cubes(
        np.asarray(field, dtype=np.float32), level=level, spacing=spacing, step_size=step_size, allow_degenerate=False
    )
    vertices += np.asarray(origin)[None, :]
    collection = Poly3DCollection(vertices[faces], facecolor=color, edgecolor="none", alpha=alpha, linewidth=0.0)
    collection.set_rasterized(True)
    axis.add_collection3d(collection)
    return collection


def select_volume(dataset, requested_time):
    if requested_time is None:
        requested_time = float(dataset.time.values[-1])
    if "cloud_water_3d" in dataset:
        available = np.asarray(dataset.volume_time.values)
        if available.size == 0:
            raise ValueError("The artifact contains no stored 3-D volumes")
        index = int(np.argmin(np.abs(available - requested_time)))
        if not np.isclose(available[index], requested_time):
            raise ValueError(
                f"No 3-D volume at t={requested_time:g} s; available: " + ", ".join(f"{value:g}" for value in available)
            )
        return (float(available[index]), dataset.cloud_water_3d.values[index], dataset.rain_water_3d.values[index])
    final_time = float(dataset.time.values[-1])
    if not np.isclose(requested_time, final_time):
        raise ValueError("This older artifact stores a full 3-D volume only at the final time")
    return final_time, dataset.final_cloud_water.values, dataset.final_rain_water.values


def render(
    artifact: Path,
    output: Path,
    mesh_step: int,
    elevation: float,
    azimuth: float,
    requested_time: float | None,
    boundary_mask_km: float,
    cloud_threshold: float,
    cloud_alpha: float,
):
    with xr.open_dataset(artifact) as source:
        dataset = source.load()

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

    fig = plt.figure(figsize=(12.0, 7.2))
    axis = fig.add_subplot(111, projection="3d", computed_zorder=False)

    xx, yy = np.meshgrid(x, y, indexing="ij")
    theta_limit = 3.0
    norm = colors.TwoSlopeNorm(vmin=-theta_limit, vcenter=0.0, vmax=theta_limit)
    cmap = plt.get_cmap("RdBu_r")
    surface_colors = np.asarray(cmap(norm(np.clip(theta_surface, -theta_limit, theta_limit))), dtype=np.float64)
    if boundary_mask_km > 0.0:
        boundary_x = np.minimum(x - x.min(), x.max() - x)
        boundary_y = np.minimum(y - y.min(), y.max() - y)
        plot_interior = (boundary_x[:, None] >= boundary_mask_km) & (boundary_y[None, :] >= boundary_mask_km)
        surface_colors[..., 3] = np.where(plot_interior, 0.88, 0.0)
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

    # ERF displays theta_d' on the central y=0 vertical plane in addition to
    # the ground slice. Put this plane behind the hydrometeor isosurfaces.
    xx_vertical, zz_vertical = np.meshgrid(x, z, indexing="ij")
    y_vertical = np.full_like(xx_vertical, y[len(y) // 2])
    vertical_colors = np.asarray(cmap(norm(np.clip(theta_xz, -theta_limit, theta_limit))), dtype=np.float64)
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
        axis, cloud, cloud_threshold, origin, spacing, color=CLOUD_COLOR, alpha=cloud_alpha, step_size=mesh_step
    )

    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_zlim(0.0, float(z.max() + 0.5 * dz))
    axis.set_xlabel("$x$ (km)", labelpad=8)
    axis.set_ylabel("$y$ (km)", labelpad=8)
    axis.set_zlabel("$z$ (km)", labelpad=7)
    axis.set_box_aspect((150.0, 100.0, 48.0), zoom=1.22)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#bbbbbb")
    axis.set_title(f"Three-dimensional supercell at $t={selected_time / 3600.0:g}$ h", pad=16)

    legend = axis.legend(
        handles=[
            Patch(
                facecolor=CLOUD_COLOR,
                edgecolor="#777777",
                alpha=cloud_alpha,
                label=(
                    rf"$q_c={cloud_threshold / 1.0e-5:g}"
                    r"\times10^{-5}$ kg kg$^{-1}$"
                ),
            ),
            Patch(facecolor="#b2182b", alpha=0.65, label=r"$q_r=10^{-4}$ kg kg$^{-1}$"),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
    )
    legend.set_zorder(20)
    fig.subplots_adjust(left=0.01, right=0.84, bottom=0.01, top=0.92)
    colorbar_axis = fig.add_axes((0.87, 0.20, 0.022, 0.57))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    colorbar.set_label(r"$\theta_d'$ (K)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def render_rain_top_view(artifact: Path, output: Path):
    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    rain = dataset.accumulated_rain.values[-1]
    cloud = dataset.column_maximum_cloud_water.values[-1]
    rain_water = dataset.column_maximum_rain_water.values[-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, axis = plt.subplots(figsize=(8.0, 5.4))
    levels = np.linspace(0.0, max(1.0, float(np.nanmax(rain))), 21)
    filled = axis.contourf(xx, yy, rain, levels=levels, cmap="Blues", extend="max")
    if np.nanmax(cloud) >= QC_THRESHOLD:
        axis.contour(xx, yy, cloud, levels=[QC_THRESHOLD], colors="#777777", linewidths=0.9)
    if np.nanmax(rain_water) >= QR_THRESHOLD:
        axis.contour(xx, yy, rain_water, levels=[QR_THRESHOLD], colors="#8c2d04", linewidths=1.0)
    bar = fig.colorbar(filled, ax=axis, pad=0.02)
    bar.set_label("Accumulated rain (mm)")
    axis.set_xlabel("$x$ (km)")
    axis.set_ylabel("$y$ (km)")
    axis.set_aspect("equal")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_title(f"Surface rain accumulation at $t={float(dataset.time.values[-1]) / 3600.0:g}$ h")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=Path("output/squall_line_3d_dx400_nodiv/artifact.nc"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/squall_line_3d_dx400_nodiv/squall_line_3d_isosurfaces.png")
    )
    parser.add_argument(
        "--mesh-step", type=int, default=2, help="marching-cubes stride; use 1 for the highest-quality mesh"
    )
    parser.add_argument("--elevation", type=float, default=14.0, help="camera elevation in degrees")
    parser.add_argument("--azimuth", type=float, default=-108.0, help="camera azimuth in degrees")
    parser.add_argument("--time", type=float, default=None, help="time in seconds of the stored 3-D volume to render")
    parser.add_argument(
        "--rain-output", type=Path, default=None, help="optional standalone top-view accumulated-rain figure"
    )
    parser.add_argument(
        "--boundary-mask-km",
        type=float,
        default=0.0,
        help=(
            "optional transparent outer buffer on the horizontal theta slice; "
            "disabled by default and never applied to cloud or rain fields"
        ),
    )
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=VISUAL_QC_THRESHOLD,
        help=(
            "cloud-water isosurface used only for visualization (kg kg-1); "
            "ERF diagnostics retain the canonical 1e-5 threshold"
        ),
    )
    parser.add_argument("--cloud-alpha", type=float, default=0.32, help="opacity of the visual cloud-water isosurface")
    args = parser.parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(
            f"Artifact is not a file: {args.artifact!s}. If using $ARTIFACT, define it in the current shell first."
        )
    if args.mesh_step < 1:
        raise ValueError("--mesh-step must be positive")
    if args.cloud_threshold <= 0.0:
        raise ValueError("--cloud-threshold must be positive")
    if not 0.0 <= args.cloud_alpha <= 1.0:
        raise ValueError("--cloud-alpha must lie in [0, 1]")
    render(
        args.artifact,
        args.output,
        args.mesh_step,
        args.elevation,
        args.azimuth,
        args.time,
        args.boundary_mask_km,
        args.cloud_threshold,
        args.cloud_alpha,
    )
    if args.rain_output is not None:
        render_rain_top_view(args.artifact, args.rain_output)


if __name__ == "__main__":
    main()
