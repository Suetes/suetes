#!/usr/bin/env python3
"""Render ERF supercell panels as separate figures for later assembly.

Layout produced:

    [ volume a | volume b | theta colorbar ]
    [   legend | rain | rain colorbar, centred as a group  ]

Every panel is bare. No titles, no captions, no inline colorbars, no
inline legend. Panel letters are composited by make_fig.sh, which also
trims the dead space a 3-D projection leaves around its box.

Do NOT add bbox_inches='tight' to any savefig here. The colorbar strips
are sized in inches to match the panels and tight bounding boxes break
that correspondence.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes
import xarray as xr

from suetes.shared.experiment import ExperimentLayout, artifact_from_bundle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_NAME = "artifact.nc"
# Bundle written by the paper suite, used when no artifact sits beside this file.
DEFAULT_BUNDLE = ExperimentLayout(kind="experiments", case="squall_line_3d", execution="paper").root

# --- Physics / thresholds ------------------------------------------------
QC_THRESHOLD = 1.0e-5
QR_THRESHOLD = 1.0e-4
VISUAL_QC_THRESHOLD = QC_THRESHOLD

CLOUD_COLOR = "#b8b8b8"
RAIN_COLOR = "#b2182b"
CLOUD_ALPHA = 0.32
RAIN_ALPHA = 0.70
CLOUD_EDGE_COLOR = "#777777"
RAIN_EDGE_COLOR = "#8c2d04"
CLOUD_CONTOUR_COLOR = "#777777"
RAIN_CONTOUR_COLOR = "#8c2d04"

THETA_LIMIT = 3.0
THETA_CMAP = "RdBu_r"
THETA_LABEL = r"$\theta_d^\prime$ (K)"
THETA_TICKS = np.arange(-3.0, 3.1, 1.0)
GROUND_ALPHA = 0.88
SLICE_ALPHA = 0.72

RAIN_CMAP = "Blues"
RAIN_LABEL = "Accumulated rain (mm)"
RAIN_NLEVELS = 21

# --- Geometry, all in inches --------------------------------------------
# Panels are rendered generously and trimmed downstream, because a 3-D
# projection leaves unpredictable dead space around its box. Canvas size
# here sets the text-to-content ratio, not the final panel size.
PANEL_W_IN = 9.00
PANEL_H_IN = 5.20
PANEL_INSET = 0.01  # fraction, keeps the projection off the canvas edge

# Bar heights are the one thing trim cannot recover, since a colorbar has
# no slack to remove. Tune THETA_BAR_H_IN to the trimmed panel height.
CBAR_W_IN = 1.70
CBAR_BAR_W_IN = 0.42
CBAR_LEFT_IN = 0.08
CBAR_PAD_IN = 0.30
THETA_BAR_H_IN = 3.60

RAIN_W_IN = 7.00
RAIN_AX_LEFT_IN = 0.95
RAIN_AX_BOTTOM_IN = 0.75
RAIN_AX_PAD_IN = 0.20
RAIN_AX_W_IN = RAIN_W_IN - RAIN_AX_LEFT_IN - RAIN_AX_PAD_IN

# Sized to its own content only. The bottom row is centred as a group, so
# the legend must not carry padding of its own or panel c drifts right.
LEGEND_W_IN = 3.40
LEGEND_H_IN = 1.50

# --- Camera --------------------------------------------------------------
VIEW_ELEV = 14.0
VIEW_AZIM = -108.0
BOX_ZOOM = 1.30
Z_EXAGGERATION = 2.0  # vertical stretch of the box aspect only

# Axis labels sit outside the tick text on a tilted projection, so they
# need far more pad than a 2-D axes does.
LABELPAD_X = 26
LABELPAD_Y = 26
LABELPAD_Z = 14
TICKPAD_3D = 4

# (time in seconds of the stored volume, output stem, draw the y and z axes).
# Panel b shares panel a's vertical and cross-stream extent, so repeating
# those axes is redundant and costs horizontal space.
PANELS = [
    (1800.0, "supercell_volume_a_main", True),
    (7200.0, "supercell_volume_b_main", False),
]


def set_style():
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "font.family": "sans-serif",
        }
    )


def theta_norm():
    return colors.TwoSlopeNorm(vmin=-THETA_LIMIT, vcenter=0.0, vmax=THETA_LIMIT)


def resolve_artifact(requested: Path | None) -> Path:
    """Explicit path or bundle, else a sibling artifact, else the paper bundle."""
    if requested is None:
        local = SCRIPT_DIR / DEFAULT_ARTIFACT_NAME
        if local.is_file():
            return local
        return artifact_from_bundle(DEFAULT_BUNDLE, DEFAULT_ARTIFACT_NAME)
    requested = Path(requested)
    if requested.is_dir():
        return artifact_from_bundle(requested, DEFAULT_ARTIFACT_NAME)
    if not requested.is_file():
        raise FileNotFoundError(
            f"Artifact is not a file: {requested!s}. If using $ARTIFACT, define it in the current shell first."
        )
    return requested


def add_isosurface(axis, field, level, origin, spacing, color, alpha, step_size, zorder):
    """Marching-cubes isosurface. Warns rather than failing silently when
    the level lies outside the field, so an empty panel is distinguishable
    from a genuinely cloud-free one."""
    field = np.asarray(field, dtype=np.float32)
    if not np.isfinite(field).any():
        warnings.warn("isosurface skipped, the field is entirely non-finite")
        return None
    low, high = float(np.nanmin(field)), float(np.nanmax(field))
    if not low <= level <= high:
        warnings.warn(f"isosurface at {level:g} skipped, field spans [{low:g}, {high:g}]")
        return None
    vertices, faces, _, _ = marching_cubes(
        field, level=level, spacing=spacing, step_size=step_size, allow_degenerate=False
    )
    vertices += np.asarray(origin)[None, :]
    collection = Poly3DCollection(
        vertices[faces], facecolor=color, edgecolor="none", alpha=alpha, linewidth=0.0, zorder=zorder
    )
    collection.set_rasterized(True)
    axis.add_collection3d(collection)
    return collection


def select_volume(dataset, requested_time):
    """Nearest stored 3-D volume. With no request, take the last one stored
    rather than the last diagnostic time, which need not coincide."""
    if "cloud_water_3d" in dataset:
        available = np.asarray(dataset.volume_time.values, dtype=float)
        if available.size == 0:
            raise ValueError("The artifact contains no stored 3-D volumes")
        if requested_time is None:
            index = available.size - 1
        else:
            index = int(np.argmin(np.abs(available - requested_time)))
            if not np.isclose(available[index], requested_time):
                raise ValueError(
                    f"No 3-D volume at t={requested_time:g} s; available: "
                    + ", ".join(f"{value:g}" for value in available)
                )
        return (float(available[index]), dataset.cloud_water_3d.values[index], dataset.rain_water_3d.values[index])

    final_time = float(dataset.time.values[-1])
    if requested_time is not None and not np.isclose(requested_time, final_time):
        raise ValueError("This older artifact stores a full 3-D volume only at the final time")
    return final_time, dataset.final_cloud_water.values, dataset.final_rain_water.values


def surface_reference(dataset, mode, surface_shape):
    """Reference removed from the (nx, ny) surface theta_d field.

    'initial' subtracts the t=0 surface field, which equals the base state
    only when the initialization put no perturbation at the ground.
    'environment' subtracts the lowest level of the stored environment
    profile, matching what the vertical slice removes. That profile is
    (nz,) or (nx, nz), never (nx, ny), so it needs explicit extraction.
    """
    if mode == "initial":
        return dataset.surface_dry_potential_temperature.values[0]

    env = np.asarray(dataset.environment_dry_potential_temperature.values)
    nz = int(dataset.z.size)
    nx = int(surface_shape[0])

    if env.ndim == 1 and env.size == nz:
        return float(env[0])
    if env.ndim == 2 and env.shape == (nx, nz):
        return env[:, 0][:, None]  # broadcast the surface level over y
    if env.ndim == 2 and env.shape == (nz, nx):
        return env[0, :][:, None]
    raise ValueError(
        f"Cannot read a surface reference from an environment field of shape "
        f"{env.shape} with nx={nx}, nz={nz}. Use --surface-reference initial."
    )


def save_volume_panel(
    dataset,
    requested_time,
    path,
    show_yz,
    mesh_step,
    boundary_mask_km,
    cloud_threshold,
    cloud_alpha,
    reference_mode,
    elevation,
    azimuth,
):
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    z = dataset.z.values / 1000.0
    dx = float(dataset.attrs["dx_m"]) / 1000.0
    dy = float(dataset.attrs.get("dy_m", dataset.attrs["dx_m"])) / 1000.0
    dz = float(dataset.attrs["dz_m"]) / 1000.0

    selected_time, cloud, rain = select_volume(dataset, requested_time)
    time_index = int(np.argmin(np.abs(np.asarray(dataset.time.values) - selected_time)))

    theta_xz = (
        dataset.dry_potential_temperature_xz.values[time_index] - dataset.environment_dry_potential_temperature.values
    )
    surface = dataset.surface_dry_potential_temperature.values[time_index]
    theta_surface = surface - surface_reference(dataset, reference_mode, surface.shape)

    norm = theta_norm()
    cmap = plt.get_cmap(THETA_CMAP)

    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    axis = fig.add_axes(
        [PANEL_INSET, PANEL_INSET, 1.0 - 2 * PANEL_INSET, 1.0 - 2 * PANEL_INSET],
        projection="3d",
        computed_zorder=False,
    )

    # Ground slice. Per-face alpha is baked into the RGBA array; passing
    # alpha= to plot_surface would overwrite it and silently kill the mask.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ground = np.asarray(cmap(norm(np.clip(theta_surface, -THETA_LIMIT, THETA_LIMIT))), dtype=np.float64)
    ground[..., 3] = GROUND_ALPHA
    if boundary_mask_km > 0.0:
        edge_x = np.minimum(x - x.min(), x.max() - x)
        edge_y = np.minimum(y - y.min(), y.max() - y)
        interior = (edge_x[:, None] >= boundary_mask_km) & (edge_y[None, :] >= boundary_mask_km)
        ground[..., 3] = np.where(interior, GROUND_ALPHA, 0.0)
    axis.plot_surface(
        xx,
        yy,
        np.zeros_like(xx),
        facecolors=ground,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=0,
    )

    # Vertical theta slice at the mid-domain y, behind the hydrometeors.
    xx_v, zz_v = np.meshgrid(x, z, indexing="ij")
    y_v = np.full_like(xx_v, y[len(y) // 2])
    slice_colors = np.asarray(cmap(norm(np.clip(theta_xz, -THETA_LIMIT, THETA_LIMIT))), dtype=np.float64)
    slice_colors[..., 3] = SLICE_ALPHA
    axis.plot_surface(
        xx_v,
        y_v,
        zz_v,
        facecolors=slice_colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=1,
    )

    origin = (float(x[0]), float(y[0]), float(z[0]))
    spacing = (dx, dy, dz)
    add_isosurface(axis, rain, QR_THRESHOLD, origin, spacing, RAIN_COLOR, RAIN_ALPHA, mesh_step, zorder=2)
    add_isosurface(axis, cloud, cloud_threshold, origin, spacing, CLOUD_COLOR, cloud_alpha, mesh_step, zorder=3)

    z_top = float(z.max() + 0.5 * dz)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_zlim(0.0, z_top)

    axis.set_xlabel("$x$ (km)", labelpad=LABELPAD_X)
    axis.tick_params(axis="x", pad=TICKPAD_3D)

    if show_yz:
        axis.set_ylabel("$y$ (km)", labelpad=LABELPAD_Y)
        axis.set_zlabel("$z$ (km)", labelpad=LABELPAD_Z)
        axis.tick_params(axis="y", pad=TICKPAD_3D)
        axis.tick_params(axis="z", pad=TICKPAD_3D)
    else:
        # Ticks stay, labels go, so the box keeps its shape but the panel
        # loses the text width it was duplicating from its neighbour.
        axis.set_yticklabels([])
        axis.set_zticklabels([])
        axis.tick_params(axis="y", length=0, pad=0)
        axis.tick_params(axis="z", length=0, pad=0)

    # Aspect from the data, with a fixed vertical exaggeration.
    axis.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), Z_EXAGGERATION * z_top), zoom=BOX_ZOOM)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#bbbbbb")

    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return selected_time


def save_theta_colorbar(path):
    cmap = plt.get_cmap(THETA_CMAP)
    fig_h = THETA_BAR_H_IN + 2 * CBAR_PAD_IN
    rect = [CBAR_LEFT_IN / CBAR_W_IN, CBAR_PAD_IN / fig_h, CBAR_BAR_W_IN / CBAR_W_IN, THETA_BAR_H_IN / fig_h]

    fig = plt.figure(figsize=(CBAR_W_IN, fig_h))
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(
        ScalarMappable(norm=theta_norm(), cmap=cmap), cax=cax, orientation="vertical", ticks=THETA_TICKS
    )
    cbar.set_label(THETA_LABEL, labelpad=10)
    cbar.outline.set_linewidth(1.5)
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def save_legend(path, cloud_threshold, cloud_alpha):
    """Trimmed to its own content and appended directly beside panel c, so
    it must carry no slack of its own."""
    handles = [
        Patch(
            facecolor=CLOUD_COLOR,
            edgecolor=CLOUD_EDGE_COLOR,
            alpha=cloud_alpha,
            linewidth=1.2,
            label=(rf"$q_c = {cloud_threshold / 1.0e-5:g}" r"\times 10^{-5}$ kg kg$^{-1}$"),
        ),
        Patch(
            facecolor=RAIN_COLOR,
            edgecolor=RAIN_EDGE_COLOR,
            alpha=RAIN_ALPHA,
            linewidth=1.2,
            label=r"$q_r = 10^{-4}$ kg kg$^{-1}$",
        ),
    ]
    fig = plt.figure(figsize=(LEGEND_W_IN, LEGEND_H_IN))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        handleheight=1.2,
        labelspacing=1.2,
    )
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def rain_geometry(x, y):
    """Equal aspect, so the canvas height follows the domain aspect ratio."""
    ax_h = RAIN_AX_W_IN * float(np.ptp(y)) / float(np.ptp(x))
    fig_h = RAIN_AX_BOTTOM_IN + ax_h + RAIN_AX_PAD_IN
    rect = [RAIN_AX_LEFT_IN / RAIN_W_IN, RAIN_AX_BOTTOM_IN / fig_h, RAIN_AX_W_IN / RAIN_W_IN, ax_h / fig_h]
    return fig_h, rect, ax_h


def save_rain_panel(dataset, path, requested_time):
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    times = np.asarray(dataset.time.values, dtype=float)
    index = -1 if requested_time is None else int(np.argmin(np.abs(times - requested_time)))

    rain = dataset.accumulated_rain.values[index]
    cloud = dataset.column_maximum_cloud_water.values[index]
    rain_water = dataset.column_maximum_rain_water.values[index]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    rain_max = max(1.0, float(np.nanmax(rain)))
    levels = np.linspace(0.0, rain_max, RAIN_NLEVELS)

    fig_h, rect, ax_h = rain_geometry(x, y)
    fig = plt.figure(figsize=(RAIN_W_IN, fig_h))
    axis = fig.add_axes(rect)
    axis.contourf(xx, yy, rain, levels=levels, cmap=RAIN_CMAP, extend="max")
    if np.nanmax(cloud) >= QC_THRESHOLD:
        axis.contour(xx, yy, cloud, levels=[QC_THRESHOLD], colors=CLOUD_CONTOUR_COLOR, linewidths=0.9)
    if np.nanmax(rain_water) >= QR_THRESHOLD:
        axis.contour(xx, yy, rain_water, levels=[QR_THRESHOLD], colors=RAIN_CONTOUR_COLOR, linewidths=1.0)
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_xlabel("$x$ (km)")
    axis.set_ylabel("$y$ (km)")

    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)
    return float(times[index]), rain_max, ax_h


def save_rain_colorbar(path, rain_max, bar_h_in):
    """Padded to mirror the rain panel's own canvas, so the bar sits level
    with the axes rather than with the canvas centre. Centring the two
    trimmed images instead misaligns them, because panel c carries an
    x-label strip below its axes and the bar does not."""
    cmap = plt.get_cmap(RAIN_CMAP)
    norm = colors.BoundaryNorm(np.linspace(0.0, rain_max, RAIN_NLEVELS), ncolors=cmap.N, extend="max")

    fig_h = RAIN_AX_BOTTOM_IN + bar_h_in + RAIN_AX_PAD_IN
    rect = [CBAR_LEFT_IN / CBAR_W_IN, RAIN_AX_BOTTOM_IN / fig_h, CBAR_BAR_W_IN / CBAR_W_IN, bar_h_in / fig_h]

    fig = plt.figure(figsize=(CBAR_W_IN, fig_h))
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical")
    cbar.set_label(RAIN_LABEL, labelpad=10)
    cbar.outline.set_linewidth(1.5)
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def render(
    artifact: Path | None = None,
    output_dir: Path | None = None,
    mesh_step: int = 2,
    elevation: float = VIEW_ELEV,
    azimuth: float = VIEW_AZIM,
    cloud_alpha: float = CLOUD_ALPHA,
    boundary_mask_km: float = 0.0,
    cloud_threshold: float = VISUAL_QC_THRESHOLD,
    reference_mode: str = "initial",
    rain_time: float | None = None,
) -> list[Path]:
    """Write the six bare panels that make_fig.sh composites."""
    artifact = resolve_artifact(artifact)
    output_dir = Path(output_dir) if output_dir is not None else SCRIPT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    with xr.open_dataset(artifact) as source:
        dataset = source.load()

    written = []
    for requested_time, stem, show_yz in PANELS:
        path = output_dir / f"{stem}.png"
        rendered = save_volume_panel(
            dataset,
            requested_time,
            path,
            show_yz,
            mesh_step,
            boundary_mask_km,
            cloud_threshold,
            cloud_alpha,
            reference_mode,
            elevation,
            azimuth,
        )
        written.append(path)
        print(f"{path.name}  |  t = {rendered:g} s ({rendered / 3600.0:g} h)")

    theta_bar_path = output_dir / "supercell_theta_colorbar.png"
    save_theta_colorbar(theta_bar_path)
    written.append(theta_bar_path)

    rain_path = output_dir / "supercell_rain_main.png"
    rain_bar_path = output_dir / "supercell_rain_colorbar.png"
    legend_path = output_dir / "supercell_legend.png"
    selected_rain_time, rain_max, rain_ax_h = save_rain_panel(dataset, rain_path, rain_time)
    save_rain_colorbar(rain_bar_path, rain_max, rain_ax_h)
    save_legend(legend_path, cloud_threshold, cloud_alpha)
    written.extend([rain_path, rain_bar_path, legend_path])

    print(f"{rain_path.name}  |  t = {selected_rain_time:g} s  |  peak {rain_max:.2f} mm")
    print(f"theta bar {THETA_BAR_H_IN:.2f} in tall, rain bar {rain_ax_h:.2f} in tall")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", type=Path, default=None, help="Artifact NetCDF file or execution bundle directory"
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory receiving the bare panels")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Legacy single-file path from the paper suite; only its parent directory is used",
    )
    parser.add_argument("--mesh-step", type=int, default=2, help="marching-cubes stride; 1 gives the finest mesh")
    parser.add_argument("--elevation", type=float, default=VIEW_ELEV, help="camera elevation in degrees")
    parser.add_argument("--azimuth", type=float, default=VIEW_AZIM, help="camera azimuth in degrees")
    parser.add_argument(
        "--cloud-alpha", type=float, default=CLOUD_ALPHA, help="opacity of the visual cloud-water isosurface"
    )
    parser.add_argument(
        "--boundary-mask-km",
        type=float,
        default=0.0,
        help="transparent outer buffer on the ground theta slice only, never on cloud or rain",
    )
    parser.add_argument(
        "--cloud-threshold",
        type=float,
        default=VISUAL_QC_THRESHOLD,
        help="visual cloud-water isosurface (kg kg-1)",
    )
    parser.add_argument(
        "--surface-reference",
        choices=("environment", "initial"),
        default="initial",
        help="reference removed from the surface theta field",
    )
    parser.add_argument(
        "--rain-time",
        type=float,
        default=None,
        help="time in seconds for the rain panel; defaults to the last diagnostic time",
    )
    args = parser.parse_args()
    if args.mesh_step < 1:
        raise ValueError("--mesh-step must be positive")
    if args.cloud_threshold <= 0.0:
        raise ValueError("--cloud-threshold must be positive")
    if not 0.0 <= args.cloud_alpha <= 1.0:
        raise ValueError("--cloud-alpha must lie in [0, 1]")

    output_dir = args.output_dir
    if output_dir is None and args.output is not None:
        output_dir = Path(args.output).parent
    for path in render(
        args.artifact,
        output_dir,
        args.mesh_step,
        args.elevation,
        args.azimuth,
        args.cloud_alpha,
        args.boundary_mask_km,
        args.cloud_threshold,
        args.surface_reference,
        args.rain_time,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
