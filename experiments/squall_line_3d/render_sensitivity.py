#!/usr/bin/env python3
"""Render squall-line precipitation sensitivities as separate panels.

Figure 8, assembled by make_fig.sh:

    [ theta map | theta section | theta bar ]
    [   qv map  |   qv section  |   qv bar  ]
    [              legend strip             ]

Each row carries one colorbar and its own colormap. Note that the map is a
column sum over the boundary layer while the section is a local value, so
a shared norm necessarily compresses one of them; --separate-limits gives
each panel its own bar instead.

Every panel is bare. No titles, no inline colorbars, no inline legend.
Panel letters are composited by make_fig.sh.

Do NOT add bbox_inches='tight' to any savefig here, and do not -trim these
files in the shell. Each colorbar strip is built on the same canvas height
as the row it belongs to, which is what makes a plain top-aligned append
line the bar up with the axes.

With no arguments the artifact is taken from beside this script, otherwise
from the standard experiment bundle, and the panels are written to
./figures next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import xarray as xr

from suetes.shared.experiment import ExperimentLayout, artifact_from_bundle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_NAME = "sensitivity.nc"
# Bundle written by the paper suite, used when no artifact sits beside this file.
DEFAULT_BUNDLE = ExperimentLayout(kind="experiments", case="squall_line_3d_sensitivity", execution="paper").root

# --- Plot settings -------------------------------------------------------
# Figure 7 already spends RdBu_r on theta_d', so these rows use different
# diverging maps to keep the two figures from being read as the same field.
THETA_CMAP = "PiYG_r"
VAPOR_CMAP = "PuOr_r"

# Short labels. The perturbation is stated once per bar as "per X" rather
# than folded into a long phrase with the layer depth.
THETA_LABEL = r"$\frac{\partial J}{\partial \theta}$ (mm h$^{-1}$K$^{-1}$) "
VAPOR_LABEL = r"$\frac{\partial J}{\partial q}$ (mm h$^{-1}$g kg$^{{-1}}$) "

CLOUD_CONTOUR_COLOR = "#e69f00"
CLOUD_CONTOUR_LW = 1.1
UPDRAFT_CONTOUR_COLOR = "k"
UPDRAFT_CONTOUR_LW = 1.0
CLOUD_LEVEL = 1.0e-4  # kg kg-1
UPDRAFT_LEVEL = 5.0  # m s-1
STORM_PAD_KM = 15.0
SECTION_Z_TOP_KM = 15.0

# --- Geometry, all in inches --------------------------------------------
# Map and section sit side by side, so they share one axes height and the
# map's width follows from its equal aspect. The upper row omits the
# distance axis, which the lower row carries for both.
ROW_AX_H_IN = 3.00
ROW_AX_TOP_PAD_IN = 0.50  # clearance for the colorbar offset text
ROW_AX_BOTTOM_LABELLED_IN = 0.72
ROW_AX_BOTTOM_BARE_IN = 0.22

MAP_AX_LEFT_IN = 1.05
MAP_AX_RIGHT_PAD_IN = 0.30

SEC_W_IN = 7.60
SEC_AX_LEFT_IN = 1.05
SEC_AX_RIGHT_PAD_IN = 0.30
SEC_AX_W_IN = SEC_W_IN - SEC_AX_LEFT_IN - SEC_AX_RIGHT_PAD_IN

# Colorbar strips. Labels are deliberately short, so these stay narrow.
CBAR_W_IN = 1.90
CBAR_BAR_W_IN = 0.40
CBAR_LEFT_IN = 0.10

LEGEND_W_IN = 8.00
LEGEND_H_IN = 0.85

# Taylor panels. Written for reference; not part of figure 8.
TAY_W_IN = 5.20
TAY_AX_LEFT_IN = 1.15
TAY_AX_BOTTOM_IN = 0.78
TAY_AX_PAD_IN = 0.22
TAY_AX_H_IN = 3.00
TAY_FIG_H_IN = TAY_AX_BOTTOM_IN + TAY_AX_H_IN + TAY_AX_PAD_IN
TAY_LEGEND_W_IN = 6.00
TAY_LEGEND_H_IN = 0.80


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
        raise FileNotFoundError(requested)
    return requested


def symmetric_limit(*fields):
    """Robust symmetric colour limit over one or more fields. The 99th
    percentile keeps a handful of extreme cells from flattening everything
    else to white."""
    pooled = []
    for field in fields:
        values = np.asarray(field)
        pooled.append(np.abs(values[np.isfinite(values)]).ravel())
    finite = np.concatenate(pooled) if pooled else np.array([])
    if not finite.size:
        return 1.0
    return max(float(np.percentile(finite, 99.0)), 1.0e-20)


def storm_limits(coordinate, occupied, padding, domain_limit=None):
    indices = np.flatnonzero(occupied)
    if not indices.size:
        return float(coordinate[0]), float(coordinate[-1])
    lower = max(float(coordinate[indices[0]]) - padding, float(coordinate[0]))
    upper = min(float(coordinate[indices[-1]]) + padding, float(coordinate[-1]))
    if domain_limit is not None:
        lower = max(lower, domain_limit[0])
        upper = min(upper, domain_limit[1])
    return lower, upper


def scientific_formatter():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    return formatter


def row_bottom(show_xlabel):
    return ROW_AX_BOTTOM_LABELLED_IN if show_xlabel else ROW_AX_BOTTOM_BARE_IN


def row_height(show_xlabel):
    return row_bottom(show_xlabel) + ROW_AX_H_IN + ROW_AX_TOP_PAD_IN


def save_colorbar(out_dir, out, norm, cmap, label, show_xlabel):
    """Padded to mirror the row it accompanies, so the bar sits level with
    the axes rather than with the canvas centre. Do not trim this file."""
    fig_h = row_height(show_xlabel)
    bottom = row_bottom(show_xlabel)
    rect = [CBAR_LEFT_IN / CBAR_W_IN, bottom / fig_h, CBAR_BAR_W_IN / CBAR_W_IN, ROW_AX_H_IN / fig_h]

    fig = plt.figure(figsize=(CBAR_W_IN, fig_h))
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
        cax=cax,
        orientation="vertical",
        format=scientific_formatter(),
    )
    cbar.set_label(label, labelpad=10)
    cbar.outline.set_linewidth(1.5)
    cbar.update_ticks()
    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def map_geometry(x_limits, y_limits, show_xlabel):
    """Equal aspect with a fixed axes height, so the canvas width follows
    from the plotted extent."""
    fig_h = row_height(show_xlabel)
    span_x = float(x_limits[1] - x_limits[0])
    span_y = float(y_limits[1] - y_limits[0])
    ax_w = ROW_AX_H_IN * span_x / span_y
    fig_w = MAP_AX_LEFT_IN + ax_w + MAP_AX_RIGHT_PAD_IN
    rect = [MAP_AX_LEFT_IN / fig_w, row_bottom(show_xlabel) / fig_h, ax_w / fig_w, ROW_AX_H_IN / fig_h]
    return fig_w, fig_h, rect


def save_map_panel(out_dir, out, x, y, field, norm, cmap, x_limits, y_limits, show_xlabel):
    fig_w, fig_h, rect = map_geometry(x_limits, y_limits, show_xlabel)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axis = fig.add_axes(rect)

    axis.pcolormesh(x, y, field.T, shading="auto", cmap=cmap, norm=norm)

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal")
    axis.set_ylabel("$y$ (km)")
    if show_xlabel:
        axis.set_xlabel("$x$ (km)")
    else:
        axis.set_xticklabels([])

    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def save_section_panel(
    out_dir, out, x, z, field, norm, cmap, cloud, vertical_velocity, x_limits, z_top, show_xlabel
):
    fig_h = row_height(show_xlabel)
    rect = [SEC_AX_LEFT_IN / SEC_W_IN, row_bottom(show_xlabel) / fig_h, SEC_AX_W_IN / SEC_W_IN, ROW_AX_H_IN / fig_h]

    fig = plt.figure(figsize=(SEC_W_IN, fig_h))
    axis = fig.add_axes(rect)

    axis.pcolormesh(x, z, field.T, shading="auto", cmap=cmap, norm=norm)
    if float(np.nanmax(cloud)) >= CLOUD_LEVEL:
        axis.contour(x, z, cloud.T, levels=[CLOUD_LEVEL], colors=CLOUD_CONTOUR_COLOR, linewidths=CLOUD_CONTOUR_LW)
    if float(np.nanmax(vertical_velocity)) >= UPDRAFT_LEVEL:
        axis.contour(
            x,
            z,
            vertical_velocity.T,
            levels=[UPDRAFT_LEVEL],
            colors=UPDRAFT_CONTOUR_COLOR,
            linewidths=UPDRAFT_CONTOUR_LW,
        )

    axis.set_xlim(x_limits)
    axis.set_ylim(0.0, z_top)
    axis.set_ylabel("$z$ (km)")
    if show_xlabel:
        axis.set_xlabel("$x$ (km)")
    else:
        axis.set_xticklabels([])

    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def save_legend(out_dir, out):
    """Covers the section overlays, which are the only ones drawn."""
    handles = [
        Line2D([0], [0], color=CLOUD_CONTOUR_COLOR, lw=CLOUD_CONTOUR_LW, label=r"$q_c = 0.1$ g kg$^{-1}$"),
        Line2D([0], [0], color=UPDRAFT_CONTOUR_COLOR, lw=UPDRAFT_CONTOUR_LW, label=r"$w = 5$ m s$^{-1}$"),
    ]
    fig = plt.figure(figsize=(LEGEND_W_IN, LEGEND_H_IN))
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=16, handlelength=2.4, columnspacing=3.0)
    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def save_taylor_panel(out_dir, out, amplitude, actual, linear, unit, symbol, show_ylabel):
    left = TAY_AX_LEFT_IN if show_ylabel else 0.45
    ax_w = TAY_W_IN - left - TAY_AX_PAD_IN
    rect = [left / TAY_W_IN, TAY_AX_BOTTOM_IN / TAY_FIG_H_IN, ax_w / TAY_W_IN, TAY_AX_H_IN / TAY_FIG_H_IN]

    fig = plt.figure(figsize=(TAY_W_IN, TAY_FIG_H_IN))
    axis = fig.add_axes(rect)
    axis.plot(amplitude, actual, "o-", color="#1f78b4", lw=1.8, ms=6)
    axis.plot(amplitude, linear, "s--", color="#b2182b", lw=1.8, ms=6)
    axis.set_xscale("log")
    axis.set_xlabel(f"Maximum perturbation ({unit})")
    if show_ylabel:
        axis.set_ylabel(rf"$\frac{{\partial J}}{{\partial {symbol}}}$ (mm h$^{{-1}}$)")
    axis.grid(alpha=0.25)

    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def save_taylor_legend(out_dir, out):
    handles = [
        Line2D([0], [0], color="#1f78b4", lw=1.8, marker="o", ms=6, label="Centered nonlinear"),
        Line2D([0], [0], color="#b2182b", lw=1.8, marker="s", ms=6, linestyle="--", label="Adjoint"),
    ]
    fig = plt.figure(figsize=(TAY_LEGEND_W_IN, TAY_LEGEND_H_IN))
    fig.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=16, handlelength=2.6, columnspacing=2.6)
    output = out_dir / f"{out}.png"
    fig.savefig(output, transparent=False, facecolor="white")
    plt.close(fig)
    return output


def render(
    artifact: Path | None = None,
    output_dir: Path | None = None,
    theta_perturbation=0.1,
    vapor_perturbation=1.0e-4,
    boundary_layer_top=3000.0,
    separate_limits=False,
) -> list[Path]:
    """Write the bare panels, colorbars and legends that make_fig.sh composites."""
    artifact = resolve_artifact(artifact)
    output_dir = Path(output_dir) if output_dir is not None else SCRIPT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    with xr.open_dataset(artifact) as source:
        dataset = source.load()

    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    z = dataset.z.values / 1000.0

    # The target mask no longer appears in the figure, but it still defines
    # the plotting extent for every panel.
    target = dataset.target_mask.values.astype(bool)
    boundary_layer = dataset.z.values <= boundary_layer_top
    theta_map = theta_perturbation * np.sum(dataset.theta_sensitivity.values[:, :, boundary_layer], axis=2)
    vapor_map = vapor_perturbation * np.sum(dataset.water_vapor_sensitivity.values[:, :, boundary_layer], axis=2)
    theta_section = theta_perturbation * dataset.theta_sensitivity_xz.values
    vapor_section = vapor_perturbation * dataset.water_vapor_sensitivity_xz.values

    cloud = dataset.spinup_cloud_water_xz.values
    vertical_velocity = dataset.spinup_vertical_velocity_xz.values
    x_limits = storm_limits(x, np.any(target, axis=1), STORM_PAD_KM)
    y_limits = storm_limits(y, np.any(target, axis=0), STORM_PAD_KM)
    z_top = min(SECTION_Z_TOP_KM, float(z[-1]))

    written = []

    # (map, section, cmap, label, map stem, section stem, bar stem, x axis).
    # The upper row omits the distance axis, which the lower row carries.
    rows = [
        (
            theta_map,
            theta_section,
            THETA_CMAP,
            THETA_LABEL,
            "sens_theta_map",
            "sens_theta_section",
            "sens_theta_cbar",
            False,
        ),
        (
            vapor_map,
            vapor_section,
            VAPOR_CMAP,
            VAPOR_LABEL,
            "sens_vapor_map",
            "sens_vapor_section",
            "sens_vapor_cbar",
            True,
        ),
    ]

    for field_map, field_section, cmap, label, map_stem, sec_stem, bar_stem, show_xlabel in rows:
        map_limit = symmetric_limit(field_map)
        sec_limit = symmetric_limit(field_section)

        if separate_limits:
            map_norm = colors.TwoSlopeNorm(vmin=-map_limit, vcenter=0.0, vmax=map_limit)
            sec_norm = colors.TwoSlopeNorm(vmin=-sec_limit, vcenter=0.0, vmax=sec_limit)
        else:
            shared = symmetric_limit(field_map, field_section)
            map_norm = sec_norm = colors.TwoSlopeNorm(vmin=-shared, vcenter=0.0, vmax=shared)

        written.append(
            save_map_panel(
                output_dir, f"{map_stem}_main", x, y, field_map, map_norm, cmap, x_limits, y_limits, show_xlabel
            )
        )
        written.append(
            save_section_panel(
                output_dir,
                f"{sec_stem}_main",
                x,
                z,
                field_section,
                sec_norm,
                cmap,
                cloud,
                vertical_velocity,
                x_limits,
                z_top,
                show_xlabel,
            )
        )

        if separate_limits:
            written.append(save_colorbar(output_dir, f"{map_stem}_cbar", map_norm, cmap, label, show_xlabel))
            written.append(save_colorbar(output_dir, f"{sec_stem}_cbar", sec_norm, cmap, label, show_xlabel))
        else:
            written.append(save_colorbar(output_dir, bar_stem, map_norm, cmap, label, show_xlabel))

        # The map is a column sum and the section is a local value, so a
        # large ratio here means the shared bar is flattening one of them.
        print(
            f"{map_stem}  |  map limit {map_limit:.3e}  |  "
            f"section limit {sec_limit:.3e}  |  "
            f"ratio {map_limit / sec_limit:.1f}"
        )

    written.append(save_legend(output_dir, "sens_legend"))

    # Written for reference. make_fig.sh does not assemble these into
    # figure 8.
    taylor = [
        (
            dataset.theta_taylor_amplitude.values,
            dataset.theta_taylor_actual_change.values,
            dataset.theta_taylor_linear_change.values,
            "K",
            r"\theta",
            "sens_taylor_a_main",
            True,
        ),
        (
            dataset.q_taylor_amplitude.values * 1000.0,
            dataset.water_vapor_taylor_actual_change.values,
            dataset.water_vapor_taylor_linear_change.values,
            r"g kg$^{-1}$",
            "q",
            "sens_taylor_b_main",
            False,
        ),
    ]
    for amplitude, actual, linear, unit, symbol, stem, show_ylabel in taylor:
        written.append(save_taylor_panel(output_dir, stem, amplitude, actual, linear, unit, symbol, show_ylabel))
    written.append(save_taylor_legend(output_dir, "sens_taylor_legend"))

    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-perturbation", type=float, default=0.1)
    parser.add_argument(
        "--vapor-perturbation", type=float, default=1.0e-4, help="display perturbation in kg kg-1 (default: 0.1 g kg-1)"
    )
    parser.add_argument(
        "--boundary-layer-top", type=float, default=3000.0, help="top of the layer included in column-response maps (m)"
    )
    parser.add_argument(
        "--separate-limits",
        action="store_true",
        help="give the map and the section their own colour limits and their own bars, not one shared bar per row",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for path in render(
        args.artifact,
        args.output_dir,
        args.theta_perturbation,
        args.vapor_perturbation,
        args.boundary_layer_top,
        args.separate_limits,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
