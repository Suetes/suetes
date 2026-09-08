#!/usr/bin/env python3
# Case renderer.
"""Render publication panels for the gravity-wave topography inversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

from suetes.shared.experiment import ExperimentLayout, artifact_from_bundle

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parents[1]
CASE = "gravity_wave_optimal_topography"
ARTIFACT_PATTERN = "gravity_wave_optimal_topography_*.nc"

# --- Geometry, all in inches --------------------------------------------
# Every panel gets an identically sized plotting area; only the surrounding
# margins change. Margins are the minimum needed for the text that is
# actually drawn, so nothing is left over as gutter when the panels are
# appended. Do NOT add bbox_inches="tight" to savefig, it undoes all of it.
PANEL_W_IN = 10.00
AX_LEFT_IN = 0.90  # y tick labels + rotated y axis label
AX_RIGHT_PAD_IN = 0.25  # overhang of the last x tick label
AX_W_IN = PANEL_W_IN - AX_LEFT_IN - AX_RIGHT_PAD_IN
AX_H_IN = 3.20
AX_BOTTOM_IN = 0.68  # x tick labels + x axis label
AX_PAD_IN = 0.15  # tick overhang only, used where no labels are drawn

FIG_H_LABELLED_IN = AX_BOTTOM_IN + AX_H_IN + AX_PAD_IN
FIG_H_BARE_IN = AX_PAD_IN + AX_H_IN + AX_PAD_IN
BLOCK_H_IN = FIG_H_LABELLED_IN + FIG_H_BARE_IN

CBAR_W_IN = 1.50
CBAR_BAR_W_IN = 0.45
CBAR_LEFT_IN = 0.10

# The evolution panel spans the assembled width and its axes are aligned to
# the plotting area of the wave panels above, leaving the colorbar column clear.
EVO_WIDTH_IN = 2 * PANEL_W_IN + CBAR_W_IN
EVO_HEIGHT_IN = 3.40
EVO_AX_W_IN = PANEL_W_IN + AX_W_IN

# --- Plot settings -------------------------------------------------------
XLIM = [-30, 40]
ZLIM = [0, 12]
X_TICKS = [-30, -20, -10, 0, 10, 20, 30, 40]
Y_TICKS = [0, 2, 4, 6, 8, 10, 12]

# Recovered from the data, not stored as an attribute: summing w**2 over
# exactly this region reproduces validation_energy for all four cases.
TARGET_X_KM = (15.0, 25.0)
TARGET_Z_KM = (3.0, 8.0)

# Scale set by the three controls, so they span the full range and the
# optimized case saturates. Raise to 8.4 to clip nothing.
W_MAX = 4.3
CB_TICKS = np.arange(-4.0, 4.1, 1.0)
CMAP_FIELD = "RdBu_r"
CBAR_LABEL = r"$W$ (m/s)"

EVO_XLIM = XLIM
EVO_X_TICKS = X_TICKS
EVO_YLIM = [0, 1550]
EVO_Y_TICKS = [0, 500, 1000, 1500]
EVO_COLOR = "royalblue"
EVO_BEST_COLOR = "red"

# (name in the file, output stem, y axis labelled, x axis labelled).
# The bottom row carries the distance axis for both rows.
PANELS = [
    ("Optimal topography", "gravity_wave_optimal_main", True, False),
    ("Random topography 1", "gravity_wave_random1_main", False, False),
    ("Random topography 2", "gravity_wave_random2_main", True, True),
    ("Random topography 3", "gravity_wave_random3_main", False, True),
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


def default_source() -> Path:
    """An artifact sitting beside this script wins, otherwise the bundle copy."""
    beside = sorted(SCRIPT_DIR.glob(ARTIFACT_PATTERN))
    if beside:
        return beside[0]
    layout = ExperimentLayout(kind="experiments", case=CASE, output_root=REPOSITORY / "output")
    return artifact_from_bundle(layout.data, ARTIFACT_PATTERN)


def load(artifact: Path) -> dict:
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    return {
        "labels": [str(label) for label in data["configuration"].values],
        "x_m": data["x"].values,
        "z_km": data["physical_height"].values / 1000.0,
        "w": data["vertical_velocity"].values,
        "energy": data["validation_energy"].values,
        "centers_m": data["rbf_center"].values,
        "sigma_m": float(data.attrs["rbf_sigma_m"]),
        "coef_history": data["coefficient_history"].values,
        "opt_energy": data["optimization_energy"].values,
        "best_index": int(data.attrs["best_optimization_index"]),
    }


def panel_geometry(show_xlabel):
    """Figure size and axes rect. The plotting area is the same either way,
    the canvas just loses the strip reserved for the distance axis."""
    height = FIG_H_LABELLED_IN if show_xlabel else FIG_H_BARE_IN
    bottom = AX_BOTTOM_IN if show_xlabel else AX_PAD_IN
    rect = [AX_LEFT_IN / PANEL_W_IN, bottom / height, AX_W_IN / PANEL_W_IN, AX_H_IN / height]
    return (PANEL_W_IN, height), rect


def draw_target_box(ax):
    ax.add_patch(
        Rectangle(
            (TARGET_X_KM[0], TARGET_Z_KM[0]),
            TARGET_X_KM[1] - TARGET_X_KM[0],
            TARGET_Z_KM[1] - TARGET_Z_KM[0],
            fill=False,
            edgecolor="black",
            linestyle="--",
            linewidth=2.0,
            zorder=5,
        )
    )


def save_panel(x_km, z_km, w, levels, output, show_ylabel, show_xlabel) -> Path:
    figsize, rect = panel_geometry(show_xlabel)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    # physical_height is terrain-following, so x needs broadcasting to the
    # full 2D mesh. Level 0 of that mesh is the surface.
    x_plot = np.broadcast_to(x_km[:, None], z_km.shape)
    ax.contourf(x_plot, z_km, w, levels=levels, cmap=CMAP_FIELD, extend="both")
    ax.fill_between(x_km, 0, z_km[:, 0], color="black")
    draw_target_box(ax)

    ax.set_xlim(XLIM)
    ax.set_ylim(ZLIM)
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)

    if show_ylabel:
        ax.set_ylabel("Altitude (km)")
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("Distance (km)")
    else:
        ax.set_xticklabels([])

    fig.savefig(output, transparent=False)
    plt.close(fig)
    return output


def save_colorbar(levels, output) -> Path:
    cmap = plt.get_cmap(CMAP_FIELD)
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend="both")

    # Spans from the bottom of the lower row's axes to the top of the
    # upper row's axes, so it ends exactly level with the panels.
    height = BLOCK_H_IN - AX_BOTTOM_IN - AX_PAD_IN
    rect = [
        CBAR_LEFT_IN / CBAR_W_IN,
        AX_BOTTOM_IN / BLOCK_H_IN,
        CBAR_BAR_W_IN / CBAR_W_IN,
        height / BLOCK_H_IN,
    ]

    fig = plt.figure(figsize=(CBAR_W_IN, BLOCK_H_IN))
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=CB_TICKS)
    cbar.set_label(CBAR_LABEL, labelpad=10)
    cbar.outline.set_linewidth(1.5)

    fig.savefig(output, transparent=False)
    plt.close(fig)
    return output


def save_evolution(d, output) -> Path:
    """Terrain at every optimization step, rebuilt from the RBF coefficients.
    Verified against the stored terrain_height to within 2e-4 m."""
    basis = np.exp(-((d["x_m"][:, None] - d["centers_m"][None, :]) ** 2) / (2.0 * d["sigma_m"] ** 2))
    terrain = basis @ d["coef_history"].T  # (nx, nsteps)
    x_km = d["x_m"] / 1000.0
    nsteps = terrain.shape[1]
    best = d["best_index"]

    rect = [
        AX_LEFT_IN / EVO_WIDTH_IN,
        AX_BOTTOM_IN / EVO_HEIGHT_IN,
        EVO_AX_W_IN / EVO_WIDTH_IN,
        (EVO_HEIGHT_IN - AX_BOTTOM_IN - AX_PAD_IN) / EVO_HEIGHT_IN,
    ]

    fig = plt.figure(figsize=(EVO_WIDTH_IN, EVO_HEIGHT_IN))
    ax = fig.add_axes(rect)

    for k in range(nsteps):
        if k == best:
            continue
        ax.plot(x_km, terrain[:, k], color=EVO_COLOR, lw=1.2, alpha=0.15 + 0.55 * k / (nsteps - 1))

    ax.plot(x_km, terrain[:, best], color=EVO_BEST_COLOR, lw=2.5, label=f"Best (E = {d['opt_energy'][best]:.1f} J)")

    ax.set_xlim(EVO_XLIM)
    ax.set_ylim(EVO_YLIM)
    ax.set_xticks(EVO_X_TICKS)
    ax.set_yticks(EVO_Y_TICKS)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="upper right", fontsize=14, frameon=True)

    fig.savefig(output, transparent=False, bbox_inches="tight")
    plt.close(fig)
    return output


def render(source: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    artifact = artifact_from_bundle(source, ARTIFACT_PATTERN) if source is not None else default_source()
    output_dir = Path(output_dir) if output_dir is not None else SCRIPT_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    d = load(artifact)
    levels_field = np.linspace(-W_MAX, W_MAX, 41)

    outputs = []
    for name, stem, show_ylabel, show_xlabel in PANELS:
        index = d["labels"].index(name)
        outputs.append(
            save_panel(
                d["x_m"] / 1000.0,
                d["z_km"][index],
                d["w"][index],
                levels_field,
                output_dir / f"{stem}.png",
                show_ylabel,
                show_xlabel,
            )
        )

    outputs.append(save_colorbar(levels_field, output_dir / "gravity_wave_colorbar.png"))
    outputs.append(save_evolution(d, output_dir / "gravity_wave_evolution_main.png"))
    print(f"colorbar at +/- {W_MAX} m/s, true peak is {np.abs(d['w']).max():.2f} m/s")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        outputs = render(args.source, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
