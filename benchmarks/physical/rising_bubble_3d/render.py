#!/usr/bin/env python3
"""Render the rising-bubble comparison from its plot-ready artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr

from suetes.shared.artifacts import SCHEMA


def format_sci(value: float) -> str:
    """Format a float into LaTeX scientific notation, e.g., 1.23 \times 10^{-4}."""
    base, exp = f"{value:.2e}".split("e")
    return f"{base} \\times 10^{{{int(exp)}}}"


def render(artifact: Path, output_dir: Path | None = None) -> list[Path]:
    """Render figures without importing JAX or rerunning either core."""
    if artifact.is_dir():
        artifact = artifact / "data" / "artifact.nc"
    output_dir = output_dir or artifact.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    if dataset.attrs.get("artifact_schema") != SCHEMA:
        raise ValueError(f"{artifact} is not a {SCHEMA} artifact")
    required = {"theta_perturbation", "mass_drift", "symmetry_error"}
    missing = required.difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Artifact is missing variables: {sorted(missing)}")

    theta = dataset["theta_perturbation"]
    if set(theta.dims) != {"core", "time", "x", "z"}:
        raise ValueError(f"Unexpected theta_perturbation dimensions: {theta.dims}")
    cores = [str(value) for value in dataset["core"].values]
    if cores != ["sisl", "split-explicit"]:
        raise ValueError(f"Expected SISL and Split-Explicit data, found {cores}")

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

    final = theta.isel(time=-1)
    theta_prime_sisl = final.sel(core="sisl").values.T
    theta_prime_split = final.sel(core="split-explicit").values.T
    theta_diff = theta_prime_sisl - theta_prime_split

    x_km = dataset["x"].values / 1000.0
    z_km = dataset["z"].values / 1000.0

    cmap_field = "RdBu_r"
    field_limit = max(float(abs(final.min())), float(abs(final.max())))
    vmin_field, vmax_field = -field_limit, field_limit

    cmap_diff = "PRGn"
    diff_limit = max(float(abs(theta_diff.min())), float(abs(theta_diff.max())))
    vmin_diff, vmax_diff = -diff_limit, diff_limit

    saved_paths = []

    # --- 2. Process and save the SISL plot (Panel a) ---
    fig_sisl, ax_sisl = plt.subplots()
    im_sisl = ax_sisl.pcolormesh(
        x_km, z_km, theta_prime_sisl, shading="gouraud", cmap=cmap_field, vmin=vmin_field, vmax=vmax_field
    )
    ax_sisl.set_xlabel("Horizontal distance (km)")
    ax_sisl.set_ylabel("Height (km)")
    ax_sisl.set_aspect("equal")

    # Add an invisible colorbar axis to perfectly match the width reduction of panels B and C
    divider_sisl = make_axes_locatable(ax_sisl)
    cax_sisl = divider_sisl.append_axes("right", size="5%", pad=0.1)
    cax_sisl.axis("off")

    plt.tight_layout()
    path_sisl = output_dir / "rising_bubble_sisl_main.png"
    plt.savefig(path_sisl, bbox_inches="tight", transparent=False)
    plt.close(fig_sisl)
    saved_paths.append(path_sisl)

    # --- 3. Process and save the Split-Explicit plot (Panel b) ---
    fig_split, ax_split = plt.subplots()
    im_split = ax_split.pcolormesh(
        x_km, z_km, theta_prime_split, shading="gouraud", cmap=cmap_field, vmin=vmin_field, vmax=vmax_field
    )
    ax_split.set_xlabel("Horizontal distance (km)")

    # Apply transparent y-labels to maintain left-side bounding box padding
    ax_split.set_ylabel("Height (km)", color=(0, 0, 0, 0))
    ax_split.tick_params(axis="y", labelcolor=(0, 0, 0, 0))
    ax_split.set_aspect("equal")

    # Add vertical colorbar
    divider_split = make_axes_locatable(ax_split)
    cax_split = divider_split.append_axes("right", size="5%", pad=0.1)
    cbar_split = fig_split.colorbar(im_split, cax=cax_split)
    cbar_split.locator = ticker.MaxNLocator(nbins=4)
    cbar_split.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar_split.formatter.set_powerlimits((-2, 3))
    cbar_split.update_ticks()
    cbar_split.set_label(r"$\theta_v^\prime$ (K)", labelpad=10)
    cbar_split.outline.set_linewidth(1.5)

    plt.tight_layout()
    path_split = output_dir / "rising_bubble_split_explicit_main.png"
    plt.savefig(path_split, bbox_inches="tight", transparent=False)
    plt.close(fig_split)
    saved_paths.append(path_split)

    # --- 4. Process and save the Difference plot (Panel c) ---
    spatial_dims = ("x", "z")
    difference_series = theta.sel(core="sisl") - theta.sel(core="split-explicit")
    relative_l2_series = np.sqrt((difference_series**2).sum(spatial_dims)) / np.sqrt(
        ((0.5 * theta.sum("core")) ** 2).sum(spatial_dims)
    ).clip(min=np.finfo(float).tiny)
    maximum_series = abs(difference_series).max(spatial_dims)

    l2_rel = float(relative_l2_series.isel(time=-1))
    l_inf = float(maximum_series.isel(time=-1))

    fig_diff, ax_diff = plt.subplots()
    im_diff = ax_diff.pcolormesh(
        x_km, z_km, theta_diff, shading="gouraud", cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff
    )

    ax_diff.set_xlabel("Horizontal distance (km)")
    ax_diff.set_ylabel("Height (km)")
    ax_diff.set_aspect("equal")

    # Add vertical colorbar
    divider_diff = make_axes_locatable(ax_diff)
    cax_diff = divider_diff.append_axes("right", size="5%", pad=0.1)
    cbar_diff = fig_diff.colorbar(im_diff, cax=cax_diff)
    cbar_diff.locator = ticker.MaxNLocator(nbins=4)
    cbar_diff.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar_diff.formatter.set_powerlimits((0, 0))
    cbar_diff.update_ticks()
    cbar_diff.set_label(r"$\Delta\theta_v^\prime$ (K)", labelpad=10)
    cbar_diff.outline.set_linewidth(1.5)

    # Add text box for error metrics
    textstr = f"relative $L_2 = {format_sci(l2_rel)}$\n$L_\\infty = {format_sci(l_inf)}$ K"
    props = dict(boxstyle="square,pad=0.5", facecolor="white", alpha=0.9, edgecolor="none")
    ax_diff.text(0.05, 0.95, textstr, transform=ax_diff.transAxes, fontsize=14, verticalalignment="top", bbox=props)

    plt.tight_layout()
    path_diff = output_dir / "rising_bubble_difference_main.png"
    plt.savefig(path_diff, bbox_inches="tight", transparent=False)
    plt.close(fig_diff)
    saved_paths.append(path_diff)

    # --- 5. Process and save Evolution plot ---
    fig_evol, ax_evol = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax_evol.plot(dataset["time"], relative_l2_series, "o-", label=r"relative $L_2$")
    ax_evol.plot(dataset["time"], maximum_series, "s--", label=r"$L_\infty$ (K)")
    ax_evol.set_xlabel("Simulation time (s)")
    ax_evol.set_ylabel("Difference metric")
    ax_evol.grid(True, ls="--", alpha=0.4)
    # Move legend outside the plot
    ax_evol.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    path_evol = output_dir / "evolution.png"
    fig_evol.savefig(path_evol, dpi=300, bbox_inches="tight")
    plt.close(fig_evol)
    saved_paths.append(path_evol)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Artifact NetCDF file or execution bundle directory")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.artifact, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
