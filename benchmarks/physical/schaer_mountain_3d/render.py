#!/usr/bin/env python3
"""Render the 3-D Schär mountain comparison from its saved artifact."""

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

EVALUATION_HALF_WIDTH = 20_000.0


def format_sci(value: float) -> str:
    """Format a float into LaTeX scientific notation, e.g., 1.23 \times 10^{-4}."""
    if value == 0.0:
        return "0"
    base, exp = f"{value:.2e}".split("e")
    return f"{base} \\times 10^{{{int(exp)}}}"


def render(artifact: Path, output_dir: Path | None = None) -> list[Path]:
    if artifact.is_dir():
        artifact = artifact / "data" / "artifact.nc"
    output_dir = output_dir or artifact.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    if dataset.attrs.get("artifact_schema") != SCHEMA:
        raise ValueError(f"{artifact} is not a {SCHEMA} artifact")
    required = {"vertical_velocity", "momentum_flux", "physical_height_interface", "terrain_height"}
    missing = required.difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Artifact is missing variables: {sorted(missing)}")

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

    velocity = dataset["vertical_velocity"]
    cores = [str(value) for value in dataset["core"].values]
    if cores != ["sisl", "split-explicit"]:
        raise ValueError(f"Expected SISL and Split-Explicit data, found {cores}")
    final = velocity.isel(time=-1)
    difference = final.sel(core="sisl") - final.sel(core="split-explicit")
    evaluation = abs(dataset["x"]) <= EVALUATION_HALF_WIDTH
    final_evaluation = final.where(evaluation, drop=True)
    difference_evaluation = difference.where(evaluation, drop=True)

    field_limit = max(float(abs(final_evaluation.min())), float(abs(final_evaluation.max())))
    difference_limit = max(
        float(abs(difference_evaluation.min())), float(abs(difference_evaluation.max())), np.finfo(float).eps
    )

    x_km = dataset["x"].where(evaluation, drop=True).values / 1000.0
    terrain_km = dataset["terrain_height"].where(evaluation, drop=True).values / 1000.0
    height_km = dataset["physical_height_interface"].where(evaluation, drop=True).values / 1000.0
    horizontal_km = np.broadcast_to(x_km[:, None], height_km.shape)
    sisl = final_evaluation.sel(core="sisl").values
    split = final_evaluation.sel(core="split-explicit").values
    difference_values = sisl - split

    reference_norm = np.linalg.norm(0.5 * (sisl + split))
    relative_l2 = np.linalg.norm(difference_values) / max(reference_norm, np.finfo(float).tiny)
    cosine = np.vdot(sisl, split) / max(np.linalg.norm(sisl) * np.linalg.norm(split), np.finfo(float).tiny)

    cmap_field = "RdBu_r"
    cmap_diff = "PRGn"
    levels_field = np.linspace(-field_limit, field_limit, 61)
    levels_diff = np.linspace(-difference_limit, difference_limit, 61)

    saved_paths = []

    # --- 1. Process and save the SISL plot (Panel a) ---
    fig_sisl, ax_sisl = plt.subplots(figsize=(10, 4))
    im_sisl = ax_sisl.contourf(horizontal_km, height_km, sisl, levels=levels_field, cmap=cmap_field, extend="both")
    ax_sisl.fill_between(x_km, 0.0, terrain_km, color="black")
    ax_sisl.set_xlim(-EVALUATION_HALF_WIDTH / 1000.0, EVALUATION_HALF_WIDTH / 1000.0)
    ax_sisl.set_xlabel("Horizontal distance (km)")
    ax_sisl.set_ylabel("Height (km)")

    divider_sisl = make_axes_locatable(ax_sisl)
    cax_sisl = divider_sisl.append_axes("right", size="3%", pad=0.1)
    cax_sisl.axis("off")

    plt.tight_layout()
    path_sisl = output_dir / "schaer_mountain_sisl_main.png"
    fig_sisl.savefig(path_sisl, bbox_inches="tight", transparent=False)
    plt.close(fig_sisl)
    saved_paths.append(path_sisl)

    # --- 2. Process and save the Split-Explicit plot (Panel b) ---
    fig_split, ax_split = plt.subplots(figsize=(10, 4))
    im_split = ax_split.contourf(horizontal_km, height_km, split, levels=levels_field, cmap=cmap_field, extend="both")
    ax_split.fill_between(x_km, 0.0, terrain_km, color="black")
    ax_split.set_xlim(-EVALUATION_HALF_WIDTH / 1000.0, EVALUATION_HALF_WIDTH / 1000.0)
    ax_split.set_xlabel("Horizontal distance (km)")

    # Transparent y-labels
    ax_split.set_ylabel("Height (km)", color=(0, 0, 0, 0))
    ax_split.tick_params(axis="y", labelcolor=(0, 0, 0, 0))

    divider_split = make_axes_locatable(ax_split)
    cax_split = divider_split.append_axes("right", size="3%", pad=0.1)
    cbar_split = fig_split.colorbar(im_split, cax=cax_split)
    cbar_split.locator = ticker.MaxNLocator(nbins=4)
    cbar_split.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar_split.formatter.set_powerlimits((-2, 3))
    cbar_split.update_ticks()
    cbar_split.set_label(r"$W$ (m s$^{-1}$)", labelpad=10)
    cbar_split.outline.set_linewidth(1.5)

    plt.tight_layout()
    path_split = output_dir / "schaer_mountain_split_explicit_main.png"
    fig_split.savefig(path_split, bbox_inches="tight", transparent=False)
    plt.close(fig_split)
    saved_paths.append(path_split)

    # --- 3. Process and save the Difference plot (Panel c) ---
    fig_diff, ax_diff = plt.subplots(figsize=(10, 4))
    im_diff = ax_diff.contourf(
        horizontal_km, height_km, difference_evaluation.values, levels=levels_diff, cmap=cmap_diff, extend="both"
    )
    ax_diff.fill_between(x_km, 0.0, terrain_km, color="black")
    ax_diff.set_xlim(-EVALUATION_HALF_WIDTH / 1000.0, EVALUATION_HALF_WIDTH / 1000.0)
    ax_diff.set_xlabel("Horizontal distance (km)")
    ax_diff.set_ylabel("Height (km)")

    divider_diff = make_axes_locatable(ax_diff)
    cax_diff = divider_diff.append_axes("right", size="3%", pad=0.1)
    cbar_diff = fig_diff.colorbar(im_diff, cax=cax_diff)
    cbar_diff.locator = ticker.MaxNLocator(nbins=4)
    cbar_diff.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar_diff.formatter.set_powerlimits((0, 0))
    cbar_diff.update_ticks()
    cbar_diff.set_label(r"$\Delta W$ (m s$^{-1}$)", labelpad=10)
    cbar_diff.outline.set_linewidth(1.5)

    textstr = (
        rf"relative $L_2 = {format_sci(relative_l2)}$"
        + "\n"
        + rf"cosine $= {cosine:.5f}$"
        + "\n"
        + rf"$L_\infty = {format_sci(difference_limit)}$ m s$^{{-1}}$"
    )
    props = dict(boxstyle="square,pad=0.5", facecolor="white", alpha=0.9, edgecolor="none")
    ax_diff.text(0.03, 0.97, textstr, transform=ax_diff.transAxes, fontsize=14, verticalalignment="top", bbox=props)

    plt.tight_layout()
    path_diff = output_dir / "schaer_mountain_difference_main.png"
    fig_diff.savefig(path_diff, bbox_inches="tight", transparent=False)
    plt.close(fig_diff)
    saved_paths.append(path_diff)

    # --- 4. Momentum Flux ---
    flux = dataset["momentum_flux"]
    fig_flux, ax_flux = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for core in cores:
        ls = "-" if core == "sisl" else "--"
        ax_flux.plot(dataset["z_level"], flux.sel(core=core).isel(time=-1), label=core, linestyle=ls)
    ax_flux.set(xlabel="Model level", ylabel="Orographic momentum flux")
    ax_flux.grid(True, ls="--", alpha=0.4)
    ax_flux.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    flux_path = output_dir / "momentum_flux.png"
    fig_flux.savefig(flux_path, dpi=300, bbox_inches="tight")
    plt.close(fig_flux)
    saved_paths.append(flux_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.artifact, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
