#!/usr/bin/env python3
"""Render the 3-D Schär mountain comparison from its saved artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.artifacts import SCHEMA

EVALUATION_HALF_WIDTH = 20_000.0


def render(artifact: Path, output_dir: Path | None = None) -> list[Path]:
    if artifact.is_dir():
        artifact = artifact / "data" / "artifact.nc"
    output_dir = output_dir or artifact.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    if dataset.attrs.get("artifact_schema") != SCHEMA:
        raise ValueError(f"{artifact} is not a {SCHEMA} artifact")
    required = {
        "vertical_velocity", "momentum_flux",
        "physical_height_interface", "terrain_height",
    }
    missing = required.difference(dataset.data_vars)
    if missing:
        raise ValueError(f"Artifact is missing variables: {sorted(missing)}")

    velocity = dataset["vertical_velocity"]
    cores = [str(value) for value in dataset["core"].values]
    if cores != ["sisl", "split-explicit"]:
        raise ValueError(f"Expected SISL and Split-Explicit data, found {cores}")
    final = velocity.isel(time=-1)
    difference = final.sel(core="sisl") - final.sel(core="split-explicit")
    evaluation = abs(dataset["x"]) <= EVALUATION_HALF_WIDTH
    final_evaluation = final.where(evaluation, drop=True)
    difference_evaluation = difference.where(evaluation, drop=True)
    field_limit = max(
        float(abs(final_evaluation.min())),
        float(abs(final_evaluation.max())),
    )
    difference_limit = max(
        float(abs(difference_evaluation.min())),
        float(abs(difference_evaluation.max())),
        np.finfo(float).eps,
    )
    x_km = dataset["x"].where(evaluation, drop=True).values / 1000.0
    terrain_km = (
        dataset["terrain_height"].where(evaluation, drop=True).values / 1000.0
    )
    height_km = (
        dataset["physical_height_interface"].where(evaluation, drop=True).values
        / 1000.0
    )
    horizontal_km = np.broadcast_to(x_km[:, None], height_km.shape)
    sisl = final_evaluation.sel(core="sisl").values
    split = final_evaluation.sel(core="split-explicit").values
    difference_values = sisl - split
    reference_norm = np.linalg.norm(0.5 * (sisl + split))
    relative_l2 = np.linalg.norm(difference_values) / max(
        reference_norm, np.finfo(float).tiny
    )
    cosine = np.vdot(sisl, split) / max(
        np.linalg.norm(sisl) * np.linalg.norm(split),
        np.finfo(float).tiny,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, core, title in zip(
        axes[:2], cores, ("(a) SISL", "(b) Split-Explicit")
    ):
        image = axis.contourf(
            horizontal_km, height_km,
            final_evaluation.sel(core=core).values,
            levels=np.linspace(-field_limit, field_limit, 61),
            cmap="RdBu_r", extend="both",
        )
        axis.set_title(title)
    difference_image = axes[2].contourf(
        horizontal_km, height_km, difference_evaluation.values,
        levels=np.linspace(-difference_limit, difference_limit, 61),
        cmap="coolwarm", extend="both",
    )
    axes[2].set_title("(c) SISL $-$ Split-Explicit")
    axes[2].text(
        0.03, 0.97,
        rf"relative $L_2={relative_l2:.2e}$" + "\n"
        + rf"cosine $={cosine:.5f}$" + "\n"
        + rf"$L_\infty={difference_limit:.2e}$ m s$^{{-1}}$",
        transform=axes[2].transAxes,
        ha="left", va="top",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    for axis in axes:
        axis.fill_between(
            x_km, 0.0, terrain_km,
            color="0.25",
        )
        axis.set_xlabel("Horizontal distance (km)")
        axis.set_xlim(-EVALUATION_HALF_WIDTH / 1000.0, EVALUATION_HALF_WIDTH / 1000.0)
    axes[0].set_ylabel("Height (km)")
    fig.colorbar(image, ax=axes[:2], label="Vertical velocity (m s$^{-1}$)")
    fig.colorbar(
        difference_image, ax=axes[2],
        label="Vertical-velocity difference (m s$^{-1}$)",
    )
    fig.suptitle(
        rf"Schär mountain waves at $t={float(dataset.time[-1]):g}$ s "
        rf"($\Delta x={float(dataset.attrs['dx_m']):g}$ m, "
        rf"$\Delta z={float(dataset.attrs['dz_m']):g}$ m, "
        rf"$\Delta t={float(dataset.attrs['dt_s']):g}$ s)"
    )
    comparison_path = output_dir / "comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    flux = dataset["momentum_flux"]
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for core in cores:
        axis.plot(
            dataset["z_level"], flux.sel(core=core).isel(time=-1),
            label=core,
        )
    axis.set(
        xlabel="Model level",
        ylabel="Orographic momentum flux",
        title="Final momentum-flux profile",
    )
    axis.grid(True, ls="--", alpha=0.4)
    axis.legend()
    flux_path = output_dir / "momentum_flux.png"
    fig.savefig(flux_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [comparison_path, flux_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.artifact, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
