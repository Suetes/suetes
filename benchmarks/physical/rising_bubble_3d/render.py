#!/usr/bin/env python3
"""Render the rising-bubble comparison from its plot-ready artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.artifacts import SCHEMA


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

    final = theta.isel(time=-1)
    difference = final.sel(core="sisl") - final.sel(core="split-explicit")
    field_limit = max(float(abs(final.min())), float(abs(final.max())))
    difference_limit = max(
        float(abs(difference.min())), float(abs(difference.max())),
        np.finfo(float).eps,
    )
    x_km = dataset["x"].values / 1000.0
    z_km = dataset["z"].values / 1000.0

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for axis, core, title in zip(
        axes[:2], cores, ("(a) SISL", "(b) Split-Explicit")
    ):
        image = axis.contourf(
            x_km, z_km, final.sel(core=core).values.T,
            levels=np.linspace(-field_limit, field_limit, 61),
            cmap="RdBu_r", extend="both",
        )
        axis.set_title(title)
    difference_image = axes[2].contourf(
        x_km, z_km, difference.values.T,
        levels=np.linspace(-difference_limit, difference_limit, 61),
        cmap="coolwarm", extend="both",
    )
    axes[2].set_title("(c) SISL $-$ Split-Explicit")
    for axis in axes:
        axis.set_xlabel("Horizontal distance (km)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("Height (km)")
    fig.colorbar(image, ax=axes[:2], label=r"$\theta_v'$ (K)")
    fig.colorbar(difference_image, ax=axes[2], label=r"Difference (K)")
    comparison_path = output_dir / "comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    difference_series = (
        theta.sel(core="sisl") - theta.sel(core="split-explicit")
    )
    spatial_dims = ("x", "z")
    relative_l2 = np.sqrt((difference_series**2).sum(spatial_dims)) / np.sqrt(
        ((0.5 * theta.sum("core")) ** 2).sum(spatial_dims)
    ).clip(min=np.finfo(float).tiny)
    maximum = abs(difference_series).max(spatial_dims)
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.plot(dataset["time"], relative_l2, "o-", label=r"relative $L_2$")
    axis.plot(dataset["time"], maximum, "s--", label=r"$L_\infty$ (K)")
    axis.set(xlabel="Simulation time (s)", ylabel="Difference metric")
    axis.grid(True, ls="--", alpha=0.4)
    axis.legend()
    evolution_path = output_dir / "evolution.png"
    fig.savefig(evolution_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [comparison_path, evolution_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        type=Path,
        help="Artifact NetCDF file or execution bundle directory",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.artifact, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
