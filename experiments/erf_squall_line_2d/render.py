#!/usr/bin/env python3
"""Render the saved ERF two-dimensional squall-line benchmark artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.experiment import SCHEMA


def render(artifact: Path, output_dir: Path) -> list[Path]:
    if not artifact.exists():
        raise FileNotFoundError(
            f"{artifact} does not exist. The simulation must complete before the rendering command is run."
        )
    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    if dataset.attrs.get("artifact_schema") != SCHEMA:
        raise ValueError(f"{artifact} is not a {SCHEMA} artifact")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )

    times = np.asarray(dataset.time)
    selected = [int(np.where(np.isclose(times, value))[0][0]) for value in (3000.0, 6000.0, 9000.0)]
    x_km = np.asarray(dataset.x) / 1000.0
    z_km = np.asarray(dataset.z) / 1000.0
    x_mesh, z_mesh = np.meshgrid(x_km, z_km, indexing="ij")
    theta_prime = dataset.dry_potential_temperature - dataset.environment_dry_potential_temperature
    rain_gkg = 1000.0 * dataset.rain_water
    cloud = dataset.cloud_water

    rain_limit = max(float(rain_gkg.isel(time=selected).max()), np.finfo(float).eps)
    rain_levels = np.linspace(0.05, rain_limit, 17)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True, sharey=True)
    color = None
    for axis, index in zip(axes, selected):
        th = theta_prime.isel(time=index).values
        qr = rain_gkg.isel(time=index).values
        qc = cloud.isel(time=index).values
        grey_limit = max(np.nanpercentile(np.abs(th), 99.0), 0.1)
        axis.contourf(x_mesh, z_mesh, th, levels=np.linspace(-grey_limit, grey_limit, 31), cmap="Greys", extend="both")
        masked_rain = np.ma.masked_less(qr, rain_levels[0])
        color = axis.contourf(x_mesh, z_mesh, masked_rain, levels=rain_levels, cmap="turbo", extend="max")
        if np.nanmax(qc) >= 1.0e-5:
            axis.contour(x_mesh, z_mesh, qc, levels=[1.0e-5], colors=["darkorange"], linewidths=1.1)
        axis.set_xlim(-60.0, 60.0)
        axis.set_ylim(0.0, 18.0)
        axis.set_ylabel("Height (km)")
        axis.set_title(f"$t={times[index]:.0f}$ s")
    axes[-1].set_xlabel("Horizontal distance (km)")
    if color is not None:
        bar = fig.colorbar(color, ax=axes, pad=0.018, fraction=0.025)
        bar.set_label(r"$q_r$ (g kg$^{-1}$)")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.07, top=0.96, hspace=0.16)
    state_path = output_dir / "erf_squall_line_evolution.png"
    fig.savefig(state_path, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    for index in selected:
        axis.plot(x_km, dataset.accumulated_rain.isel(time=index), linewidth=2.0, label=f"{times[index]:.0f} s")
    axis.set_xlim(-60.0, 60.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Horizontal distance (km)")
    axis.set_ylabel("Accumulated rain (mm)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    rain_path = output_dir / "erf_squall_line_rain_profiles.png"
    fig.savefig(rain_path, bbox_inches="tight")
    plt.close(fig)

    return [state_path, rain_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=Path("output/erf_squall_line_2d/artifact.nc"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/erf_squall_line_2d"))
    args = parser.parse_args()
    for path in render(args.artifact, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
