#!/usr/bin/env python3
# Case renderer.
"""Regenerate optimal-topography figures from a saved NetCDF artifact."""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    try:
        source = artifact_from_bundle(args.artifact, "gravity_wave_optimal_topography_*.nc")
    except ValueError as error:
        parser.error(str(error))
    if not source.exists():
        parser.error(f"artifact does not exist: {source}. Run run.py once with the updated code.")
    output = Path(args.output_dir) if args.output_dir else figure_dir_for(source)
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(source) as opened:
        data = opened.load()
    core = str(data.attrs["core"]).lower()
    x_km = data.x.values / 1000.0

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.set_title(f"Evolution of optimal topography ({core.capitalize()})")
    centers = data.rbf_center.values
    sigma = float(data.attrs["rbf_sigma_m"])
    history = data.coefficient_history.values
    profiles = []
    for index, coefficients in enumerate(history):
        height = sum(
            value * np.exp(-((data.x.values - center) ** 2) / (2 * sigma**2))
            for value, center in zip(coefficients, centers)
        )
        profiles.append(np.maximum(height, 0.0))
        axis.plot(x_km, profiles[-1], color="blue", alpha=0.5 * (index + 1) / len(history))
    best = int(data.attrs["best_optimization_index"])
    axis.plot(
        x_km, profiles[best], color="red", linewidth=2, label=f"Best (E={data.optimization_energy.values[best]:.1f} J)"
    )
    axis.set(xlabel="Distance (km)", ylabel="Elevation (m)", xlim=(-25, 25))
    axis.legend()
    fig.savefig(output / f"inverse_topography_evolution_{core}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for index, (axis, label) in enumerate(zip(axes.flat, data.configuration.values)):
        z_km = data.physical_height.isel(configuration=index).values / 1000.0
        w = data.vertical_velocity.isel(configuration=index).values
        axis.contourf(
            np.broadcast_to(x_km[:, None], z_km.shape),
            z_km,
            w,
            levels=np.linspace(-1.5, 1.5, 41),
            cmap="RdBu_r",
            extend="both",
        )
        axis.fill_between(x_km, 0, data.terrain_height.isel(configuration=index).values / 1000.0, color="black")
        rect = patches.Rectangle(
            (x_km[180], z_km[180, 8]),
            x_km[200] - x_km[180],
            z_km[180, 20] - z_km[180, 8],
            linewidth=2,
            edgecolor="k",
            facecolor="none",
            linestyle="--",
        )
        axis.add_patch(rect)
        axis.set_title(f"{label}\nTarget energy (J): {data.validation_energy.values[index]:.2f}", fontweight="bold")
        axis.set(xlim=(-30, 40), ylim=(0, 12))
        if index >= 2:
            axis.set_xlabel("Distance (km)")
        if index % 2 == 0:
            axis.set_ylabel("Altitude (km)")
    fig.tight_layout()
    fig.savefig(output / f"inverse_topography_validation_{core}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
