#!/usr/bin/env python3
# Case renderer.
"""Plot the complex 3-D tracer inversion from its saved NetCDF artifact."""

import argparse
from pathlib import Path

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
        artifact = artifact_from_bundle(
            args.artifact, "tracer_inversion_3d_complex_*.nc"
        )
    except ValueError as error:
        parser.error(str(error))
    if not artifact.exists():
        parser.error(
            f"artifact does not exist: {artifact}. Run "
            "experiments/tracer_inversion_3d_complex/run.py first."
        )
    output = (
        Path(args.output_dir) if args.output_dir else figure_dir_for(artifact)
    )
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as source:
        data = source.load()

    core = str(data.attrs["core"]).lower()
    x, y = data.x.values, data.y.values
    terrain = data.terrain_height.values
    z = data.physical_height.values
    true = data.true_parameter.values
    recovered = data.recovered_parameter.values
    sensors = data.sensor_location.values
    true_y_index = int(np.argmin(np.abs(y - true[1])))
    center_z = z[len(x) // 2, len(y) // 2]
    true_z_index = int(np.argmin(np.abs(center_z - true[2])))
    levels = np.linspace(0.01, max(10.0, float(data.plume_snapshot.max())), 50)

    count = data.sizes["snapshot"]
    fig, axes = plt.subplots(
        2, count, figsize=(4 * count + 1.5, 7.5), squeeze=False
    )
    mappable = None
    for column in range(count):
        plume = data.plume_snapshot.isel(snapshot=column).values
        axis = axes[0, column]
        mappable = axis.contourf(
            x, y, plume[:, :, true_z_index].T,
            levels=levels, cmap="Blues", extend="max",
        )
        axis.contourf(x, y, terrain.T, levels=20, cmap="binary", alpha=.15)
        axis.contour(
            x, y, terrain.T, levels=[.2, .5, .8, 1.1, 1.4, 1.7],
            colors="black", alpha=.2, linewidths=.8,
        )
        axis.scatter(true[0], true[1], color="green", marker="x", s=100)
        axis.scatter(
            sensors[:, 0], sensors[:, 1], color="red", marker="^", s=80,
            edgecolor="black",
        )
        axis.set_title(
            f"t = {data.snapshot_time.values[column] / 60.0:.1f} min"
        )
        axis.set(xlim=(-12.5, 12.5), ylim=(-12.5, 12.5), aspect="equal")
        if column == 0:
            axis.set_ylabel("y (km)")

        axis = axes[1, column]
        terrain_y = terrain[:, true_y_index]
        z_slice = z[:, true_y_index, :]
        z_padded = np.concatenate([terrain_y[:, None], z_slice], axis=1)
        plume_padded = np.concatenate(
            [plume[:, true_y_index, :1], plume[:, true_y_index, :]], axis=1
        )
        axis.contourf(
            np.broadcast_to(x[:, None], z_padded.shape).T, z_padded.T,
            plume_padded.T, levels=levels, cmap="Blues", extend="max",
        )
        axis.fill_between(x, 0, terrain_y, color="gray", alpha=.5)
        axis.scatter(true[0], true[2], color="green", marker="x", s=100)
        axis.set(xlim=(-12.5, 12.5), ylim=(0, 5), xlabel="x (km)")
        if column == 0:
            axis.set_ylabel("Altitude z (km)")
    fig.suptitle("Forward tracer plume evolution", fontsize=16)
    fig.tight_layout(rect=[0, 0, .88, .95])
    color_axis = fig.add_axes([.90, .15, .02, .70])
    fig.colorbar(mappable, cax=color_axis, label="Tracer concentration")
    fig.savefig(output / f"tracer_3d_complex_plume_{core}.png", dpi=150)
    plt.close(fig)

    history = data.parameter_history.values
    fig, (horizontal, vertical) = plt.subplots(1, 2, figsize=(15, 6))
    horizontal.contourf(x, y, terrain.T, levels=20, cmap="binary", alpha=.15)
    horizontal.contour(
        x, y, terrain.T, levels=[.2, .5, .8, 1.1, 1.4, 1.7],
        colors="black", alpha=.2, linewidths=.8,
    )
    horizontal.scatter(
        sensors[:, 0], sensors[:, 1], color="red", marker="^", s=80,
        edgecolor="black", label="Sensor towers",
    )
    horizontal.plot(
        history[:, 0], history[:, 1], "o-", color="purple", markersize=4,
        label="Optimizer trajectory",
    )
    horizontal.scatter(*history[0, :2], color="orange", s=100,
                       label="Initial guess")
    horizontal.scatter(*recovered[:2], color="purple", marker="*", s=150,
                       label="Recovered source")
    horizontal.scatter(*true[:2], color="green", marker="x", s=120,
                       label="True source")
    horizontal.set(
        title="Source location trajectory (horizontal x-y plane)",
        xlabel="x (km)", ylabel="y (km)", xlim=(-12.5, 12.5),
        ylim=(-12.5, 12.5),
    )
    horizontal.grid(True)
    horizontal.legend()

    vertical.plot(
        history[:, 0], history[:, 2], "o-", color="purple", markersize=4,
        label="Optimizer trajectory",
    )
    vertical.scatter(history[0, 0], history[0, 2], color="orange", s=100,
                     label="Initial guess")
    vertical.scatter(recovered[0], recovered[2], color="purple", marker="*",
                     s=150, label="Recovered source")
    vertical.scatter(true[0], true[2], color="green", marker="x", s=120,
                     label="True source")
    vertical.fill_between(x, 0, terrain.max(axis=1), color="gray", alpha=.3,
                          label="Max terrain profile")
    vertical.set(
        title="Source location trajectory (vertical x-z plane)",
        xlabel="x (km)", ylabel="Altitude z (km)", xlim=(-12.5, 12.5),
        ylim=(0, 5),
    )
    vertical.grid(True)
    vertical.legend()
    fig.tight_layout()
    fig.savefig(output / f"tracer_3d_complex_optimization_{core}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    minutes = data.time.values / 60.0
    target = data.target_sensor_concentration.values
    inferred = data.recovered_sensor_concentration.values
    for sensor, axis in enumerate(axes.flat):
        axis.plot(minutes, target[:, sensor], "k-", linewidth=2.5,
                  label="True observations")
        axis.plot(minutes, inferred[:, sensor], "r--", linewidth=2,
                  label="Inferred plume")
        location = tuple(sensors[sensor])
        axis.set(
            title=f"Tower {sensor + 1} at {location} km",
            xlabel="Time (min)", ylabel="Tracer concentration",
        )
        axis.grid(True)
        if sensor == 0:
            axis.legend()
    fig.suptitle("Time series of concentration at monitoring towers", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / f"tracer_3d_complex_sensors_{core}.png", dpi=150)
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
