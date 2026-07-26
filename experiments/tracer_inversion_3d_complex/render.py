#!/usr/bin/env python3
# Case renderer.
"""Plot the complex 3-D tracer inversion from its saved NetCDF artifact."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'font.family': 'sans-serif'
    })

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
    
    # --- Figure 1: Plume Snapshot ---
    fig_plume, axes_plume = plt.subplots(
        2, count, figsize=(4 * count + 1.5, 7.5),
        sharex=True, squeeze=False
    )
    mappable = None
    for column in range(count):
        plume = data.plume_snapshot.isel(snapshot=column).values
        axis = axes_plume[0, column]
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
        
        props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
        axis.text(0.05, 0.95, f"t = {data.snapshot_time.values[column] / 60.0:.1f} min", 
                  transform=axis.transAxes, fontsize=14, verticalalignment='top', bbox=props)
                  
        axis.set(xlim=(-12.5, 12.5), ylim=(-12.5, 12.5), aspect="equal")
        if column == 0:
            axis.set_ylabel("y (km)")
        else:
            axis.tick_params(axis='y', labelleft=False)

        axis_z = axes_plume[1, column]
        terrain_y = terrain[:, true_y_index]
        z_slice = z[:, true_y_index, :]
        z_padded = np.concatenate([terrain_y[:, None], z_slice], axis=1)
        plume_padded = np.concatenate(
            [plume[:, true_y_index, :1], plume[:, true_y_index, :]], axis=1
        )
        axis_z.contourf(
            np.broadcast_to(x[:, None], z_padded.shape).T, z_padded.T,
            plume_padded.T, levels=levels, cmap="Blues", extend="max",
        )
        axis_z.fill_between(x, 0, terrain_y, color="gray", alpha=.5)
        axis_z.scatter(true[0], true[2], color="green", marker="x", s=100)
        axis_z.set(xlim=(-12.5, 12.5), ylim=(0, 5), xlabel="x (km)")
        if column == 0:
            axis_z.set_ylabel("Altitude z (km)")
        else:
            axis_z.tick_params(axis='y', labelleft=False)

    fig_plume.tight_layout(rect=[0, 0, .88, 1.0])
    color_axis = fig_plume.add_axes([.90, .15, .02, .70])
    cbar_plume = fig_plume.colorbar(mappable, cax=color_axis, label="Tracer concentration", ticks=[0, 2, 4, 6, 8, 10])
    cbar_plume.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    fig_plume.savefig(output / f"tracer_3d_complex_plume_{core}.png", dpi=300)
    plt.close(fig_plume)

    # --- Figure 2: Optimization Trajectory ---
    history = data.parameter_history.values
    fig_opt, (horizontal, vertical) = plt.subplots(1, 2, figsize=(15, 6))
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
        xlabel="x (km)", ylabel="y (km)", xlim=(-12.5, 12.5),
        ylim=(-12.5, 12.5),
    )
    horizontal.grid(True, ls="--", alpha=0.4)

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
        xlabel="x (km)", ylabel="Altitude z (km)", xlim=(-12.5, 12.5),
        ylim=(0, 5),
    )
    vertical.grid(True, ls="--", alpha=0.4)
    
    # Legend outside
    handles, labels = horizontal.get_legend_handles_labels()
    handles_vert, labels_vert = vertical.get_legend_handles_labels()
    # combine unique labels
    by_label = dict(zip(labels + labels_vert, handles + handles_vert))
    
    fig_opt.tight_layout(rect=[0, 0, 0.75, 1.0])
    fig_opt.legend(by_label.values(), by_label.keys(), loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    fig_opt.savefig(output / f"tracer_3d_complex_optimization_{core}.png", dpi=300)
    plt.close(fig_opt)

    # --- Figure 3: Sensors Time Series ---
    fig_sens, axes_sens = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    minutes = data.time.values / 60.0
    target = data.target_sensor_concentration.values
    inferred = data.recovered_sensor_concentration.values
    for sensor, axis in enumerate(axes_sens.flat):
        axis.plot(minutes, target[:, sensor], "k-", linewidth=2.5,
                  label="True observations")
        axis.plot(minutes, inferred[:, sensor], "r--", linewidth=2.0,
                  label="Inferred plume")
                  
        # Shade error
        axis.fill_between(minutes, target[:, sensor], inferred[:, sensor], color='red', alpha=0.2)
        
        props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
        loc = tuple(float(x) for x in sensors[sensor])
        location_str = f"({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})"
        axis.text(0.05, 0.95, f"Tower {sensor + 1}\n{location_str} km", 
                  transform=axis.transAxes, fontsize=12, verticalalignment='top', bbox=props)
                  
        if sensor >= 3:
            axis.set_xlabel("Time (min)")
        if sensor % 3 == 0:
            axis.set_ylabel("Concentration")
            
        axis.grid(True, ls="--", alpha=0.4)
        
    handles, labels = axes_sens[0, 0].get_legend_handles_labels()
    fig_sens.tight_layout(rect=[0, 0, 0.75, 1.0])
    fig_sens.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5))
    
    fig_sens.savefig(output / f"tracer_3d_complex_sensors_{core}.png", dpi=300)
    plt.close(fig_sens)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
