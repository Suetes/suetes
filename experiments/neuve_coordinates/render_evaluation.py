#!/usr/bin/env python3
# Evaluation renderer.
"""Regenerate NEUVE coordinate-evaluation figures without rerunning the model."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

from suetes.shared.experiment import artifact_from_bundle, figure_dir_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="coordinate_evaluation.nc")
    parser.add_argument("--output-dir")
    parser.add_argument("--plot-samples", type=int, default=3)
    args = parser.parse_args()
    try:
        artifact = artifact_from_bundle(args.artifact, "coordinate_evaluation.nc")
    except ValueError as error:
        parser.error(str(error))
    if not artifact.exists():
        parser.error(
            f"artifact does not exist: {artifact}. Run "
            "experiments/neuve_coordinates/evaluate.py first; "
            "'output/EXPERIMENT' is a placeholder, not a literal directory."
        )
    output = Path(args.output_dir) if args.output_dir else figure_dir_for(artifact)
    output.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        data = source.load()
    names = [str(value) for value in data.coordinate.values]
    count = min(args.plot_samples, data.sizes["sample"])
    x = data.x.values / 1000.0

    fig, axes = plt.subplots(count, len(names), figsize=(5.0 * len(names), 4.0 * count), sharex=True, sharey=True, squeeze=False)
    props = {"boxstyle": "square,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"}
    for sample in range(count):
        for column, name in enumerate(names):
            axis = axes[sample, column]
            z = data.z_interface.sel(sample=sample, coordinate=name).values / 1000.0
            for level in range(z.shape[-1]):
                axis.plot(x, z[:, level], color="C0", linewidth=0.65)
            axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)

            # Label instead of title
            if sample == 0:
                axis.text(0.05, 0.95, f"{name}", transform=axis.transAxes, fontsize=16, verticalalignment="top", bbox=props)
            if column == 0:
                axis.set_ylabel("Height (km)")
            if sample == count - 1:
                axis.set_xlabel("x (km)")

    fig.tight_layout()
    fig.savefig(output / "coordinate_sample_gallery.png", dpi=300)
    plt.close(fig)

    # Focused comparison of the two optimized coordinates. Their absolute
    # interface geometries are deliberately similar, so pair a lower-domain
    # overlay with their signed displacement to expose the learned adjustment.
    comparison_count = min(3, count)
    displacement_limit = 200.0
    fig = plt.figure(figsize=(5.0 * comparison_count, 7.8), layout="constrained")
    comparison_grid = fig.add_gridspec(
        3, comparison_count, height_ratios=(1.0, 1.05, 0.08), hspace=0.08
    )
    axes = np.empty((2, comparison_count), dtype=object)
    for sample in range(comparison_count):
        axes[0, sample] = fig.add_subplot(comparison_grid[0, sample])
        axes[1, sample] = fig.add_subplot(
            comparison_grid[1, sample], sharex=axes[0, sample]
        )
    colorbar_axis = fig.add_subplot(comparison_grid[2, :])
    residual_images = []
    for sample in range(comparison_count):
        galchen = data.z_interface.sel(
            sample=sample, coordinate="Gal-Chen"
        ).values
        sleve = data.z_interface.sel(
            sample=sample, coordinate="Tuned SLEVE"
        ).values
        neuve = data.z_interface.sel(sample=sample, coordinate="NEUVE").values
        terrain = data.terrain.sel(sample=sample).values / 1000.0

        upper = axes[0, sample]
        upper.fill_between(x, 0.0, terrain, color="0.55", alpha=0.7)
        for level in range(galchen.shape[-1]):
            if np.nanmin(galchen[:, level]) <= 6500.0:
                upper.plot(
                    x, galchen[:, level] / 1000.0,
                    color="C0", linestyle=":", linewidth=1.05,
                    label="Gal-Chen" if level == 0 else None,
                )
        for level in range(sleve.shape[-1]):
            if min(np.nanmin(sleve[:, level]), np.nanmin(neuve[:, level])) <= 6500.0:
                upper.plot(
                    x, sleve[:, level] / 1000.0,
                    color="C1", linestyle="--", linewidth=1.15,
                    label="Tuned SLEVE" if level == 0 else None,
                )
                upper.plot(
                    x, neuve[:, level] / 1000.0,
                    color="C2", linewidth=0.9,
                    label="NEUVE" if level == 0 else None,
                )
        upper.set_ylim(0.0, 6.0)
        upper.set_title(f"Terrain {sample + 1}")
        upper.grid(True, alpha=0.2)
        if sample == 0:
            upper.set_ylabel("Height (km)")

        lower = axes[1, sample]
        midpoint = 0.5 * (sleve + neuve) / 1000.0
        displacement = neuve - sleve
        x_mesh = np.broadcast_to(x[:, None], midpoint.shape)
        residual_images.append(
            lower.contourf(
                x_mesh,
                midpoint,
                displacement,
                levels=np.linspace(-displacement_limit, displacement_limit, 21),
                cmap="RdBu_r",
                extend="neither",
            )
        )
        lower.fill_between(x, 0.0, terrain, color="0.55", alpha=0.7)
        lower.set_ylim(0.0, 6.0)
        lower.set_xlabel("x (km)")
        if sample == 0:
            lower.set_ylabel("Height (km)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=True)
    colorbar = fig.colorbar(
        residual_images[0], cax=colorbar_axis, orientation="horizontal",
    )
    colorbar.set_label(r"Interface displacement $z_{\mathrm{NEUVE}}-z_{\mathrm{SLEVE}}$ (m)")
    fig.savefig(output / "coordinate_sleve_neuve_comparison.pdf", dpi=300)
    fig.savefig(output / "coordinate_sleve_neuve_comparison.png", dpi=300)
    plt.close(fig)

    target = data.attrs["target"]
    time = data.time.values
    line_styles = ["-", "--", "-.", ":"] * 3
    pgf_ratio_limits = None
    if target == "pgf_rest":
        visible_ratios = []
        for sample in range(count):
            series = {name: data.diagnostic_series.sel(sample=sample, coordinate=name).values for name in names}
            cumulative = {name: np.cumsum(values) / np.arange(1, len(values) + 1) for name, values in series.items()}
            reference = cumulative["Tuned SLEVE"]
            valid = reference > np.finfo(float).tiny
            for name in ("Tuned SLEVE", "NEUVE"):
                visible_ratios.append(cumulative[name][valid] / reference[valid])
        visible_ratios = np.concatenate(visible_ratios)
        span = float(np.nanmax(visible_ratios) - np.nanmin(visible_ratios))
        padding = max(0.02, 0.08 * span)
        pgf_ratio_limits = (max(0.0, float(np.nanmin(visible_ratios)) - padding), max(1.02, float(np.nanmax(visible_ratios)) + padding))
    if target == "pgf_rest":
        fig = plt.figure(figsize=(6.0 * count, 6.0))
        grid = fig.add_gridspec(2, count, height_ratios=(1.0, 2.4), hspace=0.05, wspace=0.12)
        upper_axes = [fig.add_subplot(grid[0, 0])]
        lower_axes = [fig.add_subplot(grid[1, 0], sharex=upper_axes[0])]
        for sample in range(1, count):
            upper_axes.append(fig.add_subplot(grid[0, sample], sharey=upper_axes[0]))
            lower_axes.append(fig.add_subplot(grid[1, sample], sharex=upper_axes[sample], sharey=lower_axes[0]))
            upper_axes[sample].tick_params(labelleft=False)
            lower_axes[sample].tick_params(labelleft=False)
        axes = np.asarray([lower_axes])
    else:
        fig, axes = plt.subplots(1, count, figsize=(6.0 * count, 5.0), sharex=True, sharey=True, squeeze=False)

    for sample in range(count):
        axis = axes[0, sample]
        plotted = {}
        for name in names:
            series = data.diagnostic_series.sel(sample=sample, coordinate=name).values
            plotted[name] = np.cumsum(series) / np.arange(1, len(series) + 1) if target == "pgf_rest" else series

        if target == "mountain_flux":
            for i, name in enumerate(names):
                axis.plot(time, plotted[name], label=name, ls=line_styles[i], linewidth=2)
        elif target == "pgf_rest":
            reference = plotted["Tuned SLEVE"]
            ratios = {
                name: np.divide(plotted[name], reference, out=np.full_like(reference, np.nan), where=reference > np.finfo(float).tiny)
                for name in names
            }
            upper = upper_axes[sample]
            upper.plot(time, ratios["Gal-Chen"], label="Gal-Chen", color="C0", linewidth=2)
            upper.set_yscale("log")
            upper.set_ylim(2.5, 200.0)
            upper.grid(True, ls="--", alpha=0.4)
            upper.tick_params(labelbottom=False)
            for name, color, style in (("Tuned SLEVE", "C1", "--"), ("NEUVE", "C2", "-")):
                axis.plot(time, ratios[name], label=name, color=color, ls=style, linewidth=2)
            axis.set_ylim(*pgf_ratio_limits)
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1.5)
        else:
            reference = plotted["Tuned SLEVE"]
            for i, name in enumerate(names):
                axis.plot(
                    time,
                    np.divide(plotted[name], reference, out=np.full_like(reference, np.nan), where=reference > np.finfo(float).tiny),
                    label=name,
                    ls=line_styles[i],
                    linewidth=2,
                )
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1.5)
        start = 0.5 * time[-1] if target == "mountain_flux" else 0.0
        axis.set(xlabel="Time (s)", xlim=(start, time[-1]))
        axis.grid(True, ls="--", alpha=0.4)

        # Add sample label
        props = {"boxstyle": "square,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"}
        label_axis = upper_axes[sample] if target == "pgf_rest" else axis
        label_axis.text(
            0.05, 0.95, f"Sample {sample + 1}", transform=label_axis.transAxes, fontsize=16, verticalalignment="top", bbox=props
        )

    if target == "pgf_rest":
        fig.supylabel("TKE / SLEVE")
    else:
        axes[0, 0].set_ylabel(
            "Tracer return error / tuned SLEVE" if target == "tracer_reversibility" else "Momentum-flux transmission error"
        )

    if target == "pgf_rest":
        upper_handles, upper_labels = upper_axes[0].get_legend_handles_labels()
        lower_handles, lower_labels = axes[0, 0].get_legend_handles_labels()
        handles = upper_handles + lower_handles
        labels = upper_labels + lower_labels
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output / "metric_sample_gallery.png", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
