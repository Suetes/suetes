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

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for


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
    output = (
        Path(args.output_dir) if args.output_dir else figure_dir_for(artifact)
    )
    output.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        data = source.load()
    names = [str(value) for value in data.coordinate.values]
    count = min(args.plot_samples, data.sizes["sample"])
    x = data.x.values / 1000.0

    fig, axes = plt.subplots(
        count, len(names), figsize=(5.0 * len(names), 4.0 * count),
        sharex=True, sharey=True, squeeze=False,
    )
    props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
    for sample in range(count):
        for column, name in enumerate(names):
            axis = axes[sample, column]
            z = data.z_interface.sel(sample=sample, coordinate=name).values / 1000.0
            for level in range(z.shape[-1]):
                axis.plot(x, z[:, level], color="C0", linewidth=0.65)
            axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)
            
            # Label instead of title
            if sample == 0:
                axis.text(0.05, 0.95, f"{name}", transform=axis.transAxes, 
                          fontsize=16, verticalalignment='top', bbox=props)
            
            if column == 0:
                axis.set_ylabel("Height (km)")
            if sample == count - 1:
                axis.set_xlabel("x (km)")
                
    fig.tight_layout()
    fig.savefig(output / "coordinate_sample_gallery.png", dpi=300)
    plt.close(fig)

    target = data.attrs["target"]
    time = data.time.values
    fig, axes = plt.subplots(
        1, count, figsize=(6.0 * count, 5.0),
        sharex=True, sharey=True, squeeze=False,
    )
    line_styles = ['-', '--', '-.', ':'] * 3
    for sample in range(count):
        axis = axes[0, sample]
        plotted = {}
        for name in names:
            series = data.diagnostic_series.sel(
                sample=sample, coordinate=name
            ).values
            plotted[name] = (
                np.cumsum(series) / np.arange(1, len(series) + 1)
                if target == "pgf_rest" else series
            )
            
        if target == "mountain_flux":
            for i, name in enumerate(names):
                axis.plot(time, plotted[name], label=name, ls=line_styles[i], linewidth=2)
        else:
            reference = plotted["Tuned SLEVE"]
            for i, name in enumerate(names):
                axis.plot(
                    time, np.divide(
                        plotted[name], reference,
                        out=np.full_like(reference, np.nan),
                        where=reference > np.finfo(float).tiny,
                    ), label=name, ls=line_styles[i], linewidth=2
                )
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=1.5)
        start = 0.5 * time[-1] if target == "mountain_flux" else 0.0
        axis.set(xlabel="Time (s)", xlim=(start, time[-1]))
        axis.grid(True, ls="--", alpha=0.4)
        
        # Add sample label
        props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
        axis.text(0.05, 0.95, f"Sample {sample + 1}", transform=axis.transAxes, 
                  fontsize=16, verticalalignment='top', bbox=props)
                  
    axes[0, 0].set_ylabel(
        "Cumulative mean TKE / tuned SLEVE" if target == "pgf_rest"
        else ("Tracer return error / tuned SLEVE"
              if target == "tracer_reversibility"
              else "Momentum-flux transmission error")
    )
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(names))
    
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.savefig(output / "metric_sample_gallery.png", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
