#!/usr/bin/env python3
"""Regenerate NEUVE coordinate-evaluation figures without rerunning the model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="coordinate_evaluation.nc")
    parser.add_argument("--output-dir")
    parser.add_argument("--plot-samples", type=int, default=3)
    args = parser.parse_args()
    artifact = Path(args.artifact)
    if not artifact.exists():
        parser.error(
            f"artifact does not exist: {artifact}. Run "
            "experiments/evaluate_neuve_coordinates.py first; "
            "'output/EXPERIMENT' is a placeholder, not a literal directory."
        )
    output = Path(args.output_dir) if args.output_dir else artifact.parent
    output.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(artifact) as source:
        data = source.load()
    names = [str(value) for value in data.coordinate.values]
    count = min(args.plot_samples, data.sizes["sample"])
    x = data.x.values / 1000.0

    fig, axes = plt.subplots(
        count, len(names), figsize=(3.7 * len(names), 3.0 * count),
        sharex=True, sharey=True, squeeze=False,
    )
    for sample in range(count):
        for column, name in enumerate(names):
            axis = axes[sample, column]
            z = data.z_interface.sel(sample=sample, coordinate=name).values / 1000.0
            for level in range(z.shape[-1]):
                axis.plot(x, z[:, level], color="C0", linewidth=0.65)
            axis.fill_between(x, 0.0, z[:, 0], color="0.55", alpha=0.7)
            if sample == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel("Height (km)")
            if sample == count - 1:
                axis.set_xlabel("x (km)")
    fig.tight_layout()
    fig.savefig(output / "coordinate_sample_gallery.png", dpi=250)
    plt.close(fig)

    target = data.attrs["target"]
    time = data.time.values
    fig, axes = plt.subplots(
        1, count, figsize=(4.0 * count, 3.5),
        sharex=True, sharey=True, squeeze=False,
    )
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
            for name in names:
                axis.plot(time, plotted[name], label=name)
        else:
            reference = plotted["Tuned SLEVE"]
            for name in names:
                axis.plot(
                    time, np.divide(
                        plotted[name], reference,
                        out=np.full_like(reference, np.nan),
                        where=reference > np.finfo(float).tiny,
                    ), label=name,
                )
            axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.8)
        start = 0.5 * time[-1] if target == "mountain_flux" else 0.0
        axis.set(xlabel="Time (s)", xlim=(start, time[-1]))
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel(
        "Cumulative mean TKE / tuned SLEVE" if target == "pgf_rest"
        else ("Tracer return error / tuned SLEVE"
              if target == "tracer_reversibility"
              else "Momentum-flux transmission error")
    )
    axes[0, -1].legend()
    fig.tight_layout()
    fig.savefig(output / "metric_sample_gallery.png", dpi=250)
    plt.close(fig)
    print(f"Saved figures to {output}")


if __name__ == "__main__":
    main()
