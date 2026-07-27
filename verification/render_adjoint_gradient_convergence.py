"""Render Taylor-remainder and temporal adjoint-convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suetes.shared.experiment import ExperimentLayout

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    args = parser.parse_args()
    layout = ExperimentLayout(
        kind="verification",
        case="adjoint_gradient_convergence",
        execution=args.name,
        output_root=args.output_root,
    ).create()
    summary = json.loads((layout.data / "summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"split-explicit": "#D55E00", "sisl": "#0072B2"}
    for core, result in summary["results"].items():
        eps = np.asarray(result["epsilons"])
        axes[0].loglog(
            eps, result["first_order_remainder"], "o-",
            color=colors[core],
            label="Split-Explicit" if core == "split-explicit" else "SISL",
        )
    eps = np.asarray(next(iter(summary["results"].values()))["epsilons"])
    reference = np.asarray(
        next(iter(summary["results"].values()))["first_order_remainder"]
    )
    axes[0].loglog(eps, reference[0] * (eps / eps[0]) ** 2, "k--", label="Order 2")
    axes[0].set_xlabel(r"Perturbation amplitude $\varepsilon$")
    axes[0].set_ylabel(r"Taylor remainder $R_1$")
    props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
    axes[0].text(0.05, 0.95, "(a) Discrete-adjoint Taylor test", transform=axes[0].transAxes, fontsize=16, verticalalignment='top', bbox=props)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    import matplotlib.ticker as ticker
    axes[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0].xaxis.set_major_locator(ticker.LogLocator(numticks=4))
    axes[0].tick_params(axis="x", which="both", rotation=0)
    handles0, labels0 = axes[0].get_legend_handles_labels()

    for core, result in summary["results"].items():
        dt = np.asarray(result["time_steps"][:-1])
        axes[1].loglog(
            dt, result["successive_gradient_differences"], "o-",
            color=colors[core],
            label=("Split-Explicit" if core == "split-explicit" else "SISL"),
        )
    dt = np.asarray(next(iter(summary["results"].values()))["time_steps"][:-1])
    errors = np.asarray(
        next(iter(summary["results"].values()))["successive_gradient_differences"]
    )
    axes[1].loglog(dt, errors[0] * (dt / dt[0]) ** 2, "k--", label="Order 2")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Large time step $\Delta t$ [s]")
    axes[1].set_ylabel(r"$\|\Delta g\|_2$")
    axes[1].text(0.05, 0.95, "(b) Temporal adjoint convergence", transform=axes[1].transAxes, fontsize=16, verticalalignment='top', bbox=props)
    axes[1].grid(True, which="both", ls="--", alpha=0.4)
    axes[1].xaxis.set_minor_formatter(ticker.NullFormatter())
    ticks1 = sorted(list(set([dt[0], dt[len(dt)//2], dt[-1]])))
    axes[1].set_xticks(ticks1)
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].tick_params(axis="x", which="both", rotation=0)
    
    # Use one common legend for both subplots at the bottom
    # Assuming both subplots have similar labels, we just take from the first one
    fig.legend(handles0, labels0, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels0))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    output = layout.figures / "adjoint_gradient_convergence.png"
    fig.savefig(output, dpi=300)
    print(output)


if __name__ == "__main__":
    main()
