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
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
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
    parser.add_argument("--paper-figure", type=Path)
    args = parser.parse_args()
    layout = ExperimentLayout(
        kind="verification",
        case="adjoint_gradient_convergence",
        execution=args.name,
        output_root=args.output_root,
    ).create()
    summary = json.loads((layout.data / "summary.json").read_text())

    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.7))
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
    axes[0].set_xlabel(r"Perturbation $\varepsilon$")
    axes[0].set_ylabel(r"Taylor remainder $R_1$")
    props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
    axes[0].text(0.05, 0.95, "(a) Taylor test", transform=axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    import matplotlib.ticker as ticker
    axes[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0].xaxis.set_major_locator(ticker.LogLocator(numticks=4))
    axes[0].tick_params(axis="x", which="both", rotation=0)
    handles0, labels0 = axes[0].get_legend_handles_labels()

    fields = ("u", "w", "pi", "th_v")
    titles = (r"(b) $u_0$", r"(c) $w_0$", r"(d) $\pi'_0$", r"(e) $\theta_{v,0}$")
    first_result = next(iter(summary["results"].values()))
    dt = np.asarray(first_result["time_steps"][:-1])
    for panel, field, title in zip(axes[1:], fields, titles):
        for core, result in summary["results"].items():
            differences = np.asarray(
                result["successive_gradient_differences"][field]
            )
            point_count = np.prod(result["gradient_shapes"][field])
            panel.loglog(
                dt, differences / np.sqrt(point_count), "o-",
                color=colors[core],
            )
        reference = np.asarray(
            first_result["successive_gradient_differences"][field]
        ) / np.sqrt(np.prod(first_result["gradient_shapes"][field]))
        panel.loglog(
            dt, reference[0] * (dt / dt[0]) ** 2, "k--", label="Order 2"
        )
        panel.invert_xaxis()
        panel.set_xlabel(r"$\Delta t$ [s]")
        panel.text(0.05, 0.95, title, transform=panel.transAxes, fontsize=12,
                   verticalalignment="top", bbox=props)
        panel.grid(True, which="both", ls="--", alpha=0.4)
        panel.xaxis.set_minor_formatter(ticker.NullFormatter())
        panel.set_xticks(sorted(set([dt[0], dt[len(dt)//2], dt[-1]])))
        panel.xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].set_ylabel(r"$\|\Delta g\|_2$")
    
    # Use one common legend for both subplots at the bottom
    # Assuming both subplots have similar labels, we just take from the first one
    fig.legend(
        handles0, labels0, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels0), frameon=False,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=0.8)
    output = layout.figures / "adjoint_gradient_convergence.png"
    fig.savefig(output, dpi=300)
    print(output)
    if args.paper_figure:
        args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.paper_figure, dpi=300)
        print(args.paper_figure)
    plt.close(fig)


if __name__ == "__main__":
    main()
