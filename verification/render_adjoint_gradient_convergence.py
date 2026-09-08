"""Render Taylor-remainder and temporal adjoint-convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
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

COLORS = {"split-explicit": "#D55E00", "sisl": "#0072B2"}
LABELS = {"split-explicit": "Split-Explicit", "sisl": "SISL"}

# Both panels are built on one canvas size with identical margins, so a plain
# `-gravity north +append` in make_fig.sh leaves their axes level. Do not swap
# these for tight_layout or bbox_inches="tight".
PANEL_SIZE = (5.5, 4.3)
MARGINS = dict(left=0.21, right=0.97, bottom=0.18, top=0.97)


def new_panel():
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    fig.subplots_adjust(**MARGINS)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    return fig, ax


def render_taylor(summary, output):
    fig, ax = new_panel()
    for core, result in summary["results"].items():
        eps = np.asarray(result["epsilons"])
        ax.loglog(eps, result["first_order_remainder"], "o-", color=COLORS[core])
    first = next(iter(summary["results"].values()))
    eps = np.asarray(first["epsilons"])
    reference = np.asarray(first["first_order_remainder"])
    ax.loglog(eps, reference[0] * (eps / eps[0]) ** 2, "k--")
    ax.set_xlabel(r"Perturbation amplitude $\varepsilon$")
    ax.set_ylabel(r"Taylor remainder $R_1$")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_locator(ticker.LogLocator(numticks=4))
    ax.tick_params(axis="x", which="both", rotation=0)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def render_temporal(summary, output):
    fig, ax = new_panel()
    for core, result in summary["results"].items():
        dt = np.asarray(result["time_steps"][:-1])
        ax.loglog(dt, result["successive_gradient_differences"], "o-", color=COLORS[core])
    first = next(iter(summary["results"].values()))
    dt = np.asarray(first["time_steps"][:-1])
    errors = np.asarray(first["successive_gradient_differences"])
    ax.loglog(dt, errors[0] * (dt / dt[0]) ** 2, "k--")
    ax.invert_xaxis()
    ax.set_xlabel(r"Large time step $\Delta t$ [s]")
    ax.set_ylabel(r"$\|\Delta g\|_2$")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(sorted(set([dt[0], dt[len(dt) // 2], dt[-1]])))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", which="both", rotation=0)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def render_legend(summary, output):
    handles = [
        Line2D([], [], color=COLORS[core], marker="o", ls="-", label=LABELS[core])
        for core in summary["results"]
    ]
    handles.append(Line2D([], [], color="k", ls="--", label="Order 2"))
    fig = plt.figure(figsize=(PANEL_SIZE[0] * 2, 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--paper-figure", type=Path, help="accepted for compatibility; unused")
    args = parser.parse_args()
    layout = ExperimentLayout(
        kind="verification",
        case="adjoint_gradient_convergence",
        execution=args.name,
        output_root=args.output_root,
    ).create()

    # A summary.json sitting beside the script wins, so the renderer runs with
    # no arguments from its own directory.
    local_summary = SCRIPT_DIR / "summary.json"
    summary_path = local_summary if local_summary.exists() else layout.data / "summary.json"
    figures_dir = SCRIPT_DIR / "figures" if local_summary.exists() else layout.figures
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text())

    outputs = [
        figures_dir / "adjoint_gradient_taylor_main.png",
        figures_dir / "adjoint_gradient_legend.png",
        figures_dir / "adjoint_gradient_temporal_main.png",
    ]
    render_taylor(summary, outputs[0])
    render_legend(summary, outputs[1])
    render_temporal(summary, outputs[2])
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
