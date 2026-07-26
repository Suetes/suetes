"""Shared manuscript plotting style for dynamical-core convergence studies."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


VARIABLES = ("u", "w", "pi", "th_v")
VARIABLE_TITLES = {
    "u": r"$u$",
    "w": r"$w$",
    "pi": r"$\pi$",
    "th_v": r"$\theta_v$",
}

SERIES_STYLES = (
    {"color": "tab:blue", "marker": "o"},
    {"color": "tab:orange", "marker": "s"},
    {"color": "tab:purple", "marker": "d"},
)


def save_four_panel_convergence(
    x_values,
    series,
    ylabels,
    xlabel,
    title,
    output_path,
    *,
    reference_order=2,
):
    """Save a uniform one-row convergence figure for four state variables.

    ``series`` is a sequence of ``(legend_label, values_by_variable)`` pairs.
    Each values mapping must contain ``u``, ``w``, ``pi``, and ``th_v``.
    """
    x_values = np.asarray(x_values, dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.2), sharex=True)

    for ax, key in zip(axes, VARIABLES):
        plotted_values = []
        for index, (label, values_by_variable) in enumerate(series):
            values = np.asarray(values_by_variable[key], dtype=float)
            plotted_values.append(values)
            style = SERIES_STYLES[index % len(SERIES_STYLES)]
            ax.loglog(
                x_values,
                values,
                label=label,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.5,
            )

        reference_start = max(values[0] for values in plotted_values) * 1.5
        reference = reference_start * (
            x_values / x_values[0]
        ) ** reference_order
        ax.loglog(
            x_values,
            reference,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Order {reference_order}",
        )

        props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
        ax.text(0.05, 0.95, VARIABLE_TITLES[key], transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[key])
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        import matplotlib.ticker as ticker
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ticks = sorted(list(set([x_values[0], x_values[len(x_values)//2], x_values[-1]])))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis="x", which="both", rotation=0)

    # Axes share x limits, so invert once to present refinement from coarse
    # (left) to fine (right).
    axes[0].invert_xaxis()

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_labels),
        frameon=False,
    )
    # The manuscript caption supplies the figure-level title.  Keeping only
    # variable/panel labels here saves vertical space and avoids duplicating it.
    fig.tight_layout(w_pad=1.8, rect=(0, 0.15, 1, 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
