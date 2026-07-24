"""Shared manuscript plotting style for dynamical-core convergence studies."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


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

        ax.set_title(VARIABLE_TITLES[key], fontsize=13)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabels[key], fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.tick_params(axis="both", which="major", labelsize=9)

    # Axes share x limits, so invert once to present refinement from coarse
    # (left) to fine (right).
    axes[0].invert_xaxis()

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=10,
    )
    # The manuscript caption supplies the figure-level title.  Keeping only
    # variable/panel labels here saves vertical space and avoids duplicating it.
    fig.tight_layout(w_pad=1.8, rect=(0, 0, 1, 0.94))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
