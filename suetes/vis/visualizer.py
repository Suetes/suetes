"""
Backward-compatibility shim module for legacy imports from suetes.vis.visualizer.
All content has been refactored into modular sub-files under suetes/vis/.
"""

from . import (
    Visualizer,
    plot_2d_field,
    plot_quiver_field,
    plot_level_strip,
    plot_dashboard,
    plot_comparison,
    plot_adjoint_overlay,
    plot_cross_section,
    plot_slice_locator_dashboard,
    plot_hovmoller,
    plot_point_timeseries,
    plot_energy_spectrum,
)

__all__ = [
    'Visualizer',
    'plot_2d_field',
    'plot_quiver_field',
    'plot_level_strip',
    'plot_dashboard',
    'plot_comparison',
    'plot_adjoint_overlay',
    'plot_cross_section',
    'plot_slice_locator_dashboard',
    'plot_hovmoller',
    'plot_point_timeseries',
    'plot_energy_spectrum',
]
