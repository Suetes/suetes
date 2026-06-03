from .utils import _get_plot_data, _draw_domain_and_sponge, _get_level_height
from .spatial import (
    plot_2d_field,
    plot_quiver_field,
    plot_level_strip,
    plot_dashboard,
    plot_comparison,
    plot_adjoint_overlay
)
from .vertical import (
    plot_cross_section,
    plot_slice_locator_dashboard,
    plot_hovmoller
)
from .diagnostics import (
    plot_point_timeseries,
    plot_energy_spectrum
)

class Visualizer:
    """
    Backward-compatible class wrapper for the visualization functions.
    """
    def _get_plot_data(self, *args, **kwargs):
        return _get_plot_data(*args, **kwargs)

    def _draw_domain_and_sponge(self, *args, **kwargs):
        return _draw_domain_and_sponge(*args, **kwargs)

    def _get_level_height(self, *args, **kwargs):
        return _get_level_height(*args, **kwargs)

    def plot_2d_field(self, *args, **kwargs):
        return plot_2d_field(*args, **kwargs)

    def plot_quiver_field(self, *args, **kwargs):
        return plot_quiver_field(*args, **kwargs)

    def plot_dashboard(self, *args, **kwargs):
        return plot_dashboard(*args, **kwargs)

    def plot_comparison(self, *args, **kwargs):
        return plot_comparison(*args, **kwargs)

    def plot_cross_section(self, *args, **kwargs):
        return plot_cross_section(*args, **kwargs)

    def plot_level_strip(self, *args, **kwargs):
        return plot_level_strip(*args, **kwargs)

    def plot_slice_locator_dashboard(self, *args, **kwargs):
        return plot_slice_locator_dashboard(*args, **kwargs)

    def plot_hovmoller(self, *args, **kwargs):
        return plot_hovmoller(*args, **kwargs)

    def plot_energy_spectrum(self, *args, **kwargs):
        return plot_energy_spectrum(*args, **kwargs)

    def plot_adjoint_overlay(self, *args, **kwargs):
        return plot_adjoint_overlay(*args, **kwargs)

    def plot_point_timeseries(self, *args, **kwargs):
        return plot_point_timeseries(*args, **kwargs)

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
