import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from suetes.vis.utils import _get_plot_data
from suetes.vis.spatial import plot_2d_field
from suetes.vis.vertical import plot_cross_section, plot_hovmoller 
from suetes.vis.diagnostics import plot_energy_spectrum

def plot_ablation_spatial_matrix(grid, states_dict, target_state, variable='th_v', 
                                 z_idx=5, sponge_depth=30, constants=None, save_path=None):
    """Delegates 2D map drawing to spatial.plot_2d_field."""
    run_names = list(states_dict.keys())
    cols = len(run_names) + 1 
    
    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 6.5), 
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
    
    # 1. Gather data and compute global colorbar limits
    all_data = {name: _get_plot_data(grid, state, variable, z_idx, constants) for name, state in states_dict.items()}
    target_data = _get_plot_data(grid, target_state, variable, z_idx, constants)
    
    inner_vals = [d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth] if sponge_depth > 0 else d 
                  for d in list(all_data.values()) + [target_data]]
    flat_vals = np.concatenate([np.asarray(d).flatten() for d in inner_vals])
    
    vmin = -0.1 if np.all(np.isnan(flat_vals)) else float(np.nanpercentile(flat_vals, 0.5))
    vmax =  0.1 if np.all(np.isnan(flat_vals)) else float(np.nanpercentile(flat_vals, 99.5))
    if vmin == vmax: vmin -= 1e-5; vmax += 1e-5

    # 2. Draw Top Row (Absolute Fields)
    for i, name in enumerate(run_names):
        im_abs = plot_2d_field(grid, states_dict[name], variable, z_idx, sponge_depth, constants, 
                                cmap='RdBu_r', vmin=vmin, vmax=vmax, title=name, ax=axes[0, i])
        
    plot_2d_field(grid, target_state, variable, z_idx, sponge_depth, constants,
                  cmap='RdBu_r', vmin=vmin, vmax=vmax, title="Target (ERA5)", ax=axes[0, -1])
    
    cbar_ax_top = fig.add_axes([0.91, 0.55, 0.015, 0.33]) 
    fig.colorbar(im_abs, cax=cbar_ax_top, orientation='vertical', label=variable)

    # 3. Draw Bottom Row (Anomalies)
    anomalies = [all_data[name] - target_data for name in run_names]
    inner_anoms = [a[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth] if sponge_depth > 0 else a for a in anomalies]
    
    # Use percentile for anomaly scaling instead of nanmax
    flat_anoms = np.concatenate([np.asarray(a).flatten() for a in inner_anoms])
    vmax_anom = 0.1 if np.all(np.isnan(flat_anoms)) else float(np.nanpercentile(np.abs(flat_anoms), 99.0))
    if vmax_anom == 0: vmax_anom = 0.1
    
    for i, name in enumerate(run_names):
        rmse = float(np.sqrt(np.nanmean(inner_anoms[i]**2)))
        title_str = f"Error: {name}\nRMSE: {rmse:.4f}"
        
        im_anom = plot_2d_field(grid, states_dict[name], variable, z_idx, sponge_depth, constants,
                                 cmap='seismic', vmin=-vmax_anom, vmax=vmax_anom, 
                                 title=title_str, ax=axes[1, i], plot_data=anomalies[i])
        
    axes[1, -1].axis('off')
    cbar_ax_bot = fig.add_axes([0.91, 0.13, 0.015, 0.33])
    fig.colorbar(im_anom, cax=cbar_ax_bot, orientation='vertical', label=f"{variable} Error")

    plt.suptitle(f"Ablation Study: {variable.upper()} at Level {z_idx}", fontsize=18, y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.05, right=0.89)
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_ablation_cross_section_matrix(grid, states_dict, target_state, variable='th_v', 
                                       y_idx=None, sponge_depth=30, save_path=None):
    """Delegates X-Z drawing to vertical.plot_cross_section."""
    if y_idx is None: y_idx = grid.ny // 2
    run_names = list(states_dict.keys())
    
    fig, axes = plt.subplots(2, len(run_names) + 1, figsize=(6 * (len(run_names) + 1), 8))
    
    # 1. Global limit logic 
    all_data = [states_dict[name][variable][:, y_idx, :] for name in run_names] + [target_state[variable][:, y_idx, :]]
    inner_vals = [d[sponge_depth:-sponge_depth, :] if sponge_depth > 0 else d for d in all_data]
    flat_vals = np.concatenate([np.asarray(d).flatten() for d in inner_vals])
    
    if variable == 'w':
        vmax = float(np.nanmax(np.abs(flat_vals)))
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(flat_vals, 0.5))
        vmax = float(np.nanpercentile(flat_vals, 99.5))
    if vmin == vmax: vmin -= 1e-5; vmax += 1e-5

    # 2. Draw Top Row
    for i, name in enumerate(run_names):
        c_abs = plot_cross_section(grid, states_dict[name], variable, y_idx, sponge_depth, 
                                    vmin=vmin, vmax=vmax, title=name, ax=axes[0, i])
        
    plot_cross_section(grid, target_state, variable, y_idx, sponge_depth, 
                       vmin=vmin, vmax=vmax, title="Target (ERA5)", ax=axes[0, -1])
                       
    cbar_ax_top = fig.add_axes([0.91, 0.55, 0.015, 0.33])
    fig.colorbar(c_abs, cax=cbar_ax_top, orientation='vertical', label=variable)
    
    # 3. Draw Bottom Row (Anomalies)
    anomalies = [states_dict[name][variable][:, y_idx, :] - target_state[variable][:, y_idx, :] for name in run_names]
    inner_anoms = [a[sponge_depth:-sponge_depth, :] if sponge_depth > 0 else a for a in anomalies]
    
    # Use percentile for cross-section anomalies as well
    flat_anoms = np.concatenate([np.asarray(a).flatten() for a in inner_anoms])
    vmax_anom = 0.1 if np.all(np.isnan(flat_anoms)) else float(np.nanpercentile(np.abs(flat_anoms), 99.0))
    if vmax_anom == 0: vmax_anom = 0.1

    for i, name in enumerate(run_names):
        rmse = float(np.sqrt(np.nanmean(inner_anoms[i]**2)))
        c_anom = plot_cross_section(grid, states_dict[name], variable, y_idx, sponge_depth, 
                                     cmap='seismic', vmin=-vmax_anom, vmax=vmax_anom, 
                                     title=f"Error: {name}\nRMSE: {rmse:.4f}", ax=axes[1, i], plot_data=anomalies[i])
        
    axes[1, -1].axis('off')
    
    cbar_ax_bot = fig.add_axes([0.91, 0.13, 0.015, 0.33])
    fig.colorbar(c_anom, cax=cbar_ax_bot, orientation='vertical', label=f"{variable} Error")
    
    plt.suptitle(f"Cross Section Ablation: {variable.upper()} (Y-Index {y_idx})", fontsize=18, y=0.98)
    plt.subplots_adjust(wspace=0.15, hspace=0.4, right=0.89)
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()


def plot_ablation_hovmoller_matrix(grid, hov_times, hov_data_dict, variable='th_v Anomaly', 
                                   cmap='RdBu_r', save_path=None):
    """Delegates to vertical.plot_hovmoller."""
    run_names = list(hov_data_dict.keys())
    fig, axes = plt.subplots(1, len(run_names), figsize=(5 * len(run_names), 8), sharey=True)
    if len(run_names) == 1: axes = [axes]
    
    vmax = max(float(np.max(np.abs(data))) for data in hov_data_dict.values())
    if vmax <= 0.0: vmax = 0.1
    
    for ax, name in zip(axes, run_names):
        # Now safely pass the ax and title kwargs!
        im = plot_hovmoller(grid, hov_times, hov_data_dict[name], variable, cmap, 
                            vmin=-vmax, vmax=vmax, title=name, ax=ax)
        
    axes[0].set_ylabel("Simulation Time [Hours]")
    plt.suptitle(f"Hovmöller Diagram: {variable}", fontsize=18, y=0.98)
    
    # ---> ADD GLOBAL COLORBAR FOR HOVMOLLER <---
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical', label=variable)
    
    plt.subplots_adjust(right=0.89)
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_ablation_spectrum(grid, states_dict, variable='w', z_idx=5, sponge_depth=30, save_path=None):
    """Delegates spectral calculations to diagnostics.plot_energy_spectrum."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(states_dict)))
    
    for i, (name, state) in enumerate(states_dict.items()):
        # Only draw the k^-3 and k^-5/3 reference slopes for the first run to prevent clutter
        draw_refs = (i == 0) 
        
        plot_energy_spectrum(
            grid, state, variable, z_idx, sponge_depth,
            ax=ax, color=colors[i], label=name, plot_reference_slopes=draw_refs
        )

    ax.set_title(f"Ablation Power Spectra: {variable} at level {z_idx}", fontsize=16)
    ax.set_xlabel("Wavenumber $k$ [m⁻¹]", fontsize=12)
    ax.set_ylabel("Spectral power density", fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11)
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()