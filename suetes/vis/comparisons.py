import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from suetes.vis.utils import _get_plot_data, _get_projection_and_extent
from suetes.vis.spatial import plot_2d_field
from suetes.vis.vertical import plot_cross_section, plot_hovmoller 
from suetes.vis.diagnostics import plot_energy_spectrum

def plot_ablation_spatial_matrix(grid, states_dict, target_state, variable='th_v', 
                                 z_idx=5, sponge_depth=30, constants=None, save_path=None):
    """Delegates 2D map drawing to spatial.plot_2d_field."""
    run_names = list(states_dict.keys())
    cols = len(run_names) + 1 
    
    native_proj, native_extent = _get_projection_and_extent(grid)
    fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 6.5), 
                             subplot_kw={'projection': native_proj})
    
    # Gather data and compute global colorbar limits
    all_data = {name: _get_plot_data(grid, state, variable, z_idx, constants) for name, state in states_dict.items()}
    target_data = _get_plot_data(grid, target_state, variable, z_idx, constants)
    
    inner_vals = [d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth] if sponge_depth > 0 else d 
                  for d in list(all_data.values()) + [target_data]]
    flat_vals = np.concatenate([np.asarray(d).flatten() for d in inner_vals])
    
    vmin = -0.1 if np.all(np.isnan(flat_vals)) else float(np.nanpercentile(flat_vals, 0.5))
    vmax =  0.1 if np.all(np.isnan(flat_vals)) else float(np.nanpercentile(flat_vals, 99.5))
    if vmin == vmax: vmin -= 1e-5; vmax += 1e-5

    # Draw Top Row (Absolute Fields)
    for i, name in enumerate(run_names):
        im_abs = plot_2d_field(grid, states_dict[name], variable, z_idx, sponge_depth, constants, 
                                cmap='RdBu_r', vmin=vmin, vmax=vmax, title=name, ax=axes[0, i])
        
    plot_2d_field(grid, target_state, variable, z_idx, sponge_depth, constants,
                  cmap='RdBu_r', vmin=vmin, vmax=vmax, title="Target (ERA5)", ax=axes[0, -1])
    
    cbar_ax_top = fig.add_axes([0.91, 0.55, 0.015, 0.33]) 
    fig.colorbar(im_abs, cax=cbar_ax_top, orientation='vertical', label=variable)

    # Draw Bottom Row (Anomalies)
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
    
    # Global limit logic 
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

    # Draw Top Row
    for i, name in enumerate(run_names):
        c_abs = plot_cross_section(grid, states_dict[name], variable, y_idx, sponge_depth, 
                                    vmin=vmin, vmax=vmax, title=name, ax=axes[0, i])
        
    plot_cross_section(grid, target_state, variable, y_idx, sponge_depth, 
                       vmin=vmin, vmax=vmax, title="Target (ERA5)", ax=axes[0, -1])
                       
    cbar_ax_top = fig.add_axes([0.91, 0.55, 0.015, 0.33])
    fig.colorbar(c_abs, cax=cbar_ax_top, orientation='vertical', label=variable)
    
    # Draw Bottom Row (Anomalies)
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


def plot_worst_case_dashboard(grid, time_axis_mins, ts_era5, ts_baseline, ts_worst_case, 
                              pert_th_v_2d, state_baseline, state_worst_case, 
                              loc_idx, sponge_depth=30, save_path=None):
    """
    Generates a 4-panel dashboard showcasing the Adjoint worst-case impact.
    Includes Time Series (ERA5, Baseline, Worst-Case), optimal perturbation map, 
    and cross-sections of the baseline flow vs. the wind anomaly.
    """
    import matplotlib.gridspec as gridspec
    
    i_w, j_w = loc_idx
    
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)
    
    # Convert time axis to hours for the plot
    time_axis_hours = np.array(time_axis_mins) / 60.0
    
    # --- Panel A: Time Series ---
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(time_axis_hours, ts_era5, 'r--', linewidth=2, label='ERA5 driver')
    ax_ts.plot(time_axis_hours, ts_baseline, 'b-', linewidth=2, alpha=0.7, label='Baseline suetes')
    ax_ts.plot(time_axis_hours, ts_worst_case, 'k-', linewidth=2, label='Worst-case (Adjoint)')
    
    # Cleaned up title and updated x-axis label
    ax_ts.set_title("Surface wind speed evolution at target", fontsize=14)
    ax_ts.set_xlabel("Simulation time [Hours]", fontsize=12)
    ax_ts.set_ylabel("Wind speed [km/h]", fontsize=12)
    ax_ts.grid(True, linestyle='--', alpha=0.6)
    ax_ts.legend(fontsize=11)
    
    # --- Panel B: Optimal Perturbation Map ---
    native_proj, native_extent = _get_projection_and_extent(grid)
    ax_map = fig.add_subplot(gs[0, 1], projection=native_proj)
    vmax_pert = float(np.max(np.abs(pert_th_v_2d)))
    if vmax_pert == 0: vmax_pert = 0.1
    
    # Create a dummy state to satisfy the plot_2d_field signature
    dummy_state = {'th_v': np.zeros((grid.nx, grid.ny, grid.nz))} 
    im_pert = plot_2d_field(grid, dummy_state, 'th_v', z_idx=0, sponge_depth=sponge_depth, 
                            cmap='RdBu_r', vmin=-vmax_pert, vmax=vmax_pert, scale='sym',
                            title=r"Optimal upstream thermal perturbation ($\Delta \theta_v$)", 
                            ax=ax_map, plot_data=pert_th_v_2d)
    
    # Overlay Target Star
    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    if lons.max() - lons.min() > 180.0: lons = np.where(lons < 0, lons + 360.0, lons)
    ax_map.plot(lons[i_w, j_w], lats[i_w, j_w], 'k*', markersize=16, transform=ccrs.PlateCarree(), label='Target')
    ax_map.legend(loc='lower right')
    
    fig.colorbar(im_pert, ax=ax_map, orientation='horizontal', pad=0.08, fraction=0.046, label='Thermal shift [K]')
    
    # --- Prepare Cross Section Data ---
    u_base_m = 0.5 * (state_baseline['u'][:-1, :, :] + state_baseline['u'][1:, :, :])
    u_worst_m = 0.5 * (state_worst_case['u'][:-1, :, :] + state_worst_case['u'][1:, :, :])
    u_diff = (u_worst_m - u_base_m) * 3.6  # km/h
    
    # --- Panel C: Baseline Cross Section ---
    ax_base = fig.add_subplot(gs[1, 0])
    c_base = plot_cross_section(grid, state_baseline, 'u', y_idx=j_w, sponge_depth=sponge_depth,
                            title="Baseline zonal wind [km/h] across terrain", ax=ax_base, 
                            plot_data=u_base_m[:, j_w, :] * 3.6, overlay_isentropes=True,
                            vmin=-50.0, vmax=150.0) # Explicitly cap the bounds
    ax_base.set_ylim([0, 4.0]) # Cap at 4km altitude to highlight the mountain wave
    fig.colorbar(c_base, ax=ax_base, orientation='horizontal', pad=0.15, label='Baseline zonal wind [km/h]')
    
    # --- Panel D: Anomaly Cross Section ---
    ax_diff = fig.add_subplot(gs[1, 1])
    # Extract the interior to calculate color bounds without edge artifacts
    interior_diff = u_diff[sponge_depth:-sponge_depth, j_w, :] if sponge_depth > 0 else u_diff[:, j_w, :]
    vmax_diff = float(np.max(np.abs(interior_diff)))
    if vmax_diff == 0: vmax_diff = 1.0
    
    c_diff = plot_cross_section(grid, state_worst_case, 'u', y_idx=j_w, sponge_depth=sponge_depth,
                                cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff,
                                title=r"Zonal wind anomaly [$\Delta \mathbf{u}$] induced by adjoint", 
                                ax=ax_diff, plot_data=u_diff[:, j_w, :])
    ax_diff.set_ylim([0, 4.0])
    fig.colorbar(c_diff, ax=ax_diff, orientation='horizontal', pad=0.15, label='Wind difference [km/h]')
    
    plt.suptitle("Suetes downscaling: Adjoint optimal perturbation impact", fontsize=18, y=0.98)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: plt.show()
    plt.close()