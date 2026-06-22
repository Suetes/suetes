import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from suetes.vis.utils import _get_plot_data, _draw_domain_and_sponge, _get_level_height, _get_projection_and_extent

def plot_cross_section(grid, state, variable, y_idx, sponge_depth=30, overlay_isentropes=False, 
                       cmap=None, vmin=None, vmax=None, title=None, ax=None, save_path=None, 
                       plot_data=None, xlim=None, ylim=None, overlay_quiver=False, quiver_stride=4):
    """
    Generates a vertical cross-section plot along the X-axis.
    Displays the physical terrain elevation and boundary layer sponge shading.
    Allows for horizontal/vertical zooming and vector/isentrope overlays.
    """
    x_coords = grid.x_m / 1000.0  
    Z_slice = grid.Z_w[:, y_idx, :] if variable == 'w' else grid.Z_m[:, y_idx, :]
    data_slice = plot_data if plot_data is not None else state[variable][:, y_idx, :]
    
    # Pad mass-point variables down to the surface (Z_w[0]) for plotting
    if variable != 'w':
        surface_Z = grid.Z_w[:, y_idx, 0:1] 
        surface_data = data_slice[:, 0:1]   
        Z_slice = np.concatenate([surface_Z, Z_slice], axis=1)
        data_slice = np.concatenate([surface_data, data_slice], axis=1)
    
    if sponge_depth > 0:
        interior_slice = data_slice[sponge_depth:-sponge_depth, :]
    else:
        interior_slice = data_slice
        
    if cmap is None:
        cmap = 'seismic' if variable == 'w' else ('Blues' if variable == 'q_c' else 'RdBu_r')
    
    # Calculate limits strictly based on the interior domain
    if vmin is None or vmax is None:
        if variable == 'w':
            vmax_calc = float(np.max(np.abs(interior_slice)))
            vmin_calc = -vmax_calc
        else:
            vmax_calc = float(np.max(interior_slice))
            vmin_calc = float(np.min(interior_slice))
            if variable == 'q_c': vmin_calc = 0.0
        
        if vmax_calc <= vmin_calc:
            vmax_calc = vmin_calc + 1e-5
            
        vmin = vmin if vmin is not None else vmin_calc
        vmax = vmax if vmax is not None else vmax_calc

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    
    X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
    levels = np.linspace(vmin, vmax, 31)
    
    # Generate the main contour
    contour = ax.contourf(X_2d, Z_slice / 1000.0, data_slice, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    
    if overlay_isentropes and 'th_v' in state:
        th_v_slice = state['th_v'][:, y_idx, :]
        th_levels = np.arange(np.floor(np.min(th_v_slice)), np.ceil(np.max(th_v_slice)), 2.0)
        
        surf_th = th_v_slice[:, 0:1]
        th_v_pad = np.concatenate([surf_th, th_v_slice], axis=1)
        Z_m_pad = np.concatenate([grid.Z_w[:, y_idx, 0:1], grid.Z_m[:, y_idx, :]], axis=1) / 1000.0
        
        cl = ax.contour(X_2d, Z_m_pad, th_v_pad, levels=th_levels, colors='black', linewidths=1.0, alpha=0.7)
        ax.clabel(cl, inline=True, fmt='%1.0f', fontsize=8)
        
    if overlay_quiver and 'u' in state and 'w' in state:
        u_slice = 0.5 * (state['u'][:-1, y_idx, :] + state['u'][1:, y_idx, :]) if state['u'].shape[0] > grid.nx else state['u'][:, y_idx, :]
        w_slice = state['w'][:, y_idx, :]
        
        nz_min = min(u_slice.shape[1], w_slice.shape[1])
        s = quiver_stride
        
        X_q = X_2d[::s, :nz_min:s]
        Z_q = (Z_slice[:, :nz_min] / 1000.0)[::s, ::s]
        U_q = u_slice[::s, :nz_min:s]
        W_q = w_slice[::s, :nz_min:s]
        
        ax.quiver(X_q, Z_q, U_q, W_q, color='purple', alpha=0.5, width=0.002)
        
    # Draw physical terrain
    ax.fill_between(x_coords, 0, grid.Z_w[:, y_idx, 0] / 1000.0, color='dimgray', label='Topography')
    
    # Domain Zoom/Crop
    if xlim is not None: ax.set_xlim(xlim)
    else: ax.set_xlim([x_coords.min(), x_coords.max()])
        
    if ylim is not None: ax.set_ylim(ylim)
    else: ax.set_ylim([0, (grid.Z_w[:, y_idx, -1] / 1000.0).max()])

    # Draw sponges only if they are within the cropped view
    if sponge_depth > 0:
        x_left = x_coords[sponge_depth]
        x_right = x_coords[-sponge_depth-1]
        if xlim is None or (xlim[0] < x_left):
            ax.axvline(x=x_left, color='k', linestyle='--', linewidth=1.5)
            ax.axvspan(x_coords[0], x_left, color='black', alpha=0.15, zorder=4)
        if xlim is None or (xlim[1] > x_right):
            ax.axvline(x=x_right, color='k', linestyle='--', linewidth=1.5, label='Sponge' if xlim is None else None)
            ax.axvspan(x_right, x_coords[-1], color='black', alpha=0.15, zorder=4)

    ax.set_title(title or f"Cross Section: {variable} (y-index: {y_idx})", fontsize=11)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Altitude [km]")

    if show_plot:
        plt.colorbar(contour, ax=ax, label=variable)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()
        
    return contour

def plot_slice_locator_dashboard(grid, state, map_var='th_v', slice_var='w', map_z=5, 
                                 sponge_depth=30, constants=None, save_path=None,
                                 x_indices=None, y_indices=None, slice_xlim=None, slice_ylim=None):
    """Plots two locator maps and two sets of 3 vertical cross-sections."""
    import matplotlib.gridspec as gridspec
    
    # Select 3 evenly spaced indices for X and Y slices
    if y_indices is None:
        y_indices = [grid.ny // 4, grid.ny // 2, 3 * grid.ny // 4]
    if x_indices is None:
        x_indices = [grid.nx // 4, grid.nx // 2, 3 * grid.nx // 4]
    
    x_colors = ['#FF595E', '#FFCA3A', '#8AC926'] # Distinct colors for X slices (Horizontal cuts)
    y_colors = ['#1982C4', '#6A4C93', '#F15BB5'] # Distinct colors for Y slices (Vertical cuts)
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5], hspace=0.25, wspace=0.15)
    
    map_data = _get_plot_data(grid, state, map_var, map_z, constants)
    z_height = _get_level_height(grid, map_z)
    
    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    if lons.max() - lons.min() > 180.0: lons = np.where(lons < 0, lons + 360.0, lons)
    
    native_proj, native_extent = _get_projection_and_extent(grid)

    def draw_locator_map(ax, lines_indices, colors, is_x_slice):
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        im_map = ax.pcolormesh(lons, lats, map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        _draw_domain_and_sponge(ax, lons, lats, sponge_depth)
        
        for idx, color in zip(lines_indices, colors):
            if is_x_slice: # Drawing a line across constant Y
                ax.plot(lons[:, idx], lats[:, idx], color=color, linewidth=2.5, transform=ccrs.PlateCarree(), label=f'Y-idx: {idx}')
            else: # Drawing a line across constant X
                ax.plot(lons[idx, :], lats[idx, :], color=color, linewidth=2.5, transform=ccrs.PlateCarree(), label=f'X-idx: {idx}')
        
        ax.legend(loc='upper right')
        ax.set_extent(native_extent, crs=native_proj)
        ax.set_title(f"Locator Map: {map_var} at Level {map_z} (~{z_height}m)", fontsize=14)
        return im_map

    # --- Top Left: Map with X slices (constant y cuts) ---
    ax_map_x = fig.add_subplot(gs[0, 0], projection=native_proj)
    im_x = draw_locator_map(ax_map_x, y_indices, x_colors, is_x_slice=True)
    plt.colorbar(im_x, ax=ax_map_x, orientation='vertical', pad=0.02, fraction=0.03)

    # --- Top Right: Map with Y slices (constant x cuts) ---
    ax_map_y = fig.add_subplot(gs[0, 1], projection=native_proj)
    im_y = draw_locator_map(ax_map_y, x_indices, y_colors, is_x_slice=False)
    plt.colorbar(im_y, ax=ax_map_y, orientation='vertical', pad=0.02, fraction=0.03)

    def draw_slice(ax, slice_data, coords, Z_coords, title, line_color, show_x_label=False):
        inner_data = slice_data[sponge_depth:-sponge_depth, :] if sponge_depth > 0 else slice_data
        
        vmax = float(np.max(np.abs(inner_data))) if slice_var == 'w' else float(np.max(inner_data))
        vmin = -vmax if slice_var == 'w' else float(np.min(inner_data))
        if slice_var == 'q_c': vmin = 0.0
        if vmax <= vmin: vmax = vmin + 1e-5
            
        cmap = 'seismic' if slice_var == 'w' else ('Blues' if slice_var == 'q_c' else 'RdBu_r')
        levels = np.linspace(vmin, vmax, 31)
        
        X_2d = np.broadcast_to(coords[:, None], Z_coords.shape)
        c = ax.contourf(X_2d, Z_coords, slice_data, levels=levels, cmap=cmap, extend='both')
        
        ax.fill_between(coords, 0, Z_coords[:, 0], color='dimgray')
        
        if slice_xlim is not None:
            ax.set_xlim(slice_xlim)
        if slice_ylim is not None:
            ax.set_ylim(slice_ylim)
        else:
            ax.set_ylim(0, 15000)

        if sponge_depth > 0:
            ax.axvspan(coords[0], coords[sponge_depth], color='black', alpha=0.15, zorder=4)
            ax.axvspan(coords[-sponge_depth-1], coords[-1], color='black', alpha=0.15, zorder=4)

        # Color the border spine to match the map locator line
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
            spine.set_linewidth(3.0)
            
        ax.text(0.01, 0.85, title, transform=ax.transAxes, fontsize=12, color=line_color, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        if show_x_label: ax.set_xlabel("Distance [km]")
        else: ax.set_xticklabels([])
        ax.set_ylabel("Height [m]")
        return c

    # --- Bottom Left: 3 Stacked X-slices ---
    gs_x = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.1)
    for i, (y_idx, color) in enumerate(zip(y_indices, x_colors)):
        ax_slice = fig.add_subplot(gs_x[i])
        Z_x = grid.Z_w[:, y_idx, :] if slice_var == 'w' else grid.Z_m[:, y_idx, :]
        c_x = draw_slice(ax_slice, state[slice_var][:, y_idx, :], grid.x_m / 1000.0, Z_x, f"Y-idx: {y_idx}", color, show_x_label=(i==2))
    
    cbar_ax_x = fig.add_axes([0.15, 0.05, 0.3, 0.015])
    fig.colorbar(c_x, cax=cbar_ax_x, orientation='horizontal', label=slice_var)

    # --- Bottom Right: 3 Stacked Y-slices ---
    gs_y = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.1)
    for i, (x_idx, color) in enumerate(zip(x_indices, y_colors)):
        ax_slice = fig.add_subplot(gs_y[i])
        Z_y = grid.Z_w[x_idx, :, :] if slice_var == 'w' else grid.Z_m[x_idx, :, :]
        c_y = draw_slice(ax_slice, state[slice_var][x_idx, :, :], grid.y_m / 1000.0, Z_y, f"X-idx: {x_idx}", color, show_x_label=(i==2))

    cbar_ax_y = fig.add_axes([0.57, 0.05, 0.3, 0.015])
    fig.colorbar(c_y, cax=cbar_ax_y, orientation='horizontal', label=slice_var)

    # Make room for the bottom horizontal colorbars
    plt.subplots_adjust(bottom=0.12)
    
    if save_path: plt.savefig(save_path, dpi=200)
    else: plt.show()
    plt.close()

def plot_hovmoller(grid, time_hours_array, hov_data_2d, variable='th_v Anomaly', cmap='RdBu_r', 
                   vmin=None, vmax=None, title=None, ax=None, save_path=None):
    """
    Plots a Hovmöller diagram (Time vs Longitude).
    """
    x_coords = grid.x_m / 1000.0
    X, Y = np.meshgrid(x_coords, time_hours_array)
    
    # Calculate limits if not provided by the parent script
    if vmin is None or vmax is None:
        calc_vmax = float(np.max(np.abs(hov_data_2d)))
        calc_vmin = -calc_vmax
        if calc_vmax <= calc_vmin: calc_vmax = calc_vmin + 1e-5
        vmin = vmin if vmin is not None else calc_vmin
        vmax = vmax if vmax is not None else calc_vmax
        
    levels = np.linspace(vmin, vmax, 41)
    
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        show_plot = True
        
    im = ax.contourf(X, Y, hov_data_2d, levels=levels, cmap=cmap, extend='both')
    
    ax.set_title(title or f"Hovmöller Diagram: {variable}", fontsize=14)
    ax.set_ylabel("Simulation Time [Hours]")
    ax.set_xlabel("X Distance [km]")
    ax.invert_yaxis()
    
    if show_plot:
        plt.colorbar(im, ax=ax, pad=0.02, label=variable)
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()
        
    return im
