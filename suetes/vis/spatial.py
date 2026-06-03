import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from suetes.vis.utils import _get_plot_data, _draw_domain_and_sponge, _get_level_height

def plot_2d_field(grid, state, variable, z_idx=5, sponge_depth=30, constants=None, 
                  cmap='viridis', vmin=None, vmax=None, scale='linear', title=None, ax=None, save_path=None, plot_data=None):
    """
    Plots a horizontal cross-section of a 3D field on a geographic map projection.
    Includes an automatic antimeridian wrap fix for domains crossing the dateline.
    """
    data = plot_data if plot_data is not None else _get_plot_data(grid, state, variable, z_idx, constants)
    
    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    
    # Detect if the domain crosses the antimeridian (creating a massive min/max gap)
    if lons.max() - lons.min() > 180.0:
        # Convert negative longitudes back to positive (e.g., -179 -> 181) 
        # to make the array continuous for the extent calculation and pcolormesh.
        lons = np.where(lons < 0, lons + 360.0, lons)

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
        show_plot = True

    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
    ax.set_facecolor('darkgray')

    extend = 'neither'
    if vmin is None and vmax is None:
        if sponge_depth > 0:
            inner_data = data[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
        else:
            inner_data = data
            
        # Use percentiles instead of nanmax/nanmin
        data_max = float(np.nanpercentile(inner_data, 99.5))
        data_min = float(np.nanpercentile(inner_data, 0.5))
        
        if scale == 'sym': 
            limit = max(abs(data_max), abs(data_min))
            
            vmax = limit
            vmin = -limit
            extend = 'both'
        else:
            vmax = data_max
            vmin = data_min

    im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    _draw_domain_and_sponge(ax, lons, lats, sponge_depth)
    
    ax.set_title(title or f"{variable.upper()} at Level {z_idx}", fontsize=14)
    ax.gridlines(draw_labels=show_plot, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    if show_plot:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, extend=extend)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()
        
    return im

def plot_quiver_field(grid, state, bg_var='pi', u_var='u', v_var='v', z_idx=5, 
                      sponge_depth=30, constants=None, cmap='coolwarm', 
                      title=None, ax=None, save_path=None, stride=15):
    """Plots a contoured background field with a geographic wind quiver overlay."""
    
    bg_data = _get_plot_data(grid, state, bg_var, z_idx, constants)
    u_grid = _get_plot_data(grid, state, u_var, z_idx, constants)
    v_grid = _get_plot_data(grid, state, v_var, z_idx, constants)
    
    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    
    # Dateline fix
    if lons.max() - lons.min() > 180.0:
        lons = np.where(lons < 0, lons + 360.0, lons)

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
        show_plot = True

    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
    ax.set_facecolor('darkgray')

    # Background Contour Plot
    im = ax.contourf(lons, lats, bg_data, levels=100, transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.85)
    
    # Wind Vector Geographic Rotation
    # The U/V in the model state are grid-relative. We MUST rotate them back to 
    # Earth-relative (North/East) before plotting them on a map projection!
    gamma = np.array(grid.proj.get_convergence_angle(Xi, Yi))
    u_geo = u_grid * np.cos(gamma) - v_grid * np.sin(gamma)
    v_geo = u_grid * np.sin(gamma) + v_grid * np.cos(gamma)

    # Quiver Overlay
    # Slice the arrays to prevent dense black blobs
    s = stride
    q = ax.quiver(lons[::s, ::s], lats[::s, ::s], u_geo[::s, ::s], v_geo[::s, ::s], 
                  transform=ccrs.PlateCarree(), pivot='middle', color='black', 
                  width=0.003, headwidth=4, headlength=5)
    
    _draw_domain_and_sponge(ax, lons, lats, sponge_depth)
    
    ax.set_title(title or f"{bg_var.upper()} and Wind Vectors at Level {z_idx}", fontsize=14)
    ax.gridlines(draw_labels=show_plot, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    # Add a reference arrow for magnitude
    ax.quiverkey(q, X=0.9, Y=1.05, U=15, label='15 m/s', labelpos='E', coordinates='axes')

    if show_plot:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()
        
    return im

def plot_dashboard(grid, state_model, z_idx=5, sponge_depth=30, fields=None, time_hours=None, save_path=None):
    """Generates a multi-panel overview of the model state."""
    if fields is None:
        fields = [
            {'var': 'w', 'cmap': 'seismic', 'title': 'Vertical velocity [m/s]', 'scale': 'sym'},
            {'var': 'div', 'cmap': 'seismic', 'title': 'Horizontal divergence [s⁻¹]', 'scale': 'sym'},
            {'var': 'th_v', 'cmap': 'RdBu_r', 'title': 'Potential temperature [K]', 'scale': 'linear'},
            {'type': 'quiver', 'bg_var': 'pi', 'cmap': 'coolwarm', 'title': 'Exner pressure & wind field'} 
        ]

    n_vars = len(fields)
    cols = 2
    rows = int(np.ceil(n_vars / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
    axes = np.atleast_1d(axes).flatten()
    
    for i, field_def in enumerate(fields):
        if field_def.get('type') == 'quiver':
            # Route to the quiver method
            im = plot_quiver_field(
                grid, state_model, 
                bg_var=field_def.get('bg_var', 'pi'),
                u_var=field_def.get('u_var', 'u'),
                v_var=field_def.get('v_var', 'v'),
                z_idx=z_idx, sponge_depth=sponge_depth, 
                cmap=field_def.get('cmap', 'coolwarm'), 
                title=field_def['title'], ax=axes[i], stride=12
            )
            fig.colorbar(im, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.046)
        else:
            # Standard pcolormesh
            scale = field_def.get('scale', 'linear')
            im = plot_2d_field(
                grid, state_model, field_def['var'], z_idx, sponge_depth, 
                cmap=field_def.get('cmap', 'viridis'), scale=scale, 
                title=field_def['title'], ax=axes[i]
            )
            extend = 'both' if scale == 'sym' else 'neither'
            fig.colorbar(im, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.046, extend=extend)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    height_str = f"Z={z_idx}m" if isinstance(z_idx, float) else f"Level {z_idx}"
    title = f"Suetes results ({height_str})" + (f" | T={time_hours}h" if time_hours else "")
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_comparison(grid, state_model, state_era5, variable, z_idx=5, sponge_depth=30, save_path=None):
    """Plots the Model field, the target ERA5 field, and their discrete difference (Anomaly)."""
    val_model = _get_plot_data(grid, state_model, variable, z_idx)
    val_era5 = _get_plot_data(grid, state_era5, variable, z_idx)
    anomaly = val_model - val_era5

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
    titles = ["Suetes model", "ERA5 target", "Anomaly (Model - ERA5)"]
    datas = [val_model, val_era5, anomaly]
    
    vmax_main = max(float(np.max(val_model)), float(np.max(val_era5)))
    vmin_main = min(float(np.min(val_model)), float(np.min(val_era5)))
    vmax_anom = max(float(np.max(np.abs(anomaly))), 0.1)

    for ax, data, title in zip(axes, datas, titles):
        cmap = 'seismic' if 'Anomaly' in title else 'RdBu_r'
        vmin, vmax = (-vmax_anom, vmax_anom) if 'Anomaly' in title else (vmin_main, vmax_main)

        im = plot_2d_field(grid, state_model, variable, z_idx, sponge_depth, 
                           cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax, plot_data=data)
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_level_strip(grid, state, variable, z_indices=[0, 5, 15, 30], sponge_depth=30, constants=None, cmap=None, scale='linear', save_path=None):
    """Plots a 2x2 grid of the same variable at four different vertical levels, with localized colorbars."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
    axes = axes.flatten()
    
    if cmap is None:
        cmap = 'seismic' if variable in ['w', 'div'] else ('Blues' if variable == 'q_c' else 'RdBu_r')
    if variable in ['w', 'div']: scale = 'sym'

    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    if lons.max() - lons.min() > 180.0: lons = np.where(lons < 0, lons + 360.0, lons)
    extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

    for i, (ax, z) in enumerate(zip(axes, z_indices)):
        data = _get_plot_data(grid, state, variable, z, constants)
        z_height = _get_level_height(grid, z)
        
        # Isolate the interior domain (Sponge layer protection)
        if sponge_depth > 0:
            inner_data = data[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
        else:
            inner_data = data
        
        # Dynamic Percentile Scaling (Spike protection)
        # This automatically adapts to the magnitude of the data
        vmax = float(np.percentile(inner_data, 99.5))
        vmin = float(np.percentile(inner_data, 0.5))
        
        extend = 'neither'
        if scale == 'sym':
            limit = max(abs(vmax), abs(vmin))
            vmin, vmax = -limit, limit
            extend = 'both'
        elif variable == 'q_c': 
            vmin = 0.0

        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':', edgecolor='gray')
        
        im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        _draw_domain_and_sponge(ax, lons, lats, sponge_depth)
        
        ax.set_title(f"Level {z} (~{z_height} m)", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add an individual colorbar to each subplot
        fig.colorbar(im, ax=ax, orientation='vertical', extend=extend, pad=0.02, fraction=0.046)
        
    plt.suptitle(f"Vertical Profile: {variable.upper()}", fontsize=18, y=0.98)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()

def plot_adjoint_overlay(grid, initial_state, sensitivity_2d, u_var='u', v_var='v', z_idx=5, sponge_depth=30, stride=15, save_path=None):
    """Overlays the forward initial wind field on top of the adjoint sensitivities."""
    u_grid = _get_plot_data(grid, initial_state, u_var, z_idx)
    v_grid = _get_plot_data(grid, initial_state, v_var, z_idx)
    
    Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
    lats, lons = grid.proj.get_lat_lon(Xi, Yi)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    
    # 1. Plot Adjoint Sensitivity as the background
    vmax = float(np.max(np.abs(sensitivity_2d)))
    im = ax.contourf(lons, lats, sensitivity_2d, levels=50, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.85)
    
    # 2. Plot Forward Winds on top
    gamma = np.array(grid.proj.get_convergence_angle(Xi, Yi))
    u_geo = u_grid * np.cos(gamma) - v_grid * np.sin(gamma)
    v_geo = u_grid * np.sin(gamma) + v_grid * np.cos(gamma)

    s = stride
    q = ax.quiver(lons[::s, ::s], lats[::s, ::s], u_geo[::s, ::s], v_geo[::s, ::s], 
                  transform=ccrs.PlateCarree(), pivot='middle', color='black', alpha=0.6)
    
    _draw_domain_and_sponge(ax, lons, lats, sponge_depth)
    ax.set_title(r"Adjoint Sensitivity overlaid with Initial Wind Field at $t=0$", fontsize=14)
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=r'Absolute Impact on Wave Energy per $+1K$ Perturbation')
    
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    else: plt.show()
    plt.close()
