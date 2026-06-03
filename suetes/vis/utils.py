import numpy as np
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

def _get_plot_data(grid, state, variable, z_level, constants=None):
    """Helper to extract data, compute 3D derived fields, and slice/interpolate."""
    is_height = isinstance(z_level, float)
    
    # Compute the full 3D field on the mass points
    if variable == 'div':
        u, v = state['u'], state['v']
        data_3d = (u[1:, :, :] - u[:-1, :, :]) / grid.dx + (v[:, 1:, :] - v[:, :-1, :]) / grid.dy
        z_coords = grid.Z_m
    elif variable == 'rh' and constants is not None:
        qv, pi, th_v = state['q'], state['pi'], state['th_v']
        epsilon = constants.get('epsilon', 0.622)
        T = (th_v * pi) / (1.0 + (1.0 / epsilon - 1.0) * qv)
        p = constants['p0'] * (pi ** (constants['cp'] / constants['Rd']))
        e_s = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (epsilon * e_s) / (p - (1.0 - epsilon) * e_s)
        data_3d = np.clip((qv / q_s) * 100.0, 0, 100)
        z_coords = grid.Z_m
    elif variable == 'u': 
        data_3d = 0.5 * (state['u'][:-1, :, :] + state['u'][1:, :, :])
        z_coords = grid.Z_m
    elif variable == 'v': 
        data_3d = 0.5 * (state['v'][:, :-1, :] + state['v'][:, 1:, :])
        z_coords = grid.Z_m
    else:
        data_3d = state[variable]
        z_coords = grid.Z_w if variable == 'w' else grid.Z_m
        
    # Extract the 2D plane
    if is_height:
        # Interpolate to constant geometric height (returns np.nan below ground)
        data_2d = grid.interp_to_height(data_3d, z_coords, float(z_level))
        return np.array(data_2d) # Convert to numpy for matplotlib NaN handling
    else:
        # Extract logical grid index
        return data_3d[:, :, int(z_level)]

def _draw_domain_and_sponge(ax, lons, lats, sponge_depth):
    """Helper to draw domain boundaries and Davies sponge extents."""
    # Domain Boundaries
    ax.plot(lons[0, :], lats[0, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
    ax.plot(lons[-1, :], lats[-1, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
    ax.plot(lons[:, 0], lats[:, 0], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
    ax.plot(lons[:, -1], lats[:, -1], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)

    # Sponge Boundaries
    if sponge_depth > 0:
        sd = sponge_depth
        kwargs = {'color': 'k', 'linestyle': '--', 'transform': ccrs.PlateCarree(), 'linewidth': 1.0, 'alpha': 0.7}
        ax.plot(lons[sd, sd:-sd], lats[sd, sd:-sd], **kwargs)
        ax.plot(lons[-sd-1, sd:-sd], lats[-sd-1, sd:-sd], **kwargs)
        ax.plot(lons[sd:-sd, sd], lats[sd:-sd, sd], **kwargs)
        ax.plot(lons[sd:-sd, -sd-1], lats[sd:-sd, -sd-1], **kwargs)

        # Grid-perfect dimming mask
        ny, nx = lons.shape
        mask = np.ones((ny, nx))
        mask[sd:-sd, sd:-sd] = np.nan # Keep interior transparent
        
        cmap_white = mcolors.ListedColormap(['white'])
        # zorder=4 ensures it draws over the data but under the coastlines/borders
        ax.pcolormesh(lons, lats, mask, transform=ccrs.PlateCarree(), 
                      cmap=cmap_white, alpha=0.5, zorder=4)

def _get_level_height(grid, z_idx):
    """Returns the domain-averaged physical height of a logical model level."""
    return int(np.mean(grid.Z_m[:, :, z_idx]))
