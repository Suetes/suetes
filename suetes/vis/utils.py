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

def _draw_domain_and_sponge(ax, grid, sponge_depth):
    """Helper to draw domain boundaries and Davies sponge extents using native coordinates."""
    # Since the axis projection is ccrs.Stereographic matching grid.proj, we can plot directly in grid coordinates (meters)
    x0, x1 = float(grid.x_c.min()), float(grid.x_c.max())
    y0, y1 = float(grid.y_c.min()), float(grid.y_c.max())
    
    proj, _ = _get_projection_and_extent(grid)
    
    # Outer domain boundaries
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k-', linewidth=1.5, transform=proj)
    
    if sponge_depth > 0:
        # Sponge boundaries are located at sponge_depth * dx/dy from the edges
        xs0 = x0 + sponge_depth * grid.dx
        xs1 = x1 - sponge_depth * grid.dx
        ys0 = y0 + sponge_depth * grid.dy
        ys1 = y1 - sponge_depth * grid.dy
        
        ax.plot([xs0, xs1, xs1, xs0, xs0], [ys0, ys0, ys1, ys1, ys0], 'k--', linewidth=1.0, alpha=0.7, transform=proj)
        
        # Grid-perfect dimming mask using Rectangle patches
        import matplotlib.patches as patches
        rects = [
            patches.Rectangle((x0, y0), xs0 - x0, y1 - y0, facecolor='white', alpha=0.5, zorder=4, transform=proj),
            patches.Rectangle((xs1, y0), x1 - xs1, y1 - y0, facecolor='white', alpha=0.5, zorder=4, transform=proj),
            patches.Rectangle((xs0, y0), xs1 - xs0, ys0 - y0, facecolor='white', alpha=0.5, zorder=4, transform=proj),
            patches.Rectangle((xs0, ys1), xs1 - xs0, y1 - ys1, facecolor='white', alpha=0.5, zorder=4, transform=proj)
        ]
        for r in rects:
            ax.add_patch(r)

def _get_level_height(grid, z_idx):
    """Returns the domain-averaged physical height of a logical model level."""
    return int(np.mean(grid.Z_m[:, :, z_idx]))

def _get_projection_and_extent(grid):
    """Returns the Cartopy projection object and native extent for a given grid."""
    proj = ccrs.Stereographic(
        central_latitude=float(grid.lat_c),
        central_longitude=float(grid.lon_c),
        globe=ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0)
    )
    extent = [float(grid.x_c.min()), float(grid.x_c.max()),
              float(grid.y_c.min()), float(grid.y_c.max())]
    return proj, extent

