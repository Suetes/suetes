"""
Diagnostics and Visualization Module.

Provides a comprehensive suite for analyzing and mapping the dynamical core's 
prognostic variables. Uses Cartopy for geographic map projections, SciPy for 
spectral analysis, and Matplotlib for multi-panel dashboards.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.signal as signal

class Visualizer:
    """
    Handles the post-processing and plotting of 3D fluid states.
    """
    def _get_plot_data(self, grid, state, variable, z_level, constants=None):
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

    def _draw_domain_and_sponge(self, ax, lons, lats, sponge_depth):
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


    def _get_level_height(self, grid, z_idx):
        """Returns the domain-averaged physical height of a logical model level."""
        return int(np.mean(grid.Z_m[:, :, z_idx]))


    def plot_2d_field(self, grid, state, variable, z_idx=5, sponge_depth=30, constants=None, 
                      cmap='viridis', vmin=None, vmax=None, scale='linear', title=None, ax=None, save_path=None, plot_data=None):
        """
        Plots a horizontal cross-section of a 3D field on a geographic map projection.
        Includes an automatic antimeridian wrap fix for domains crossing the dateline.
        """
        data = plot_data if plot_data is not None else self._get_plot_data(grid, state, variable, z_idx, constants)
        
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        
        # Detect if the domain crosses the antimeridian (creating a massive min/max gap)
        if lons.max() - lons.min() > 180.0:
            # Convert negative longitudes back to positive (e.g., -179 -> 181) 
            # to make the array continuous for the extent calculation and pcolormesh.
            lons = np.where(lons < 0, lons + 360.0, lons)
        
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

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
        self._draw_domain_and_sponge(ax, lons, lats, sponge_depth)
        
        ax.set_title(title or f"{variable.upper()} at Level {z_idx}", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=show_plot, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        if show_plot:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, extend=extend)
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=200)
            else: plt.show()
            plt.close()
            
        return im
    
    def plot_quiver_field(self, grid, state, bg_var='pi', u_var='u', v_var='v', z_idx=5, 
                          sponge_depth=30, constants=None, cmap='coolwarm', 
                          title=None, ax=None, save_path=None, stride=15):
        """Plots a contoured background field with a geographic wind quiver overlay."""
        
        bg_data = self._get_plot_data(grid, state, bg_var, z_idx, constants)
        u_grid = self._get_plot_data(grid, state, u_var, z_idx, constants)
        v_grid = self._get_plot_data(grid, state, v_var, z_idx, constants)
        
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        
        # Dateline fix
        if lons.max() - lons.min() > 180.0:
            lons = np.where(lons < 0, lons + 360.0, lons)
            
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

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
        
        self._draw_domain_and_sponge(ax, lons, lats, sponge_depth)
        
        ax.set_title(title or f"{bg_var.upper()} and Wind Vectors at Level {z_idx}", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
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

    def plot_dashboard(self, grid, state_model, z_idx=5, sponge_depth=30, fields=None, time_hours=None, save_path=None):
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
                im = self.plot_quiver_field(
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
                im = self.plot_2d_field(
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
        
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def plot_comparison(self, grid, state_model, state_era5, variable, z_idx=5, sponge_depth=30, save_path=None):
        """Plots the Model field, the target ERA5 field, and their discrete difference (Anomaly)."""
        val_model = self._get_plot_data(grid, state_model, variable, z_idx)
        val_era5 = self._get_plot_data(grid, state_era5, variable, z_idx)
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

            im = self.plot_2d_field(grid, state_model, variable, z_idx, sponge_depth, 
                                    cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax, plot_data=data)
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def plot_cross_section(self, grid, state, variable, y_idx, sponge_depth=30, overlay_isentropes=False, save_path=None):
        """
        Generates a vertical cross-section plot along the X-axis.
        Displays the physical terrain elevation and boundary layer sponge shading.
        """
        x_coords = grid.x_m / 1000.0  
        Z_slice = grid.Z_w[:, y_idx, :] if variable == 'w' else grid.Z_m[:, y_idx, :]
        data_slice = state[variable][:, y_idx, :]
        
        # Pad mass-point variables down to the surface (Z_w[0]) for plotting
        if variable != 'w':
            surface_Z = grid.Z_w[:, y_idx, 0:1] # Get exact topography height
            surface_data = data_slice[:, 0:1]   # Duplicate the lowest model level data
            
            # Prepend the surface values to the arrays
            Z_slice = np.concatenate([surface_Z, Z_slice], axis=1)
            data_slice = np.concatenate([surface_data, data_slice], axis=1)
        
        if sponge_depth > 0:
            interior_slice = data_slice[sponge_depth:-sponge_depth, :]
        else:
            interior_slice = data_slice
            
        cmap = 'seismic' if variable == 'w' else ('Blues' if variable == 'q_c' else 'RdBu_r')
        
        # Calculate limits strictly based on the interior domain
        if variable == 'w':
            vmax = float(np.max(np.abs(interior_slice)))
            vmin = -vmax
        else:
            vmax = float(np.max(interior_slice))
            vmin = float(np.min(interior_slice))
            if variable == 'q_c': vmin = 0.0
        
        if vmax <= vmin:
            vmax = vmin + 1e-5

        fig, ax = plt.subplots(figsize=(12, 6))
        
        X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
        levels = np.linspace(vmin, vmax, 31)
        
        contour = ax.contourf(X_2d, Z_slice, data_slice, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        plt.colorbar(contour, ax=ax, label=variable)
        
        if overlay_isentropes and 'th_v' in state:
            th_v_slice = state['th_v'][:, y_idx, :]
            levels = np.arange(np.floor(np.min(th_v_slice)), np.ceil(np.max(th_v_slice)), 2.0)
            ax.contour(np.broadcast_to(x_coords[:, None], grid.Z_m[:, y_idx, :].shape), 
                       grid.Z_m[:, y_idx, :], th_v_slice, levels=levels, colors='black', linewidths=1.0)
            
        ax.fill_between(x_coords, 0, grid.Z_w[:, y_idx, 0], color='dimgray', label='Topography')
        
        if sponge_depth > 0:
            x_left = x_coords[sponge_depth]
            x_right = x_coords[-sponge_depth-1]
            
            ax.axvline(x=x_left, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(x=x_right, color='k', linestyle='--', linewidth=1.5, label='Sponge')
            
            # Use the dark shadow approach so it doesn't wash out the white center of the seismic cmap
            ax.axvspan(x_coords[0], x_left, color='black', alpha=0.15, zorder=4)
            ax.axvspan(x_right, x_coords[-1], color='black', alpha=0.15, zorder=4)

        ax.set_title(f"Cross Section: {variable} (y-index: {y_idx})", fontsize=14)
        ax.set_xlabel("X distance [km]")
        ax.set_ylabel("Geometric height [m]")
        ax.set_ylim(0, 15000)
        ax.legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()

    def plot_level_strip(self, grid, state, variable, z_indices=[0, 5, 15, 30], sponge_depth=30, constants=None, cmap=None, scale='linear', save_path=None):
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
            data = self._get_plot_data(grid, state, variable, z, constants)
            z_height = self._get_level_height(grid, z)
            
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
                
                # Removed the hardcoded 1.0 and 1e-4 limits!
                vmin, vmax = -limit, limit
                extend = 'both'
            elif variable == 'q_c': 
                vmin = 0.0

            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':', edgecolor='gray')
            
            im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
            self._draw_domain_and_sponge(ax, lons, lats, sponge_depth)
            
            ax.set_title(f"Level {z} (~{z_height} m)", fontsize=14)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Add an individual colorbar to each subplot
            fig.colorbar(im, ax=ax, orientation='vertical', extend=extend, pad=0.02, fraction=0.046)
            
        plt.suptitle(f"Vertical Profile: {variable.upper()}", fontsize=18, y=0.98)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def plot_slice_locator_dashboard(self, grid, state, map_var='th_v', slice_var='w', map_z=5, sponge_depth=30, constants=None, save_path=None):
        """Plots two locator maps and two sets of 3 vertical cross-sections."""
        import matplotlib.gridspec as gridspec
        
        # Select 3 evenly spaced indices for X and Y slices
        y_indices = [grid.ny // 4, grid.ny // 2, 3 * grid.ny // 4]
        x_indices = [grid.nx // 4, grid.nx // 2, 3 * grid.nx // 4]
        
        x_colors = ['#FF595E', '#FFCA3A', '#8AC926'] # Distinct colors for X slices (Horizontal cuts)
        y_colors = ['#1982C4', '#6A4C93', '#F15BB5'] # Distinct colors for Y slices (Vertical cuts)
        
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5], hspace=0.25, wspace=0.15)
        
        map_data = self._get_plot_data(grid, state, map_var, map_z, constants)
        z_height = self._get_level_height(grid, map_z)
        
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        if lons.max() - lons.min() > 180.0: lons = np.where(lons < 0, lons + 360.0, lons)
        
        def draw_locator_map(ax, lines_indices, colors, is_x_slice):
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            im_map = ax.pcolormesh(lons, lats, map_data, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            self._draw_domain_and_sponge(ax, lons, lats, sponge_depth)
            
            for idx, color in zip(lines_indices, colors):
                if is_x_slice: # Drawing a line across constant Y
                    ax.plot(lons[:, idx], lats[:, idx], color=color, linewidth=2.5, transform=ccrs.PlateCarree(), label=f'Y-idx: {idx}')
                else: # Drawing a line across constant X
                    ax.plot(lons[idx, :], lats[idx, :], color=color, linewidth=2.5, transform=ccrs.PlateCarree(), label=f'X-idx: {idx}')
            
            ax.legend(loc='upper right')
            ax.set_title(f"Locator Map: {map_var} at Level {map_z} (~{z_height}m)", fontsize=14)
            return im_map

        # --- Top Left: Map with X slices (constant y cuts) ---
        ax_map_x = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=grid.lon_c))
        im_x = draw_locator_map(ax_map_x, y_indices, x_colors, is_x_slice=True)
        plt.colorbar(im_x, ax=ax_map_x, orientation='vertical', pad=0.02, fraction=0.03)

        # --- Top Right: Map with Y slices (constant x cuts) ---
        ax_map_y = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=grid.lon_c))
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
            ax.set_ylim(0, 15000)
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


    def plot_hovmoller(self, grid, time_hours_array, hov_data_2d, variable='th_v Anomaly', cmap='RdBu_r', save_path=None):
        """
        Plots a Hovmöller diagram (Time vs Longitude).
        
        Args:
            time_hours_array: 1D array of time in hours (e.g., [0, 1, 2, ... 24]).
            hov_data_2d: 2D array of shape (len(time_hours), nx) containing the 
                         y-averaged, z-averaged parameter to plot.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine physical X limits (convert to km for readability)
        x_coords = grid.x_m / 1000.0
        X, Y = np.meshgrid(x_coords, time_hours_array)
        
        # Calculate limits strictly based on the interior
        vmax = float(np.max(np.abs(hov_data_2d)))
        vmin = -vmax
        if vmax <= vmin: vmax = vmin + 1e-5
        
        levels = np.linspace(vmin, vmax, 41)
        
        im = ax.contourf(X, Y, hov_data_2d, levels=levels, cmap=cmap, extend='both')
        plt.colorbar(im, ax=ax, pad=0.02, label=variable)
        
        ax.set_title(f"Hovmöller Diagram: {variable}", fontsize=14)
        ax.set_ylabel("Simulation Time [Hours]")
        ax.set_xlabel("X Distance [km]")
        
        # Invert Y axis so time goes from top to bottom (standard meteorological convention)
        ax.invert_yaxis()
        
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def plot_energy_spectrum(self, grid, state_or_states, variable='w', z_idx=5, sponge_depth=30, save_path=None):
        r"""
        Computes and plots the 1D spatial power spectrum using FFT.

        A fundamental check for the physical accuracy of the dynamical core's 
        dissipation schemes is comparing the resolved kinetic energy spectrum 
        against the theoretical Kolmogorov isotropic turbulence cascade:

        $$ E(k) \propto k^{-5/3} $$

        Accepts either a single state dict (one snapshot) or a list of state
        dicts (time-averages the spectra across snapshots). Time averaging
        reduces the per-wavenumber variance of the spectral estimate.
        """
        # Accept a single state dict for backward compatibility, or a list/tuple
        # of state dicts for time-averaging.
        if isinstance(state_or_states, dict):
            states = [state_or_states]
        else:
            states = list(state_or_states)

        dx = grid.dx
        spectra = []
        for state in states:
            data_2d = state[variable][:, :, z_idx]
            if sponge_depth > 0:
                data_inner = data_2d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
            else:
                data_inner = data_2d

            for row in data_inner.T:
                row_windowed = signal.detrend(row) * np.hanning(len(row))
                spectra.append(np.abs(np.fft.rfft(row_windowed))**2)

        avg_power = np.mean(spectra, axis=0)[1:]
        k = np.fft.rfftfreq(data_inner.shape[0], d=dx)[1:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(k, avg_power, 'b-', label=f"Spectrum ({variable})", linewidth=2)
        
        # Reference slopes are placed in their physically expected regimes:
        # k^-3 at synoptic/large scales, k^-5/3 at mesoscale. The transition
        # is set at the empirical Nastrom-Gage wavelength of ~400 km. For a
        # small regional domain this puts the split at very low k (few
        # synoptic modes resolved), which is the honest physical picture.
        k_transition = 1.0 / 400e3
        split = int(np.clip(np.searchsorted(k, k_transition), 1, len(k) - 1))

        ref_k_low = k[: split + 1]
        i_low = max(1, split // 2)
        ax.loglog(ref_k_low,
                  avg_power[i_low] * (ref_k_low / k[i_low]) ** (-3),
                  'k:', label="$k^{-3}$ slope")

        ref_k_high = k[split:]
        i_high = split + (len(k) - split) // 2
        ax.loglog(ref_k_high,
                  avg_power[i_high] * (ref_k_high / k[i_high]) ** (-5/3),
                  'k--', label="$k^{-5/3}$ slope")
        
        ax.axvline(x=1.0/(2*dx), color='r', linestyle=':', label=rf'$2\Delta x$ ({2*dx/1000:.1f} km)')
        ax.axvline(x=1.0/(6*dx), color='orange', linestyle=':', label=rf'$6\Delta x$ ({6*dx/1000:.1f} km)')

        title_suffix = f" (averaged over {len(states)} snapshots)" if len(states) > 1 else ""
        ax.set_title(f"Power spectrum: {variable} at level {z_idx}{title_suffix}")
        ax.set_xlabel("Wavenumber $k$ [m⁻¹]")
        ax.set_ylabel("Spectral power density")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        
        # Safe inverse function to avoid divide-by-zero RuntimeWarnings
        def safe_inv(x):
            x_arr = np.array(x, dtype=float)
            return np.divide(1.0, x_arr, out=np.full_like(x_arr, np.inf), where=(x_arr != 0))
            
        secax = ax.secondary_xaxis('top', functions=(safe_inv, safe_inv))
        secax.set_xlabel('Wavelength [m]')
        secax.set_xticks([1e5, 5e4, 2e4, 1.2e4])
        secax.set_xticklabels(['100km', '50km', '20km', '12km'])
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()

    def plot_adjoint_overlay(self, grid, initial_state, sensitivity_2d, u_var='u', v_var='v', z_idx=5, sponge_depth=30, stride=15, save_path=None):
        """Overlays the forward initial wind field on top of the adjoint sensitivities."""
        u_grid = self._get_plot_data(grid, initial_state, u_var, z_idx)
        v_grid = self._get_plot_data(grid, initial_state, v_var, z_idx)
        
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
        
        self._draw_domain_and_sponge(ax, lons, lats, sponge_depth)
        ax.set_title(r"Adjoint Sensitivity overlaid with Initial Wind Field at $t=0$", fontsize=14)
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=r'Absolute Impact on Wave Energy per $+1K$ Perturbation')
        
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()
        
    def plot_point_timeseries(self, times_hours, suetes_values, era5_values,
                              location_name, units='K', save_path=None):
        """
        Plot a single-cell timeseries comparing Suetes against the ERA5 driver.

        Args:
            times_hours: 1D iterable of times in hours.
            suetes_values: 1D iterable of Suetes values at each time.
            era5_values: 1D iterable of ERA5 values at each time.
            location_name (str): label for the title.
            units (str): units string for the y-axis.
            save_path (str): if set, saves the figure to this path.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times_hours, suetes_values, 'b-',  linewidth=2, label='Suetes')
        ax.plot(times_hours, era5_values,   'r--', linewidth=2, label='ERA5')
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel(f'Temperature [{units}]')
        ax.set_title(f'Near-surface temperature over {location_name}')
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()
