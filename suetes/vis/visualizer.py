import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.signal as signal

class Visualizer:
    
    def _get_plot_data(self, grid, state, variable, z_idx, constants=None):
        """Helper to extract data, compute derived fields, and interpolate to mass points."""
        if variable == 'div':
            u = state['u'][:, :, z_idx]
            v = state['v'][:, :, z_idx]
            data = (u[1:, :] - u[:-1, :]) / grid.dx + (v[:, 1:] - v[:, :-1]) / grid.dy
        elif variable == 'rh' and constants is not None:
            qv, pi, th_v = state['q'][:, :, z_idx], state['pi'][:, :, z_idx], state['th_v'][:, :, z_idx]
            epsilon = constants.get('epsilon', 0.622)
            T = (th_v * pi) / (1.0 + (1.0 / epsilon - 1.0) * qv)
            p = constants['p0'] * (pi ** (constants['cp'] / constants['Rd']))
            e_s = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
            q_s = (epsilon * e_s) / (p - (1.0 - epsilon) * e_s)
            data = np.clip((qv / q_s) * 100.0, 0, 100)
        elif variable == 'u': # Interpolate U to mass points
            data = 0.5 * (state['u'][:-1, :, z_idx] + state['u'][1:, :, z_idx])
        elif variable == 'v': # Interpolate V to mass points
            data = 0.5 * (state['v'][:, :-1, z_idx] + state['v'][:, 1:, z_idx])
        else:
            data = state[variable][:, :, z_idx]
            
        return data

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

    def plot_2d_field(self, grid, state, variable, z_idx=5, sponge_depth=30, constants=None, 
                      cmap='viridis', vmin=None, vmax=None, scale='linear', title=None, ax=None, save_path=None, plot_data=None):
    
        # Use the explicitly passed data if available, otherwise extract it from the state
        data = plot_data if plot_data is not None else self._get_plot_data(grid, state, variable, z_idx, constants)
        
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        
        # --- THE DATELINE FIX ---
        # Detect if the domain crosses the antimeridian (creating a massive min/max gap)
        if lons.max() - lons.min() > 180.0:
            # Convert negative longitudes back to positive (e.g., -179 -> 181) 
            # to make the array continuous for the extent calculation and pcolormesh.
            lons = np.where(lons < 0, lons + 360.0, lons)
        # ------------------------
        
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
            show_plot = True

        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')

        extend = 'neither'
        if vmin is None and vmax is None:
            if sponge_depth > 0:
                inner_data = data[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
            else:
                inner_data = data
                
            data_max = float(np.max(inner_data))
            data_min = float(np.min(inner_data))
            
            if scale == 'sym': 
                limit = max(abs(data_max), abs(data_min))
                
                # Safety caps to prevent extreme outliers from washing out the plot
                capped_limit = limit
                if variable == 'div': capped_limit = min(limit, 1e-4)
                if variable == 'w': capped_limit = min(limit, 1.0)
                
                # Tell matplotlib to draw arrows on the colorbar if we capped the visual range
                if capped_limit < limit:
                    extend = 'both'
                    
                vmax = capped_limit
                vmin = -capped_limit
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

    def plot_dashboard(self, grid, state_model, z_idx=5, sponge_depth=30, fields=None, time_hours=None, save_path=None):
        if fields is None:
            fields = [
                {'var': 'w', 'cmap': 'seismic', 'title': 'Vertical Velocity [m/s]', 'scale': 'sym'},
                {'var': 'div', 'cmap': 'seismic', 'title': 'Horizontal Divergence [s⁻¹]', 'scale': 'sym'},
                {'var': 'q_c', 'cmap': 'Blues', 'title': 'Cloud Water [kg/kg]', 'scale': 'linear'},
                {'var': 'u', 'cmap': 'seismic', 'title': 'Zonal Wind [m/s]', 'scale': 'sym'} 
            ]

        n_vars = len(fields)
        cols = 2
        rows = int(np.ceil(n_vars / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
        axes = np.atleast_1d(axes).flatten()
        
        for i, field_def in enumerate(fields):
            scale = field_def.get('scale', 'linear')
            im = self.plot_2d_field(grid, state_model, field_def['var'], z_idx, sponge_depth, 
                                    cmap=field_def.get('cmap', 'viridis'), scale=scale, 
                                    title=field_def['title'], ax=axes[i])
                                    
            # Extract the extend parameter from the mappable so subplots show arrows correctly
            extend = im.cmap.name if hasattr(im, 'extend') else 'neither' 
            # (Note: pcolormesh doesn't strictly store extend, but colorbar handles out-of-bounds automatically if limits are set)
            fig.colorbar(im, ax=axes[i], orientation='horizontal', pad=0.05, fraction=0.046, extend='both' if scale == 'sym' else 'neither')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        title = f"Suetes results (Level {z_idx})" + (f" | T={time_hours}h" if time_hours else "")
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else: plt.show()
        plt.close()

    def plot_comparison(self, grid, state_model, state_era5, variable, z_idx=5, sponge_depth=30, save_path=None):
        """Plots Model, ERA5, and the Anomaly (Difference)."""
        val_model = self._get_plot_data(grid, state_model, variable, z_idx)
        val_era5 = self._get_plot_data(grid, state_era5, variable, z_idx)
        anomaly = val_model - val_era5

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=grid.lon_c)})
        titles = ["Suetes Model", "ERA5 Target", "Anomaly (Model - ERA5)"]
        datas = [val_model, val_era5, anomaly]
        
        vmax_main = max(float(np.max(val_model)), float(np.max(val_era5)))
        vmin_main = min(float(np.min(val_model)), float(np.min(val_era5)))
        vmax_anom = max(float(np.max(np.abs(anomaly))), 0.1)

        for ax, data, title in zip(axes, datas, titles):
            cmap = 'seismic' if 'Anomaly' in title else 'viridis'
            vmin, vmax = (-vmax_anom, vmax_anom) if 'Anomaly' in title else (vmin_main, vmax_main)

            # Pass the pre-calculated 'data' array into plot_data
            im = self.plot_2d_field(grid, state_model, variable, z_idx, sponge_depth, 
                                    cmap=cmap, vmin=vmin, vmax=vmax, title=title, ax=ax, plot_data=data)
            fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()

    def plot_cross_section(self, grid, state, variable, y_idx, sponge_depth=30, overlay_isentropes=False, save_path=None):
        """Generalized vertical cross-section plotting."""
        x_coords = grid.x_m / 1000.0  
        Z_slice = grid.Z_w[:, y_idx, :] if variable == 'w' else grid.Z_m[:, y_idx, :]
        data_slice = state[variable][:, y_idx, :]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cmap = 'seismic' if variable == 'w' else ('Blues' if variable == 'q_c' else 'viridis')
        vmax = float(np.max(np.abs(data_slice))) if variable == 'w' else float(np.max(data_slice))
        vmin = -vmax if variable == 'w' else float(np.min(data_slice))
        if variable == 'q_c': vmin = 0.0
        
        X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
        contour = ax.contourf(X_2d, Z_slice, data_slice, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
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
            
            # Dim the sponge zones
            ax.axvspan(x_coords[0], x_left, color='white', alpha=0.5, zorder=4)
            ax.axvspan(x_right, x_coords[-1], color='white', alpha=0.5, zorder=4)

        ax.set_title(f"Cross Section: {variable} (y-index: {y_idx})", fontsize=14)
        ax.set_xlabel("X distance [km]")
        ax.set_ylabel("Geometric height [m]")
        ax.set_ylim(0, 15000)
        ax.legend()
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()

    def plot_energy_spectrum(self, grid, state_model, variable='w', z_idx=5, sponge_depth=30, save_path=None):
        """Computes and plots the 1D spatial power spectrum."""
        data_2d = state_model[variable][:, :, z_idx]
        if sponge_depth > 0:
            data_inner = data_2d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
        else:
            data_inner = data_2d
            
        dx = grid.dx
        spectra = []
        for row in data_inner.T:
            row_windowed = signal.detrend(row) * np.hanning(len(row))
            spectra.append(np.abs(np.fft.rfft(row_windowed))**2)
            
        avg_power = np.mean(spectra, axis=0)[1:]
        k = np.fft.rfftfreq(data_inner.shape[0], d=dx)[1:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(k, avg_power, 'b-', label=f"Spectrum ({variable})", linewidth=2)
        
        ref_k = k[len(k)//10:] 
        ax.loglog(ref_k, avg_power[len(k)//10] * (ref_k / ref_k[0])**(-5/3), 'k--', label="$k^{-5/3}$ cascade")
        
        ax.axvline(x=1.0/(2*dx), color='r', linestyle=':', label=rf'$2\Delta x$ ({2*dx/1000:.1f} km)')
        ax.axvline(x=1.0/(6*dx), color='orange', linestyle=':', label=rf'$6\Delta x$ ({6*dx/1000:.1f} km)')
        
        ax.set_title(f"Power Spectrum: {variable} at Level {z_idx}")
        ax.set_xlabel("Wavenumber $k$ [m⁻¹]")
        ax.set_ylabel("Spectral Power Density")
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