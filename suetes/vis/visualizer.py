import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import scipy.signal as signal

class Visualizer:
    
    def plot_cross_section(self, state, lat_idx, save_path=None):
        """Plots a longitudinal cross-section of Temperature and Topography."""
        lons = state['longitude']
        
        T_slice = state['T'][:, lat_idx, :]
        Z_slice = state['geopotential'][:, lat_idx, :] / 9.81 
        
        surface_height = Z_slice[-1, :]
        
        # Sort columns vertically
        sort_idx = np.argsort(Z_slice, axis=0)
        Z_slice_sorted = np.take_along_axis(Z_slice, sort_idx, axis=0)
        T_slice_sorted = np.take_along_axis(T_slice, sort_idx, axis=0)

        lons_2d = np.broadcast_to(lons, Z_slice_sorted.shape)

        fig, ax = plt.subplots(figsize=(12, 6))

        contour = ax.contourf(lons_2d, Z_slice_sorted, T_slice_sorted, levels=20, cmap='RdYlBu_r')
        cbar = plt.colorbar(contour, ax=ax, label='Temperature [K]')

        ax.fill_between(lons, 0, surface_height, color='dimgray', label='Surface Topography')
        
        ax.set_title(f"ERA5 vertical cross-section (Latitude: {state['latitude'][lat_idx]:.2f}°)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Geometric height [m]")
        ax.set_ylim(0, 15000) 
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved cross-section to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_map(self, ds_pl, ds_sl, variable, time_idx=0, level=None, cmap='RdYlBu_r', save_path=None):
        """Plots a geographical 2D map of ERA5 data."""
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
        
        time_dim_sl = 'valid_time' if 'valid_time' in ds_sl.dims else 'time'
        time_dim_pl = 'valid_time' if 'valid_time' in ds_pl.dims else 'time'
        level_dim = 'pressure_level' if 'pressure_level' in ds_pl.dims else 'level'

        if level is None or level == 'surface':
            data = ds_sl[variable].isel({time_dim_sl: time_idx})
            title_str = "Surface"
        else:
            data = ds_pl[variable].isel({time_dim_pl: time_idx}).sel({level_dim: level})
            title_str = f"{level} hPa"

        contour = ax.contourf(data.longitude, data.latitude, data.values, 
                              levels=30, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
        
        units = data.attrs.get('units', 'unknown units')
        cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
        cbar.set_label(f"{variable} [{units}]", fontsize=12)
        
        ax.set_title(f"ERA5 {variable} - {title_str} (Timestep: {time_idx})", fontsize=14)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved ERA5 map to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_topography_comparison(self, ds_era5_sl, ds_gebco, time_idx=0, save_path=None):
        """Compares ERA5 topography with GEBCO topography side-by-side."""
        time_dim = 'valid_time' if 'valid_time' in ds_era5_sl.dims else 'time'
        era5_topo = ds_era5_sl['z'].isel({time_dim: time_idx}) / 9.81
        
        min_lon, max_lon = float(era5_topo.longitude.min()), float(era5_topo.longitude.max())
        min_lat, max_lat = float(era5_topo.latitude.min()), float(era5_topo.latitude.max())
        
        lat_slice = slice(min_lat, max_lat) if ds_gebco.lat[0] < ds_gebco.lat[-1] else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        
        gebco_topo = ds_gebco['elevation'].sel(lat=lat_slice, lon=lon_slice)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        vmax = max(float(era5_topo.max()), float(gebco_topo.max()))
        vmin = min(float(era5_topo.min()), float(gebco_topo.min()))
        
        # ERA5
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        im1 = ax1.pcolormesh(era5_topo.longitude, era5_topo.latitude, era5_topo.values, 
                             transform=ccrs.PlateCarree(), cmap='terrain', vmin=vmin, vmax=vmax)
        ax1.set_title("ERA5 topography (~31 km)", fontsize=14)
        ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # GEBCO
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        im2 = ax2.pcolormesh(gebco_topo.lon, gebco_topo.lat, gebco_topo.values, 
                             transform=ccrs.PlateCarree(), cmap='terrain', vmin=vmin, vmax=vmax)
        ax2.set_title("GEBCO topography (High resolution)", fontsize=14)
        ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Elevation [m]', fontsize=12)
        
        plt.subplots_adjust(bottom=0.15)
        if save_path:
            plt.savefig(save_path)
            print(f"Saved topography comparison to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_model_vs_era5_map(self, grid, state_model, state_era5, variable='th_v', z_idx=5, sponge_depth=30, save_path=None):
        """Plots a side-by-side horizontal comparison of the Model vs ERA5."""
        if variable == 'u': x_coords, y_coords = grid.x_c, grid.y_m
        elif variable == 'v': x_coords, y_coords = grid.x_m, grid.y_c
        else: x_coords, y_coords = grid.x_m, grid.y_m

        Xi, Yi = np.meshgrid(x_coords, y_coords, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        lons = (lons + 180.0) % 360.0 - 180.0

        val_model = state_model[variable][:, :, z_idx]
        val_era5 = state_era5[variable][:, :, z_idx]
        z_approx = grid.z_m[z_idx] if variable != 'w' else grid.z_c[z_idx]

        vmin = min(float(np.min(val_model)), float(np.min(val_era5)))
        vmax = max(float(np.max(val_model)), float(np.max(val_era5)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

        for ax, data, title in zip([ax1, ax2], [val_model, val_era5], ["Suetes Simulation (T=1h)", "ERA5 Target (T=1h)"]):
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
            
            im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), 
                               cmap='seismic', vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=14)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

            # --- Domain Boundaries (Solid Black) ---
            ax.plot(lons[0, :], lats[0, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[-1, :], lats[-1, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[:, 0], lats[:, 0], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[:, -1], lats[:, -1], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)

            # --- Sponge Boundaries (Dashed Black) ---
            if sponge_depth > 0:
                sd = sponge_depth
                ax.plot(lons[sd, sd:-sd], lats[sd, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
                ax.plot(lons[-sd-1, sd:-sd], lats[-sd-1, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
                ax.plot(lons[sd:-sd, sd], lats[sd:-sd, sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
                ax.plot(lons[sd:-sd, -sd-1], lats[sd:-sd, -sd-1], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)

        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f"Variable: {variable} | Approx height: {z_approx:.0f} m", fontsize=12)

        plt.subplots_adjust(bottom=0.20)
        
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved comparison to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_anomaly(self, grid, state_model, state_era5, variable='u', z_idx=15, sponge_depth=30, save_path=None):
        """Plots the explicit difference between the model and ERA5."""
        if variable == 'u': x_coords, y_coords = grid.x_c, grid.y_m
        elif variable == 'v': x_coords, y_coords = grid.x_m, grid.y_c
        else: x_coords, y_coords = grid.x_m, grid.y_m

        Xi, Yi = np.meshgrid(x_coords, y_coords, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        lons = (lons + 180.0) % 360.0 - 180.0

        val_model = state_model[variable][:, :, z_idx]
        val_era5 = state_era5[variable][:, :, z_idx]
        anomaly = val_model - val_era5

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
        
        vmax = max(float(np.max(np.abs(anomaly))), 0.1) 
        im = ax.pcolormesh(lons, lats, anomaly, transform=ccrs.PlateCarree(), cmap='seismic', vmin=-vmax, vmax=vmax)
        
        ax.set_title(f"Mesoscale anomaly ({variable}): Suetes - ERA5", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # --- Domain Boundaries (Solid Black) ---
        ax.plot(lons[0, :], lats[0, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
        ax.plot(lons[-1, :], lats[-1, :], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
        ax.plot(lons[:, 0], lats[:, 0], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)
        ax.plot(lons[:, -1], lats[:, -1], 'k-', transform=ccrs.PlateCarree(), linewidth=1.5)

        # --- Sponge Boundaries (Dashed Black) ---
        if sponge_depth > 0:
            sd = sponge_depth
            ax.plot(lons[sd, sd:-sd], lats[sd, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[-sd-1, sd:-sd], lats[-sd-1, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[sd:-sd, sd], lats[sd:-sd, sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[sd:-sd, -sd-1], lats[sd:-sd, -sd-1], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label(f"{variable} anomaly", fontsize=12)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved anomaly plot to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_divergence(self, grid, state_model, z_idx=5, sponge_depth=30, save_path=None):
        """Calculates and plots the horizontal divergence of the wind field."""
        # Calculate horizontal divergence (du/dx + dv/dy) on the mass points
        u = state_model['u'][:, :, z_idx]
        v = state_model['v'][:, :, z_idx]
        
        du_dx = (u[1:, :] - u[:-1, :]) / grid.dx
        dv_dy = (v[:, 1:] - v[:, :-1]) / grid.dy
        
        div = du_dx + dv_dy
        
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        lons = (lons + 180.0) % 360.0 - 180.0

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
        
        # Scale divergence for visibility (usually on the order of 1e-5 to 1e-4)
        vmax = 1e-4 
        im = ax.pcolormesh(lons, lats, div, transform=ccrs.PlateCarree(), cmap='seismic', vmin=-vmax, vmax=vmax)
        
        ax.set_title(f"Horizontal Divergence at Level {z_idx}", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # --- Sponge Boundaries (Dashed Black) ---
        if sponge_depth > 0:
            sd = sponge_depth
            ax.plot(lons[sd, sd:-sd], lats[sd, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[-sd-1, sd:-sd], lats[-sd-1, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[sd:-sd, sd], lats[sd:-sd, sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)
            ax.plot(lons[sd:-sd, -sd-1], lats[sd:-sd, -sd-1], 'k--', transform=ccrs.PlateCarree(), linewidth=1.5)

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label("Divergence [s⁻¹]", fontsize=12)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved divergence plot to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_suetes_w_cross_section(self, grid, state_model, y_idx, sponge_depth=30, save_path=None):
        """Plots a vertical cross-section of Vertical Velocity (W) through the Suetes grid."""
        x_coords = grid.x_m / 1000.0  # Convert to km
        
        Z_slice = grid.Z_w[:, y_idx, :]
        W_slice = state_model['w'][:, y_idx, :]
        terrain_z = grid.Z_w[:, y_idx, 0]
        
        X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        vmax = max(float(np.max(W_slice)), float(np.abs(np.min(W_slice))))
        vmax = min(max(vmax, 0.1), 3.0) 
        
        contour = ax.contourf(X_2d, Z_slice, W_slice, levels=30, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(contour, ax=ax, label='Vertical velocity (w) [m/s]')
        
        ax.fill_between(x_coords, 0, terrain_z, color='dimgray', label='Suetes Topography')
        
        # --- Sponge Boundaries (Dashed Black Vertical Lines) ---
        if sponge_depth > 0:
            # We use x_coords to find the correct physical distance of the sponge edges
            ax.axvline(x=x_coords[sponge_depth], color='k', linestyle='--', linewidth=1.5, label='Sponge Boundary')
            ax.axvline(x=x_coords[-sponge_depth-1], color='k', linestyle='--', linewidth=1.5)

        ax.set_title(f"Suetes vertical velocity (y-index: {y_idx})")
        ax.set_xlabel("X distance from domain center [km]")
        ax.set_ylabel("Geometric height [m]")
        ax.set_ylim(0, 15000)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved cross-section to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_w_and_isentropes(self, grid, state_model, y_idx, sponge_depth=30, save_path=None):
        """Plots W with overlaid isentropes (Virtual Potential Temperature) to diagnose wave breaking."""
        x_coords = grid.x_m / 1000.0  # Convert to km
        
        # W is on w-levels, th_v is on mass-levels
        Z_w_slice = grid.Z_w[:, y_idx, :]
        Z_m_slice = grid.Z_m[:, y_idx, :]
        W_slice = state_model['w'][:, y_idx, :]
        th_v_slice = state_model['th_v'][:, y_idx, :]
        terrain_z = grid.Z_w[:, y_idx, 0]
        
        X_w_2d = np.broadcast_to(x_coords[:, None], Z_w_slice.shape)
        X_m_2d = np.broadcast_to(x_coords[:, None], Z_m_slice.shape)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 1. Plot W as the background color
        vmax = max(float(np.max(W_slice)), float(np.abs(np.min(W_slice))))
        vmax = min(max(vmax, 0.1), 3.0) 
        contour_w = ax.contourf(X_w_2d, Z_w_slice, W_slice, levels=30, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(contour_w, ax=ax, label='Vertical velocity (w) [m/s]', pad=0.02)
        
        # 2. Plot Isentropes (Potential Temperature) as black contour lines
        # Determine a good contour interval based on the data
        th_min, th_max = np.min(th_v_slice), np.max(th_v_slice)
        levels = np.arange(np.floor(th_min), np.ceil(th_max), 2.0) # Every 2 Kelvin
        
        contour_th = ax.contour(X_m_2d, Z_m_slice, th_v_slice, levels=levels, 
                                colors='black', linewidths=1.0, alpha=0.8)
        
        ax.fill_between(x_coords, 0, terrain_z, color='dimgray', label='Suetes Topography')
        
        # --- Sponge Boundaries (Dashed Black Vertical Lines) ---
        if sponge_depth > 0:
            ax.axvline(x=x_coords[sponge_depth], color='k', linestyle='--', linewidth=2.0)
            ax.axvline(x=x_coords[-sponge_depth-1], color='k', linestyle='--', linewidth=2.0, label='Sponge Boundary')

        ax.set_title(f"Wave breaking diagnostics: W and Isentropes (y-index: {y_idx})", fontsize=14)
        ax.set_xlabel("X distance from domain center [km]", fontsize=12)
        ax.set_ylabel("Geometric height [m]", fontsize=12)
        ax.set_ylim(0, 15000)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved isentrope cross-section to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_energy_spectrum(self, grid, state_model, variable='w', z_idx=5, sponge_depth=30, save_path=None):
        """
        Computes and plots the 1D spatial power spectrum along the X-axis.
        Averages the spectra across all valid Y-rows to create a robust signal.
        """
        
        # Extract the 2D horizontal slice
        data_2d = state_model[variable][:, :, z_idx]
        
        # Crop out the sponge zone. The sponge heavily damps waves; including it will skew the spectrum.
        if sponge_depth > 0:
            data_inner = data_2d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
        else:
            data_inner = data_2d
            
        nx_inner = data_inner.shape[0]
        dx = grid.dx
        
        # Transpose so we iterate over y-rows. Each row is an x-slice.
        rows_to_compute = data_inner.T
        spectra = []
        
        for row in rows_to_compute:
            # Detrend the data to remove large-scale gradients
            row_detrended = signal.detrend(row)
            
            # Apply a Hanning window to enforce periodicity and prevent spectral leakage
            window = np.hanning(len(row_detrended))
            row_windowed = row_detrended * window
            
            # Compute FFT and power density
            fft_vals = np.fft.rfft(row_windowed)
            power = np.abs(fft_vals)**2
            spectra.append(power)
            
        # Average the power spectra across all y-rows for a smooth curve
        avg_power = np.mean(spectra, axis=0)
        
        # Compute corresponding wavenumbers
        k = np.fft.rfftfreq(nx_inner, d=dx)
        
        # Ignore the zero frequency (mean) for log-log plotting
        k = k[1:]
        avg_power = avg_power[1:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the actual model spectrum
        ax.loglog(k, avg_power, 'b-', label=f"Suetes '{variable}' Spectrum", linewidth=2)
        
        # Add theoretical reference line
        ref_k = k[len(k)//10:] # Start the reference line a bit away from the largest scales
        ref_power_53 = avg_power[len(k)//10] * (ref_k / ref_k[0])**(-5/3)
        ax.loglog(ref_k, ref_power_53, 'k--', label="$k^{-5/3}$ physical cascade", alpha=0.7)
        
        # Mark resolution limits
        k_nyquist = 1.0 / (2 * dx)
        k_effective = 1.0 / (6 * dx)
        
        ax.axvline(x=k_nyquist, color='r', linestyle=':', label=rf'$2\Delta x$ Nyquist ({2*dx/1000:.1f} km)')
        ax.axvline(x=k_effective, color='orange', linestyle=':', label=rf'$6\Delta x$ Effective ({6*dx/1000:.1f} km)')
        
        ax.set_title(f"Spatial Power Spectrum of '{variable}' at Level {z_idx}", fontsize=14)
        ax.set_xlabel("Wavenumber $k$ [m⁻¹]", fontsize=12)
        ax.set_ylabel("Spectral Power Density", fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=11)
        
        # Add secondary axis on top for Wavelengths to make it intuitive
        def k_to_lambda(x): 
            return np.divide(1.0, x, out=np.full_like(np.array(x, dtype=float), np.inf), where=(x!=0))
        def lambda_to_k(x): 
            return np.divide(1.0, x, out=np.full_like(np.array(x, dtype=float), np.inf), where=(x!=0))
        secax = ax.secondary_xaxis('top', functions=(k_to_lambda, lambda_to_k))
        secax.set_xlabel('Wavelength [m]', fontsize=12)
        secax.set_xticks([1e5, 5e4, 2e4, 1.2e4])
        secax.set_xticklabels(['100km', '50km', '20km', '12km'])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
            print(f"Saved energy spectrum to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_qc_cross_section(self, grid, state_model, y_idx, sponge_depth=30, save_path=None):
        """Plots a vertical cross-section of Cloud Liquid Water (q_c)."""
        x_coords = grid.x_m / 1000.0  # Convert to km
        
        Z_slice = grid.Z_m[:, y_idx, :]
        # Convert kg/kg to g/kg for readability
        qc_slice = state_model['q_c'][:, y_idx, :] * 1000.0 
        terrain_z = grid.Z_w[:, y_idx, 0]
        
        X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Only plot where qc > 0.01 g/kg (typical cloud threshold)
        vmax = max(float(np.max(qc_slice)), 0.05)
        levels = np.linspace(0.01, vmax, 20)
        
        contour = ax.contourf(X_2d, Z_slice, qc_slice, levels=levels, cmap='Blues', extend='max')
        plt.colorbar(contour, ax=ax, label='Cloud Water ($q_c$) [g/kg]')
        
        ax.fill_between(x_coords, 0, terrain_z, color='dimgray', label='Suetes Topography')
        
        # --- Sponge Boundaries ---
        if sponge_depth > 0:
            ax.axvline(x=x_coords[sponge_depth], color='k', linestyle='--', linewidth=1.5, label='Sponge Boundary')
            ax.axvline(x=x_coords[-sponge_depth-1], color='k', linestyle='--', linewidth=1.5)

        ax.set_title(f"Cloud Liquid Water (y-index: {y_idx})", fontsize=14)
        ax.set_xlabel("X distance from domain center [km]", fontsize=12)
        ax.set_ylabel("Geometric height [m]", fontsize=12)
        ax.set_ylim(0, 15000)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved qc cross-section to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_rh_map(self, grid, state_model, constants, z_idx=5, save_path=None):
        """Plots Relative Humidity to diagnose dry-air environments."""
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        lons = (lons + 180.0) % 360.0 - 180.0
        
        qv = np.array(state_model['q'][:, :, z_idx])
        pi = np.array(state_model['pi'][:, :, z_idx])
        th_v = np.array(state_model['th_v'][:, :, z_idx])
        
        # Reverse engineer T and p to find the saturation threshold
        epsilon = constants.get('epsilon', 0.622)
        Tv = th_v * pi
        T = Tv / (1.0 + (1.0 / epsilon - 1.0) * qv)
        p = constants['p0'] * (pi ** (constants['cp'] / constants['Rd']))
        
        # Calculate Saturation Specific Humidity
        e_s = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (epsilon * e_s) / (p - (1.0 - epsilon) * e_s)
        
        # Calculate RH (capped at 100% for plotting)
        rh = np.clip((qv / q_s) * 100.0, 0, 100)
        
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
        
        # Use a brown (dry) to green (wet) colormap
        im = ax.pcolormesh(lons, lats, rh, transform=ccrs.PlateCarree(), cmap='BrBG', vmin=0, vmax=100)
        
        ax.set_title(f"Relative Humidity [%] at Level {z_idx}", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label("Relative Humidity [%]", fontsize=12)

        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=200)
            print(f"Saved RH plot to {save_path}")
        else: 
            plt.show()
        plt.close()

    def plot_dashboard(self, grid, state_model, z_idx=5, sponge_depth=30, fields=None, time_hours=None, save_path=None):
        """
        Creates a dynamic multi-panel dashboard for a specific model level.
        
        Args:
            fields (list of dicts): E.g., [{'var': 'w', 'cmap': 'seismic', 'title': 'Vertical Vel'}]
        """
        if fields is None:
            # Default diagnostic panel for fields not strictly compared to ERA5
            fields = [
                {'var': 'w', 'cmap': 'seismic', 'title': 'Vertical Velocity [m/s]', 'scale': 'sym'},
                {'var': 'div', 'cmap': 'seismic', 'title': 'Horizontal Divergence [s⁻¹]', 'scale': 'sym'},
                {'var': 'th_v', 'cmap': 'plasma', 'title': 'Virtual Pot. Temp [K]', 'scale': 'linear'},
                {'var': 'u', 'cmap': 'seismic', 'title': 'Zonal Wind [m/s]', 'scale': 'linear'}
            ]

        n_vars = len(fields)
        cols = 2
        rows = int(np.ceil(n_vars / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        axes = np.atleast_1d(axes).flatten()
        
        # Common coordinates (project everything to mass points for the dashboard)
        Xi, Yi = np.meshgrid(grid.x_m, grid.y_m, indexing='ij')
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        lons = (lons + 180.0) % 360.0 - 180.0
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, float(lats.min()) - 0.5, float(lats.max()) + 0.5]

        # Helper to compute divergence on the fly
        def get_divergence(u, v):
            du_dx = (u[1:, :] - u[:-1, :]) / grid.dx
            dv_dy = (v[:, 1:] - v[:, :-1]) / grid.dy
            return du_dx + dv_dy

        for i, field_def in enumerate(fields):
            ax = axes[i]
            var_name = field_def['var']
            
            # Extract and interpolate data to mass points if necessary
            if var_name == 'div':
                data = get_divergence(state_model['u'][:, :, z_idx], state_model['v'][:, :, z_idx])
            elif var_name == 'u':
                data = 0.5 * (state_model['u'][:-1, :, z_idx] + state_model['u'][1:, :, z_idx])
            elif var_name == 'v':
                data = 0.5 * (state_model['v'][:, :-1, z_idx] + state_model['v'][:, 1:, z_idx])
            else:
                data = state_model[var_name][:, :, z_idx]

            # Determine colorbar scaling
            if field_def.get('scale') == 'sym':
                vmax = max(float(np.max(data)), float(np.abs(np.min(data))))
                # Cap divergence/w blowouts for cleaner visualization
                if var_name == 'div': vmax = min(vmax, 1e-4)
                if var_name == 'w': vmax = min(vmax, 2.0)
                vmin = -vmax
            else:
                vmin, vmax = float(np.min(data)), float(np.max(data))

            ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='gray')
            
            im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), 
                               cmap=field_def.get('cmap', 'viridis'), vmin=vmin, vmax=vmax)
            
            ax.set_title(field_def['title'], fontsize=12)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Sponge boundaries (dashed black)
            if sponge_depth > 0:
                sd = sponge_depth
                ax.plot(lons[sd, sd:-sd], lats[sd, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.7)
                ax.plot(lons[-sd-1, sd:-sd], lats[-sd-1, sd:-sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.7)
                ax.plot(lons[sd:-sd, sd], lats[sd:-sd, sd], 'k--', transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.7)
                ax.plot(lons[sd:-sd, -sd-1], lats[sd:-sd, -sd-1], 'k--', transform=ccrs.PlateCarree(), linewidth=1.0, alpha=0.7)

            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
            cbar.ax.tick_params(labelsize=9)

        # Turn off any unused axes if n_vars is odd
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        title = f"Suetes results (Level {z_idx})"
        if time_hours is not None:
            title += f" | T={time_hours}h"
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path: 
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved results plot to {save_path}")
        else: 
            plt.show()
        plt.close()