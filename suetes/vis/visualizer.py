import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

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
        else:
            plt.show()
        plt.close()

    def plot_model_vs_era5_map(self, grid, state_model, state_era5, variable='th_v', z_idx=5, save_path=None):
        """
        Plots a side-by-side horizontal comparison of the Model vs ERA5.
        """
        # 1. Determine horizontal coordinates based on Arakawa-C staggering
        if variable == 'u':
            x_coords, y_coords = grid.x_c, grid.y_m
        elif variable == 'v':
            x_coords, y_coords = grid.x_m, grid.y_c
        else:
            x_coords, y_coords = grid.x_m, grid.y_m

        Xi, Yi = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Get geographic coordinates
        lats, lons = grid.proj.get_lat_lon(Xi, Yi)
        
        # --- THE FIX: Wrap longitudes to [-180, 180] ---
        # This prevents the bounding box from exploding if the domain crosses the Prime Meridian (0°).
        lons = (lons + 180.0) % 360.0 - 180.0

        # 2. Extract the 2D horizontal slices
        val_model = state_model[variable][:, :, z_idx]
        val_era5 = state_era5[variable][:, :, z_idx]
        
        # Calculate approximate height for the title
        z_approx = grid.z_m[z_idx] if variable != 'w' else grid.z_c[z_idx]

        # Find common color limits for a fair 1:1 comparison
        vmin = min(float(np.min(val_model)), float(np.min(val_era5)))
        vmax = max(float(np.max(val_model)), float(np.max(val_era5)))

        # 3. Plotting (Adjusted figsize for better aspect ratio)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # Buffer for map extent
        extent = [float(lons.min()) - 0.5, float(lons.max()) + 0.5, 
                  float(lats.min()) - 0.5, float(lats.max()) + 0.5]

        for ax, data, title in zip([ax1, ax2], [val_model, val_era5], ["Suetes Simulation (T=1h)", "ERA5 Target (T=1h)"]):
            ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=':', edgecolor='gray')
            
            # Use pcolormesh for fast, native grid plotting
            im = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), 
                               cmap='RdYlBu_r' if variable in ['u', 'v', 'w'] else 'RdYlBu_r', 
                               vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=14)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Shared colorbar
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

    def plot_anomaly(self, grid, state_model, state_era5, variable='u', z_idx=15, save_path=None):
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
        
        # Use a diverging colormap centered tightly on 0
        vmax = max(float(np.max(np.abs(anomaly))), 0.1) 
        im = ax.pcolormesh(lons, lats, anomaly, transform=ccrs.PlateCarree(), cmap='seismic', vmin=-vmax, vmax=vmax)
        
        ax.set_title(f"Mesoscale anomaly ({variable}): Suetes - ERA5", fontsize=14)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label(f"{variable} anomaly", fontsize=12)

        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()

    def plot_suetes_w_cross_section(self, grid, state_model, y_idx, save_path=None):
        """Plots a vertical cross-section of Vertical Velocity (W) through the Suetes grid."""
        x_coords = grid.x_m / 1000.0  # Convert to km
        
        # W sits on Z_w (the vertical cell faces). We plot it using X and Z_w.
        Z_slice = grid.Z_w[:, y_idx, :]
        W_slice = state_model['w'][:, y_idx, :]
        
        # Surface topography is the bottom of Z_w
        terrain_z = grid.Z_w[:, y_idx, 0]
        
        X_2d = np.broadcast_to(x_coords[:, None], Z_slice.shape)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use a diverging colormap for vertical velocity (red=up, blue=down)
        vmax = max(float(np.max(W_slice)), float(np.abs(np.min(W_slice))))
        # Clamp vmax so it highlights the mountain waves perfectly
        vmax = min(max(vmax, 0.1), 3.0) 
        
        contour = ax.contourf(X_2d, Z_slice, W_slice, levels=30, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(contour, ax=ax, label='Vertical velocity (w) [m/s]')
        
        ax.fill_between(x_coords, 0, terrain_z, color='dimgray', label='Suetes Topography')
        
        ax.set_title(f"Suetes vertical velocity (y-index: {y_idx})")
        ax.set_xlabel("X distance from domain center [km]")
        ax.set_ylabel("Geometric height [m]")
        ax.set_ylim(0, 15000)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()