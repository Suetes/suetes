import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

class ERA5Visualizer:
    
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
        
        ax.set_title(f"ERA5 Vertical Cross-Section (Latitude: {state['latitude'][lat_idx]:.2f}°)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Geometric Height [m]")
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
        """Compares ERA5 Topography with GEBCO Topography side-by-side."""
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
        ax1.set_title("ERA5 Topography (~31 km)", fontsize=14)
        ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        
        # GEBCO
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        im2 = ax2.pcolormesh(gebco_topo.lon, gebco_topo.lat, gebco_topo.values, 
                             transform=ccrs.PlateCarree(), cmap='terrain', vmin=vmin, vmax=vmax)
        ax2.set_title("GEBCO Topography (High Resolution)", fontsize=14)
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