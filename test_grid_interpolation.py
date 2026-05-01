import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from suetes.preprocessing.topography import TopographyProcessor
from suetes.regional3d.geometry import RegionalGrid3D

def plot_grid_comparison():
    # Setup the Suetes base grid
    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 2000.0, 2000.0, 500.0
    lat_c, lon_c = 45.0, 5.0
    sponge_depth = 20

    print("Initializing Suetes geometry...")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)

    # Process and blend the topography
    print("Interpolating and blending topography...")
    topo_proc = TopographyProcessor(
        era5_sl_path="suetes/data/suetes_test_run_single_levels.nc",
        gebco_path="suetes/data/gebco_data.nc"
    )
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth)

    # Evaluate the topography on the Suetes 2D mesh
    # Xi_m and Yi_m are the Cartesian coordinates of the cell centers
    Xi_m, Yi_m = np.meshgrid(np.array(base_grid.x_m), np.array(base_grid.y_m), indexing='ij')
    
    # Use h_func to get the height at every cell
    suetes_h = h_func(Xi_m, Yi_m)
    
    # Get the Lat/Lon coordinates of every Suetes cell for plotting
    suetes_lat, suetes_lon = base_grid.proj.get_lat_lon(Xi_m, Yi_m)
    suetes_lon = np.mod(suetes_lon, 360.0) 

    # Extract ERA5 topography
    time_dim = 'valid_time' if 'valid_time' in topo_proc.ds_era5.dims else 'time'
    era5_h = topo_proc.ds_era5['z'].isel({time_dim: 0}).values / 9.81
    era5_lat = topo_proc.ds_era5.latitude.values
    era5_lon = topo_proc.ds_era5.longitude.values

    # Plotting
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})

    vmin = 0
    vmax = max(float(np.max(era5_h)), float(np.max(suetes_h)))

    # Calculate extent so we zoom in directly on the Suetes domain
    # (ERA5 domain is much larger, we want to see the overlap)
    min_lon, max_lon = float(suetes_lon.min()) - 0.2, float(suetes_lon.max()) + 0.2
    min_lat, max_lat = float(suetes_lat.min()) - 0.2, float(suetes_lat.max()) + 0.2
    extent = [min_lon, max_lon, min_lat, max_lat]

    # ERA5 plot
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='red')
    # edgecolors='black' draws the boundary of the 31km grid cells
    im1 = ax1.pcolormesh(era5_lon, era5_lat, era5_h, transform=ccrs.PlateCarree(),
                         cmap='terrain', vmin=vmin, vmax=vmax, 
                         edgecolors='black', linewidth=0.5)
    ax1.set_title("ERA5 native grid (~31km)", fontsize=14)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())

    # Suetes plot
    ax2.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='red')
    
    # Define a decimation factor for the grid lines (e.g., plot every 10th line)
    line_decim = 10 
    
    # Plot the full resolution field
    im2 = ax2.pcolormesh(suetes_lon, suetes_lat, suetes_h, 
                         transform=ccrs.PlateCarree(),
                         cmap='terrain', vmin=vmin, vmax=vmax, 
                         edgecolors='none') # Turn off the default 2km grid lines

    # Overlay "ghost" grid lines at a coarser interval (e.g., every 20km)
    # to show the orientation and curvature of the Oblique Stereographic projection
    ax2.pcolormesh(suetes_lon[::line_decim, ::line_decim], 
                   suetes_lat[::line_decim, ::line_decim], 
                   suetes_h[::line_decim, ::line_decim],
                   transform=ccrs.PlateCarree(),
                   facecolor='none', # Don't overwrite the colors
                   edgecolors='black', 
                   linewidth=0.2, 
                   alpha=0.5) # Make lines subtle

    ax2.set_title(f"Suetes Grid (2km data, {line_decim*2}km mesh lines)", fontsize=14)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elevation [m]', fontsize=12)

    plt.subplots_adjust(bottom=0.18)
    plt.savefig("out_grid_comparison.png", dpi=200)
    print("Done! Saved to out_grid_comparison.png")

if __name__ == "__main__":
    plot_grid_comparison()