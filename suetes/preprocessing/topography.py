"""
Topography Blending Module.

Merges high-resolution GEBCO surface elevation data for the model interior with 
the low-resolution ERA5 topography at the lateral boundaries.
"""
import os
import urllib.request
import zipfile
import xarray as xr

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import scipy.ndimage as ndimage

class TopographyProcessor:
    """
    Processes and blends topography data for the computational domain.
    
    This class handles the interpolation of high-resolution GEBCO global 
    topography onto the computational grid and merges it with low-resolution 
    ERA5 topography at the boundaries using a smooth cosine taper (sponge).
    """
    
    def __init__(self, era5_sl_path, gebco_path):
        """
        Initializes the topography processor.

        Args:
            era5_sl_path (str): File path to ERA5 single-level data.
            gebco_path (str): File path to GEBCO topography data.
        """
        self._ensure_gebco_data(gebco_path)
        self.ds_era5 = xr.open_dataset(era5_sl_path)
        self.ds_gebco = xr.open_dataset(gebco_path)

    def _ensure_gebco_data(self, gebco_path):
        """Downloads and extracts GEBCO data if it doesn't already exist."""
        if os.path.exists(gebco_path):
            return
            
        print(f"[DOWNLOAD] GEBCO data not found. Downloading to {gebco_path}...")
        print("Note: This is a very large file (~8GB) and may take a while depending on your connection.")
        
        url = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.zip?download=1"
        zip_path = gebco_path.replace(".nc", ".zip")
        
        # Ensure the target directory exists (e.g., suetes/data/)
        os.makedirs(os.path.dirname(gebco_path), exist_ok=True)
        
        try:
            # Download the zip archive
            urllib.request.urlretrieve(url, zip_path)
            
            print("[EXTRACT] Download complete. Extracting NetCDF file...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Dynamically find the .nc file inside the zip archive
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_files:
                    raise FileNotFoundError("No .nc file found in the downloaded GEBCO zip.")
                
                # Extract the file to the data directory
                extracted_file_path = zip_ref.extract(nc_files[0], path=os.path.dirname(gebco_path))
                
                # Rename the extracted file to exactly match what the user requested ('gebco_data.nc')
                os.rename(extracted_file_path, gebco_path)
                
            print("[SUCCESS] GEBCO data extracted and ready for the model!")
            
        finally:
            # Clean up the zip file to save disk space, even if an error occurs
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def _get_era5_topo(self):
        r"""
        Extracts the first timestep of ERA5 topography and converts to meters.

        ERA5 provides geopotential height ($Z$) in $m^2/s^2$. This is converted
        to geometric height ($h$) by dividing by the standard acceleration due to
        gravity, $g \approx 9.81 \, m/s^2$.

        Returns:
            tuple: (lats, lons, z_era5_m)
        """
        time_dim = 'valid_time' if 'valid_time' in self.ds_era5.dims else 'time'
        # z is geopotential [m^2/s^2]. Divide by g (9.81) for geometric height.
        z_era5 = self.ds_era5['z'].isel({time_dim: 0}).values / 9.81
        lats = self.ds_era5.latitude.values
        lons = self.ds_era5.longitude.values
        
        # RegularGridInterpolator requires strictly ascending coordinates
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            z_era5 = z_era5[::-1, :]
            
        return lats, lons, z_era5

    def _get_gebco_topo(self, min_lat, max_lat, min_lon, max_lon):
        """
        Extracts a subset of GEBCO topography for the required region.

        Args:
            min_lat (float): Minimum latitude for the subset.
            max_lat (float): Maximum latitude for the subset.
            min_lon (float): Minimum longitude for the subset.
            max_lon (float): Maximum longitude for the subset.

        Returns:
            tuple: (lats, lons, z_gebco_m)
        """
        lat_slice = slice(min_lat, max_lat) if self.ds_gebco.lat[0] < self.ds_gebco.lat[-1] else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        
        subset = self.ds_gebco['elevation'].sel(lat=lat_slice, lon=lon_slice)
        
        lats = subset.lat.values
        lons = subset.lon.values
        z_gebco = subset.values
        
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            z_gebco = z_gebco[::-1, :]
            
        return lats, lons, z_gebco

    def _create_blending_mask(self, nx, ny, depth):
        r"""
        Creates a cosine-tapered blending mask matching the Davies Sponge layer.

        $$ M(d) = \cos^2\left(\frac{\pi}{2} \frac{d}{D}\right) $$
        
        where $d$ is distance to the boundary, returning 1.0 (ERA5) at the edge 
        and 0.0 (GEBCO) in the interior.

        Args:
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            depth (int): Width of the sponge layer in grid cells.

        Returns:
            np.ndarray: The 2D blending mask.
        """
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        
        dist_x = np.minimum(X, nx - 1 - X)
        dist_y = np.minimum(Y, ny - 1 - Y)
        dist_to_bound = np.minimum(dist_x, dist_y)
        
        # Use a smooth cosine taper for the blend
        mask = np.where(dist_to_bound < depth, 
                        np.cos(0.5 * np.pi * dist_to_bound / depth) ** 2, 
                        0.0)
        return mask

    def process_and_blend(self, grid, sponge_depth=30, smooth_sigma=1.0):
        """
        Main processing pipeline.

        1. Generates the target latitude/longitude grid from the model's 
           Cartesian coordinates.
        2. Interpolates the high-resolution GEBCO data onto this grid.
        3. Interpolates the lower-resolution ERA5 topography onto the grid.
        4. Blends the two datasets using a cosine taper (Davies Sponge).
        5. Generates a continuous, differentiable height function for the 
           dynamical core.

        Args:
            grid (BareGrid or Grid): The computational grid object.
            sponge_depth (int): Width of the boundary sponge layer [cells]. Defaults to 30.
            smooth_sigma (float): Standard deviation for Gaussian smoothing [cells].
                                   Defaults to 1.0. Set to 0.0 to disable.

        Returns:
            np.ndarray: The final 2D array of blended heights $H$ [m].
        """
        print("[GEOMETRY] Generating target Lat/Lon grid...")
        Xi_m, Yi_m = np.meshgrid(np.array(grid.x_m), np.array(grid.y_m), indexing='ij')
        target_lat, target_lon = grid.proj.get_lat_lon(Xi_m, Yi_m)
        target_lon = (target_lon + 180.0) % 360.0 - 180.0

        print("[GEOMETRY] Interpolating GEBCO topography...")
        min_lat, max_lat = float(target_lat.min()) - 2.0, float(target_lat.max()) + 2.0
        min_lon, max_lon = float(target_lon.min()) - 2.0, float(target_lon.max()) + 2.0

        gebco_lat, gebco_lon, gebco_z = self._get_gebco_topo(min_lat, max_lat, min_lon, max_lon)
        interp_gebco = RegularGridInterpolator((gebco_lat, gebco_lon), gebco_z, bounds_error=False, fill_value=None)
        
        pts = np.stack([target_lat, target_lon], axis=-1)
        z_base = np.maximum(interp_gebco(pts), 0.0)

        print("[GEOMETRY] Smoothing topography...")
        z_interior = ndimage.gaussian_filter(z_base, sigma=smooth_sigma) if smooth_sigma > 0 else z_base
        
        # Retrieve and interpolate ERA5 topography
        era5_lat, era5_lon, z_era5 = self._get_era5_topo()
        
        # Handle potential ERA5 [0, 360] longitude format to match target_lon
        era5_lon = (era5_lon + 180.0) % 360.0 - 180.0
        # Re-sort if wrapping ruined the strictly ascending requirement
        sort_idx = np.argsort(era5_lon)
        era5_lon = era5_lon[sort_idx]
        z_era5 = z_era5[:, sort_idx]

        interp_era5 = RegularGridInterpolator((era5_lat, era5_lon), z_era5, bounds_error=False, fill_value=None)
        z_boundary = np.maximum(interp_era5(pts), 0.0)
        
        # Blend the interior (GEBCO) with the external boundary (ERA5)
        mask = self._create_blending_mask(grid.nx, grid.ny, sponge_depth)
        z_final = (1.0 - mask) * z_interior + mask * z_boundary
        
        H_array = jnp.array(z_final)
        
        print("[GEOMETRY] Generating continuous h_func for dynamical core...")
        def h_func(x, y):
            idx_x = (x - grid.x_m[0]) / grid.dx
            idx_y = (y - grid.y_m[0]) / grid.dy
            return jnd.map_coordinates(H_array, [idx_x, idx_y], order=1, mode='nearest')

        return h_func