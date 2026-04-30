import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

class TopographyProcessor:
    def __init__(self, era5_sl_path, gebco_path):
        """Loads the raw datasets for topography processing."""
        self.ds_era5 = xr.open_dataset(era5_sl_path)
        self.ds_gebco = xr.open_dataset(gebco_path)

    def _get_era5_topo(self):
        """Extracts the first timestep of ERA5 topography and converts to meters."""
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
        """Subsets and extracts GEBCO topography."""
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
        """
        Creates a 2D mask identical to the DaviesSponge.
        1.0 at the boundaries (100% ERA5), 0.0 in the interior (100% GEBCO).
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

    def process_and_blend(self, grid, sponge_depth=10):
        """
        Main pipeline: Interpolates both topographies to the model grid and blends them.
        Returns a callable h_func(x, y) for JAX geometry generation.
        """
        print("1. Generating target Lat/Lon grid...")
        Xi_m, Yi_m = np.meshgrid(np.array(grid.x_m), np.array(grid.y_m), indexing='ij')
        target_lat, target_lon = grid.proj.get_lat_lon(Xi_m, Yi_m)
        target_lon = np.mod(target_lon, 360.0) # Ensure 0-360 standard

        print("2. Interpolating ERA5 Topography...")
        era5_lat, era5_lon, era5_z = self._get_era5_topo()
        interp_era5 = RegularGridInterpolator((era5_lat, era5_lon), era5_z, bounds_error=False, fill_value=None)
        
        # We must pass coordinates to the interpolator as (lat, lon) pairs
        pts = np.stack([target_lat, target_lon], axis=-1)
        z_era5_grid = interp_era5(pts)

        print("3. Interpolating GEBCO Topography...")
        # Add a 2-degree buffer and CAST JAX ARRAYS TO FLOAT so xarray/pandas can read them
        min_lat = float(target_lat.min()) - 2.0
        max_lat = float(target_lat.max()) + 2.0
        min_lon = float(target_lon.min()) - 2.0
        max_lon = float(target_lon.max()) + 2.0

        gebco_lat, gebco_lon, gebco_z = self._get_gebco_topo(min_lat, max_lat, min_lon, max_lon)
        
        interp_gebco = RegularGridInterpolator((gebco_lat, gebco_lon), gebco_z, bounds_error=False, fill_value=None)
        z_gebco_grid = interp_gebco(pts)

        print("4. Blending Topographies...")
        mask = self._create_blending_mask(grid.nx, grid.ny, sponge_depth)
        
        # blended = 100% ERA5 at boundaries, 100% GEBCO in the middle
        z_blended = (1.0 - mask) * z_gebco_grid + mask * z_era5_grid
        
        # Flatten ocean bathymetry (no negative heights in our atmospheric model)
        z_blended = np.maximum(z_blended, 0.0)
        
        # Convert to a fixed JAX array so it's ready for fast compilation
        H_array = jnp.array(z_blended)
        
        print("5. Generating continuous h_func for JAX Geometry...")
        
        # We create a closure that captures H_array and grid properties.
        # This allows JAX to evaluate height at ANY Cartesian (x, y) coordinate 
        # (e.g., staggered u and v points).
        def h_func(x, y):
            # Convert physical Cartesian coordinates back to fractional array indices
            idx_x = (x - grid.x_m[0]) / grid.dx
            idx_y = (y - grid.y_m[0]) / grid.dy
            
            # Map coordinates supports JAX transformations (vmap, grad, etc.)
            return jnd.map_coordinates(H_array, [idx_x, idx_y], order=1, mode='nearest')

        return h_func