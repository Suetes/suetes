import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

class ERA5Bridge:
    def __init__(self, grid, constants):
        self.grid = grid
        self.c = constants
        
        # JAX's jnp.interp is 1D. We map it over the X (axis 0) and Y (axis 1) dimensions.
        # in_axes=(0, 0, 0) means we map over the first dimension of target_z, z_era5, and var_era5
        self._column_interp = jax.vmap(
            jax.vmap(jnp.interp, in_axes=(0, 0, 0)), 
            in_axes=(0, 0, 0)
        )

    def _thermodynamics(self, T, p, q):
        """Converts standard meteorology variables to Suetes dry-core variables."""
        # Epsilon is the ratio of gas constants (Rd/Rv ~= 0.622)
        epsilon = self.c.get('epsilon', 0.622)
        
        # Virtual temperature (Tv)
        Tv = T * (1.0 + (1.0 / epsilon - 1.0) * q)
        
        # Density (Ideal Gas Law: P = rho * Rd * Tv)
        rho = p / (self.c['Rd'] * Tv)
        
        # Exner Pressure (pi)
        pi = (p / self.c['p0']) ** (self.c['Rd'] / self.c['cp'])
        
        # Virtual Potential Temperature (th_v)
        th_v = Tv / pi
        
        return th_v, pi, rho

    def _interp_3d(self, target_z, z_era5, var_era5):
        """
        Interpolates a 3D ERA5 field onto the target Suetes Z grid.
        Note: jnp.interp requires the x-coordinates (z_era5) to be strictly increasing.
        """
        # ERA5 pressure levels typically go from top-of-atmosphere to surface 
        # (meaning height is descending). We must flip them to be strictly increasing.
        is_descending = z_era5[0, 0, 0] > z_era5[0, 0, -1]
        
        z_era5_sorted = jnp.where(is_descending, jnp.flip(z_era5, axis=-1), z_era5)
        var_era5_sorted = jnp.where(is_descending, jnp.flip(var_era5, axis=-1), var_era5)

        return self._column_interp(target_z, z_era5_sorted, var_era5_sorted)

    def process_era5_slice(self, era5_state):
        """
        Processes a dict of ERA5 arrays that have already been horizontally regridded 
        to the model's (nx, ny) domain.
        
        era5_state requires:
        - 'T': Temperature [K]
        - 'q': Specific Humidity [kg/kg]
        - 'p': Pressure [Pa] (3D array or 1D broadcasted to 3D)
        - 'geopotential': [m^2/s^2]
        - 'u', 'v': Horizontal winds [m/s]
        - 'omega': Pressure vertical velocity [Pa/s]
        """
        # 1. Physical Height of ERA5 Levels
        z_era5 = era5_state['geopotential'] / self.c['g']

        # 2. Thermodynamic Conversion (on the native ERA5 vertical grid)
        th_v_era5, pi_era5, rho_era5 = self._thermodynamics(
            era5_state['T'], era5_state['p'], era5_state['q']
        )

        # 3. Convert Vertical Velocity (omega -> w)
        # Using hydrostatic approximation: omega ~ -rho * g * w 
        w_era5 = -era5_state['omega'] / (rho_era5 * self.c['g'])

        # 4. Vertical Interpolation to Suetes Grid
        model_state = {}
        
        # Interpolate variables to the Mass (M) points
        model_state['u'] = self._interp_3d(self.grid.Z_m, z_era5, era5_state['u'])
        model_state['v'] = self._interp_3d(self.grid.Z_m, z_era5, era5_state['v'])
        model_state['th_v'] = self._interp_3d(self.grid.Z_m, z_era5, th_v_era5)
        model_state['pi'] = self._interp_3d(self.grid.Z_m, z_era5, pi_era5)
        model_state['rho'] = self._interp_3d(self.grid.Z_m, z_era5, rho_era5)
        
        # Specific humidity is passed through as a tracer 
        # (It will be conserved by the FFSL advector you wrote)
        model_state['q'] = self._interp_3d(self.grid.Z_m, z_era5, era5_state['q'])
        
        # Interpolate Vertical velocity to the vertical faces (W points)
        model_state['w'] = self._interp_3d(self.grid.Z_w, z_era5, w_era5)
        
        # Initialize contra-variant vertical velocity to 0 (the solver will adjust it)
        model_state['eta_dot'] = jnp.zeros_like(model_state['w'])

        return model_state


class HorizontalRegridder:
    def __init__(self, grid, era5_lats, era5_lons):
        """
        grid: RegionalGrid3D instance
        era5_lats: 1D array of ERA5 latitudes (typically descending, e.g., 90 to -90)
        era5_lons: 1D array of ERA5 longitudes (typically 0 to 359.75)
        """
        self.grid = grid
        
        # Calculate the grid spacing of the ERA5 data
        self.lat_0 = era5_lats[0]
        self.dlat = era5_lats[1] - era5_lats[0] # Will be negative if descending
        
        self.lon_0 = era5_lons[0]
        self.dlon = era5_lons[1] - era5_lons[0]

        # Precompute the fractional ERA5 indices for every point on the regional grid.
        # We compute this for the staggered grids (m, u, v) separately.
        self.target_indices = {
            'm': self._compute_fractional_indices(self.grid.x_m, self.grid.y_m),
            'u': self._compute_fractional_indices(self.grid.x_c, self.grid.y_m),
            'v': self._compute_fractional_indices(self.grid.x_m, self.grid.y_c)
        }

    def _compute_fractional_indices(self, x_coords, y_coords):
        # 1. Generate local Cartesian grid
        Xi, Yi = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        
        # 2. Project to Lat/Lon using your existing Oblique Stereographic math
        target_lat, target_lon = self.grid.proj.get_lat_lon(Xi, Yi)
        
        # 3. Normalize longitude to 0-360 to match standard ERA5 formatting
        target_lon = jnp.mod(target_lon, 360.0)
        
        # 4. Convert target coordinates into floating-point ERA5 array indices
        idx_lat = (target_lat - self.lat_0) / self.dlat
        idx_lon = (target_lon - self.lon_0) / self.dlon
        
        # Stack into shape (2, nx, ny) required by jnd.map_coordinates
        return jnp.stack([idx_lat, idx_lon], axis=0)

    def regrid_2d(self, field_era5, loc='m'):
        """
        Regrids a 2D surface field (like surface pressure or topography).
        field_era5: shape (n_lat, n_lon)
        """
        coords = self.target_indices[loc]
        # order=1 is bilinear interpolation
        return jnd.map_coordinates(field_era5, coords, order=1, mode='nearest')
        
    def regrid_3d(self, field_era5_3d, loc='m'):
        """
        Regrids a 3D atmospheric field (like T, q, u, v).
        field_era5_3d: shape (n_levels, n_lat, n_lon)
        """
        # Vectorize the 2D regridding over the vertical levels (axis 0)
        vmap_regrid = jax.vmap(lambda f: self.regrid_2d(f, loc), in_axes=0, out_axes=0)
        
        # Output is (n_levels, nx, ny). 
        # We transpose to (nx, ny, n_levels) so it is ready for the vertical interpolation 
        # in ERA5Bridge._interp_3d.
        regridded = vmap_regrid(field_era5_3d)
        return jnp.transpose(regridded, (1, 2, 0))