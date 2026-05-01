import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

class HorizontalRegridder:
    def __init__(self, grid, era5_lats, era5_lons):
        self.grid = grid
        
        self.lat_0 = era5_lats[0]
        self.dlat = era5_lats[1] - era5_lats[0] 
        self.lon_0 = era5_lons[0]
        self.dlon = era5_lons[1] - era5_lons[0]

        # Check if the ERA5 dataset uses negative longitudes
        self.is_negative_lon = self.lon_0 < 0 

        self.target_indices = {
            'm': self._compute_fractional_indices(self.grid.x_m, self.grid.y_m),
            'u': self._compute_fractional_indices(self.grid.x_c, self.grid.y_m),
            'v': self._compute_fractional_indices(self.grid.x_m, self.grid.y_c)
        }

    def _compute_fractional_indices(self, x_coords, y_coords):
        Xi, Yi = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        target_lat, target_lon = self.grid.proj.get_lat_lon(Xi, Yi)
        
        target_lon = jnp.where(
            self.is_negative_lon,
            (target_lon + 180.0) % 360.0 - 180.0, # Map to [-180, 180]
            jnp.mod(target_lon, 360.0)            # Map to [0, 360]
        )
        
        idx_lat = (target_lat - self.lat_0) / self.dlat
        idx_lon = (target_lon - self.lon_0) / self.dlon
        
        return jnp.stack([idx_lat, idx_lon], axis=0)

    def regrid_3d(self, field_era5_3d, loc='m'):
        """Regrids a (levels, lat, lon) array to (levels, nx, ny)."""
        coords = self.target_indices[loc]
        vmap_regrid = jax.vmap(lambda f: jnd.map_coordinates(f, coords, order=1, mode='nearest'), in_axes=0)
        regridded = vmap_regrid(field_era5_3d)
        return jnp.transpose(regridded, (1, 2, 0))


class BoundaryProcessor:
    def __init__(self, grid, era5_lats, era5_lons, constants):
        self.grid = grid
        self.c = constants
        self.regridder = HorizontalRegridder(grid, era5_lats, era5_lons)
        
        # JAX's 1D interpolator vectorized over X (axis 0) and Y (axis 1)
        self._column_interp = jax.vmap(
            jax.vmap(jnp.interp, in_axes=(0, 0, 0)), 
            in_axes=(0, 0, 0)
        )

    def _thermodynamics(self, T, p, q):
        """Converts standard meteorology variables to dry-core prognostic variables."""
        epsilon = self.c.get('epsilon', 0.622)
        Tv = T * (1.0 + (1.0 / epsilon - 1.0) * q)
        rho = p / (self.c['Rd'] * Tv)
        pi = (p / self.c['p0']) ** (self.c['Rd'] / self.c['cp'])
        th_v = Tv / pi
        return th_v, pi, rho

    def _interp_3d(self, target_z, z_era5, var_era5):
        """Interpolates an ERA5 column to the Suetes terrain-following column."""
        # Use argsort to guarantee strictly ascending coordinates
        sort_idx = jnp.argsort(z_era5, axis=-1)
        z_era5_sorted = jnp.take_along_axis(z_era5, sort_idx, axis=-1)
        var_era5_sorted = jnp.take_along_axis(var_era5, sort_idx, axis=-1)

        return self._column_interp(target_z, z_era5_sorted, var_era5_sorted)

    def process(self, stitched_era5_state):
        """
        Takes the raw numpy arrays from the ERA5Processor, applies horizontal regridding, 
        thermodynamic conversion, and vertical interpolation, returning a model-ready state.
        """
        # --- 1. Horizontal Regridding ---
        # We must regrid the ERA5 heights to the staggered locations so the 
        # vertical interpolator has the correct physical z-coordinate for every face!
        z_era5_m = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='m')
        z_era5_u = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='u')
        z_era5_v = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='v')
        
        p_era5 = self.regridder.regrid_3d(stitched_era5_state['p'], loc='m')
        T_era5 = self.regridder.regrid_3d(stitched_era5_state['T'], loc='m')
        q_era5 = self.regridder.regrid_3d(stitched_era5_state['q'], loc='m')
        omega_era5 = self.regridder.regrid_3d(stitched_era5_state['omega'], loc='m')
        
        # --- 1a. U-Face Wind Rotation ---
        # Interpolate BOTH geographic u and v to the staggered u-points
        u_geo_at_u = self.regridder.regrid_3d(stitched_era5_state['u'], loc='u')
        v_geo_at_u = self.regridder.regrid_3d(stitched_era5_state['v'], loc='u')
        
        Xi_u, Yi_u = jnp.meshgrid(self.grid.x_c, self.grid.y_m, indexing='ij')
        gamma_u = self.grid.proj.get_convergence_angle(Xi_u, Yi_u)
        gamma_u_3d = jnp.expand_dims(gamma_u, axis=-1)
        
        # Rotate into Grid X/Y basis and keep only the u-component
        u_era5 = u_geo_at_u * jnp.cos(gamma_u_3d) + v_geo_at_u * jnp.sin(gamma_u_3d)

        # --- 1b. V-Face Wind Rotation ---
        # Interpolate BOTH geographic u and v to the staggered v-points
        u_geo_at_v = self.regridder.regrid_3d(stitched_era5_state['u'], loc='v')
        v_geo_at_v = self.regridder.regrid_3d(stitched_era5_state['v'], loc='v')
        
        Xi_v, Yi_v = jnp.meshgrid(self.grid.x_m, self.grid.y_c, indexing='ij')
        gamma_v = self.grid.proj.get_convergence_angle(Xi_v, Yi_v)
        gamma_v_3d = jnp.expand_dims(gamma_v, axis=-1)
        
        # Rotate into Grid X/Y basis and keep only the v-component
        v_era5 = -u_geo_at_v * jnp.sin(gamma_v_3d) + v_geo_at_v * jnp.cos(gamma_v_3d)

        # --- 2. Thermodynamics ---
        # Thermodynamics are computed entirely on the mass points
        th_v_era5, pi_era5, rho_era5 = self._thermodynamics(T_era5, p_era5, q_era5)

        # Convert omega (Pa/s) to geometric w (m/s) using hydrostatic approx
        w_era5 = -omega_era5 / (rho_era5 * self.c['g'])

        # --- 3. Vertical Interpolation to 3D Grid ---
        state = {}
        # Notice we pair the _u fields with z_era5_u, and _v fields with z_era5_v
        state['u'] = self._interp_3d(self.grid.Z_u, z_era5_u, u_era5)
        state['v'] = self._interp_3d(self.grid.Z_v, z_era5_v, v_era5)
        
        state['th_v'] = self._interp_3d(self.grid.Z_m, z_era5_m, th_v_era5)
        state['pi'] = self._interp_3d(self.grid.Z_m, z_era5_m, pi_era5)
        state['rho'] = self._interp_3d(self.grid.Z_m, z_era5_m, rho_era5)
        state['q'] = self._interp_3d(self.grid.Z_m, z_era5_m, q_era5)
        
        # W sits on the vertical cell faces but shares the horizontal coordinates of M
        state['w'] = self._interp_3d(self.grid.Z_w, z_era5_m, w_era5)
        
        # --- THE FIX 2: STRICT KINEMATIC BOUNDARIES ---
        # ERA5 has tiny non-zero velocities at the edges. 
        # We must zero them out so the Sponge doesn't fight the implicit solver!
        state['w'] = state['w'].at[:, :, 0].set(0.0)
        state['w'] = state['w'].at[:, :, -1].set(0.0)
        
        # --- 4. Kinematics ---
        state['eta_dot'] = jnp.zeros_like(state['w'])

        return state