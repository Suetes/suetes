import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

import numpy as np
import scipy.ndimage as ndimage_cpu


class TimeManager:
    def __init__(self, states_list, times_sec_list, grid):
        self.grid = grid
        self.times_sec = jnp.array(times_sec_list, dtype=jnp.float32)
        
        # Stack the list of dicts into a single dict of 4D arrays (time, X, Y, Z)
        # This allows JAX to dynamically slice the correct time index during the scan loop
        self.stacked_states = {}
        for k in states_list[0].keys():
            self.stacked_states[k] = jnp.stack([state[k] for state in states_list], axis=0)

    def get_forcing(self, t):
        # 1. Find the left bounding time index for the current t
        # (e.g., if t=4000s, idx will be 1, representing the 3600s boundary)
        idx = jnp.searchsorted(self.times_sec, t, side='right') - 1
        
        # Clip to prevent out-of-bounds errors if the simulation runs slightly past the last ERA5 state
        idx = jnp.clip(idx, 0, len(self.times_sec) - 2)
        
        t0 = self.times_sec[idx]
        t1 = self.times_sec[idx + 1]
        
        # 2. Calculate interpolation weight
        alpha = (t - t0) / (t1 - t0)
        alpha = jnp.clip(alpha, 0.0, 1.0)
        
        # 3. Dynamically slice the two bounding states and interpolate
        interp_state = {}
        for k in self.stacked_states.keys():
            state_t0 = self.stacked_states[k][idx]
            state_t1 = self.stacked_states[k][idx + 1]
            interp_state[k] = (1.0 - alpha) * state_t0 + alpha * state_t1
            
        return interp_state

class HorizontalRegridder:
    def __init__(self, grid, era5_lats, era5_lons):
        self.grid = grid
        
        self.lat_0 = float(era5_lats[0])
        self.dlat = float(era5_lats[1] - era5_lats[0]) 
        self.lon_0 = float(era5_lons[0])
        self.dlon = float(era5_lons[1] - era5_lons[0])

        self.is_negative_lon = self.lon_0 < 0 

        # Compute target indices as standard numpy arrays
        self.target_indices = {
            'm': self._compute_fractional_indices(np.array(self.grid.x_m), np.array(self.grid.y_m)),
            'u': self._compute_fractional_indices(np.array(self.grid.x_c), np.array(self.grid.y_m)),
            'v': self._compute_fractional_indices(np.array(self.grid.x_m), np.array(self.grid.y_c))
        }

    def _compute_fractional_indices(self, x_coords, y_coords):
        Xi, Yi = np.meshgrid(x_coords, y_coords, indexing='ij')
        target_lat, target_lon = self.grid.proj.get_lat_lon(Xi, Yi)
        
        # Ensure we are using numpy here, not jnp
        target_lat = np.array(target_lat)
        target_lon = np.array(target_lon)
        
        target_lon = np.where(
            self.is_negative_lon,
            (target_lon + 180.0) % 360.0 - 180.0,
            np.mod(target_lon, 360.0)            
        )
        
        idx_lat = (target_lat - self.lat_0) / self.dlat
        idx_lon = (target_lon - self.lon_0) / self.dlon
        
        return np.stack([idx_lat, idx_lon], axis=0)

    def regrid_3d(self, field_era5_3d, loc='m'):
        """Regrids a (levels, lat, lon) array using SciPy bicubic interpolation."""
        coords = self.target_indices[loc]
        field_np = np.array(field_era5_3d)
        
        # Map coordinates layer by layer using standard SciPy with order=3
        regridded_layers = []
        for z in range(field_np.shape[0]):
            layer = ndimage_cpu.map_coordinates(field_np[z], coords, order=3, mode='nearest')
            regridded_layers.append(layer)
            
        regridded = np.stack(regridded_layers, axis=0)
        
        # Convert back to JAX array and transpose to (X, Y, Z) expected by the model
        return jnp.array(np.transpose(regridded, (1, 2, 0)))


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

    def _balance_global_mass(self, state):
        """
        Calculates the net mass flux through the four lateral boundaries and 
        applies a barotropic correction to ensure exact global mass conservation.
        """
        # Approximate density on the boundaries (using the outermost interior cells)
        rho_w = state['rho'][0, :, :]
        rho_e = state['rho'][-1, :, :]
        rho_s = state['rho'][:, 0, :]
        rho_n = state['rho'][:, -1, :]

        # Get the vertical cell heights at the boundaries
        dz_w = self.grid.dz_m_full[0, :, :]
        dz_e = self.grid.dz_m_full[-1, :, :]
        dz_s = self.grid.dz_m_full[:, 0, :]
        dz_n = self.grid.dz_m_full[:, -1, :]

        # Calculate absolute mass flux (kg/s) through each face
        # Flux = sum(rho * v_normal * Area)
        # Note: West/South are inflow (+), East/North are outflow (-)
        flux_west  = jnp.sum(state['u'][0, :, :] * rho_w * self.grid.dy * dz_w)
        flux_east  = jnp.sum(state['u'][-1, :, :] * rho_e * self.grid.dy * dz_e)
        flux_south = jnp.sum(state['v'][:, 0, :] * rho_s * self.grid.dx * dz_s)
        flux_north = jnp.sum(state['v'][:, -1, :] * rho_n * self.grid.dx * dz_n)

        # Net mass accumulation in the domain (kg/s)
        net_flux = (flux_west - flux_east) + (flux_south - flux_north)

        # Calculate total boundary surface mass-area to distribute the correction
        area_west  = jnp.sum(rho_w * self.grid.dy * dz_w)
        area_east  = jnp.sum(rho_e * self.grid.dy * dz_e)
        area_south = jnp.sum(rho_s * self.grid.dx * dz_s)
        area_north = jnp.sum(rho_n * self.grid.dx * dz_n)
        
        total_mass_area = area_west + area_east + area_south + area_north

        # Calculate the uniform velocity correction (m/s)
        V_c = net_flux / total_mass_area

        # Distribute the correction as a linear gradient across the domain to avoid 
        # a localized divergence shock at the boundaries.
        
        # Create normalized coordinates spanning [-1.0, 1.0]
        x_norm = (jnp.arange(self.grid.nx + 1) / self.grid.nx) * 2.0 - 1.0
        y_norm = (jnp.arange(self.grid.ny + 1) / self.grid.ny) * 2.0 - 1.0
        
        # Reshape for 3D broadcasting
        x_norm_3d = jnp.expand_dims(x_norm, axis=(1, 2))
        y_norm_3d = jnp.expand_dims(y_norm, axis=(0, 2))

        # Apply linearly: West face gets -V_c (reduces inflow), East gets +V_c (increases outflow)
        state['u'] = state['u'] + (V_c * x_norm_3d)
        
        # South face gets -V_c, North gets +V_c
        state['v'] = state['v'] + (V_c * y_norm_3d)

        return state

    def process(self, stitched_era5_state):
        """
        Takes the raw numpy arrays from the ERA5Processor, applies horizontal regridding, 
        thermodynamic conversion, and vertical interpolation, returning a model-ready state.
        """
        # Horizontal Regridding 
        # We must regrid the ERA5 heights to the staggered locations so the 
        # vertical interpolator has the correct physical z-coordinate for every face!
        z_era5_m = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='m')
        z_era5_u = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='u')
        z_era5_v = self.regridder.regrid_3d(stitched_era5_state['geopotential'] / self.c['g'], loc='v')
        
        p_era5 = self.regridder.regrid_3d(stitched_era5_state['p'], loc='m')
        T_era5 = self.regridder.regrid_3d(stitched_era5_state['T'], loc='m')
        q_era5 = self.regridder.regrid_3d(stitched_era5_state['q'], loc='m')
        omega_era5 = self.regridder.regrid_3d(stitched_era5_state['omega'], loc='m')
        
        # U-Face Wind Rotation
        # Interpolate BOTH geographic u and v to the staggered u-points
        u_geo_at_u = self.regridder.regrid_3d(stitched_era5_state['u'], loc='u')
        v_geo_at_u = self.regridder.regrid_3d(stitched_era5_state['v'], loc='u')
        
        Xi_u, Yi_u = jnp.meshgrid(self.grid.x_c, self.grid.y_m, indexing='ij')
        gamma_u = self.grid.proj.get_convergence_angle(Xi_u, Yi_u)
        gamma_u_3d = jnp.expand_dims(gamma_u, axis=-1)
        
        # Rotate into Grid X/Y basis and keep only the u-component
        u_era5 = u_geo_at_u * jnp.cos(gamma_u_3d) + v_geo_at_u * jnp.sin(gamma_u_3d)

        # V-Face Wind Rotation
        # Interpolate BOTH geographic u and v to the staggered v-points
        u_geo_at_v = self.regridder.regrid_3d(stitched_era5_state['u'], loc='v')
        v_geo_at_v = self.regridder.regrid_3d(stitched_era5_state['v'], loc='v')
        
        Xi_v, Yi_v = jnp.meshgrid(self.grid.x_m, self.grid.y_c, indexing='ij')
        gamma_v = self.grid.proj.get_convergence_angle(Xi_v, Yi_v)
        gamma_v_3d = jnp.expand_dims(gamma_v, axis=-1)
        
        # Rotate into Grid X/Y basis and keep only the v-component
        v_era5 = -u_geo_at_v * jnp.sin(gamma_v_3d) + v_geo_at_v * jnp.cos(gamma_v_3d)

        # THERMODYNAMICS
        # Thermodynamics are computed entirely on the mass points
        th_v_era5, pi_era5, rho_era5 = self._thermodynamics(T_era5, p_era5, q_era5)

        # Convert omega (Pa/s) to geometric w (m/s) using hydrostatic approx
        w_era5 = -omega_era5 / (rho_era5 * self.c['g'])

        # Vertical Interpolation to 3D Grid
        state = {}
        state['u'] = self._interp_3d(self.grid.Z_u, z_era5_u, u_era5)
        state['v'] = self._interp_3d(self.grid.Z_v, z_era5_v, v_era5)
        state['th_v'] = self._interp_3d(self.grid.Z_m, z_era5_m, th_v_era5)
        state['q'] = self._interp_3d(self.grid.Z_m, z_era5_m, q_era5)
        state['w'] = self._interp_3d(self.grid.Z_w, z_era5_m, w_era5)
        
        # Hydrostatic pressure reconstruction
        pi_interp = self._interp_3d(self.grid.Z_m, z_era5_m, pi_era5)
        pi_anchor_top = pi_interp[:, :, -1] 
        
        delta_z = self.grid.Z_m[:, :, 1:] - self.grid.Z_m[:, :, :-1]
        th_v_w = 0.5 * (state['th_v'][:, :, 1:] + state['th_v'][:, :, :-1])
        delta_pi = -(self.c['g'] * delta_z) / (self.c['cp'] * th_v_w)
        
        # Reverse cumsum to subtract pressure increments from the top down
        # (JAX doesn't have a native reverse_cumsum, so we flip, sum, and flip back)
        pi_cumsum_rev = jnp.cumsum(delta_pi[..., ::-1], axis=-1)[..., ::-1]
        
        state['pi'] = jnp.concatenate([
            jnp.expand_dims(pi_anchor_top, axis=2) - pi_cumsum_rev,
            jnp.expand_dims(pi_anchor_top, axis=2)
        ], axis=2)
        
        # Re-derive rho to strictly satisfy the Equation of State
        state['rho'] = self.c['p0'] / (self.c['Rd'] * state['th_v']) * \
                       (state['pi'] ** (self.c['cvd'] / self.c['Rd']))

        # KINEMATIC BOUNDARIES
        # Calculate terrain-following w at the surface
        u_m_surf = 0.5 * (state['u'][:-1, :, 0] + state['u'][1:, :, 0])
        v_m_surf = 0.5 * (state['v'][:, :-1, 0] + state['v'][:, 1:, 0])
        
        kinematic_bottom = (
            u_m_surf * self.grid.z_xi_w[:, :, 0] + 
            v_m_surf * self.grid.z_eta_w[:, :, 0]
        )
        
        # Apply strict boundary conditions
        state['w'] = state['w'].at[:, :, 0].set(kinematic_bottom) 
        state['w'] = state['w'].at[:, :, -1].set(0.0)             
        state['eta_dot'] = jnp.zeros_like(state['w'])
        
        # Enforce global mass conservation (probably a bad idea for open systems!)
        # state = self._balance_global_mass(state)
        
        return state