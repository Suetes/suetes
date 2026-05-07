import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, sponge_depth=30, dt=30.0, tau_bndy_factor=10.0):
        self.grid = grid
        self.depth = sponge_depth
        
        # Automatically scale the relaxation timescale with the timestep
        tau_bndy = tau_bndy_factor * dt
        
        # Calculate the maximum relaxation coefficient
        # If tau_bndy_factor is 10.0, max_c is strictly bounded to 0.1
        self.max_c = dt / tau_bndy
        
        # Precompute the three possible lateral staggered shapes
        self.masks = {
            'u': self._compute_mask(grid.nx + 1, grid.ny, sponge_depth),
            'v': self._compute_mask(grid.nx, grid.ny + 1, sponge_depth),
            'm': self._compute_mask(grid.nx, grid.ny, sponge_depth)
        }

    def _compute_mask(self, nx, ny, depth):
        X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        
        # Calculate distance to the closest X and Y boundaries
        dist_x = jnp.minimum(X, nx - 1 - X) if nx > 2 * depth else jnp.full_like(X, 9999.0)
        dist_y = jnp.minimum(Y, ny - 1 - Y) if ny > 2 * depth else jnp.full_like(Y, 9999.0)
        
        # Find the absolute shortest distance to any domain edge
        dist_to_bound = jnp.minimum(dist_x, dist_y)
        
        # Create a smooth cosine taper: 
        # Equals 1.0 at the absolute boundary (dist == 0)
        # Tapers smoothly to 0.0 at the interior edge (dist >= depth)
        spatial_weight = jnp.where(
            dist_to_bound < depth, 
            jnp.cos(0.5 * jnp.pi * dist_to_bound / depth) ** 2, 
            0.0
        )
        
        # Scale the continuous profile by the maximum stable nudging factor
        # No more jnp.where() cliffs! The outermost cell gets max_c, tapering to 0.
        scaled_alpha = spatial_weight * self.max_c
                
        return jnp.expand_dims(scaled_alpha, axis=-1)

    def blend(self, intermediate_state, external_state):
        blended = {}
        
        # 'w' dampens the vertical acoustic jets.
        # 'pi' provides the synoptic pressure gradient to maintain geostrophic balance.
        blend_vars = ['u', 'v', 'w', 'th_v', 'q', 'pi'] 
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                c = self.masks.get(k, self.masks['m'])
                blended[k] = (1.0 - c) * intermediate_state[k] + c * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        return blended