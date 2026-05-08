import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, sponge_depth=30, dt=30.0, tau_bndy_factor=10.0):
        self.grid = grid
        self.depth = sponge_depth
        
        tau_bndy = tau_bndy_factor * dt
        self.max_c = dt / tau_bndy
        
        # Pass the 'loc' identifier so the mask knows its staggering
        self.masks = {
            'u': self._compute_mask(grid.nx, grid.ny, sponge_depth, loc='u'),
            'v': self._compute_mask(grid.nx, grid.ny, sponge_depth, loc='v'),
            'm': self._compute_mask(grid.nx, grid.ny, sponge_depth, loc='m')
        }

    def _compute_mask(self, nx, ny, depth, loc='m'):
        nx_pts = nx + 1 if loc == 'u' else nx
        ny_pts = ny + 1 if loc == 'v' else ny
        
        X, Y = jnp.meshgrid(jnp.arange(nx_pts, dtype=jnp.float32), 
                            jnp.arange(ny_pts, dtype=jnp.float32), 
                            indexing='ij')
        
        offset_x = 0.0 if loc == 'u' else 0.5
        offset_y = 0.0 if loc == 'v' else 0.5
        
        dist_x = jnp.minimum(X + offset_x, nx - (X + offset_x))
        dist_y = jnp.minimum(Y + offset_y, ny - (Y + offset_y))
        
        # Normalize distance: 0.0 at the boundary, 1.0 at the inner edge of the sponge
        norm_x = jnp.clip(dist_x / depth, 0.0, 1.0)
        norm_y = jnp.clip(dist_y / depth, 0.0, 1.0)
        
        # Pure cosine-squared: exactly 1.0 at the boundary, smoothly reaching 0.0 at the interior
        weight_x = jnp.cos(0.5 * jnp.pi * norm_x) ** 2
        weight_y = jnp.cos(0.5 * jnp.pi * norm_y) ** 2
        
        # Apply 2D corner blending
        spatial_weight = 1.0 - (1.0 - weight_x) * (1.0 - weight_y)
        
        # Could multiply this with self.max_c if you want relaxation instead of overwrite
        scaled_alpha = spatial_weight 
                
        return jnp.expand_dims(scaled_alpha, axis=-1)

    def blend(self, intermediate_state, external_state):
        blended = {}
        blend_vars = ['u', 'v', 'w', 'eta_dot', 'th_v', 'q', 'pi']
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                c = self.masks.get(k, self.masks['m'])
                blended[k] = (1.0 - c) * intermediate_state[k] + c * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        return blended