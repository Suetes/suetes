import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, sponge_depth=10):
        self.grid = grid
        self.depth = sponge_depth
        
        # Precompute the three possible lateral staggered shapes
        self.masks = {
            'u': self._compute_mask(grid.nx + 1, grid.ny, sponge_depth),
            'v': self._compute_mask(grid.nx, grid.ny + 1, sponge_depth),
            'm': self._compute_mask(grid.nx, grid.ny, sponge_depth)
        }

    def _compute_mask(self, nx, ny, depth):
        X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        
        # Only apply the sponge to an axis if the domain is wide enough to support it
        dist_x = jnp.minimum(X, nx - 1 - X) if nx > 2 * depth else jnp.full_like(X, 9999)
        dist_y = jnp.minimum(Y, ny - 1 - Y) if ny > 2 * depth else jnp.full_like(Y, 9999)
        
        dist_to_bound = jnp.minimum(dist_x, dist_y)
        alpha = jnp.where(dist_to_bound < depth, jnp.cos(0.5 * jnp.pi * dist_to_bound / depth) ** 2, 0.0)
        
        # Add Z-dimension for 3D broadcasting
        return jnp.expand_dims(alpha, axis=-1)

    def blend(self, model_state, external_state):
        def apply_relaxation(mod_val, ext_val, mask):
            return (1.0 - mask) * mod_val + mask * ext_val

        blended = {}
        for k in model_state.keys():
            # Assign the correct staggered mask
            if k == 'u':
                mask = self.masks['u']
            elif k == 'v':
                mask = self.masks['v']
            else:
                # w, pi, rho, th_v, eta_dot, and ALL tracers use the mass-grid laterally
                mask = self.masks['m']
            
            # Apply relaxation if the external boundary state has this variable
            if k in external_state:
                blended[k] = apply_relaxation(model_state[k], external_state[k], mask)
            else:
                blended[k] = model_state[k]
                
        return blended