import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, sponge_depth=30, dt=30.0, tau_bndy=300.0):
        self.grid = grid
        self.depth = sponge_depth
        
        # Calculate the maximum relaxation coefficient (fraction replaced per step)
        # e.g., dt=30s, tau=300s -> max_c = 0.1 (10% blend per step at absolute boundary)
        self.max_c = dt / tau_bndy
        
        # Precompute the three possible lateral staggered shapes
        self.masks = {
            'u': self._compute_mask(grid.nx + 1, grid.ny, sponge_depth),
            'v': self._compute_mask(grid.nx, grid.ny + 1, sponge_depth),
            'm': self._compute_mask(grid.nx, grid.ny, sponge_depth)
        }

    def _compute_mask(self, nx, ny, depth):
        X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        
        dist_x = jnp.minimum(X, nx - 1 - X) if nx > 2 * depth else jnp.full_like(X, 9999)
        dist_y = jnp.minimum(Y, ny - 1 - Y) if ny > 2 * depth else jnp.full_like(Y, 9999)
        
        # Identify the absolute outermost boundary cells
        is_boundary = (dist_x == 0) | (dist_y == 0)
        
        # 1D weights: 1.0 in the free interior, tapering to 0.0 at the absolute boundary
        wx = jnp.where(dist_x < depth, jnp.sin(0.5 * jnp.pi * dist_x / depth) ** 2, 1.0)
        wy = jnp.where(dist_y < depth, jnp.sin(0.5 * jnp.pi * dist_y / depth) ** 2, 1.0)
        
        # Base spatial alpha (0.0 in interior, 1.0 at absolute boundary)
        alpha = 1.0 - (wx * wy)
        
        # Apply strict ERA5 overwrite at the edges, and nudging in the sponge
        scaled_alpha = jnp.where(is_boundary, 1.0, alpha * self.max_c)
                
        return jnp.expand_dims(scaled_alpha, axis=-1)

    def blend(self, intermediate_state, external_state):
        blended = {}
        
        blend_vars = ['u', 'v', 'th_v', 'q']
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                c = self.masks.get(k, self.masks['m'])
                # Gently nudge towards the external state based on the relaxation timescale
                blended[k] = (1.0 - c) * intermediate_state[k] + c * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        return blended