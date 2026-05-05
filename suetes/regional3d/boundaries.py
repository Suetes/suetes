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
        
        dist_x = jnp.minimum(X, nx - 1 - X) if nx > 2 * depth else jnp.full_like(X, 9999)
        dist_y = jnp.minimum(Y, ny - 1 - Y) if ny > 2 * depth else jnp.full_like(Y, 9999)
        
        # 1D weights: 1.0 in the free interior, tapering to 0.0 at the absolute boundary
        wx = jnp.where(dist_x < depth, jnp.sin(0.5 * jnp.pi * dist_x / depth) ** 2, 1.0)
        wy = jnp.where(dist_y < depth, jnp.sin(0.5 * jnp.pi * dist_y / depth) ** 2, 1.0)
        
        # The blending alpha needs to be 1.0 at the boundary, 0.0 in interior
        # The product of sines ensures the corners are mathematically smooth
        alpha = 1.0 - (wx * wy)
                
        return jnp.expand_dims(alpha, axis=-1)

    def blend(self, intermediate_state, external_state):
        blended = {}
        
        # MASS CONSERVATION:
        # We only blend horizontal momentum, temperature, and tracers.
        # pi, w, and eta_dot are left completely untouched for the implicit solver.
        blend_vars = ['u', 'v', 'th_v', 'q']
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                # Use the mass mask as the default for scalar variables
                mask = self.masks.get(k, self.masks['m'])
                blended[k] = (1.0 - mask) * intermediate_state[k] + mask * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        # The sponge simply relaxes the explicitly advected variables toward the forcing.
        # The Equation of State and kinematic boundaries are dynamically handled by the implicit solver.
        return blended