import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, sponge_depth=10):
        self.grid = grid
        self.depth = sponge_depth
        self.masks = {
            'u': self._compute_mask(grid.nx + 1, grid.ny, sponge_depth),
            'v': self._compute_mask(grid.nx, grid.ny + 1, sponge_depth),
            'w': self._compute_mask(grid.nx, grid.ny, sponge_depth),
            'pi': self._compute_mask(grid.nx, grid.ny, sponge_depth),
            'rho': self._compute_mask(grid.nx, grid.ny, sponge_depth),
            'th_v': self._compute_mask(grid.nx, grid.ny, sponge_depth),
            'eta_dot': self._compute_mask(grid.nx, grid.ny, sponge_depth)
        }

    def _compute_mask(self, nx, ny, depth):
        X, Y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
        dist_to_bound = jnp.minimum(jnp.minimum(X, nx - 1 - X), jnp.minimum(Y, ny - 1 - Y))
        alpha = jnp.where(dist_to_bound < depth, jnp.cos(0.5 * jnp.pi * dist_to_bound / depth) ** 2, 0.0)
        return jnp.expand_dims(alpha, axis=-1)

    def blend(self, model_state, external_state):
        def apply_relaxation(mod_val, ext_val, mask):
            return (1.0 - mask) * mod_val + mask * ext_val

        return {k: apply_relaxation(model_state[k], external_state[k], self.masks[k]) for k in model_state.keys()}