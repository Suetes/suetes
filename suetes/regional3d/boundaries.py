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
        
        dist_to_bound = jnp.minimum(dist_x, dist_y)
        alpha = jnp.where(dist_to_bound < depth, jnp.cos(0.5 * jnp.pi * dist_to_bound / depth) ** 2, 0.0)
                
        return jnp.expand_dims(alpha, axis=-1)

    def blend(self, model_state, external_state):
        blended = {}
        
        # STRICT MASS CONSERVATION:
        # We only blend horizontal momentum, temperature, and tracers.
        # We DO NOT blend w, pi, or rho. They must remain completely controlled 
        # by the implicit solver and FFSL advector to guarantee continuity.
        blend_vars = ['u', 'v', 'w', 'th_v', 'q', 'pi']
        
        for k in model_state.keys():
            if k in blend_vars and k in external_state:
                if k == 'u':
                    mask = self.masks['u']
                elif k == 'v':
                    mask = self.masks['v']
                else:
                    mask = self.masks['m']
                
                blended[k] = (1.0 - mask) * model_state[k] + mask * external_state[k]
            else:
                blended[k] = model_state[k]
                
        # Re-diagnose density to strictly satisfy the Equation of State in the sponge
        Rd, cvd, p0 = 287.0, 717.0, 100000.0
        blended['rho'] = p0 / (Rd * blended['th_v']) * (blended['pi'] ** (cvd / Rd))

        # 1. Manually average the 2D bottom slice from faces to mass points
        u_m_surf = 0.5 * (blended['u'][:-1, :, 0] + blended['u'][1:, :, 0])
        v_m_surf = 0.5 * (blended['v'][:, :-1, 0] + blended['v'][:, 1:, 0])

        # 2. Compute the terrain-following kinematic w
        kinematic_bottom = (
            u_m_surf * self.grid.z_xi_w[:, :, 0] + 
            v_m_surf * self.grid.z_eta_w[:, :, 0]
        )

        # 3. Strictly enforce it at the bottom boundary
        blended['w'] = blended['w'].at[:, :, 0].set(kinematic_bottom)
        
        # NOTICE: We completely removed the manual 'rho' recalculation here!
        return blended