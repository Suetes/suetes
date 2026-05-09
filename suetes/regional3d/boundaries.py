import jax.numpy as jnp

class DaviesSponge:
    def __init__(self, grid, operators, sponge_depth=30, dt=30.0, tau_bndy_factor=10.0, outflow_factor=0.01):
        self.grid = grid
        self.op = operators # Save the operators here
        self.depth = sponge_depth
        self.outflow_factor = outflow_factor # How much to relax during outflow (0.01 = very weak)
        
        tau_bndy = tau_bndy_factor * dt
        self.max_c = dt / tau_bndy
        
        # Precompute the isolated directional masks for each variable staggering
        self.masks = {
            'u': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='u'),
            'v': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='v'),
            'm': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='m')
        }

    def _compute_directional_masks(self, nx, ny, depth, loc='m'):
        """Computes four separate masks (West, East, South, North)."""
        nx_pts = nx + 1 if loc == 'u' else nx
        ny_pts = ny + 1 if loc == 'v' else ny
        
        idx_x = jnp.arange(nx_pts, dtype=jnp.float32)
        idx_y = jnp.arange(ny_pts, dtype=jnp.float32)
        
        offset_x = 0.0 if loc == 'u' else 0.5
        offset_y = 0.0 if loc == 'v' else 0.5
        
        dist_west = jnp.clip((idx_x + offset_x) / depth, 0.0, 1.0)
        dist_east = jnp.clip((nx - (idx_x + offset_x)) / depth, 0.0, 1.0)
        dist_south = jnp.clip((idx_y + offset_y) / depth, 0.0, 1.0)
        dist_north = jnp.clip((ny - (idx_y + offset_y)) / depth, 0.0, 1.0)
        
        # Cosine-squared profiles (1.0 at boundary, 0.0 at interior)
        weight_west = jnp.cos(0.5 * jnp.pi * dist_west) ** 2
        weight_east = jnp.cos(0.5 * jnp.pi * dist_east) ** 2
        weight_south = jnp.cos(0.5 * jnp.pi * dist_south) ** 2
        weight_north = jnp.cos(0.5 * jnp.pi * dist_north) ** 2
        
        # Expand dims to 3D for broadcasting (X, Y, Z)
        return {
            'west': jnp.expand_dims(jnp.expand_dims(weight_west, axis=-1), axis=-1),
            'east': jnp.expand_dims(jnp.expand_dims(weight_east, axis=-1), axis=-1),
            'south': jnp.expand_dims(jnp.expand_dims(weight_south, axis=-1), axis=0),
            'north': jnp.expand_dims(jnp.expand_dims(weight_north, axis=-1), axis=0)
        }

    def blend(self, intermediate_state, external_state):
        blended = {}
        # Strictly exclude w and eta_dot to preserve the local kinematic boundary condition
        blend_vars = ['u', 'v', 'th_v', 'q', 'pi']
        
        u_bndy = intermediate_state['u']
        v_bndy = intermediate_state['v']
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                masks = self.masks.get(k, self.masks['m'])
                
                # Accurately map the winds to the specific grid staggering of variable 'k'
                if k == 'u':
                    u_eval = u_bndy
                    # Map v -> m, then m -> u (pads the x-axis)
                    v_m = self.op.avg(v_bndy, axis=1, from_loc='v', to_loc='m')
                    v_eval = self.op.avg(v_m, axis=0, from_loc='m', to_loc='u')
                elif k == 'v':
                    # Map u -> m, then m -> v (pads the y-axis)
                    u_m = self.op.avg(u_bndy, axis=0, from_loc='u', to_loc='m')
                    u_eval = self.op.avg(u_m, axis=1, from_loc='m', to_loc='v')
                    v_eval = v_bndy
                else: 
                    # 'th_v', 'q', 'pi' are all on the mass ('m') grid
                    u_eval = self.op.avg(u_bndy, axis=0, from_loc='u', to_loc='m')
                    v_eval = self.op.avg(v_bndy, axis=1, from_loc='v', to_loc='m')

                # Apply the outflow factor dynamically based on local wind direction
                w_west  = masks['west']  * jnp.where(u_eval > 0, 1.0, self.outflow_factor)
                w_east  = masks['east']  * jnp.where(u_eval < 0, 1.0, self.outflow_factor)
                w_south = masks['south'] * jnp.where(v_eval > 0, 1.0, self.outflow_factor)
                w_north = masks['north'] * jnp.where(v_eval < 0, 1.0, self.outflow_factor)

                # Combine masks
                c_total = jnp.maximum(jnp.maximum(w_west, w_east), jnp.maximum(w_south, w_north))
                
                blended[k] = (1.0 - c_total) * intermediate_state[k] + c_total * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        return blended