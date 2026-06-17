"""
Lateral Boundary Conditions Module.

Implements relaxation zones (sponge layers) near the lateral boundaries of the 
regional domain. These layers smoothly blend internal prognostic variables with 
external large-scale forcing (e.g., from global models like ERA5) and absorb 
outgoing gravity and acoustic waves to prevent non-physical domain reflections.
"""

import jax
import jax.numpy as jnp

class DaviesSponge:
    r"""
    Davies-style lateral boundary relaxation sponge.

    Applies a spatial relaxation toward a specified external state over a boundary 
    zone of width D. Employs a cosine-squared weighting profile and dynamic 
    outflow scaling to minimize spurious wave reflection at the domain edges.
    """
    def __init__(self, grid, operators, sponge_depth=30, dt=30.0, tau_bndy_factor=10.0, outflow_factor=0.01):
        self.grid = grid
        self.op = operators 
        self.depth = sponge_depth
        self.outflow_factor = outflow_factor 
        
        tau_bndy = tau_bndy_factor * dt
        self.max_c = dt / tau_bndy
        
        self.masks = {
            'u': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='u'),
            'v': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='v'),
            'm': self._compute_directional_masks(grid.nx, grid.ny, sponge_depth, loc='m')
        }

    def _compute_directional_masks(self, nx, ny, depth, loc='m'):
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
        
        weight_west = jnp.cos(0.5 * jnp.pi * dist_west) ** 2
        weight_east = jnp.cos(0.5 * jnp.pi * dist_east) ** 2
        weight_south = jnp.cos(0.5 * jnp.pi * dist_south) ** 2
        weight_north = jnp.cos(0.5 * jnp.pi * dist_north) ** 2
        
        return {
            'west': jnp.expand_dims(jnp.expand_dims(weight_west, axis=-1), axis=-1),
            'east': jnp.expand_dims(jnp.expand_dims(weight_east, axis=-1), axis=-1),
            'south': jnp.expand_dims(jnp.expand_dims(weight_south, axis=-1), axis=0),
            'north': jnp.expand_dims(jnp.expand_dims(weight_north, axis=-1), axis=0)
        }

    def get_interior_mask(self):
        mask = {}
        for loc in ('u', 'v', 'm'):
            masks_loc = self.masks[loc]
            combined = jnp.maximum(
                jnp.maximum(masks_loc['west'], masks_loc['east']),
                jnp.maximum(masks_loc['south'], masks_loc['north']),
            )
            interior = 1.0 - combined
            
            if loc == 'u':
                mask['u'] = interior
            elif loc == 'v':
                mask['v'] = interior
            else:
                mask['th_v'] = interior
                mask['w'] = interior
                
        return mask

    def blend(self, intermediate_state, external_state):
        blended = {}
        blend_vars = ['u', 'v', 'th_v', 'q', 'pi']
        
        u_bndy = intermediate_state['u']
        v_bndy = intermediate_state['v']
        
        for k in intermediate_state.keys():
            if k in blend_vars and k in external_state:
                masks = self.masks.get(k, self.masks['m'])
                
                if k == 'u':
                    u_eval = u_bndy
                    v_m = self.op.avg(v_bndy, axis=1, from_loc='v', to_loc='m')
                    v_eval = self.op.avg(v_m, axis=0, from_loc='m', to_loc='u')
                elif k == 'v':
                    u_m = self.op.avg(u_bndy, axis=0, from_loc='u', to_loc='m')
                    u_eval = self.op.avg(u_m, axis=1, from_loc='m', to_loc='v')
                    v_eval = v_bndy
                else: 
                    u_eval = self.op.avg(u_bndy, axis=0, from_loc='u', to_loc='m')
                    v_eval = self.op.avg(v_bndy, axis=1, from_loc='v', to_loc='m')

                # Stop the gradients on the directional conditions
                u_dir = jax.lax.stop_gradient(u_eval)
                v_dir = jax.lax.stop_gradient(v_eval)

                w_west  = masks['west']  * jnp.where(u_dir > 0, 1.0, self.outflow_factor)
                w_east  = masks['east']  * jnp.where(u_dir < 0, 1.0, self.outflow_factor)
                w_south = masks['south'] * jnp.where(v_dir > 0, 1.0, self.outflow_factor)
                w_north = masks['north'] * jnp.where(v_dir < 0, 1.0, self.outflow_factor)

                c_total = jnp.maximum(jnp.maximum(w_west, w_east), jnp.maximum(w_south, w_north))
                
                blended[k] = (1.0 - c_total) * intermediate_state[k] + c_total * external_state[k]
            else:
                blended[k] = intermediate_state[k]
                
        return blended

class BenchmarkXSponge:
    """
    Simplified 1D relaxation sponge for pseudo-2D benchmark cases.
    """
    def __init__(self, nx, sponge_depth=10):
        self.nx = nx
        self.sponge_depth = sponge_depth
        
        x_idx = jnp.arange(nx, dtype=jnp.float32)
        dist_x = jnp.minimum(x_idx, nx - x_idx)
        weight_x = jnp.where(dist_x < sponge_depth, 
                             jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
        self.mask_m = weight_x[:, None, None]
        self.mask_u = jnp.pad(self.mask_m, ((0, 1), (0, 0), (0, 0)), mode='edge')

    def blend(self, state_in, ext_state):
        blended = {}
        blend_vars = ['u', 'v', 'th_v', 'pi', 'rho', 'q_tr'] 
        
        for k in state_in.keys():
            if k in blend_vars and k in ext_state:
                m = self.mask_u if k == 'u' else self.mask_m
                blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
            else:
                blended[k] = state_in[k]
        return blended