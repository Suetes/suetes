"""
Lateral Boundary Conditions Module.

Implements relaxation zones (sponge layers) near the lateral boundaries of the 
regional domain. These layers smoothly blend internal prognostic variables with 
external large-scale forcing (e.g., from global models like ERA5) and absorb 
outgoing gravity and acoustic waves to prevent non-physical domain reflections.
"""

import jax.numpy as jnp

class DaviesSponge:
    r"""
    Davies-style lateral boundary relaxation sponge.

    Applies a spatial relaxation toward a specified external state over a boundary 
    zone of width $D$. Employs a cosine-squared weighting profile and dynamic 
    outflow scaling to minimize spurious wave reflection at the domain edges.
    """
    def __init__(self, grid, operators, sponge_depth=30, dt=30.0, tau_bndy_factor=10.0, outflow_factor=0.01):
        r"""
        Initializes the sponge layer masks and relaxation timescales.

        Args:
            grid (RegionalGrid3D): Computational grid geometry.
            operators (CGridOperator3D): Spatial operators for staggered grid alignments.
            sponge_depth (int): Width of the sponge zone in grid cells ($D$).
            dt (float): Integration time step $\Delta t$ [s].
            tau_bndy_factor (float): Boundary relaxation timescale relative to $\Delta t$.
            outflow_factor (float): Scaling factor $\alpha_{out}$ applied to the 
                relaxation weight where the local flow is directed out of the domain.
        """
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
        r"""
        Computes the spatial relaxation weights for the lateral boundaries.

        Evaluates the spatial weight profile $W(d)$ as:

        $$
        W(d) = 
        \begin{cases} 
        \cos^2\left(\frac{\pi}{2} \frac{d}{D}\right) & \text{if } d \le D \\
        0 & \text{if } d > D 
        \end{cases}
        $$

        where $d$ is the perpendicular distance from the boundary in grid cells, 
        and $D$ is the `sponge_depth`.

        Args:
            nx (int): Number of points in the x-direction.
            ny (int): Number of points in the y-direction.
            depth (int): Sponge depth $D$ in grid cells.
            loc (str): Target grid staggering ('m', 'u', or 'v').

        Returns:
            dict: Directional mask arrays ('west', 'east', 'south', 'north') of shape (X, Y, Z).
        """
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

    def get_interior_mask(self):
        """
        Builds interior masks for u, v, w, th_v from the sponge masks.
        Physics tendencies are multiplied by this mask to shut them off inside the sponge.
        """
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
        r"""
        Blends the internally integrated state with the external boundary forcing.

        The final blended state $\phi_{next}$ is computed via:

        $$ \phi_{next} = (1 - C)\phi_{int} + C\phi_{ext} $$

        To minimize spurious reflections, the total relaxation coefficient $C$ 
        dynamically accounts for local inflow and outflow conditions:

        $$ C = \max(W_{inflow}, \alpha_{out} W_{outflow}) $$

        where $\alpha_{out}$ heavily reduces the relaxation strength when the wind 
        vector is directed out of the regional domain.

        Args:
            intermediate_state (dict): Prognostic state predicted by the dynamical core.
            external_state (dict): Target boundary state (e.g., from ERA5).

        Returns:
            dict: The relaxed prognostic state dictionary.
        """
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

class BenchmarkXSponge:
    """
    Simplified 1D relaxation sponge for pseudo-2D benchmark cases.
    
    Applies strict cosine-squared relaxation along the X-axis boundaries without 
    dynamic outflow scaling. Designed for idealized vertical slice experiments 
    (e.g., Schär mountain waves).
    """
    def __init__(self, nx, sponge_depth=10):
        """
        Initializes the 1D sponge layer masks.

        Args:
            nx (int): Number of internal mass points in the x-direction.
            sponge_depth (int): Width of the lateral sponge zone in grid cells.
        """
        self.nx = nx
        self.sponge_depth = sponge_depth
        
        # Precompute the mass-grid mask (m, v, w, th_v, rho, pi, q_tr)
        x_idx = jnp.arange(nx, dtype=jnp.float32)
        dist_x = jnp.minimum(x_idx, nx - x_idx)
        weight_x = jnp.where(dist_x < sponge_depth, 
                             jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
        self.mask_m = weight_x[:, None, None]
        
        # Precompute the u-grid mask (padded for the staggered X grid)
        self.mask_u = jnp.pad(self.mask_m, ((0, 1), (0, 0), (0, 0)), mode='edge')

    def blend(self, state_in, ext_state):
        r"""
        Blends the interior state with the idealized external benchmark state.

        $$ \phi_{next} = (1 - W(x))\phi_{int} + W(x)\phi_{ext} $$

        Args:
            state_in (dict): Prognostic state predicted by the dynamical core.
            ext_state (dict): Analytical or steady-state background conditions.

        Returns:
            dict: The relaxed prognostic state dictionary.
        """
        blended = {}
        # Define variables that should be relaxed (strictly excluding w, eta_dot)
        blend_vars = ['u', 'v', 'th_v', 'pi', 'rho', 'q_tr'] 
        
        for k in state_in.keys():
            if k in blend_vars and k in ext_state:
                m = self.mask_u if k == 'u' else self.mask_m
                blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
            else:
                blended[k] = state_in[k]
        return blended