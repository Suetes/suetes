"""
Grid Diffusion and Stabilization Module.

Provides spatial filters to damp grid-scale noise, prevent non-linear instability 
(aliasing), and parameterize subgrid-scale turbulent mixing.
"""

import jax.numpy as jnp

class SpatialFilter:
    r"""
    Applies a 2nd-order Laplacian diffusion to damp grid-scale noise.
    
    This filter acts as a simple constant-coefficient subgrid-scale (SGS) 
    turbulence parameterization, generating tendencies of the form:

    $$ \frac{\partial f}{\partial t} = \nu_h \nabla_h^2 f + \nu_v \frac{\partial^2 f}{\partial z^2} $$
    """
    def __init__(self, grid, nu_h=75.0, nu_v=75.0):
        """
        Initializes the 2nd-order spatial filter.

        Args:
            grid (RegionalGrid3D): The computational grid.
            nu_h (float): Horizontal eddy viscosity coefficient [$m^2 s^{-1}$].
            nu_v (float): Vertical eddy viscosity coefficient [$m^2 s^{-1}$].
        """
        self.grid = grid
        self.nu_h = nu_h 
        self.nu_v = nu_v

    def _laplacian_1d(self, f, axis, dx):
        r"""
        Computes the discrete 1D Laplacian using central differences.

        $$ \frac{\partial^2 f}{\partial x^2} \approx \frac{f_{i+1} - 2f_i + f_{i-1}}{\Delta x^2} $$
        """
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[axis] = (1, 1)
        f_pad = jnp.pad(f, pad_width, mode='edge')
        
        if axis == 0:
            d2f = f_pad[2:, :, :] - 2.0 * f + f_pad[:-2, :, :]
        elif axis == 1:
            d2f = f_pad[:, 2:, :] - 2.0 * f + f_pad[:, :-2, :]
        else:
            d2f = f_pad[:, :, 2:] - 2.0 * f + f_pad[:, :, :-2]
            
        return d2f / (dx**2)

    def get_diffusion_tendencies(self, state_prime):
        """
        Calculates the diffusion tendencies for all prognostic variables.

        Args:
            state_prime (dict): The prognostic state variables.

        Returns:
            dict: Dictionary of diffusion tendencies.
        """
        diff_tends = {}
        
        # We diffuse momentum and thermodynamics, but NOT pressure.
        for k in ['u', 'v', 'w', 'th_v']:
            if k in state_prime:
                f = state_prime[k]
                
                # Horizontal Diffusion
                diff_x = self.nu_h * self._laplacian_1d(f, axis=0, dx=self.grid.dx)
                diff_y = self.nu_h * self._laplacian_1d(f, axis=1, dx=self.grid.dy)
                
                # Vertical Diffusion (Using logical dz for computational stability)
                diff_z = self.nu_v * self._laplacian_1d(f, axis=2, dx=self.grid.dz)
                
                diff_tends[k] = diff_x + diff_y + diff_z
                
        return diff_tends

class HyperFilter:
    """
    Applies 4th-order Hyperdiffusion using true Cartesian gradients.
    
    Corrects for terrain-following coordinates by incorporating grid metric 
    terms, preventing spurious diffusion along steep topography.
    """
    def __init__(self, grid, operators, nu_h=1e6, nu_v=1e6):
        self.grid = grid
        self.op = operators
        self.nu_h = nu_h 
        self.nu_v = nu_v

    def _cartesian_horizontal_laplacian(self, f_m, bg):
        # True horizontal gradient in X (at u-faces)
        df_dxi_u = self.op.diff(f_m, axis=0, from_loc='m', to_loc='u') / self.grid.dx
        df_dz_m = self.op.avg(
            self.op.diff(f_m, axis=2, from_loc='m', to_loc='w') / bg['dz_w_full'], 
            axis=2, from_loc='w', to_loc='m'
        )
        df_dz_u = self.op.avg(df_dz_m, axis=0, from_loc='m', to_loc='u')
        z_xi_u = self.op.diff(self.grid.Z_m, axis=0, from_loc='m', to_loc='u') / self.grid.dx
        
        Gx_u = df_dxi_u - z_xi_u * df_dz_u

        # True horizontal gradient in Y (at v-faces)
        df_deta_v = self.op.diff(f_m, axis=1, from_loc='m', to_loc='v') / self.grid.dy
        df_dz_v = self.op.avg(df_dz_m, axis=1, from_loc='m', to_loc='v')
        z_eta_v = self.op.diff(self.grid.Z_m, axis=1, from_loc='m', to_loc='v') / self.grid.dy
        
        Gy_v = df_deta_v - z_eta_v * df_dz_v

        # Divergence of the Cartesian gradients (back to mass points)
        dGx_dxi_m = self.op.diff(Gx_u, axis=0, from_loc='u', to_loc='m') / self.grid.dx
        Gx_m = self.op.avg(Gx_u, axis=0, from_loc='u', to_loc='m')
        dGx_dz_w = self.op.diff(Gx_m, axis=2, from_loc='m', to_loc='w') / bg['dz_w_full']
        dGx_dz_m = self.op.avg(dGx_dz_w, axis=2, from_loc='w', to_loc='m')
        z_xi_m = self.op.avg(z_xi_u, axis=0, from_loc='u', to_loc='m')
        
        div_Gx = dGx_dxi_m - z_xi_m * dGx_dz_m

        dGy_deta_m = self.op.diff(Gy_v, axis=1, from_loc='v', to_loc='m') / self.grid.dy
        Gy_m = self.op.avg(Gy_v, axis=1, from_loc='v', to_loc='m')
        dGy_dz_w = self.op.diff(Gy_m, axis=2, from_loc='m', to_loc='w') / bg['dz_w_full']
        dGy_dz_m = self.op.avg(dGy_dz_w, axis=2, from_loc='w', to_loc='m')
        z_eta_m = self.op.avg(z_eta_v, axis=1, from_loc='v', to_loc='m')

        div_Gy = dGy_deta_m - z_eta_m * dGy_dz_m

        return div_Gx + div_Gy

    def _vertical_laplacian(self, f_m, bg):
        # Pure vertical diffusion
        df_dz_w = self.op.diff(f_m, axis=2, from_loc='m', to_loc='w') / bg['dz_w_full']
        return self.op.diff(df_dz_w, axis=2, from_loc='w', to_loc='m') / bg['dz_m_full']

    def get_tendencies(self, state_prime, bg_precomputed):
        diff_tends = {}
        for k in ['u', 'v', 'w', 'th_v']:
            if k in state_prime:
                # Map field to mass points for stable tensor math
                if k == 'u':
                    f_m = self.op.avg(state_prime['u'], axis=0, from_loc='u', to_loc='m')
                elif k == 'v':
                    f_m = self.op.avg(state_prime['v'], axis=1, from_loc='v', to_loc='m')
                elif k == 'w':
                    f_m = self.op.avg(state_prime['w'], axis=2, from_loc='w', to_loc='m')
                else: 
                    f_m = state_prime['th_v'] - bg_precomputed['th_v']

                # 1st Laplacian
                lap1_h = self._cartesian_horizontal_laplacian(f_m, bg_precomputed)
                lap1_v = self._vertical_laplacian(f_m, bg_precomputed)

                # 2nd Laplacian (Hyperdiffusion)
                hyper_h = -self.nu_h * self._cartesian_horizontal_laplacian(lap1_h, bg_precomputed)
                hyper_v = -self.nu_v * self._vertical_laplacian(lap1_v, bg_precomputed)

                tend_m = hyper_h + hyper_v

                # Stagger back to appropriate face
                if k == 'u':
                    diff_tends['u'] = self.op.avg(tend_m, axis=0, from_loc='m', to_loc='u')
                elif k == 'v':
                    diff_tends['v'] = self.op.avg(tend_m, axis=1, from_loc='m', to_loc='v')
                elif k == 'w':
                    diff_tends['w'] = self.op.avg(tend_m, axis=2, from_loc='m', to_loc='w')
                else:
                    diff_tends['th_v'] = tend_m
                    
        return diff_tends