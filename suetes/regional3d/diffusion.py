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
    r"""
    Applies 4th-order Hyperdiffusion to selectively damp the smallest resolvable scales.

    Unlike 2nd-order diffusion, hyperdiffusion strongly preserves physical gradients 
    at larger scales while heavily penalizing $2\Delta x$ numerical noise. The 
    governing continuous equation is:

    $$ \frac{\partial f}{\partial t} = -\nu_h \nabla_h^4 f - \nu_v \frac{\partial^4 f}{\partial z^4} $$
    """
    def __init__(self, grid, nu_h=1e6, nu_v=1e6):
        """
        Initializes the 4th-order hyper-filter.

        Args:
            grid (RegionalGrid3D): The computational grid.
            nu_h (float): Horizontal hyper-viscosity coefficient [$m^4 s^{-1}$].
            nu_v (float): Vertical hyper-viscosity coefficient [$m^4 s^{-1}$].
        """
        self.grid = grid
        self.nu_h = nu_h 
        self.nu_v = nu_v

    def _laplacian_comp(self, f, axis, ds):
        """
        Computes the discrete 1D Laplacian using central differences.

        $$ \frac{\partial^2 f}{\partial x^2} \approx \frac{f_{i+1} - 2f_i + f_{i-1}}{\Delta x^2} $$
        """
        pad_width = [(0, 0), (0, 0), (0, 0)]
        pad_width[axis] = (1, 1)
        f_pad = jnp.pad(f, pad_width, mode='edge')
        
        if axis == 0:
            return (f_pad[2:, :, :] - 2.0 * f + f_pad[:-2, :, :]) / (ds**2)
        elif axis == 1:
            return (f_pad[:, 2:, :] - 2.0 * f + f_pad[:, :-2, :]) / (ds**2)
        else:
            return (f_pad[:, :, 2:] - 2.0 * f + f_pad[:, :, :-2]) / (ds**2)

    def get_tendencies(self, state_prime, bg_precomputed=None):
        r"""
        Computes the hyperdiffusion tendencies for momentum and thermodynamics.

        To accurately compute $\nabla^4 f$, the Laplacian operator is applied 
        iteratively: $\nabla^4 f = \nabla^2(\nabla^2 f)$.

        Args:
            state_prime (dict): The prognostic state variables.
            bg_precomputed (dict, optional): Background state for thermodynamic variables.
                Defaults to None.

        Returns:
            dict: Dictionary of hyperdiffusion tendencies.
        """
        diff_tends = {}
        for k in ['u', 'v', 'w', 'th_v']:
            if k in state_prime:
                if k == 'th_v' and bg_precomputed is not None:
                    f = state_prime['th_v'] - bg_precomputed['th_v']
                else:
                    f = state_prime[k]
                
                # Compute Horizontal Hyperdiffusion
                lap_x = self._laplacian_comp(f, axis=0, ds=self.grid.dx)
                hyper_x = -self.nu_h * self._laplacian_comp(lap_x, axis=0, ds=self.grid.dx)
                
                lap_y = self._laplacian_comp(f, axis=1, ds=self.grid.dy)
                hyper_y = -self.nu_h * self._laplacian_comp(lap_y, axis=1, ds=self.grid.dy)
                
                # Compute Vertical Hyperdiffusion
                lap_z = self._laplacian_comp(f, axis=2, ds=self.grid.dz)
                hyper_z = -self.nu_v * self._laplacian_comp(lap_z, axis=2, ds=self.grid.dz)
                
                diff_tends[k] = hyper_x + hyper_y + hyper_z
                
        return diff_tends