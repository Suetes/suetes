import jax.numpy as jnp

class SpatialFilter:
    """
    Applies a 2nd-order Laplacian diffusion to damp grid-scale noise.
    Acts as a simple subgrid-scale (SGS) turbulence parameterization.
    """
    def __init__(self, grid, nu_h=75.0, nu_v=75.0):
        self.grid = grid
        # nu_h and nu_v are the eddy viscosity coefficients [m^2 / s]
        self.nu_h = nu_h 
        self.nu_v = nu_v

    def _laplacian_1d(self, f, axis, dx):
        # 2nd order derivative: (f_{i+1} - 2f_i + f_{i-1}) / dx^2
        # We use 'edge' padding to enforce a zero-gradient boundary condition
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


import jax.numpy as jnp

class HyperFilter:
    """
    Applies 4th-order Hyperdiffusion (-nu * grad^4) to stabilize the grid
    while preserving sharp physical gradients.
    """
    def __init__(self, grid, nu_h=1e6, nu_v=1e6):
        self.grid = grid
        # Hyper-viscosity coefficients [m^4 / s]
        # For 125m grid, 1e6 to 1e7 is a good starting range.
        self.nu_h = nu_h 
        self.nu_v = nu_v

    def _laplacian_comp(self, f, axis, ds):
        """Standard 2nd-order Laplacian in computational space."""
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
        diff_tends = {}
        for k in ['u', 'v', 'w', 'th_v']:
            if k in state_prime:
                # If we are diffusing thermodynamics, subtract the background first!
                if k == 'th_v' and bg_precomputed is not None:
                    # Assuming state_prime['th_v'] is the absolute field here
                    f = state_prime['th_v'] - bg_precomputed['th_v']
                else:
                    f = state_prime[k]
                
                # Compute Horizontal Hyperdiffusion: -nu * d4/dx4
                # We iterate the Laplacian twice to get the 4th derivative
                lap_x = self._laplacian_comp(f, axis=0, ds=self.grid.dx)
                hyper_x = -self.nu_h * self._laplacian_comp(lap_x, axis=0, ds=self.grid.dx)
                
                lap_y = self._laplacian_comp(f, axis=1, ds=self.grid.dy)
                hyper_y = -self.nu_h * self._laplacian_comp(lap_y, axis=1, ds=self.grid.dy)
                
                # Compute Vertical Hyperdiffusion
                lap_z = self._laplacian_comp(f, axis=2, ds=self.grid.dz)
                hyper_z = -self.nu_v * self._laplacian_comp(lap_z, axis=2, ds=self.grid.dz)
                
                diff_tends[k] = hyper_x + hyper_y + hyper_z
                
        return diff_tends