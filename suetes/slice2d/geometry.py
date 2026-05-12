import jax
import jax.numpy as jnp

from suetes.shared.transforms import GalChenSigma

class StaggeredGrid:
    r"""
    Constructs a 2D (X-Z) Arakawa C-grid with a terrain-following vertical coordinate.
    
    In an Arakawa C-grid, thermodynamic scalars and density (mass points, 'm') are 
    located at the cell centers, while the velocity components are staggered onto 
    the cell faces:
    - $u$ resides on the vertical faces (left/right).
    - $w$ resides on the horizontal faces (top/bottom).
    
    The physical coordinates $(x, z)$ are mapped to a rectangular computational 
    domain $(\xi, \zeta)$ via a vertical transformation $z = H(\xi, \zeta)$, 
    such as the standard Gal-Chen & Somerville (1975) $\sigma_z$ mapping:
    $$ z = h(x) + \zeta \frac{L_z - h(x)}{L_z} $$
    where $h(x)$ is the underlying topography.
    """
    def __init__(self, nx, nz, Lx, Lz, h_func, transform=None):
        r"""
        Initializes the grid domains, coordinate arrays, and topological limits.
        
        Parameters:
            nx (int): Number of grid cells in the horizontal X-direction.
            nz (int): Number of grid cells in the vertical Z-direction.
            Lx (float): Total length of the domain in meters.
            Lz (float): Total height of the domain in meters.
            h_func (Callable): Function returning the terrain height $h(x)$.
            transform (Callable, optional): The vertical coordinate mapping function.
        """
        self.nx, self.nz = nx, nz
        self.Lx, self.Lz = Lx, Lz
        self.dx = Lx / nx
        self.dz = Lz / nz
        self.h_func = h_func

        if transform is None:
            self.transform_op = GalChenSigma()
        else:
            self.transform_op = transform
        
        xi_c = jnp.linspace(0, Lx, nx+1)
        zeta_c = jnp.linspace(0, Lz, nz+1)
        
        xi_m = 0.5 * (xi_c[:-1] + xi_c[1:])
        zeta_m = 0.5 * (zeta_c[:-1] + zeta_c[1:])
        Xi_m, Zeta_m = jnp.meshgrid(xi_m, zeta_m, indexing='ij')
        
        xi_u = xi_c
        Xi_u, Zeta_u = jnp.meshgrid(xi_u, zeta_m, indexing='ij')
        
        zeta_w = zeta_c
        Xi_w, Zeta_w = jnp.meshgrid(xi_m, zeta_w, indexing='ij')
        
        self.X_m, self.Z_m = Xi_m, self._apply_transform(Xi_m, Zeta_m)
        self.X_u, self.Z_u = Xi_u, self._apply_transform(Xi_u, Zeta_u)
        self.X_w, self.Z_w = Xi_w, self._apply_transform(Xi_w, Zeta_w)
        
        Xi_c_grid, Zeta_c_grid = jnp.meshgrid(xi_c, zeta_c, indexing='ij')
        self.X_corner = Xi_c_grid
        self.Z_corner = self._apply_transform(Xi_c_grid, Zeta_c_grid)

        dz_min = float(jnp.min(self.Z_w[:, 1:] - self.Z_w[:, :-1]))
        if dz_min <= 0.0:
            raise ValueError(
                f"Grid Tangling Detected! Minimum dz is {dz_min:.2f} m.\n"
                f"The topography is too steep for the current vertical resolution "
                f"and coordinate transform. Please smooth the terrain or increase Lz."
            )

    def _apply_transform(self, xi, zeta):
        r"""Applies the vertical coordinate transformation mapping $(\xi, \zeta) \to z$."""
        h = self.h_func(xi)
        return self.transform_op(xi, zeta, h, self.Lz)

    def _compute_discrete_metrics(Z_grid, dx, dz, periodic_x=False):
        r"""
        Calculates the discrete grid metrics necessary for the covariant/contravariant 
        tensor transformations on the curvilinear mesh.
        
        Calculates the metric terms:
        $$ z_\xi = \frac{\partial z}{\partial \xi} \quad \text{and} \quad z_\zeta = \frac{\partial z}{\partial \zeta} $$
        """
        Z_pad_z = jnp.pad(Z_grid, ((0, 0), (1, 1)), mode='edge')
        z_zeta = (Z_pad_z[:, 2:] - Z_pad_z[:, :-2]) / (2.0 * dz)

        if periodic_x:
            z_xi = (jnp.roll(Z_grid, -1, axis=0) - jnp.roll(Z_grid, 1, axis=0)) / (2.0 * dx)
        else:
            Z_pad_x = jnp.pad(Z_grid, ((1, 1), (0, 0)), mode='edge')
            z_xi = (Z_pad_x[2:, :] - Z_pad_x[:-2, :]) / (2.0 * dx)
            
        return {'z_xi': z_xi, 'z_zeta': z_zeta}