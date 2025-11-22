import jax
import jax.numpy as jnp
from .transforms import GalChenSigma

class StaggeredGrid:
    def __init__(self, nx, nz, Lx, Lz, h_func, transform=None):
        self.nx, self.nz = nx, nz
        self.Lx, self.Lz = Lx, Lz
        self.dx = Lx / nx
        self.dz = Lz / nz
        self.h_func = h_func

        if transform is None:
            self.transform_op = GalChenSigma()
        else:
            self.transform_op = transform
        
        # 1. Logical Grids (Flat 0..L)
        # used as input for the coordinate transform z = T(x, z_flat)
        xi_c = jnp.linspace(0, Lx, nx+1)
        zeta_c = jnp.linspace(0, Lz, nz+1)
        
        # 2. Define Logical Centers/Edges
        # Mass Points (Centers)
        xi_m = 0.5 * (xi_c[:-1] + xi_c[1:])
        zeta_m = 0.5 * (zeta_c[:-1] + zeta_c[1:])
        Xi_m, Zeta_m = jnp.meshgrid(xi_m, zeta_m, indexing='ij')
        
        # U Points (Faces in X)
        xi_u = xi_c # 0..nx (nx+1 points)
        Xi_u, Zeta_u = jnp.meshgrid(xi_u, zeta_m, indexing='ij')
        
        # W Points (Faces in Z)
        zeta_w = zeta_c
        Xi_w, Zeta_w = jnp.meshgrid(xi_m, zeta_w, indexing='ij')
        
        # 3. Generate Physical Grids (Using Transform)
        # We store these for plotting/physics
        self.X_m, self.Z_m = Xi_m, self._apply_transform(Xi_m, Zeta_m)
        self.X_u, self.Z_u = Xi_u, self._apply_transform(Xi_u, Zeta_u)
        self.X_w, self.Z_w = Xi_w, self._apply_transform(Xi_w, Zeta_w)
        
        # Corner grid (for completeness)
        Xi_c_grid, Zeta_c_grid = jnp.meshgrid(xi_c, zeta_c, indexing='ij')
        self.X_corner = Xi_c_grid
        self.Z_corner = self._apply_transform(Xi_c_grid, Zeta_c_grid)

        # 4. Compute Analytic Metrics (The Fix)
        self.metrics = {}
        self.metrics['m'] = self._compute_analytic_metrics(Xi_m, Zeta_m)
        self.metrics['u'] = self._compute_analytic_metrics(Xi_u, Zeta_u)
        self.metrics['w'] = self._compute_analytic_metrics(Xi_w, Zeta_w)

    def _apply_transform(self, xi, zeta):
        h = self.h_func(xi)
        return self.transform_op(xi, zeta, h, self.Lz)

    def _compute_analytic_metrics(self, Xi, Zeta):
        """
        Computes z_xi and z_zeta using exact JAX autodiff of the transform.
        This eliminates discretization errors in the metric terms.
        """
        # Wrap transform to be scalar-valued z(xi, zeta) for autodiff
        def _z_scalar(xi, zeta):
            h = self.h_func(xi)
            return self.transform_op(xi, zeta, h, self.Lz)
        
        # Vectorized gradients: scalar -> scalar
        grad_x_fn = jax.vmap(jax.vmap(jax.grad(_z_scalar, argnums=0)))
        grad_z_fn = jax.vmap(jax.vmap(jax.grad(_z_scalar, argnums=1)))
        
        z_xi = grad_x_fn(Xi, Zeta)
        z_zeta = grad_z_fn(Xi, Zeta)
        
        return {'z_xi': z_xi, 'z_zeta': z_zeta}