import jax.numpy as jnp

class StaggeredGrid:
    def __init__(self, nx, nz, Lx, Lz, h_func):
        self.nx, self.nz = nx, nz
        self.Lx, self.Lz = Lx, Lz
        self.dx = Lx / nx
        self.dz = Lz / nz
        self.h_func = h_func
        
        # 1. Master Grid (Corners) - Pure JAX
        xi_c = jnp.linspace(0, Lx, nx+1)
        zeta_c = jnp.linspace(0, Lz, nz+1)
        Xi_c, Zeta_c = jnp.meshgrid(xi_c, zeta_c, indexing='ij')
        
        self.X_corner = Xi_c
        self.Z_corner = self._transform(Xi_c, Zeta_c)
        
        # 2. Derived Grids (Averaging)
        self.X_u = 0.5 * (self.X_corner[:, :-1] + self.X_corner[:, 1:])
        self.Z_u = 0.5 * (self.Z_corner[:, :-1] + self.Z_corner[:, 1:])
        
        self.X_w = 0.5 * (self.X_corner[:-1, :] + self.X_corner[1:, :])
        self.Z_w = 0.5 * (self.Z_corner[:-1, :] + self.Z_corner[1:, :])
        
        self.X_m = 0.5 * (self.X_w[:, :-1] + self.X_w[:, 1:])
        self.Z_m = 0.5 * (self.Z_w[:, :-1] + self.Z_w[:, 1:])
        
        # 3. Metrics
        self.metrics = {}
        self.metrics['m'] = self._compute_metrics(self.Z_m, self.dx, self.dz)
        self.metrics['u'] = self._compute_metrics(self.Z_u, self.dx, self.dz)
        self.metrics['w'] = self._compute_metrics(self.Z_w, self.dx, self.dz)

    def _transform(self, xi, zeta):
        h = self.h_func(xi)
        return h + zeta * (self.Lz - h) / self.Lz

    def _compute_metrics(self, Z, dx, dz):
        # Centered differences with padding to maintain shape
        # X-derivative
        z_xi_inner = (Z[2:, :] - Z[:-2, :]) / (2*dx)
        z_xi = jnp.pad(z_xi_inner, ((1, 1), (0, 0)), mode='edge')
        
        # Z-derivative
        z_zeta_inner = (Z[:, 2:] - Z[:, :-2]) / (2*dz)
        z_zeta = jnp.pad(z_zeta_inner, ((0, 0), (1, 1)), mode='edge')
        
        return {'z_xi': z_xi, 'z_zeta': z_zeta}