import jax
import jax.numpy as jnp


class ObliqueStereographic:
    def __init__(self, lat_center, lon_center, R_earth=6371229.0):
        self.phi_c = jnp.radians(lat_center)
        self.lam_c = jnp.radians(lon_center)
        self.R = R_earth

    def get_map_factor(self, x, y):
        rho_sq = x**2 + y**2
        return 1.0 + rho_sq / (4.0 * self.R**2)

    def get_lat_lon(self, x, y):
        rho = jnp.sqrt(x**2 + y**2)
        rho = jnp.where(rho == 0, 1e-15, rho) 
        
        c = 2.0 * jnp.arctan(rho / (2.0 * self.R))
        sin_c, cos_c = jnp.sin(c), jnp.cos(c)
        sin_phic, cos_phic = jnp.sin(self.phi_c), jnp.cos(self.phi_c)

        lat = jnp.arcsin(cos_c * sin_phic + (y * sin_c * cos_phic) / rho)
        lon = self.lam_c + jnp.arctan2(x * sin_c, rho * cos_phic * cos_c - y * sin_phic * sin_c)
        
        return jnp.degrees(lat), jnp.degrees(lon)

    def get_convergence_angle(self, x, y):
        """
        Uses JAX Auto-Diff to find the exact True North vector.
        True North points in the direction of steepest increasing latitude.
        """
        def get_phi(x_val, y_val):
            # The exact same math as get_lat_lon, but scalar and returns radians
            rho = jnp.sqrt(x_val**2 + y_val**2) + 1e-15
            c = 2.0 * jnp.arctan(rho / (2.0 * self.R))
            sin_c = jnp.sin(c)
            cos_c = jnp.cos(c)
            sin_phic = jnp.sin(self.phi_c)
            cos_phic = jnp.cos(self.phi_c)
            
            return jnp.arcsin(cos_c * sin_phic + (y_val * sin_c * cos_phic) / rho)
            
        # vmap handles the 2D meshgrid arrays (axes 0 and 1)
        grad_fn = jax.vmap(jax.vmap(jax.grad(get_phi, argnums=(0, 1))))
        dphi_dx, dphi_dy = grad_fn(x, y)
        
        # arctan2(X, Y) gives the exact clockwise angle from the Y-axis
        return jnp.arctan2(dphi_dx, dphi_dy)


class RegionalGrid3D:
    def __init__(self, nx, ny, nz, dx, dy, dz, lat_center, lon_center, h_func=None, transform=None):
        self.shape = (nx, ny, nz)
        self.delta = (dx, dy, dz)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        
        Lx, Ly = nx * dx, ny * dy
        self.Lz = nz * dz
        
        self.x_c = jnp.arange(nx + 1) * dx - (Lx / 2.0)
        self.y_c = jnp.arange(ny + 1) * dy - (Ly / 2.0)
        self.z_c = jnp.arange(nz + 1) * dz
        
        self.x_m = 0.5 * (self.x_c[:-1] + self.x_c[1:])
        self.y_m = 0.5 * (self.y_c[:-1] + self.y_c[1:])
        self.z_m = 0.5 * (self.z_c[:-1] + self.z_c[1:])
        
        # --- MAP PROJECTIONS ---
        Xi_m_2d, Yi_m_2d = jnp.meshgrid(self.x_m, self.y_m, indexing='ij')
        Xi_u_2d, Yi_u_2d = jnp.meshgrid(self.x_c, self.y_m, indexing='ij')
        Xi_v_2d, Yi_v_2d = jnp.meshgrid(self.x_m, self.y_c, indexing='ij')
        Xi_w_2d, Yi_w_2d = Xi_m_2d, Yi_m_2d

        self.proj = ObliqueStereographic(lat_center, lon_center)
        self.m_factors = {
            'm': self.proj.get_map_factor(Xi_m_2d, Yi_m_2d),
            'u': self.proj.get_map_factor(Xi_u_2d, Yi_u_2d),
            'v': self.proj.get_map_factor(Xi_v_2d, Yi_v_2d),
            'w': self.proj.get_map_factor(Xi_w_2d, Yi_w_2d)
        }
        
        Omega = 7.2921e-5
        lat_m, _ = self.proj.get_lat_lon(Xi_m_2d, Yi_m_2d)
        lat_u, _ = self.proj.get_lat_lon(Xi_u_2d, Yi_u_2d)
        lat_v, _ = self.proj.get_lat_lon(Xi_v_2d, Yi_v_2d)
        
        self.f_m = 2.0 * Omega * jnp.sin(jnp.radians(lat_m))
        self.f_u = 2.0 * Omega * jnp.sin(jnp.radians(lat_u))
        self.f_v = 2.0 * Omega * jnp.sin(jnp.radians(lat_v))

        # --- 3D TOPOGRAPHY AND TRANSFORMS ---
        self.h_func = h_func if h_func is not None else lambda x, y: 0.0
        
        if transform is None:
            from suetes.shared.transforms import GalChenSigma
            self.transform_op = GalChenSigma()
        else:
            self.transform_op = transform

        # Generate logical 3D coordinates (xi, eta, zeta)
        Xi_m, Yi_m, Zeta_m = jnp.meshgrid(self.x_m, self.y_m, self.z_m, indexing='ij')
        Xi_u, Yi_u, Zeta_u = jnp.meshgrid(self.x_c, self.y_m, self.z_m, indexing='ij')
        Xi_v, Yi_v, Zeta_v = jnp.meshgrid(self.x_m, self.y_c, self.z_m, indexing='ij')
        Xi_w, Yi_w, Zeta_w = jnp.meshgrid(self.x_m, self.y_m, self.z_c, indexing='ij')

        def apply_transform(xi, eta, zeta):
            h = self.h_func(xi, eta)
            # We pass xi to the transform (compatible with 2D transforms)
            return self.transform_op(xi, zeta, h, self.Lz)

        self.Z_m = apply_transform(Xi_m, Yi_m, Zeta_m)
        self.Z_u = apply_transform(Xi_u, Yi_u, Zeta_u)
        self.Z_v = apply_transform(Xi_v, Yi_v, Zeta_v)
        self.Z_w = apply_transform(Xi_w, Yi_w, Zeta_w)

        # --- 3D METRIC TENSORS ---
        # 1. Vertical Metrics (dz)
        self.dz_m_full = self.Z_w[:, :, 1:] - self.Z_w[:, :, :-1]
        
        Z_m_extrap_bottom = self.Z_m[:, :, 0] - (self.Z_m[:, :, 1] - self.Z_m[:, :, 0])
        Z_m_extrap_top = self.Z_m[:, :, -1] + (self.Z_m[:, :, -1] - self.Z_m[:, :, -2])
        Z_m_pad = jnp.concatenate([
            jnp.expand_dims(Z_m_extrap_bottom, axis=-1), 
            self.Z_m, 
            jnp.expand_dims(Z_m_extrap_top, axis=-1)
        ], axis=-1)
        
        self.dz_w_full = Z_m_pad[:, :, 1:] - Z_m_pad[:, :, :-1]

        # 2. Horizontal Metrics at W-points for Kinematic Advection
        Z_w_pad_x = jnp.pad(self.Z_w, ((1, 1), (0, 0), (0, 0)), mode='edge')
        self.z_xi_w = (Z_w_pad_x[2:, :, :] - Z_w_pad_x[:-2, :, :]) / (2.0 * self.dx)

        Z_w_pad_y = jnp.pad(self.Z_w, ((0, 0), (1, 1), (0, 0)), mode='edge')
        self.z_eta_w = (Z_w_pad_y[:, 2:, :] - Z_w_pad_y[:, :-2, :]) / (2.0 * self.dy)

        # --- DIAGNOSTIC CHECK ---
        dz_min = float(jnp.min(self.dz_m_full))
        if dz_min <= 0.0:
            raise ValueError(
                f"Grid Tangling Detected! Minimum dz is {dz_min:.2f} m.\n"
                f"The 3D topography is too steep. Smooth the terrain or increase Lz."
            )