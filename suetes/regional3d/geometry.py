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


class RegionalGrid3D:
    def __init__(self, nx, ny, nz, dx, dy, dz, lat_center, lon_center):
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
        
        Xi_m, Yi_m = jnp.meshgrid(self.x_m, self.y_m, indexing='ij')
        Xi_u, Yi_u = jnp.meshgrid(self.x_c, self.y_m, indexing='ij')
        Xi_v, Yi_v = jnp.meshgrid(self.x_m, self.y_c, indexing='ij')
        Xi_w, Yi_w = Xi_m, Yi_m 

        self.proj = ObliqueStereographic(lat_center, lon_center)
        
        self.m_factors = {
            'm': self.proj.get_map_factor(Xi_m, Yi_m),
            'u': self.proj.get_map_factor(Xi_u, Yi_u),
            'v': self.proj.get_map_factor(Xi_v, Yi_v),
            'w': self.proj.get_map_factor(Xi_w, Yi_w)
        }
        
        Omega = 7.2921e-5
        lat_m, _ = self.proj.get_lat_lon(Xi_m, Yi_m)
        lat_u, _ = self.proj.get_lat_lon(Xi_u, Yi_u)
        lat_v, _ = self.proj.get_lat_lon(Xi_v, Yi_v)
        
        self.f_m = 2.0 * Omega * jnp.sin(jnp.radians(lat_m))
        self.f_u = 2.0 * Omega * jnp.sin(jnp.radians(lat_u))
        self.f_v = 2.0 * Omega * jnp.sin(jnp.radians(lat_v))