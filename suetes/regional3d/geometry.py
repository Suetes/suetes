"""
Geometry and Grid Construction Module.

This module provides the core spatial definitions for the Suetes dynamical core, 
handling map projections, Arakawa C-grid staggering, and terrain-following 
coordinate transformations.
"""

import jax
import jax.numpy as jnp

class ObliqueStereographic:
    """
    Defines an Oblique Stereographic projection for regional domains.

    This projection is conformal, meaning it preserves local angles and shapes,
    which is highly desirable for fluid dynamics as the map factors are isotropic
    (i.e., $m_x = m_y = m$).
    """
    def __init__(self, lat_center, lon_center, R_earth=6371229.0):
        r"""
        Initializes the projection around a central tangent point.

        Args:
            lat_center (float): Central latitude $\phi_c$ in degrees.
            lon_center (float): Central longitude $\lambda_c$ in degrees.
            R_earth (float, optional): Radius of the Earth in meters. Defaults to 6371229.0.
        """
        self.phi_c = jnp.radians(lat_center)
        self.lam_c = jnp.radians(lon_center)
        self.R = R_earth

    def get_map_factor(self, x, y):
        r"""
        Computes the isotropic map scale factor $m$ at a given grid location.

        The map factor represents the ratio of a distance on the projection plane 
        to the corresponding distance on the sphere:

        $$ m(x,y) = 1 + \frac{x^2 + y^2}{4R^2} $$

        Args:
            x (jnp.ndarray): Cartesian x-coordinates on the projection plane [m].
            y (jnp.ndarray): Cartesian y-coordinates on the projection plane [m].

        Returns:
            jnp.ndarray: The dimensionless map factor.
        """
        rho_sq = x**2 + y**2
        return 1.0 + rho_sq / (4.0 * self.R**2)

    def get_lat_lon(self, x, y):
        r"""
        Computes the geographic latitude and longitude for a given Cartesian point.

        The inverse projection maps the planar coordinates $(x, y)$ back to the 
        sphere $(\phi, \lambda)$ using the central point $(\phi_c, \lambda_c)$ 
        and the angular distance $c$:

        $$
        \begin{aligned}
        \rho &= \sqrt{x^2 + y^2} \\
        c &= 2 \arctan\left(\frac{\rho}{2R}\right) \\
        \phi &= \arcsin\left(\cos(c)\sin(\phi_c) + \frac{y \sin(c) \cos(\phi_c)}{\rho}\right)
        \end{aligned}
        $$

        Args:
            x (jnp.ndarray): Cartesian x-coordinates [m].
            y (jnp.ndarray): Cartesian y-coordinates [m].

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Arrays of latitude and longitude in degrees.
        """
        rho = jnp.sqrt(x**2 + y**2)
        rho = jnp.where(rho == 0, 1e-15, rho) 
        
        c = 2.0 * jnp.arctan(rho / (2.0 * self.R))
        sin_c, cos_c = jnp.sin(c), jnp.cos(c)
        sin_phic, cos_phic = jnp.sin(self.phi_c), jnp.cos(self.phi_c)

        lat = jnp.arcsin(cos_c * sin_phic + (y * sin_c * cos_phic) / rho)
        lon = self.lam_c + jnp.arctan2(x * sin_c, rho * cos_phic * cos_c - y * sin_phic * sin_c)
        
        return jnp.degrees(lat), jnp.degrees(lon)

    def get_convergence_angle(self, x, y):
        r"""
        Computes the grid convergence angle $\gamma$ using JAX Auto-Diff.

        The convergence angle is the angle between True North (geographic) and 
        Grid North (the y-axis of the Cartesian plane). This is required to rotate 
        geographic wind vectors (e.g., from ERA5) into the model's $X/Y$ basis.
        
        True North points in the direction of the steepest increasing latitude gradient:
        
        $$ \gamma = \arctan2\left(\frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}\right) $$

        Args:
            x (jnp.ndarray): Cartesian x-coordinates [m].
            y (jnp.ndarray): Cartesian y-coordinates [m].

        Returns:
            jnp.ndarray: The convergence angle in radians.
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
    r"""
    Constructs the 3D staggered computational grid and physical metric tensors.

    This class defines an Arakawa C-grid in the horizontal and a Lorenz staggering 
    in the vertical. It maps the logical Cartesian coordinates $(\xi, \eta, \zeta)$ 
    to the physical coordinates $(x, y, z)$ using a user-specified terrain-following 
    transformation.
    """
    def __init__(self, nx, ny, nz, dx, dy, dz, lat_center, lon_center, h_func=None, transform=None):
        r"""
        Initializes the grid geometry, map factors, Coriolis parameters, and metric tensors.

        Args:
            nx (int): Number of mass cells in the x-direction.
            ny (int): Number of mass cells in the y-direction.
            nz (int): Number of mass cells in the z-direction.
            dx (float): Grid spacing in the x-direction [m].
            dy (float): Grid spacing in the y-direction [m].
            dz (float): Nominal grid spacing in the z-direction [m].
            lat_center (float): Central latitude of the domain [deg].
            lon_center (float): Central longitude of the domain [deg].
            h_func (callable, optional): A function $h(\xi, \eta)$ providing the surface elevation. 
                Defaults to a flat surface.
            transform (callable, optional): The terrain-following coordinate transformation. 
                Defaults to `GalChenSigma`.

        Raises:
            ValueError: If the terrain transformation results in grid tangling (negative $\Delta z$).
        """
        self.shape = (nx, ny, nz)
        self.delta = (dx, dy, dz)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz

        self.lat_c = lat_center
        self.lon_c = lon_center
        
        Lx, Ly = nx * dx, ny * dy
        self.Lz = nz * dz
        
        # Logical boundary coordinates
        self.x_c = jnp.arange(nx + 1) * dx - (Lx / 2.0)
        self.y_c = jnp.arange(ny + 1) * dy - (Ly / 2.0)
        self.z_c = jnp.arange(nz + 1) * dz
        
        # Logical mass-point coordinates
        self.x_m = 0.5 * (self.x_c[:-1] + self.x_c[1:])
        self.y_m = 0.5 * (self.y_c[:-1] + self.y_c[1:])
        self.z_m = 0.5 * (self.z_c[:-1] + self.z_c[1:])
        
        # Map projections defined across all C-grid staggerings
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
        
        # Coriolis Parameter: f = 2 \Omega \sin(\phi)
        Omega = 7.2921e-5
        lat_m, _ = self.proj.get_lat_lon(Xi_m_2d, Yi_m_2d)
        lat_u, _ = self.proj.get_lat_lon(Xi_u_2d, Yi_u_2d)
        lat_v, _ = self.proj.get_lat_lon(Xi_v_2d, Yi_v_2d)
        
        self.f_m = 2.0 * Omega * jnp.sin(jnp.radians(lat_m))
        self.f_u = 2.0 * Omega * jnp.sin(jnp.radians(lat_u))
        self.f_v = 2.0 * Omega * jnp.sin(jnp.radians(lat_v))

        # 3D Topography and Transforms
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

        # Apply transformation to get physical 3D altitude (z) at all staggers
        self.Z_m = apply_transform(Xi_m, Yi_m, Zeta_m)
        self.Z_u = apply_transform(Xi_u, Yi_u, Zeta_u)
        self.Z_v = apply_transform(Xi_v, Yi_v, Zeta_v)
        self.Z_w = apply_transform(Xi_w, Yi_w, Zeta_w)

        # ---------------------------------------------------------
        # 3D METRIC TENSORS
        # ---------------------------------------------------------
        
        # Vertical Metrics: dz = \partial z / \partial \zeta
        self.dz_m_full = self.Z_w[:, :, 1:] - self.Z_w[:, :, :-1]
        
        Z_m_extrap_bottom = self.Z_m[:, :, 0] - (self.Z_m[:, :, 1] - self.Z_m[:, :, 0])
        Z_m_extrap_top = self.Z_m[:, :, -1] + (self.Z_m[:, :, -1] - self.Z_m[:, :, -2])
        Z_m_pad = jnp.concatenate([
            jnp.expand_dims(Z_m_extrap_bottom, axis=-1), 
            self.Z_m, 
            jnp.expand_dims(Z_m_extrap_top, axis=-1)
        ], axis=-1)
        
        self.dz_w_full = Z_m_pad[:, :, 1:] - Z_m_pad[:, :, :-1]

        # Horizontal Metrics: \partial z / \partial \xi and \partial z / \partial \eta
        # Evaluated at W-points. Used to compute the contravariant vertical velocity 
        # and enforce the kinematic bottom boundary condition.
        Z_w_pad_x = jnp.pad(self.Z_w, ((1, 1), (0, 0), (0, 0)), mode='edge')
        self.z_xi_w = (Z_w_pad_x[2:, :, :] - Z_w_pad_x[:-2, :, :]) / (2.0 * self.dx)

        Z_w_pad_y = jnp.pad(self.Z_w, ((0, 0), (1, 1), (0, 0)), mode='edge')
        self.z_eta_w = (Z_w_pad_y[:, 2:, :] - Z_w_pad_y[:, :-2, :]) / (2.0 * self.dy)

        # DIAGNOSTIC CHECK
        dz_min = float(jnp.min(self.dz_m_full))
        if dz_min <= 0.0:
            raise ValueError(
                f"Grid Tangling Detected! Minimum dz is {dz_min:.2f} m.\n"
                f"The 3D topography is too steep. Smooth the terrain or increase Lz."
            )