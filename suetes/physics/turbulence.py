import jax.numpy as jnp

class SmagorinskyLillySGS:
    r"""
    3D Subgrid-Scale Turbulence Closure using the Smagorinsky-Lilly model.

    This scheme calculates a localized eddy viscosity $\nu_t$ based on the magnitude 
    of the resolved strain rate tensor $|S|$ and the local atmospheric stability 
    (Richardson number, $Ri$). The resulting turbulent mixing is modeled as Fickian diffusion:

    $$ \frac{\partial \mathbf{v}}{\partial t} = \nabla \cdot (\nu_t \nabla \mathbf{v}) $$

    The eddy viscosity is parameterized as:
    $$
    \begin{align}
        \nu_t &= (C_s \Delta)^2 |S| f(Ri) \\
        f(Ri) &= \sqrt{\max\left(0, 1 - \frac{Ri}{Ri_c}\right)}
    \end{align}
    $$
    """
    def __init__(self, grid, operators, constants, dt, Cs=0.15, Pr_t=1.0, critical_Ri=0.25):
        """
        Args:
            grid (RegionalGrid3D): The computational grid.
            operators (CGridOperator3D): Spatial finite-difference operators.
            constants (dict): Physical constants.
            Cs (float): The Smagorinsky constant (typically 0.1 to 0.2).
            Pr_t (float): Turbulent Prandtl number.
            critical_Ri (float): Critical Richardson number $Ri_c$ where turbulence ceases.
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.dt = dt
        self.Cs = Cs
        self.Pr_t = Pr_t
        self.Ri_c = critical_Ri

    def get_tendencies(self, state, bg):
        r"""
        Computes the turbulent diffusion tendencies for the momentum field.
        
        Evaluates the full symmetric strain rate tensor $D_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$ 
        to find its magnitude:
        $$ |S| = \sqrt{2(D_{11}^2 + D_{22}^2 + D_{33}^2) + D_{12}^2 + D_{13}^2 + D_{23}^2} $$
        """
        u, v, w, th_v = state['u'], state['v'], state['w'], state['th_v']
        
        # Calculate Diagonal Strains natively (saves 6 expensive 3D averages)
        D11_m = self.op.diff(u, axis=0, from_loc='u', to_loc='m')
        D22_m = self.op.diff(v, axis=1, from_loc='v', to_loc='m')
        D33_m = self.op.diff(w, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])

        # We still need mass-centered winds for the off-diagonals and scalar diffusion
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        w_m = self.op.avg(w, axis=2, from_loc='w', to_loc='m')

        # Streamline off-diagonals
        D12_m = self.op.avg(self.op.diff(u_m, axis=1, from_loc='m', to_loc='v'), axis=1, from_loc='v', to_loc='m') + \
                self.op.avg(self.op.diff(v_m, axis=0, from_loc='m', to_loc='u'), axis=0, from_loc='u', to_loc='m')
                
        D13_m = self.op.avg(self.op.diff(u_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full']), axis=2, from_loc='w', to_loc='m') + \
                self.op.avg(self.op.diff(w_m, axis=0, from_loc='m', to_loc='u'), axis=0, from_loc='u', to_loc='m')
                
        D23_m = self.op.avg(self.op.diff(v_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full']), axis=2, from_loc='w', to_loc='m') + \
                self.op.avg(self.op.diff(w_m, axis=1, from_loc='m', to_loc='v'), axis=1, from_loc='v', to_loc='m')

        # Magnitude of the Strain Rate Tensor |S|
        S_mag = jnp.sqrt(2.0 * (D11_m**2 + D22_m**2 + D33_m**2) + D12_m**2 + D13_m**2 + D23_m**2 + self.c.get('eps_l', 1e-12))

        # Buoyancy Frequency (N^2) and Richardson Number (Ri)
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        N2_m = self.op.avg((self.c['g'] / bg['th_v_w']) * dth_dz_w, axis=2, from_loc='w', to_loc='m')

        Ri = N2_m / (S_mag**2)
        val = 1.0 - Ri / self.Ri_c
        safe_val = jnp.maximum(val, self.c.get('eps_l', 1e-5)) # Prevents exactly 0.0 inside the sqrt
        f_Ri = jnp.where(val > 0.0, jnp.sqrt(safe_val), 0.0)

        # Calculate Unbounded Eddy Viscosity (nu_t)
        Delta = (self.grid.dx * self.grid.dy * bg['dz_m_full']) ** (1.0/3.0)
        nu_t_m_unbounded = (self.Cs * Delta)**2 * S_mag * f_Ri

        # Explicit diffusive CFL limit: nu < dz^2 / (4 * dt). 
        # We apply a 0.8 safety factor to ensure strict stability.
        max_nu_t = (bg['dz_m_full']**2) / (4.0 * self.dt) * 0.8
        nu_t_m = jnp.minimum(nu_t_m_unbounded, max_nu_t)

        # Pre-calculate face viscosities for the flux divergence
        nu_t_u = self.op.avg(nu_t_m, axis=0, from_loc='m', to_loc='u')
        nu_t_v = self.op.avg(nu_t_m, axis=1, from_loc='m', to_loc='v')
        nu_t_w = self.op.avg(nu_t_m, axis=2, from_loc='m', to_loc='w')

        # Streamline Fickian Diffusion calculation
        def diffuse_scalar_on_m(phi_m):
            grad_x = self.op.diff(phi_m, axis=0, from_loc='m', to_loc='u')
            grad_y = self.op.diff(phi_m, axis=1, from_loc='m', to_loc='v')
            grad_z = self.op.diff(phi_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
            
            # Direct flux divergence (saves 3 intermediate assignments)
            div_x = self.op.diff(nu_t_u * grad_x, axis=0, from_loc='u', to_loc='m')
            div_y = self.op.diff(nu_t_v * grad_y, axis=1, from_loc='v', to_loc='m')
            div_z = self.op.diff(nu_t_w * grad_z, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])
            
            return div_x + div_y + div_z

        tend_u_m = diffuse_scalar_on_m(u_m)
        tend_v_m = diffuse_scalar_on_m(v_m)
        tend_w_m = diffuse_scalar_on_m(w_m)

        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')
        tend_w = self.op.avg(tend_w_m, axis=2, from_loc='m', to_loc='w')

        return {'u': tend_u, 'v': tend_v, 'w': tend_w}


class FastVerticalDiffusion:
    r"""
    1D Vertical Eddy Diffusion with a PBL-confined K-profile.
    """
    def __init__(self, grid, operators, K_z_max=15.0, h_pbl=1500.0):
        self.grid = grid
        self.op = operators
        self.K_z_max = K_z_max
        self.h_pbl = h_pbl # Height where mixing effectively stops [m]

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']
        
        # Calculate Height Above Ground Level (AGL) by subtracting the surface height (level 0) 
        # from all vertical levels. We use 0:1 to keep the z-axis dimension for broadcasting.
        Z_AGL = self.grid.Z_w - self.grid.Z_w[:, :, 0:1]
        
        # Create a spatial K-profile that decays exponentially above the PBL using AGL
        K_profile = self.K_z_max * jnp.exp(- (Z_AGL / self.h_pbl)**2)
        
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        
        du_dz_w = self.op.diff(u_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        dv_dz_w = self.op.diff(v_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])

        # Apply the restricted K-profile to the fluxes
        flux_u_w  = K_profile * du_dz_w
        flux_v_w  = K_profile * dv_dz_w
        
        # Enforce zero flux at physical boundaries
        flux_u_w  = flux_u_w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        flux_v_w  = flux_v_w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        # Calculate flux divergence back at mass points
        tend_u_m  = self.op.diff(flux_u_w, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])
        tend_v_m  = self.op.diff(flux_v_w, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])

        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')

        return {'u': tend_u, 'v': tend_v}


class McFarlaneVerticalDiffusion:
    r"""
    Stability-dependent vertical eddy diffusion of momentum and heat.

    Follows McFarlane et al. (1992) GCMII, equations (2.1) to (2.3):

    $$ K_{m,h} = l^2 \left|\frac{\partial \mathbf{V}}{\partial z}\right| f_{m,h}(Ri) $$

    with mixing length

    $$ l = \frac{k z}{1 + k z / \lambda} $$

    and a piecewise stability function in the gradient Richardson number Ri.
    GCMII uses the same functional form for momentum and heat.

    Surface flux is held at zero by this scheme; the surface stress and SHF
    are provided by a separate surface scheme (e.g. McFarlaneSurfaceDrag) as
    a tendency confined to the lowest mass layer.
    """

    def __init__(self, grid, operators, constants,
                 lambda_mix=100.0, epsilon=0.3, k_vk=0.4):
        """
        Args:
            grid (RegionalGrid3D): computational grid.
            operators (CGridOperator3D): finite-difference operators.
            constants (dict): physical constants; must contain 'g'.
            lambda_mix (float): asymptotic mixing length [m]. McFarlane: 100 m.
            epsilon (float or array-like): Stability cutoff parameter. McFarlane uses 0.3 over open
                water and 0.0 over land/ice. Can be a scalar (uniform) or a
                2D (nx, ny) array for per-cell values from a land-sea mask.
                A 2D array is promoted internally to (nx, ny, 1) so it
                broadcasts against the 3D Ri on w-points; the caller does
                not need to add a trailing axis.
            k_vk (float): von Karman constant.
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.lambda_mix = lambda_mix
        eps_arr = jnp.asarray(epsilon)
        if eps_arr.ndim == 2:
            eps_arr = eps_arr[..., None]
        self.epsilon = eps_arr
        self.k_vk = k_vk

    def _stability_function(self, Ri):
        """McFarlane (1992) eq. (2.2). Same form for f_m and f_h in GCMII.

        The unstable branch is written in the paper as "1 - 10|Ri| / (...)"
        but yields f > 1 in unstable conditions (enhanced mixing) when |Ri|
        is read with the Louis (1979) signed-Ri convention. Implemented here
        with an explicit + sign so the formula in absolute-value form gives
        the physically correct enhancement.
        """
        abs_Ri = jnp.abs(Ri)
        # Unstable branch (Ri < 0): f > 1, enhanced mixing
        safe_abs_Ri = jnp.maximum(abs_Ri, self.c.get('eps_l', 1e-5)) # Protect the sqrt
        f_unstable = 1.0 + 10.0 * abs_Ri / (1.0 + 10.0 * jnp.sqrt(safe_abs_Ri / 87.0))

        # Stable branch (0 <= Ri <= 1/(5*eps)): f < 1, reduced mixing
        f_stable = (1.0 - 5.0 * self.epsilon * Ri) ** 2 \
                   / (1.0 + 10.0 * (1.0 - self.epsilon) * Ri)
        Ri_cutoff = 1.0 / (5.0 * self.epsilon + self.c.get('eps_l', 1e-5))
        f = jnp.where(Ri < 0.0, f_unstable, f_stable)
        f = jnp.where(Ri > Ri_cutoff, 0.0, f)
        return f

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']

        # Height above ground at w-points
        Z_AGL_w = self.grid.Z_w - self.grid.Z_w[:, :, 0:1]
        kz = self.k_vk * Z_AGL_w
        l_w = kz / (1.0 + kz / self.lambda_mix)

        # Horizontal winds at mass points
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')

        # Shear at w-points
        du_dz_w = self.op.diff(u_m, axis=2, from_loc='m', to_loc='w') \
                  * (self.grid.dz / bg['dz_w_full'])
        dv_dz_w = self.op.diff(v_m, axis=2, from_loc='m', to_loc='w') \
                  * (self.grid.dz / bg['dz_w_full'])
        shear_sq_w = du_dz_w ** 2 + dv_dz_w ** 2
        shear_w = jnp.sqrt(shear_sq_w + self.c.get('eps_l', 1e-12))

        # Buoyancy at w-points
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') \
                   * (self.grid.dz / bg['dz_w_full'])
        N2_w = (self.c['g'] / bg['th_v_w']) * dth_dz_w

        # Gradient Richardson number
        Ri_w = N2_w / (shear_sq_w + self.c.get('eps_l', 1e-12))
        f_w = self._stability_function(Ri_w)

        # Eddy diffusivity at w-points
        K_w = l_w ** 2 * shear_w * f_w

        # Diffusive CFL safety cap to prevent numerical explosion
        K_w = jnp.minimum(K_w, 500.0)

        # Fluxes; zero at top and bottom (surface is handled elsewhere)
        flux_u_w = K_w * du_dz_w
        flux_v_w = K_w * dv_dz_w
        flux_th_w = K_w * dth_dz_w
        flux_u_w = flux_u_w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        flux_v_w = flux_v_w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        flux_th_w = flux_th_w.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        # Flux divergence at mass points
        tend_u_m = self.op.diff(flux_u_w, axis=2, from_loc='w', to_loc='m') \
                   * (self.grid.dz / bg['dz_m_full'])
        tend_v_m = self.op.diff(flux_v_w, axis=2, from_loc='w', to_loc='m') \
                   * (self.grid.dz / bg['dz_m_full'])
        tend_th = self.op.diff(flux_th_w, axis=2, from_loc='w', to_loc='m') \
                  * (self.grid.dz / bg['dz_m_full'])

        # Stagger horizontal momentum tendencies back to u and v faces
        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')

        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th}

class TKE15Closure:
    r"""
    1.5-Order Turbulent Kinetic Energy Closure.
    
    Computes eddy viscosity $K_m$ from a prognostic TKE variable $e$:
    $$ K_m = c_k l \sqrt{e} $$
    """
    def __init__(self, grid, operators, constants, l_mix=50.0):
        self.grid = grid
        self.op = operators
        self.c = constants
        self.l_mix = l_mix
        self.c_k = 0.1      # Empirical constant

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']
        # Extract TKE (e) or initialize if missing
        e = state.get('tke', jnp.full_like(th_v, 1e-4)) 

        # Horizontal winds to mass points
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')

        # Shear and Buoyancy at w-points
        du_dz = self.op.diff(u_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        dv_dz = self.op.diff(v_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        shear_sq = du_dz**2 + dv_dz**2

        dth_dz = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        N2 = (self.c['g'] / bg['th_v_w']) * dth_dz

        # Average TKE to w-points to compute Eddy Viscosity (Km)
        e_w = self.op.avg(e, axis=2, from_loc='m', to_loc='w')
        K_m_w = self.c_k * self.l_mix * jnp.sqrt(jnp.maximum(e_w, self.c.get('eps_l', 1e-6)))
        
        # Assume Pr_t = 1 for simplicity, so K_h = K_m
        K_h_w = K_m_w

        # TKE Tendency terms (Production - Dissipation)
        prod_shear = K_m_w * shear_sq
        prod_buoy = -K_h_w * N2
        dissipation = (jnp.maximum(e_w, self.c.get('eps_l', 1e-6))**1.5) / self.l_mix
        
        tend_e_w = prod_shear + prod_buoy - dissipation
        tend_e_m = self.op.avg(tend_e_w, axis=2, from_loc='w', to_loc='m')

        # Diffusive Fluxes for Momentum and Heat
        flux_u = K_m_w * du_dz
        flux_v = K_m_w * dv_dz
        flux_th = K_h_w * dth_dz

        # Apply zero-flux boundaries
        flux_u = flux_u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        flux_v = flux_v.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        flux_th = flux_th.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        # Flux divergences
        tend_u_m = self.op.diff(flux_u, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])
        tend_v_m = self.op.diff(flux_v, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])
        tend_th_m = self.op.diff(flux_th, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])

        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')

        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_m, 'tke': tend_e_m}
