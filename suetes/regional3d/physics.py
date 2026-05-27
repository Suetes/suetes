r"""
Subgrid-Scale Physics and Parameterizations Module.

Contains the physical closures required to model processes that occur at scales 
smaller than the grid resolution, including turbulence, surface 
friction, and moist microphysics.
"""

import jax  
import jax.numpy as jnp     
import flax.linen as nn
from flax import serialization  
import pickle

class PhysicsSuite:
    r"""
    Unified API for orchestrating all physics parameterizations.
    
    The suite separates physics into two categories:
    1. `tendency_schemes`: Continuous processes (like turbulence) evaluated alongside 
       the dynamical core to produce $\partial / \partial t$ tendencies.
    2. `update_schemes`: Instantaneous adjustments (like condensation) applied at 
       the end of the timestep to strictly enforce physical limits.
    """
    def __init__(self):
        self.tendency_schemes = []
        self.update_schemes = []
        self.tracer_keys = []  

    def add_tendency_scheme(self, scheme):
        self.tendency_schemes.append(scheme)

    def add_update_scheme(self, scheme):
        self.update_schemes.append(scheme)

    def register_tracer(self, key):
        if key not in self.tracer_keys:
            self.tracer_keys.append(key)

    def get_explicit_tendencies(self, state, bg, interior_mask=None, ml_params=None):
        """Aggregates continuous momentum and thermodynamic tendencies from all schemes.

        If `interior_mask` is provided, it should be a dict keyed by field
        name ('u', 'v', 'w', 'th_v') with values in [0, 1]. The accumulated
        tendencies are multiplied by the mask before being returned, so
        physics tendencies are zeroed inside the Davies sponge zone where
        the LBC nudging would otherwise be fighting drag and diffusion
        every step.
        """
        tends_total = {'u': jnp.zeros_like(state['u']),
                       'v': jnp.zeros_like(state['v']),
                       'w': jnp.zeros_like(state['w']),
                       'th_v': jnp.zeros_like(state['th_v'])}

        for scheme in self.tendency_schemes:
            # Force the pass explicitly. No try/except!
            if getattr(scheme, 'is_ml_closure', False): 
                scheme_tends = scheme.get_tendencies(state, bg, ml_params=ml_params)
            else:
                scheme_tends = scheme.get_tendencies(state, bg)
            
            for k in scheme_tends:
                tends_total[k] += scheme_tends[k]

        if interior_mask is not None:
            for k in tends_total:
                if k in interior_mask:
                    tends_total[k] = tends_total[k] * interior_mask[k]

        return tends_total

    def apply_state_updates(self, state):
        """Sequentially applies instantaneous thermodynamic adjustments."""
        updated_state = state.copy()
        for scheme in self.update_schemes:
            updates = scheme.apply_update(updated_state)
            updated_state.update(updates)
        return updated_state


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
    def __init__(self, grid, operators, constants, Cs=0.15, Pr_t=1.0, critical_Ri=0.25):
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
        S_mag = jnp.sqrt(2.0 * (D11_m**2 + D22_m**2 + D33_m**2) + D12_m**2 + D13_m**2 + D23_m**2 + 1e-12)

        # Buoyancy Frequency (N^2) and Richardson Number (Ri)
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        N2_m = self.op.avg((self.c['g'] / bg['th_v_w']) * dth_dz_w, axis=2, from_loc='w', to_loc='m')

        Ri = N2_m / (S_mag**2)
        f_Ri = jnp.sqrt(jnp.maximum(0.0, 1.0 - Ri / self.Ri_c))

        # Calculate Eddy Viscosity (nu_t)
        Delta = (self.grid.dx * self.grid.dy * bg['dz_m_full']) ** (1.0/3.0)
        nu_t_m = (self.Cs * Delta)**2 * S_mag * f_Ri

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

class NewtonianRelaxation:
    r"""
    Newtonian Nudging towards a target state.

    Acts as a proxy for missing diabatic physics (radiation, land-surface) 
    by gently pulling the thermodynamic field towards the ERA5 background.
    $$ \frac{\partial \theta_v}{\partial t} = -\frac{1}{\tau_R} (\theta_v - \theta_{v,\text{ERA5}}) $$
    """
    def __init__(self, tau_relax_hours=6.0):
        # Convert relaxation time to seconds
        self.tau_relax = tau_relax_hours * 3600.0
        self.target_state = None

    def update_target(self, target_state):
        """Called dynamically in the integration loop to update the target ERA5 state."""
        self.target_state = target_state

    def get_tendencies(self, state, bg):
        # Extract target from state safely
        target_th_v = state.get('target_th_v')
        
        # Fallback to internal state if missing (e.g., initial baseline calculations)
        if target_th_v is None and self.target_state is not None:
            target_th_v = self.target_state['th_v']
            
        if target_th_v is None:
            return {'th_v': jnp.zeros_like(state['th_v'])}
            
        # Calculate the linear restoring tendency
        tend_th_v = -(state['th_v'] - target_th_v) / self.tau_relax
        
        return {'th_v': tend_th_v}

class BulkAerodynamicPBL:
    r"""
    Models the frictional deceleration of the wind and Sensible Heat Flux at the surface.

    Applies a bulk aerodynamic drag formula exclusively to the lowest model layer.
    """
    def __init__(self, grid, operators, theta_surf=None, Cd_ocean=0.001, Ch_ocean=0.001):
        self.grid = grid
        self.op = operators
        self.Cd = Cd_ocean  
        self.Ch = Ch_ocean
        self.theta_surf = theta_surf # Static surface skin temperature boundary condition

    def get_tendencies(self, state, bg):
        u, v = state['u'], state['v']
        
        # Bring horizontal winds to the mass points to calculate true wind speed
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        
        # Calculate full 3D wind speed magnitude 
        speed_m_3d = jnp.sqrt(u_m**2 + v_m**2 + 1e-8) 
        
        # Map the 3D wind speed back to the staggered faces
        speed_u_3d = self.op.avg(speed_m_3d, axis=0, from_loc='m', to_loc='u')
        speed_v_3d = self.op.avg(speed_m_3d, axis=1, from_loc='m', to_loc='v')
        
        # Extract the surface layer (level 0) for the drag calculation
        speed_u_surf = speed_u_3d[:, :, 0]
        speed_v_surf = speed_v_3d[:, :, 0]
        
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        
        drag_u_surf = -self.Cd * (speed_u_surf * u[:, :, 0]) / dz_u_surf
        drag_v_surf = -self.Cd * (speed_v_surf * v[:, :, 0]) / dz_v_surf
        
        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)
        
        # Sensible Heat Flux (SHF)
        theta_surf = state.get('theta_surf', self.theta_surf) # Extract from state
        
        if theta_surf is not None:
            speed_m_surf = speed_m_3d[:, :, 0]
            th_v_surf = state['th_v'][:, :, 0]
            dz_m_surf = bg['dz_m_full'][:, :, 0]
            
            # Positive flux warms the atmosphere (Ocean is warmer than air)
            shf_kinematic = self.Ch * speed_m_surf * (theta_surf - th_v_surf)
            heat_tend_surf = shf_kinematic / dz_m_surf
            
            tend_th_v = jnp.zeros_like(state['th_v']).at[:, :, 0].set(heat_tend_surf)
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v}


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
            epsilon: stability cutoff parameter. McFarlane uses 0.3 over open
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
        f_unstable = 1.0 + 10.0 * abs_Ri / (1.0 + 10.0 * jnp.sqrt(abs_Ri / 87.0))
        # Stable branch (0 <= Ri <= 1/(5*eps)): f < 1, reduced mixing
        f_stable = (1.0 - 5.0 * self.epsilon * Ri) ** 2 \
                   / (1.0 + 10.0 * (1.0 - self.epsilon) * Ri)
        Ri_cutoff = 1.0 / (5.0 * self.epsilon + 1e-12)
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
        shear_w = jnp.sqrt(shear_sq_w + 1e-12)

        # Buoyancy at w-points
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') \
                   * (self.grid.dz / bg['dz_w_full'])
        N2_w = (self.c['g'] / bg['th_v_w']) * dth_dz_w

        # Gradient Richardson number
        Ri_w = N2_w / (shear_sq_w + 1e-12)
        f_w = self._stability_function(Ri_w)

        # Eddy diffusivity at w-points
        K_w = l_w ** 2 * shear_w * f_w

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


class McFarlaneSurfaceDrag:
    r"""
    Stability-dependent bulk surface drag and sensible heat flux.

    Follows McFarlane et al. (1992) GCMII, equations (2.4) and (2.5):

    $$ (C_m, C_h) = (C_{DM}, C_{DH}) \cdot (F_m(Ri_B), F_h(Ri_B)) $$

    The bulk Richardson number is formed from the lowest model mass level and
    the prescribed surface skin temperature theta_surf (Dirichlet condition
    from outside). Neutral coefficients come from the log law:

    $$ C_{DN} = (k / \ln(z_L / z_0))^2 $$

    GCMII uses the same stability function for momentum and heat.
    """

    def __init__(self, grid, operators, constants,
                 z_0=1e-4, epsilon=0.3, k_vk=0.4, theta_surf=None):
        """
        Args:
            grid (RegionalGrid3D): computational grid.
            operators (CGridOperator3D): finite-difference operators.
            constants (dict): physical constants; must contain 'g'.
            z_0 (float): surface roughness length [m]. McFarlane GCMII used
                per-class values from Wilson-Henderson-Sellers; here applied
                uniformly. Default 1e-4 m is the ocean value. Land values
                range roughly 0.01 to 1 m.
            epsilon (float): stability cutoff parameter. McFarlane: 0.3 over
                open water, 0.0 over land/ice.
            k_vk (float): von Karman constant.
            theta_surf: optional fallback if state['theta_surf'] is missing.
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.z_0 = z_0
        self.epsilon = epsilon
        self.k_vk = k_vk
        self.theta_surf = theta_surf

    def _stability_function(self, Ri_B, A_sq):
        """McFarlane (1992) eq. (2.4). Same form for F_m and F_h in GCMII.

        Unstable branch sign convention as in McFarlaneVerticalDiffusion: the
        paper formula written with |Ri_B| produces F > 1 under Louis (1979)
        sign conventions, implemented here as an explicit + sign.
        """
        abs_Ri = jnp.abs(Ri_B)
        F_unstable = 1.0 + 10.0 * abs_Ri \
                     / (1.0 + 10.0 * jnp.sqrt(abs_Ri / (87.0 * A_sq + 1e-12)))
        F_stable = (1.0 - 5.0 * self.epsilon * Ri_B) ** 2 \
                   / (1.0 + 10.0 * (1.0 - self.epsilon) * Ri_B)
        Ri_cutoff = 1.0 / (5.0 * self.epsilon + 1e-12)
        F = jnp.where(Ri_B < 0.0, F_unstable, F_stable)
        F = jnp.where(Ri_B > Ri_cutoff, 0.0, F)
        return F

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']

        # Mass-point wind speed throughout the column, with a soft floor
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        speed_m = jnp.sqrt(u_m ** 2 + v_m ** 2 + 1e-8)
        speed_u = self.op.avg(speed_m, axis=0, from_loc='m', to_loc='u')
        speed_v = self.op.avg(speed_m, axis=1, from_loc='m', to_loc='v')

        speed_m_surf = speed_m[:, :, 0]
        speed_u_surf = speed_u[:, :, 0]
        speed_v_surf = speed_v[:, :, 0]

        # Height of lowest mass level above the local terrain
        z_L = self.grid.Z_m[:, :, 0] - self.grid.Z_w[:, :, 0]

        # Neutral drag coefficient from log law (scalar in z_0, 2D in z_L)
        log_ratio = jnp.log(z_L / self.z_0)
        C_DN = (self.k_vk / log_ratio) ** 2

        # A^2 from eq. (2.5), used in the unstable branch denominator
        A_sq = (self.z_0 / z_L) * (self.k_vk ** 4) / (C_DN ** 2 + 1e-30)

        # Surface temperature (Dirichlet from outside)
        theta_surf = state.get('theta_surf', self.theta_surf)

        th_v_L = th_v[:, :, 0]
        speed_sq_surf = speed_m_surf ** 2

        if theta_surf is not None:
            Ri_B = self.c['g'] * (th_v_L - theta_surf) * z_L \
                   / (th_v_L * speed_sq_surf)
        else:
            Ri_B = jnp.zeros_like(z_L)

        F = self._stability_function(Ri_B, A_sq)
        C_eff = C_DN * F  # GCMII uses the same for momentum and heat

        # Promote C_eff to 3D for face averaging
        nx_m, ny_m, nz_m = u_m.shape
        C_eff_3d = jnp.broadcast_to(C_eff[:, :, None], (nx_m, ny_m, nz_m))
        C_u = self.op.avg(C_eff_3d, axis=0, from_loc='m', to_loc='u')[:, :, 0]
        C_v = self.op.avg(C_eff_3d, axis=1, from_loc='m', to_loc='v')[:, :, 0]

        # Surface momentum tendency (lowest layer only)
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        drag_u_surf = -C_u * speed_u_surf * u[:, :, 0] / dz_u_surf
        drag_v_surf = -C_v * speed_v_surf * v[:, :, 0] / dz_v_surf

        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)

        # Sensible heat flux (lowest layer only)
        if theta_surf is not None:
            dz_m_surf = bg['dz_m_full'][:, :, 0]
            shf_kin = C_eff * speed_m_surf * (theta_surf - th_v_L)
            tend_th = jnp.zeros_like(th_v).at[:, :, 0].set(shf_kin / dz_m_surf)
        else:
            tend_th = jnp.zeros_like(th_v)

        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th}


class McFarlaneGWD:
    r"""
    Orographic gravity wave drag (single column, vectorized over horizontal).

    Follows McFarlane (1987) and McFarlane et al. (1992) eq. (2.8) and (2.9).
    Subgrid-scale orographic gravity waves carry momentum flux upward; the
    flux is capped at each level by a Lindzen-style saturation criterion, and
    the excess is deposited locally as horizontal drag.

    $$ F_p(z) = \mu \rho N U \delta^2, \quad \delta_{sat} = F_c U / N $$

    $$ (\frac{dV}{dt})_g = \mathbf{n} \frac{1}{\rho} \frac{d F_p}{dz} $$

    Reference level: the lowest mass point. Reference direction n: the
    surface-projected horizontal wind there. Drag is deposited along that
    direction at each level above the reference.

    If h_variance is None or zero, this scheme returns identically zero
    tendencies.
    """

    def __init__(self, grid, operators, constants,
                 h_variance=None, F_c=0.7, mu=1.5e-5,
                 U_min=1.0, N2_min=1e-6, speed_min=0.1):
        """
        Args:
            grid (RegionalGrid3D): computational grid.
            operators (CGridOperator3D): finite-difference operators.
            constants (dict): physical constants; must contain 'g', 'p0',
                'cp', 'Rd'.
            h_variance: 2D array on mass points of subgrid orography variance
                [m^2], or None. If None, returns zero tendencies.
            F_c (float): saturation Froude factor. McFarlane: 0.7.
            mu (float): effective inverse horizontal wavelength of the
                orographic spectrum [m^-1]. McFarlane: 1.5e-5.
            U_min (float): floor on the projected wind speed used inside the
                saturation cap (avoids singularities at critical levels).
            N2_min (float): floor on N^2 below which waves are treated as
                evanescent and the saturation flux is computed with the floor.
            speed_min (float): if the surface wind speed is below this, the
                column produces no drag (avoids spurious drag in resting air).
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.h_variance = h_variance
        self.F_c = F_c
        self.mu = mu
        self.U_min = U_min
        self.N2_min = N2_min
        self.speed_min = speed_min

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']

        if self.h_variance is None:
            return {'u': jnp.zeros_like(u), 'v': jnp.zeros_like(v)}

        # Density on mass points: from state['rho'] if present, else from pi
        rho_m = state.get('rho')
        if rho_m is None:
            pi_full = state.get('pi')
            if pi_full is None:
                return {'u': jnp.zeros_like(u), 'v': jnp.zeros_like(v)}
            T_v = th_v * pi_full
            rho_m = self.c['p0'] * pi_full ** (self.c['cp'] / self.c['Rd']) \
                    / (self.c['Rd'] * T_v)

        # Winds at mass points
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')

        # Reference (surface) direction
        u0 = u_m[:, :, 0]
        v0 = v_m[:, :, 0]
        speed0 = jnp.sqrt(u0 ** 2 + v0 ** 2 + 1e-8)
        n_x = u0 / speed0
        n_y = v0 / speed0

        # Active mask: only columns with appreciable surface flow generate drag
        active = (speed0 > self.speed_min).astype(rho_m.dtype)

        # Projected wind speed at all m-levels (along the reference direction)
        U_m = u_m * n_x[:, :, None] + v_m * n_y[:, :, None]
        U_m_floor = jnp.maximum(U_m, self.U_min)

        # N^2 on w-points
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') \
                   * (self.grid.dz / bg['dz_w_full'])
        N2_w = (self.c['g'] / bg['th_v_w']) * dth_dz_w
        N2_w = jnp.maximum(N2_w, self.N2_min)
        N_w = jnp.sqrt(N2_w)

        # Place rho and U on w-points (simple centered average; edges padded)
        def m_to_w(field_m):
            interior = 0.5 * (field_m[:, :, :-1] + field_m[:, :, 1:])
            return jnp.concatenate(
                [field_m[:, :, 0:1], interior, field_m[:, :, -1:]],
                axis=2,
            )

        rho_w = m_to_w(rho_m)
        U_w = m_to_w(U_m_floor)

        # Saturation flux at every w-level
        F_sat_w = self.mu * rho_w * (self.F_c ** 2) * (U_w ** 3) / N_w

        # Reference flux at the surface, using the actual (un-floored) U
        delta_ref_sq = jnp.maximum(self.h_variance, 0.0)
        F_ref_surf = self.mu * rho_w[:, :, 0] * N_w[:, :, 0] * U_m[:, :, 0] \
                     * delta_ref_sq
        F_ref_surf = jnp.maximum(F_ref_surf, 0.0)
        F_0 = jnp.minimum(F_ref_surf, F_sat_w[:, :, 0])

        # March upward through w-points: F_k = min(F_{k-1}, F_sat_k)
        F_sat_z_first = jnp.moveaxis(F_sat_w, 2, 0)

        def scan_step(F_prev, F_sat_k):
            F_k = jnp.minimum(F_prev, F_sat_k)
            return F_k, F_k

        _, F_above_z_first = jax.lax.scan(scan_step, F_0, F_sat_z_first[1:])
        F_w_z_first = jnp.concatenate([F_0[None, :, :], F_above_z_first], axis=0)
        F_w = jnp.moveaxis(F_w_z_first, 0, 2)

        # Drag along n: a_proj = (1/rho) d F_w / dz at each m-point
        dF_dz_m = (F_w[:, :, 1:] - F_w[:, :, :-1]) / bg['dz_m_full']
        a_proj_m = dF_dz_m / rho_m

        # Mask inactive columns
        a_proj_m = a_proj_m * active[:, :, None]

        # Project back into (u, v) components on mass points, then stagger
        tend_u_m = a_proj_m * n_x[:, :, None]
        tend_v_m = a_proj_m * n_y[:, :, None]
        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')

        return {'u': tend_u, 'v': tend_v}


class SimpleMicrophysics:
    r"""
    A fast saturation adjustment microphysics scheme.

    Converts excess water vapor ($q_v$) into cloud water ($q_c$) when the air 
    becomes supersaturated, releasing latent heat back into the thermodynamic field.

    The condensation amount is approximated via a Taylor series expansion of the 
    Clausius-Clapeyron equation:
    $$ \delta q = \frac{q_v - q_s}{1 + \frac{L_v^2 q_s}{c_p R_v T^2}} $$

    The resulting temperature increase is:
    $$ \Delta T = \frac{L_v}{c_p} \delta q $$
    """
    def __init__(self, constants):
        """
        Args:
            constants (dict): Physical constants.
        """
        self.c = constants
        self.Lv = 2.5e6  # Latent heat of vaporization [J/kg]
        self.Rv = 461.5  # Gas constant for water vapor [J/(kg K)]
        self.epsilon = constants.get('epsilon', 0.622)

        # Pre-computed lookup table for the expensive Tetens exponent
        self.T_table = jnp.linspace(150.0, 330.0, 2000)
        self.es_table = 611.2 * jnp.exp(17.67 * (self.T_table - 273.15) / (self.T_table - 29.65))

    def apply_update(self, state):
        """Conforms to the PhysicsSuite state update API."""
        if 'q' in state:
            # Assume state['pi'] is the full pressure (bg + prime) at this stage
            return self.saturation_adjustment(state, state['pi'])
        return {}

    def saturation_adjustment(self, state, pi_full):
        r"""
        Computes the instantaneous thermodynamic adjustments due to condensation/evaporation.
        
        Calculates the saturation specific humidity ($q_s$) using the Tetens formula, 
        determines the moisture adjustment $\delta q$, and updates the temperature 
        (latent heat release) and humidity fields:
        
        $$
        \begin{align}
            q_c &= q_c + \delta q \\
            q_v &= q_v - \delta q \\
            T &= T + \frac{L_v}{c_p} \delta q 
        \end{align}
        $$
        """
        th_v = state['th_v']
        qv = state['q']
        qc = state.get('q_c', jnp.zeros_like(qv))
        
        # Back out the physical temperature (T) and pressure (p)
        Tv = th_v * pi_full
        T = Tv / (1.0 + (1.0 / self.epsilon - 1.0) * qv - qc)
        p = self.c['p0'] * (pi_full ** (self.c['cp'] / self.c['Rd']))

        # Pre-computed LUT Evaluation
        e_s = jnp.interp(T, self.T_table, self.es_table)
        
        # Calculate Saturation Specific Humidity (q_s)
        q_s = (self.epsilon * e_s) / (p - (1.0 - self.epsilon) * e_s)

        # Calculate Condensation/Evaporation amount (dq)
        dq = (qv - q_s) / (1.0 + (self.Lv**2 * q_s) / (self.c['cp'] * self.Rv * T**2))

        # Apply phase change only where needed
        dq = jnp.where(dq > 0, dq, jnp.maximum(dq, -qc))

        # Update the mass variables
        new_qv = qv - dq
        new_qc = qc + dq
        
        # Apply Latent Heating to Virtual Potential Temperature
        T_new = T + (self.Lv / self.c['cp']) * dq
        new_th_v = (T_new * (1.0 + (1.0 / self.epsilon - 1.0) * new_qv - new_qc)) / pi_full

        return {'q': new_qv, 'q_c': new_qc, 'th_v': new_th_v}


class ColumnPhysicsNet(nn.Module):
    """1D Neural Parameterization for subgrid tendencies."""
    hidden_dims: tuple = (128, 128, 64)
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
        x = nn.Dense(3, kernel_init=jax.nn.initializers.normal(stddev=1e-3),
             bias_init=jax.nn.initializers.zeros)(x)
        return x

class MLPhysicsClosure:
    def __init__(self, op, norm_stats):
        self.op = op
        self.mean = jnp.array(norm_stats['mean'])
        self.std = jnp.array(norm_stats['std'])
        self.model = ColumnPhysicsNet()
        self.is_ml_closure = True

    def get_tendencies(self, state, bg, ml_params=None):
        if ml_params is None:
            raise ValueError("GRAPH SEVERED: ml_params dropped before reaching the closure!")
            
        nn_params = ml_params['nn_params']
        u_m = self.op.avg(state['u'], axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(state['v'], axis=1, from_loc='v', to_loc='m')
        z_m = self.op.grid.Z_m
        X = jnp.stack([u_m, v_m, state['th_v'], z_m], axis=-1)
        X_norm = (X - self.mean) / (self.std + 1e-8)
        
        nx, ny, nz, _ = X_norm.shape
        X_flat = X_norm.reshape((nx * ny * nz, 4))
        
        preds_flat = self.model.apply({'params': nn_params}, X_flat)
        preds = preds_flat.reshape((nx, ny, nz, 3))
        
        MAX_TENDENCY = 5.0e-4 
        tend_u_m = MAX_TENDENCY * jnp.tanh(preds[..., 0])
        tend_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 1])
        tend_th_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 2])
        
        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v_m}