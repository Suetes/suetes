"""
Subgrid-Scale Physics and Parameterizations Module.

Contains the physical closures required to model processes that occur at scales 
smaller than the grid resolution ($\Delta x$), including turbulence, surface 
friction, and moist microphysics.
"""

import jax.numpy as jnp

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

    def get_explicit_tendencies(self, state, bg):
        """Aggregates continuous momentum and thermodynamic tendencies from all schemes."""
        tends_total = {'u': jnp.zeros_like(state['u']), 
                       'v': jnp.zeros_like(state['v']), 
                       'w': jnp.zeros_like(state['w']),
                       'th_v': jnp.zeros_like(state['th_v'])}
        
        for scheme in self.tendency_schemes:
            scheme_tends = scheme.get_tendencies(state, bg)
            for k in scheme_tends:
                tends_total[k] += scheme_tends[k]
                
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
        if self.target_state is None:
            return {'th_v': jnp.zeros_like(state['th_v'])}
            
        # Calculate the linear restoring tendency
        tend_th_v = -(state['th_v'] - self.target_state['th_v']) / self.tau_relax
        
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
        if self.theta_surf is not None:
            speed_m_surf = speed_m_3d[:, :, 0]
            th_v_surf = state['th_v'][:, :, 0]
            dz_m_surf = bg['dz_m_full'][:, :, 0]
            
            # Positive flux warms the atmosphere (Ocean is warmer than air)
            shf_kinematic = self.Ch * speed_m_surf * (self.theta_surf - th_v_surf)
            heat_tend_surf = shf_kinematic / dz_m_surf
            
            tend_th_v = jnp.zeros_like(state['th_v']).at[:, :, 0].set(heat_tend_surf)
        else:
            tend_th_v = jnp.zeros_like(state['th_v'])
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v}

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