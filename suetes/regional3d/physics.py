import jax.numpy as jnp

class PhysicsSuite:
    """Unified API for executing all physics parameterizations."""
    def __init__(self):
        self.tendency_schemes = []
        self.update_schemes = []
        self.tracer_keys = []  # Let the physics module own the tracers

    def add_tendency_scheme(self, scheme):
        self.tendency_schemes.append(scheme)

    def add_update_scheme(self, scheme):
        self.update_schemes.append(scheme)

    def register_tracer(self, key):
        if key not in self.tracer_keys:
            self.tracer_keys.append(key)

    def get_explicit_tendencies(self, state, bg):
        tends_total = {'u': jnp.zeros_like(state['u']), 
                       'v': jnp.zeros_like(state['v']), 
                       'w': jnp.zeros_like(state['w'])}
        
        for scheme in self.tendency_schemes:
            scheme_tends = scheme.get_tendencies(state, bg)
            for k in scheme_tends:
                tends_total[k] += scheme_tends[k]
                
        return tends_total

    def apply_state_updates(self, state):
        """Sequentially applies instantaneous thermodynamic adjustments."""
        updated_state = state.copy()
        for scheme in self.update_schemes:
            # Each scheme returns a dictionary of updated variables
            updates = scheme.apply_update(updated_state)
            updated_state.update(updates)
        return updated_state


class SmagorinskyLillySGS:
    """3D Subgrid-Scale Turbulence Closure using the Smagorinsky-Lilly model."""
    def __init__(self, grid, operators, constants, Cs=0.15, Pr_t=1.0, critical_Ri=0.25):
        self.grid = grid
        self.op = operators
        self.c = constants
        self.Cs = Cs
        self.Pr_t = Pr_t
        self.Ri_c = critical_Ri

    def get_tendencies(self, state, bg):
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

class BulkAerodynamicPBL:
    def __init__(self, grid, operators, Cd_land=0.005, Cd_ocean=0.001):
        self.grid = grid
        self.op = operators
        
        # For now, we use a uniform drag coefficient (Later, we can map Cd_land and Cd_ocean based on the topography/land-mask)
        self.Cd = Cd_ocean 

    def get_tendencies(self, state, bg):
        """
        Calculates the frictional deceleration for the lowest model layer.
        Returns tendencies in units of [m/s^2].
        """
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
        
        # Construct the 3D tendency arrays (zeros everywhere except the surface)
        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)
        
        return {'u': tend_u, 'v': tend_v}


class SimpleMicrophysics:
    def __init__(self, constants):
        self.c = constants
        self.Lv = 2.5e6  # Latent heat of vaporization [J/kg]
        self.Rv = 461.5  # Gas constant for water vapor [J/(kg K)]
        self.epsilon = constants.get('epsilon', 0.622)

        # =====================================================================
        # PRE-COMPUTED LOOKUP TABLE (LUT)
        # =====================================================================
        # We compute the expensive Tetens exponent once during initialization
        # over the realistic atmospheric temperature range (150K to 330K).
        self.T_table = jnp.linspace(150.0, 330.0, 2000)
        self.es_table = 611.2 * jnp.exp(17.67 * (self.T_table - 273.15) / (self.T_table - 29.65))

    def apply_update(self, state):
        """Conforms to the PhysicsSuite state update API."""
        if 'q' in state:
            # Assume state['pi'] is the full pressure (bg + prime) at this stage
            return self.saturation_adjustment(state, state['pi'])
        return {}

    def saturation_adjustment(self, state, pi_full):
        """
        Fast saturation adjustment using a linear interpolation LUT.
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