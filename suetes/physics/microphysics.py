import jax.numpy as jnp

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


class KesslerWarmRain:
    r"""
    Three-category bulk microphysics (Vapor, Cloud, Rain).
    
    Handles instantaneous saturation adjustment and prognostic rain formation:
    1. Condensation/Evaporation ($q_v \leftrightarrow q_c$)
    2. Autoconversion ($q_c \rightarrow q_r$)
    3. Accretion ($q_c + q_r \rightarrow q_r$)
    """
    def __init__(self, constants):
        self.c = constants
        self.Lv = 2.5e6  
        self.Rv = 461.5  
        self.epsilon = constants.get('epsilon', 0.622)

        # Kessler autoconversion parameters
        self.k1 = 1e-3       # Autoconversion rate [1/s]
        self.qc0 = 1e-3      # Autoconversion threshold [kg/kg]
        self.k2 = 2.2        # Accretion coefficient

    def apply_update(self, state):
        # Ensure tracer arrays exist
        qv = jnp.maximum(state.get('q', jnp.zeros_like(state['th_v'])), 0.0)
        qc = jnp.maximum(state.get('q_c', jnp.zeros_like(state['th_v'])), 0.0)
        qr = jnp.maximum(state.get('q_r', jnp.zeros_like(state['th_v'])), 0.0)
        th_v = state['th_v']
        pi_full = state['pi']

        # SATURATION ADJUSTMENT (Latent Heating)
        Tv = th_v * pi_full
        T = Tv / (1.0 + (1.0 / self.epsilon - 1.0) * qv - qc - qr)
        p = self.c['p0'] * (pi_full ** (self.c['cp'] / self.c['Rd']))

        e_s = 611.2 * jnp.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (self.epsilon * e_s) / (p - (1.0 - self.epsilon) * e_s)

        # Taylor expansion condensation 
        dq = (qv - q_s) / (1.0 + (self.Lv**2 * q_s) / (self.c['cp'] * self.Rv * T**2))
        dq = jnp.where(dq > 0, dq, jnp.maximum(dq, -qc))

        new_qv = qv - dq
        new_qc = qc + dq
        T_new = T + (self.Lv / self.c['cp']) * dq

        # KESSLER RAIN FORMATION
        # Autoconversion (Cloud -> Rain)
        auto = jnp.where(new_qc > self.qc0, self.k1 * (new_qc - self.qc0), 0.0)
        
        # Accretion (Rain sweeping Cloud)
        acc = self.k2 * new_qc * (jnp.maximum(qr, 0.0) ** 0.875)

        # Limit total conversion to available cloud water
        total_conversion = jnp.minimum(auto + acc, new_qc)
        
        final_qc = new_qc - total_conversion
        final_qr = qr + total_conversion

        # Recompute virtual potential temperature
        new_th_v = (T_new * (1.0 + (1.0 / self.epsilon - 1.0) * new_qv - final_qc - final_qr)) / pi_full

        return {'q': new_qv, 'q_c': final_qc, 'q_r': final_qr, 'th_v': new_th_v}


class SimplifiedBettsMiller:
    r"""
    A differentiable Convective Adjustment scheme.
    
    Relaxes the thermodynamic profile towards a reference state if the column 
    is sufficiently moist and unstable, preventing grid-scale convective storms.
    """
    def __init__(self, constants, tau_adj=7200.0, rh_ref=0.85):
        self.c = constants
        self.tau_adj = tau_adj  # Relaxation time (~2 hours)
        self.rh_ref = rh_ref    # Reference relative humidity
        self.Lv = 2.5e6
        self.epsilon = constants.get('epsilon', 0.622)

    def apply_update(self, state):
        th_v = state['th_v']
        qv = state.get('q', jnp.zeros_like(th_v))
        pi_full = state['pi']
        dt = state.get('dt', 120.0) # Ensure dt is passed in the state dict or initialize it

        # Calculate local temperature and pressure
        Tv = th_v * pi_full
        T = Tv / (1.0 + (1.0 / self.epsilon - 1.0) * qv)
        p = self.c['p0'] * (pi_full ** (self.c['cp'] / self.c['Rd']))

        # Calculate Saturation Specific Humidity
        e_s = 611.2 * jnp.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (self.epsilon * e_s) / (p - (1.0 - self.epsilon) * e_s)

        # Target states
        q_ref = q_s * self.rh_ref
        
        # Convective trigger: Is the lower troposphere moist and unstable?
        # A true scheme integrates CAPE; here we use a smooth differentiable proxy:
        # If actual q is close to q_ref, convection is active.
        moisture_trigger = jnp.clip((qv - 0.6 * q_s) / (0.3 * q_s + 1e-8), 0.0, 1.0)
        
        # Calculate adjustments
        dq = (qv - q_ref) / self.tau_adj * dt
        # Condensation heating
        dT = (self.Lv / self.c['cp']) * dq
        
        # Only apply where trigger is active and dq is positive (condensation)
        dq_applied = jnp.where(dq > 0, dq * moisture_trigger, 0.0)
        
        new_qv = qv - dq_applied
        T_new = T + (self.Lv / self.c['cp']) * dq_applied
        new_th_v = (T_new * (1.0 + (1.0 / self.epsilon - 1.0) * new_qv)) / pi_full

        # Add precipitation to a diagnostic rain field if desired
        precip_rate = jnp.sum(dq_applied, axis=-1) # Integrate vertically

        return {'q': new_qv, 'th_v': new_th_v, 'precip_conv': precip_rate}