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
    Three-category bulk warm-rain microphysics (Vapor, Cloud, Rain) following
    WRF's module_mp_kessler.F (Kessler 1969; coefficients per Klemp &
    Wilhelmson 1978).

    Per step (in this order, matching WRF):
    1. Rain sedimentation (flux-upstream, density-weighted Marshall-Palmer
       fall speed, Courant-limited sub-stepping) + surface precipitation.
    2. Autoconversion + accretion ($q_c \rightarrow q_r$), implicit-in-$q_r$ form.
    3. Saturation adjustment ($q_v \leftrightarrow q_c$, single Taylor step).
    4. Rain evaporation in sub-saturated air (Kessler ventilation formula).
    5. Latent-heat update re-encoded into virtual potential temperature.

    Sedimentation requires `grid` (for layer thickness) and the model `dt`;
    constructed without a grid the scheme degrades to the no-fallout legacy
    behaviour (rain forms but never precipitates) — fine for column tests.
    """
    def __init__(self, constants, dt=120.0, grid=None, max_cr=0.75, v_t_max=10.0):
        self.c = constants
        self.dt = float(dt)
        self.grid = grid
        self.Lv = 2.5e6
        self.Rv = 461.5
        self.epsilon = constants.get('epsilon', 0.622)

        # Kessler conversion parameters (WRF module_mp_kessler.F: c1..c4)
        self.k1 = 1e-3       # Autoconversion rate [1/s]
        self.qc0 = 1e-3      # Autoconversion threshold [kg/kg]
        self.k2 = 2.2        # Accretion coefficient [1/s per (kg/kg)^0.875]

        # Sedimentation: static substep count from the fall-Courant limit
        # V_T*dt/dz <= max_cr with V_T capped at v_t_max (WRF uses a dynamic
        # nfall; a static bound keeps the loop jit-stable).
        self.v_t_max = v_t_max
        if grid is not None:
            import numpy as _np
            dz_min = float(_np.min(_np.asarray(grid.dz_m_full)))
            self.nfall = max(1, int(_np.ceil(v_t_max * self.dt / (max_cr * dz_min))))
        else:
            self.nfall = 0

    def _fall_speed(self, rho, qr):
        """Marshall-Palmer terminal velocity [m/s]: 36.34*(rho*qr [g/cm3])^0.1364
        with the sqrt(rho_sfc/rho) density correction, capped at v_t_max."""
        rho_g = 1e-3 * rho                      # kg/m3 -> g/cm3
        vt = 36.34 * jnp.maximum(rho_g * qr, 0.0) ** 0.1364 * jnp.sqrt(rho[:, :, 0:1] / rho)
        return jnp.minimum(vt, self.v_t_max)

    def _sediment(self, qr, rho):
        """Flux-upstream rain fallout, sub-stepped to respect the fall Courant
        limit. Returns (qr_after, precip_step [mm] accumulated at the surface)."""
        import jax as _jax
        dz = self.grid.dz_m_full
        dtfall = self.dt / self.nfall

        def substep(_, carry):
            qr_k, acc = carry
            vt = self._fall_speed(rho, qr_k)
            flux = rho * jnp.maximum(qr_k, 0.0) * vt          # kg/m2/s at centers
            flux_above = jnp.concatenate([flux[:, :, 1:], jnp.zeros_like(flux[:, :, :1])], axis=2)
            qr_k = qr_k + (dtfall / (rho * dz)) * (flux_above - flux)
            acc = acc + flux[:, :, 0] * dtfall                # kg/m2 == mm of water
            return (qr_k, acc)

        qr_out, precip = _jax.lax.fori_loop(
            0, self.nfall, substep, (qr, jnp.zeros(qr.shape[:2], qr.dtype)))
        return jnp.maximum(qr_out, 0.0), precip

    def apply_update(self, state):
        # Ensure tracer arrays exist and are non-negative
        qv = jnp.maximum(state.get('q', jnp.zeros_like(state['th_v'])), 0.0)
        qc = jnp.maximum(state.get('q_c', jnp.zeros_like(state['th_v'])), 0.0)
        qr = jnp.maximum(state.get('q_r', jnp.zeros_like(state['th_v'])), 0.0)
        th_v = state['th_v']
        pi_full = state['pi']
        dt = self.dt

        # Thermodynamic back-out (full vapor + condensate loading)
        Tv = th_v * pi_full
        T = Tv / (1.0 + (1.0 / self.epsilon - 1.0) * qv - qc - qr)
        p = self.c['p0'] * (pi_full ** (self.c['cp'] / self.c['Rd']))
        rho = state.get('rho', p / (self.c['Rd'] * T))

        # 1. SEDIMENTATION + SURFACE PRECIPITATION (needs grid for dz)
        precip_step = jnp.zeros(th_v.shape[:2], th_v.dtype)
        if self.nfall > 0:
            qr, precip_step = self._sediment(qr, rho)

        # 2. AUTOCONVERSION + ACCRETION (WRF implicit-in-qr form, x dt)
        factorn = 1.0 / (1.0 + self.k2 * dt * jnp.maximum(qr, 0.0) ** 0.875)
        qrprod = qc * (1.0 - factorn) + factorn * self.k1 * dt * jnp.maximum(qc - self.qc0, 0.0)
        qrprod = jnp.minimum(qrprod, qc)
        qc = qc - qrprod
        qr = qr + qrprod

        # 3. SATURATION ADJUSTMENT (single Taylor step; latent heating)
        e_s = 611.2 * jnp.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (self.epsilon * e_s) / (p - (1.0 - self.epsilon) * e_s)
        prod = (qv - q_s) / (1.0 + (self.Lv**2 * q_s) / (self.c['cp'] * self.Rv * T**2))
        product = jnp.maximum(prod, -qc)       # evaporation limited by available qc

        # 4. RAIN EVAPORATION in sub-saturated air (WRF ventilation formula, p in Pa)
        rho_g = 1e-3 * rho                     # g/cm3
        rqr = jnp.maximum(rho_g * qr, 0.0)
        ern = (dt * ((1.6 + 124.9 * rqr ** 0.2046) * rqr ** 0.525)
               / (2.55e8 / (p * q_s) + 5.4e5)
               * jnp.maximum(q_s - qv, 0.0) / jnp.clip(rho_g * q_s, self.c.get('eps', 1e-20), None))
        ern = jnp.minimum(ern, jnp.maximum(-product - qc, 0.0))
        ern = jnp.minimum(ern, qr)

        # 5. FINAL UPDATE: latent heating, re-encode virtual potential temperature
        T_new = T + (self.Lv / self.c['cp']) * (product - ern)
        new_qv = jnp.maximum(qv - product + ern, 0.0)
        new_qc = qc + product
        new_qr = qr - ern
        new_th_v = (T_new * (1.0 + (1.0 / self.epsilon - 1.0) * new_qv - new_qc - new_qr)) / pi_full

        return {'q': new_qv, 'q_c': new_qc, 'q_r': new_qr, 'th_v': new_th_v,
                'precip_step': precip_step}

                
class SimplifiedBettsMiller:
    r"""
    A differentiable Betts-Miller-style convective adjustment.

    Relaxes moist, convectively-active columns toward a reference humidity
    profile ($q_{ref} = RH_{ref} \cdot q_s$) on a timescale $\tau_{adj}$,
    raining out the removed vapor and releasing its latent heat. This is the
    subgrid moisture vent the grid-scale (Kessler) path lacks: it condenses
    BELOW grid-box saturation wherever the smooth convective trigger is
    active, which is what keeps RH from ratcheting toward 100% everywhere.

    Run BEFORE the grid-scale microphysics in the update sequence. Returns the
    column-integrated convective precipitation of this step in
    `precip_conv_step` [mm] (mass-weighted; needs `grid` for layer depths).
    """
    def __init__(self, constants, dt=120.0, grid=None, tau_adj=7200.0, rh_ref=0.85):
        self.c = constants
        self.dt = float(dt)
        self.grid = grid
        self.tau_adj = tau_adj  # Relaxation time (~2 hours)
        self.rh_ref = rh_ref    # Reference relative humidity of the adjusted column
        self.Lv = 2.5e6
        self.Rv = 461.5
        self.epsilon = constants.get('epsilon', 0.622)

    def apply_update(self, state):
        th_v = state['th_v']
        qv = jnp.maximum(state.get('q', jnp.zeros_like(th_v)), 0.0)
        qc = jnp.maximum(state.get('q_c', jnp.zeros_like(th_v)), 0.0)
        qr = jnp.maximum(state.get('q_r', jnp.zeros_like(th_v)), 0.0)
        pi_full = state['pi']

        # Thermodynamic back-out with the FULL vapor + condensate loading
        # (the old version dropped -qc-qr, shifting th_v by the condensate mass)
        Tv = th_v * pi_full
        T = Tv / (1.0 + (1.0 / self.epsilon - 1.0) * qv - qc - qr)
        p = self.c['p0'] * (pi_full ** (self.c['cp'] / self.c['Rd']))

        # Saturation specific humidity (Bolton)
        e_s = 611.2 * jnp.exp(17.67 * (T - 273.15) / (T - 29.65))
        q_s = (self.epsilon * e_s) / (p - (1.0 - self.epsilon) * e_s)

        # Target profile and smooth convective trigger: active where the layer
        # humidity approaches saturation (differentiable CAPE proxy)
        q_ref = q_s * self.rh_ref
        moisture_trigger = jnp.clip((qv - 0.6 * q_s) / (0.3 * q_s + self.c.get('eps_s', 1e-8)), 0.0, 1.0)

        # Relax toward q_ref over tau_adj; only drying (condensing) adjustments
        dq = (qv - q_ref) / self.tau_adj * self.dt
        dq_applied = jnp.where(dq > 0.0, dq * moisture_trigger, 0.0)

        new_qv = qv - dq_applied
        T_new = T + (self.Lv / self.c['cp']) * dq_applied
        new_th_v = (T_new * (1.0 + (1.0 / self.epsilon - 1.0) * new_qv - qc - qr)) / pi_full

        # Convective precipitation: the removed vapor falls out immediately
        # (standard BM). Mass-weighted column integral -> kg/m2 == mm.
        if self.grid is not None:
            rho = p / (self.c['Rd'] * T)
            precip_conv = jnp.sum(rho * dq_applied * self.grid.dz_m_full, axis=2)
        else:
            precip_conv = jnp.zeros(th_v.shape[:2], th_v.dtype)

        return {'q': new_qv, 'th_v': new_th_v, 'precip_conv_step': precip_conv}
