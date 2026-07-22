import jax
import jax.numpy as jnp

class UpperRayleighDamping:
    r"""
    Absorbs vertically propagating waves near the model top.
    
    Applies a sine-squared relaxation profile above a specified activation height:
    $$ \frac{\partial \phi}{\partial t} = -\nu(z) (\phi - \phi_{bg}) $$
    """
    def __init__(self, grid, operators, damp_height=9000.0, tau_max_secs=300.0, constants=None):
        self.grid = grid
        self.op = operators
        self.z_damp = damp_height
        self.nu_max = 1.0 / tau_max_secs
        self.c = constants or {}

    def _get_profile(self, z_array):
        # Branchless sine-squared profile bounded between 0 and nu_max
        depth = jnp.maximum(z_array - self.z_damp, 0.0)
        max_depth = jnp.max(self.grid.Z_w) - self.z_damp
        phase = (jnp.pi / 2.0) * (depth / (max_depth + self.c.get('eps_s', 1e-8)))
        return self.nu_max * jnp.sin(phase)**2

    def get_tendencies(self, state, bg):
        u, v, w = state['u'], state['v'], state['w']
        
        nu_u = self._get_profile(self.grid.Z_m)  
        nu_v = self._get_profile(self.grid.Z_m)
        nu_w = self._get_profile(self.grid.Z_w)
        
        nu_u_face = self.op.avg(nu_u, axis=0, from_loc='m', to_loc='u')
        nu_v_face = self.op.avg(nu_v, axis=1, from_loc='m', to_loc='v')

        # Extract dynamic targets from the state (fallback to current wind if missing so damping = 0)
        target_u = state.get('target_u', u)
        target_v = state.get('target_v', v)

        tend_u = -nu_u_face * (u - target_u)
        tend_v = -nu_v_face * (v - target_v)
        
        # Always relax vertical velocity strictly to 0.0
        tend_w = -nu_w * w

        return {'u': tend_u, 'v': tend_v, 'w': tend_w}

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
        
        if rho_m is not None:
            # Reconstruct full density (Euler3D passes rho_prime in state)
            rho_bg = bg.get('rho', bg.get('rho_bg', 0.0))
            rho_m = rho_m + rho_bg
        else:
            pi_full = state.get('pi')
            if pi_full is None:
                return {'u': jnp.zeros_like(u), 'v': jnp.zeros_like(v)}
            
            # Reconstruct full thermodynamics if they are perturbations
            pi_full = pi_full + bg.get('pi', bg.get('pi_bg', 0.0))
            th_v_full = th_v + bg.get('th_v', bg.get('th_v_bg', 0.0))
            
            T_v = th_v_full * pi_full
            rho_m = self.c['p0'] * pi_full ** (self.c['cp'] / self.c['Rd']) \
                    / (self.c['Rd'] * T_v)

        # Categorically prevent 0/0 NaN if density evaluates to zero
        rho_m = jnp.maximum(rho_m, 1e-4)

        # Winds at mass points
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')

        # Reference (surface) direction
        u0 = u_m[:, :, 0]
        v0 = v_m[:, :, 0]
        speed0 = jnp.sqrt(u0 ** 2 + v0 ** 2 + self.c.get('eps_s', 1e-8))
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