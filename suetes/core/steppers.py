import jax
import jax.numpy as jnp
import jax.lax.linalg as jla

class HEVIStepper:
    def __init__(self, physics, dt):
        self.physics = physics
        self.dt = dt
        self.beta1 = 0.65 # ICON recommended off-centering
        self.beta2 = 1.0 - self.beta1
        
        def _solve_banded_wrapper(ab, b):
            m = ab.shape[1]
            du = jnp.zeros(m, dtype=ab.dtype)
            du = du.at[:-1].set(ab[0, 1:])
            dl = jnp.zeros(m, dtype=ab.dtype)
            dl = dl.at[1:].set(ab[2, :-1])
            d = ab[1, :]
            res = jla.tridiagonal_solve(dl, d, du, b.astype(ab.dtype).reshape(-1, 1))
            return res.squeeze(-1)

        self.solve_columns = jax.vmap(_solve_banded_wrapper, in_axes=(0, 0))

    def _calc_vert_div(self, w_eval, w_upwind, rho, th_v):
        # CHANGED TO TVD
        rho_w = self.physics.op.tvd_interp_m_to_w(rho, w_upwind)
        th_v_w = self.physics.op.tvd_interp_m_to_w(th_v, w_upwind)
        flux = w_eval * rho_w * th_v_w
        
        flux = flux.at[:, 0].set(0.0)
        flux = flux.at[:, -1].set(0.0)
        return self.physics.op.diff_z_w_to_m(flux)

    def _calc_mass_div(self, w_eval, w_upwind, rho):
        # CHANGED TO TVD
        rho_w = self.physics.op.tvd_interp_m_to_w(rho, w_upwind)
        flux = w_eval * rho_w
        flux = flux.at[:, 0].set(0.0)
        flux = flux.at[:, -1].set(0.0)
        return self.physics.op.diff_z_w_to_m(flux)

    def _implicit_solve(self, current_state, u_val, F_w_exp, F_rho_exp, F_pi_exp, w_upwind):
        w_tilde = current_state['w'] + self.dt * F_w_exp
        vert_div_n = self._calc_vert_div(current_state['w'], w_upwind, current_state['rho'], current_state['th_v'])
        
        C_pi = (self.physics.c['Rd'] / self.physics.c['cvd']) * (current_state['pi'] / (current_state['rho'] * current_state['th_v']))
        
        # Isolate pi_prime for the implicit solve
        pi_prime_n = current_state['pi'] - self.physics.pi_bg
        pi_tilde_prime = pi_prime_n + self.dt * F_pi_exp - (self.dt * self.beta2 * C_pi) * vert_div_n
        
        ab, K_face, Gamma, M_face = self.physics.build_helmholtz(current_state, self.dt, self.beta1)
        
        R = jnp.zeros_like(current_state['w'])
        R = R.at[:, 1:-1].set(w_tilde[:, 1:-1] - K_face * (pi_tilde_prime[:, 1:] - pi_tilde_prime[:, :-1]))
        
        w_new = self.solve_columns(ab, R)
        
        vert_div_new = self._calc_vert_div(w_new, w_upwind, current_state['rho'], current_state['th_v'])
        pi_prime_new = pi_tilde_prime - (self.dt * self.beta1 * C_pi) * vert_div_new
        
        # Add the analytical background back to the newly solved perturbation
        pi_new = pi_prime_new + self.physics.pi_bg
        rho_new = current_state['rho'] + self.dt * F_rho_exp - self.dt * self._calc_mass_div(w_new, w_upwind, current_state['rho'])
        
        th_v_new = current_state['th_v'] * (current_state['rho'] / rho_new) * \
                   (1.0 + (self.physics.c['cvd'] / self.physics.c['Rd']) * (pi_new / current_state['pi'] - 1.0))
        
        return {'u': u_val, 'w': w_new, 'rho': rho_new, 'pi': pi_new, 'th_v': th_v_new}

    def step(self, state, t, forcing, bc_fn):
        # 4th-order Hyperdiffusion to suppress noise
        nu4 = 0.001 * (self.physics.grid.dx ** 4) / self.dt
        diff_u = self.physics.op.hyper_diff_2d(state['u'], nu4)
        diff_w = self.physics.op.hyper_diff_2d(state['w'], nu4)
        
        # --- PREDICTOR ---
        adv_u_n, adv_w_n = self.physics.get_advection(state)
        P_u_n, P_w_n = self.physics.get_pressure_buoyancy(state, self.beta1)
        
        F_u_n = -adv_u_n + P_u_n + diff_u
        F_w_n = -adv_w_n + P_w_n + diff_w
        
        u_star = state['u'] + self.dt * F_u_n
        F_rho_n, F_pi_n = self.physics.explicit_scalar_forcing(state, u_star)
        
        state_star = self._implicit_solve(state, u_star, F_w_n, F_rho_n, F_pi_n, w_upwind=state['w'])
        state_star = bc_fn(state_star, forcing)
        
        # --- CORRECTOR ---
        adv_u_star, adv_w_star = self.physics.get_advection(state_star)
        
        # Only blend the advection terms
        adv_u_blend = 0.75 * adv_u_star + 0.25 * adv_u_n
        adv_w_blend = 0.75 * adv_w_star + 0.25 * adv_w_n
        
        F_u_blend = -adv_u_blend + P_u_n + diff_u
        F_w_blend = -adv_w_blend + P_w_n + diff_w
        
        u_next = state['u'] + self.dt * F_u_blend
        F_rho_star, F_pi_star = self.physics.explicit_scalar_forcing(state_star, u_next)
        
        F_rho_blend = 0.75 * F_rho_star + 0.25 * F_rho_n
        F_pi_blend = 0.75 * F_pi_star + 0.25 * F_pi_n
        
        state_next = self._implicit_solve(state, u_next, F_w_blend, F_rho_blend, F_pi_blend, w_upwind=state['w'])
        return bc_fn(state_next, forcing)