import jax
import jax.numpy as jnp

class Euler3D:
    def __init__(self, grid, operators, constants, damp_height=20000.0, max_damp=0.5, N_bv=0.01):
        self.grid = grid
        self.op = operators
        self.c = constants
        self.theta_0 = 300.0
        
        zeta_c, zeta_m = self.grid.z_c, self.grid.z_m
        theta_bg_1d = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * zeta_m) if N_bv > 0.0 else self.theta_0 * jnp.ones_like(zeta_m)
            
        def integrate_pi(pi_current, theta_val):
            dpi = -(self.c['g'] / (self.c['cp'] * theta_val)) * self.grid.dz
            return pi_current + dpi, pi_current
            
        _, pi_bg_1d = jax.lax.scan(integrate_pi, 1.0, theta_bg_1d)

        target_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        self.theta_bg = jnp.broadcast_to(theta_bg_1d, target_shape)
        self.pi_bg = jnp.broadcast_to(pi_bg_1d, target_shape)
        
        z_w_3d = jnp.broadcast_to(zeta_c, (self.grid.nx, self.grid.ny, self.grid.nz + 1))
        z_top = jnp.max(zeta_c)
        self.tau_damp = jnp.where(
            z_w_3d > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w_3d - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

    def precompute_bg(self, bg_state):
        th_v_bg, rho_bg, pi_bg = bg_state['th_v'], bg_state['rho'], bg_state['pi']
        return {
            'th_v_u': self.op.avg(th_v_bg, axis=0, from_loc='m', to_loc='u'),
            'th_v_v': self.op.avg(th_v_bg, axis=1, from_loc='m', to_loc='v'),
            'th_v_w': self.op.avg(th_v_bg, axis=2, from_loc='m', to_loc='w'),
            'rho_u':  self.op.avg(rho_bg, axis=0, from_loc='m', to_loc='u'),
            'rho_v':  self.op.avg(rho_bg, axis=1, from_loc='m', to_loc='v'),
            'rho_w':  self.op.avg(rho_bg, axis=2, from_loc='m', to_loc='w'),
            'dz_m_full': self.grid.dz_m_full,
            'dz_w_full': self.grid.dz_w_full,
            'dz_u': self.op.avg(self.grid.dz_m_full, axis=0, from_loc='m', to_loc='u'),
            'dz_v': self.op.avg(self.grid.dz_m_full, axis=1, from_loc='m', to_loc='v'),
            'C_pi': (self.c['Rd'] / self.c['cvd']) * (pi_bg / (rho_bg * th_v_bg))
        }

    def get_tendencies(self, state_prime, bg):
        u, v, w, pi, eta_dot = state_prime['u'], state_prime['v'], state_prime['w'], state_prime['pi'], state_prime['eta_dot']
        
        # Pressure Gradients
        grad_pi_x = self.op.diff(pi, axis=0, from_loc='m', to_loc='u')
        grad_pi_y = self.op.diff(pi, axis=1, from_loc='m', to_loc='v')
        grad_pi_z = self.op.diff(pi, axis=2, from_loc='m', to_loc='w')

        tend_u = -self.c['cp'] * bg['th_v_u'] * grad_pi_x
        tend_v = -self.c['cp'] * bg['th_v_v'] * grad_pi_y
        tend_w = -self.c['cp'] * bg['th_v_w'] * grad_pi_z

        # Coriolis
        v_at_u = self.op.avg(self.op.avg(v, axis=1, from_loc='v', to_loc='m'), axis=0, from_loc='m', to_loc='u')
        u_at_v = self.op.avg(self.op.avg(u, axis=0, from_loc='u', to_loc='m'), axis=1, from_loc='m', to_loc='v')
        
        f_u_3d, f_v_3d = jnp.expand_dims(self.grid.f_u, axis=-1), jnp.expand_dims(self.grid.f_v, axis=-1)
        tend_u += f_u_3d * v_at_u
        tend_v -= f_v_3d * u_at_v

        # Divergence
        m_u, m_v, m_m = self.grid.m_factors['u'][..., None], self.grid.m_factors['v'][..., None], self.grid.m_factors['m'][..., None]
        
        # --- Make sure these two lines use dz_u and dz_v! ---
        flux_x = (u * bg['rho_u'] * bg['th_v_u'] * bg['dz_u']) / m_u
        flux_y = (v * bg['rho_v'] * bg['th_v_v'] * bg['dz_v']) / m_v
        
        # (It is correct to still use dz_m_full for the diff division below)
        div_x = self.op.diff(flux_x, axis=0, from_loc='u', to_loc='m') / bg['dz_m_full']
        div_y = self.op.diff(flux_y, axis=1, from_loc='v', to_loc='m') / bg['dz_m_full']

        w_contravariant = eta_dot * bg['dz_w_full']
        flux_z = w_contravariant * bg['rho_w'] * bg['th_v_w']
        flux_z = flux_z.at[:, :, 0].set(0.0)
        flux_z = flux_z.at[:, :, -1].set(0.0)
        div_z = self.op.diff(flux_z, axis=2, from_loc='w', to_loc='m') / bg['dz_m_full']

        tend_pi = -bg['C_pi'] * (div_x + div_y + div_z)
        return {'u': tend_u, 'v': tend_v, 'w': tend_w, 'pi': tend_pi}

    def linear_operator(self, state_prime, bg, dt):
        tends = self.get_tendencies(state_prime, bg)
        
        L_u = state_prime['u'] - 0.5 * dt * tends['u']
        L_v = state_prime['v'] - 0.5 * dt * tends['v']
        L_w = (1.0 + dt * self.tau_damp) * state_prime['w'] - 0.5 * dt * tends['w']
        L_pi = state_prime['pi'] - 0.5 * dt * tends['pi']
        L_eta_dot = 0.5 * (bg['dz_w_full'] * state_prime['eta_dot'] - state_prime['w'])

        # Enforce exact implicit boundaries for flat base logic
        L_w = L_w.at[:, :, 0].set(state_prime['w'][:, :, 0]) 
        L_w = L_w.at[:, :, -1].set(state_prime['w'][:, :, -1])
        L_eta_dot = L_eta_dot.at[:, :, 0].set(state_prime['eta_dot'][:, :, 0])
        L_eta_dot = L_eta_dot.at[:, :, -1].set(state_prime['eta_dot'][:, :, -1])
        
        return {'u': L_u, 'v': L_v, 'w': L_w, 'pi': L_pi, 'eta_dot': L_eta_dot}