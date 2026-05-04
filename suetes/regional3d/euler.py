import jax
import jax.numpy as jnp
from suetes.regional3d.diffusion import HyperFilter

class Euler3D:
    def __init__(self, grid, operators, constants, damp_height=20000.0, max_damp=0.5, N_bv=0.01, nu_h=5e6, nu_v=5e6):
        self.grid = grid
        self.op = operators
        self.c = constants
        self.theta_0 = 300.0
        self.nu_h = nu_h
        self.nu_v = nu_v
        
         # USE PHYSICAL 3D HEIGHT (Z_m), NOT LOGICAL 1D HEIGHT (zeta_m)
        Z_m = self.grid.Z_m
        
        self.theta_bg = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_m) if N_bv > 0.0 else self.theta_0 * jnp.ones_like(Z_m)
        
        # CALCULATE 3D PI ANALYTICALLY
        if N_bv > 0.0:
            self.pi_bg = 1.0 + (self.c['g']**2 / (self.c['cp'] * self.theta_0 * N_bv**2)) * \
                         (jnp.exp(-N_bv**2 * Z_m / self.c['g']) - 1.0)
        else:
            self.pi_bg = 1.0 - (self.c['g'] / (self.c['cp'] * self.theta_0)) * Z_m
        
        z_w_3d = self.grid.Z_w
        z_top = self.grid.Lz 
        
        self.tau_damp = jnp.where(
            z_w_3d > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w_3d - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

        # Initialize the spatial filter
        self.diffusion = HyperFilter(self.grid, nu_h=self.nu_h, nu_v=self.nu_v)

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
            'C_pi': (self.c['Rd'] / self.c['cvd']) * (pi_bg / (rho_bg * th_v_bg)),
            'pi_bg': pi_bg
        }

    def get_tendencies(self, state_prime, bg, is_explicit=False):
        u, v, w, pi_prime, eta_dot = state_prime['u'], state_prime['v'], state_prime['w'], state_prime['pi'], state_prime['eta_dot']
        
        th_v_u = bg['th_v_u'] + state_prime.get('th_v_prime_u', 0.0)
        th_v_v = bg['th_v_v'] + state_prime.get('th_v_prime_v', 0.0)
        th_v_w = bg['th_v_w'] + state_prime.get('th_v_prime_w', 0.0)

        # --- 1. HORIZONTAL PRESSURE GRADIENTS (MUST use pi_prime) ---
        grad_pi_prime_x = self.op.diff(pi_prime, axis=0, from_loc='m', to_loc='u')
        grad_pi_prime_y = self.op.diff(pi_prime, axis=1, from_loc='m', to_loc='v')
        grad_pi_prime_z_w = self.op.diff(pi_prime, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])

        z_xi_u = self.op.diff(self.grid.Z_m, axis=0, from_loc='m', to_loc='u')
        z_eta_v = self.op.diff(self.grid.Z_m, axis=1, from_loc='m', to_loc='v')

        grad_pi_prime_z_m = self.op.avg(grad_pi_prime_z_w, axis=2, from_loc='w', to_loc='m')
        grad_pi_prime_z_u = self.op.avg(grad_pi_prime_z_m, axis=0, from_loc='m', to_loc='u')
        grad_pi_prime_z_v = self.op.avg(grad_pi_prime_z_m, axis=1, from_loc='m', to_loc='v')

        grad_pi_x_cart = grad_pi_prime_x - z_xi_u * grad_pi_prime_z_u
        grad_pi_y_cart = grad_pi_prime_y - z_eta_v * grad_pi_prime_z_v

        tend_u = -self.c['cp'] * th_v_u * grad_pi_x_cart
        tend_v = -self.c['cp'] * th_v_v * grad_pi_y_cart

        # --- 2. VERTICAL PRESSURE GRADIENT ---
        # Take gradient of the perturbation
        grad_pi_prime_z_w = self.op.diff(pi_prime, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])

        if is_explicit:
            # Perturbation form cancels discrete gravity at rest
            th_v_prime_w = state_prime.get('th_v_prime_w', 0.0)
            buoyancy = self.c['g'] * (th_v_prime_w / bg['th_v_w'])
            
            tend_w = -self.c['cp'] * th_v_w * grad_pi_prime_z_w + buoyancy
        else:
            # The implicit solver matrix requires strict linearity
            tend_w = -self.c['cp'] * bg['th_v_w'] * grad_pi_prime_z_w

        # --- 2. CORIOLIS ---
        v_at_u = self.op.avg(self.op.avg(v, axis=1, from_loc='v', to_loc='m'), axis=0, from_loc='m', to_loc='u')
        u_at_v = self.op.avg(self.op.avg(u, axis=0, from_loc='u', to_loc='m'), axis=1, from_loc='m', to_loc='v')
        
        f_u_3d, f_v_3d = jnp.expand_dims(self.grid.f_u, axis=-1), jnp.expand_dims(self.grid.f_v, axis=-1)
        tend_u += f_u_3d * v_at_u
        tend_v -= f_v_3d * u_at_v

        # --- 3. DIVERGENCE ---
        m_u, m_v, m_m = self.grid.m_factors['u'][..., None], self.grid.m_factors['v'][..., None], self.grid.m_factors['m'][..., None]
        
        flux_x = (u * bg['rho_u'] * bg['th_v_u'] * bg['dz_u']) / m_u
        flux_y = (v * bg['rho_v'] * bg['th_v_v'] * bg['dz_v']) / m_v
        
        div_x = self.op.diff(flux_x, axis=0, from_loc='u', to_loc='m') / bg['dz_m_full']
        div_y = self.op.diff(flux_y, axis=1, from_loc='v', to_loc='m') / bg['dz_m_full']

        w_contravariant = eta_dot * bg['dz_w_full']
        flux_z = w_contravariant * bg['rho_w'] * bg['th_v_w']
        flux_z = flux_z.at[:, :, 0].set(0.0)
        flux_z = flux_z.at[:, :, -1].set(0.0)
        
        # Undo the logical dz division from op.diff, divide only by physical dz_m_full
        div_z = (self.op.diff(flux_z, axis=2, from_loc='w', to_loc='m') * self.grid.dz) / bg['dz_m_full']

        tend_pi = -bg['C_pi'] * (div_x + div_y + div_z)

        # --- 4. EXPLICIT DIFFUSION ---
        diff_tends = self.diffusion.get_tendencies(state_prime)

        tend_u += diff_tends['u']
        tend_v += diff_tends['v']
        tend_w += diff_tends['w']

        return {'u': tend_u, 'v': tend_v, 'w': tend_w, 'pi': tend_pi}

    def linear_operator(self, state_prime, bg, dt):
        tends = self.get_tendencies(state_prime, bg, is_explicit=False)
        alpha = 0.55 

        L_u = state_prime['u'] - alpha * dt * tends['u']
        L_v = state_prime['v'] - alpha * dt * tends['v']
        L_w = (1.0 + dt * self.tau_damp) * state_prime['w'] - alpha * dt * tends['w']
        L_pi = state_prime['pi'] - alpha * dt * tends['pi']
        
        # Bring horizontal winds to the W-grid
        u_m = self.op.avg(state_prime['u'], axis=0, from_loc='u', to_loc='m')
        u_w = self.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
        v_m = self.op.avg(state_prime['v'], axis=1, from_loc='v', to_loc='m')
        v_w = self.op.avg(v_m, axis=2, from_loc='m', to_loc='w')

        # --- FIX: OVERRIDE L_w AT BOUNDARIES ---
        # Bottom: Kinematic constraint (w - u*dz/dx - v*dz/dy = 0)
        kinematic_bottom = state_prime['w'][:, :, 0] - (
            u_w[:, :, 0] * self.grid.z_xi_w[:, :, 0] + 
            v_w[:, :, 0] * self.grid.z_eta_w[:, :, 0]
        )
        L_w = L_w.at[:, :, 0].set(kinematic_bottom)
        
        # Top: Rigid lid (w = 0)
        L_w = L_w.at[:, :, -1].set(state_prime['w'][:, :, -1])
        # ---------------------------------------

        # Horizontal terrain advection to the implicit constraint
        L_eta_dot = alpha * (
            bg['dz_w_full'] * state_prime['eta_dot'] 
            + u_w * self.grid.z_xi_w 
            + v_w * self.grid.z_eta_w 
            - state_prime['w']
        )
        
        L_eta_dot = L_eta_dot.at[:, :, 0].set(state_prime['eta_dot'][:, :, 0])
        L_eta_dot = L_eta_dot.at[:, :, -1].set(state_prime['eta_dot'][:, :, -1])
        
        return {'u': L_u, 'v': L_v, 'w': L_w, 'pi': L_pi, 'eta_dot': L_eta_dot}