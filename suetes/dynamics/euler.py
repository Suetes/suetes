import jax.numpy as jnp
from suetes.core.operators import CGridOperator

class ICON2DSlice:
    def __init__(self, grid, constants, damp_height=20000.0, max_damp=0.5, N_bv=0.01):
        self.grid = grid
        self.c = constants
        self.op = CGridOperator(grid.dx, grid.dz, periodic_x=grid.periodic_x) 
        
        self.theta_0 = 300.0
        
        if N_bv > 0.0:
            self.theta_bg = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * self.grid.Z_m)
            pi_factor = (self.c['g']**2) / (self.c['cp'] * self.theta_0 * N_bv**2)
            self.pi_bg = 1.0 + pi_factor * (jnp.exp(-(N_bv**2 / self.c['g']) * self.grid.Z_m) - 1.0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * self.grid.Z_w))
        else:
            # FIX: Ensure theta_bg is a 2D array matching the mass grid
            self.theta_bg = self.theta_0 * jnp.ones_like(self.grid.Z_m) 
            
            self.pi_bg = 1.0 - (self.c['g'] * self.grid.Z_m) / (self.c['cp'] * self.theta_0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0) * jnp.ones_like(self.grid.Z_w)
        
        z_w = self.grid.Z_w[:, :]
        z_top = jnp.max(z_w)
        self.tau_damp = jnp.where(
            z_w > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

    def _get_metrics(self, grid_type: str):
        """Fetch the exact JAX autodiff metrics precomputed in the grid."""
        return self.grid.metrics[grid_type]

    def get_advection(self, state):
        u, w = state['u'], state['w']
        adv_u = self.op.advect_2d(u, u, w, 'u', self._get_metrics('u'))
        adv_w = self.op.advect_2d(w, u, w, 'w', self._get_metrics('w'))
        return adv_u, adv_w

    def get_pressure_buoyancy(self, state, beta1=0.65):
        pi, th_v = state['pi'], state['th_v']
        pi_prime = pi - self.pi_bg
        th_v_prime = th_v - self.theta_bg
        
        dpi_dxi = self.op.diff_x_m_to_u(pi_prime)
        z_xi_u = self.op.diff_x_m_to_u(self.grid.Z_m)
        dpi_dzeta_w = self.op.diff_z_m_to_w(pi_prime)
        dpi_dzeta_u = self.op.avg_m_to_u(self.op.avg_w_to_m(dpi_dzeta_w))
        z_zeta_w = self.op.diff_z_m_to_w(self.grid.Z_m)
        z_zeta_u = self.op.avg_m_to_u(self.op.avg_w_to_m(z_zeta_w))
        
        grad_pi_prime_x = dpi_dxi - (z_xi_u / z_zeta_u) * dpi_dzeta_u
        th_v_at_u = self.op.avg_m_to_u(th_v)
        P_u = -self.c['cp'] * th_v_at_u * grad_pi_prime_x
        
        grad_pi_prime_z = dpi_dzeta_w / z_zeta_w
        th_v_prime_at_w = self.op.avg_m_to_w(th_v_prime)
        th_v_at_w = self.op.avg_m_to_w(th_v)
        
        P_w = -self.c['cp'] * (th_v_at_w * (1.0 - beta1) * grad_pi_prime_z + th_v_prime_at_w * self.dpi0_dz_w)
        return P_u, P_w

    def explicit_scalar_forcing(self, state, u_next):
        rho, pi, th_v = state['rho'], state['pi'], state['th_v']
        
        rho_at_u = self.op.tvd_interp_m_to_u(rho, u_next)
        th_v_at_u = self.op.tvd_interp_m_to_u(th_v, u_next)

        dz_m = self.grid.Z_w[:, 1:] - self.grid.Z_w[:, :-1]
        dz_u = self.op.avg_m_to_u(dz_m)

        flux_rho_x = u_next * rho_at_u * dz_u
        flux_th_x = u_next * rho_at_u * th_v_at_u * dz_u

        F_rho = -self.op.diff_x_u_to_m(flux_rho_x) / dz_m
        div_th_x = self.op.diff_x_u_to_m(flux_th_x) / dz_m
        
        C_pi = (self.c['Rd'] / self.c['cvd']) * (pi / (rho * th_v))
        F_pi = -C_pi * div_th_x
        
        return F_rho, F_pi

    def build_helmholtz(self, state, dt, beta1=0.65):
        rho, pi, th_v = state['rho'], state['pi'], state['th_v']
        nx, nz = pi.shape
        
        rho_w = self.op.tvd_interp_m_to_w(rho, state['w'])
        th_v_w = self.op.tvd_interp_m_to_w(th_v, state['w'])
        
        dz_m = self.grid.Z_w[:, 1:] - self.grid.Z_w[:, :-1]
        dz_w = jnp.pad(self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1], ((0,0),(1,1)), mode='edge')
        
        M_face = (rho_w * th_v_w)[:, 1:-1]
        K_face = dt * self.c['cp'] * th_v_w[:, 1:-1] * (beta1 / dz_w[:, 1:-1])
        
        C_pi = (self.c['Rd'] / self.c['cvd']) * (pi / (rho * th_v))
        Gamma = dt * beta1 * C_pi / dz_m 
        
        A = -K_face[:, 1:] * Gamma[:, 1:-1] * M_face[:, :-1]
        C = -K_face[:, :-1] * Gamma[:, 1:-1] * M_face[:, 1:]
        B = 1.0 + K_face * (Gamma[:, 1:] + Gamma[:, :-1]) * M_face + (dt * self.tau_damp[:, 1:-1])
        
        ab = jnp.zeros((nx, 3, nz + 1))
        ab = ab.at[:, 1, 1:-1].set(B)
        ab = ab.at[:, 1, 0].set(1.0)
        ab = ab.at[:, 1, -1].set(1.0)
        ab = ab.at[:, 0, 2:-1].set(C)
        ab = ab.at[:, 2, 1:-2].set(A)
        
        return ab, K_face, Gamma, M_face