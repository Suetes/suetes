import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from .operators import tensor_product_interp_3d

class SemiLagrangianAdvector3D:
    def __init__(self, grid, physics, dt):
        self.grid, self.physics, self.dt, self.op = grid, physics, dt, physics.op

    def _get_logical_velocities(self, u_phys, v_phys, eta_dot, loc='m'):
        # CRITICAL FIX: Convert eta_dot (1/s) to physical logical velocity (m/s)
        eta_dot_m_s = eta_dot * self.grid.dz
        
        if loc == 'm':
            u_loc = self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m')
            v_loc = self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m')
            w_loc = self.op.avg(eta_dot_m_s, axis=2, from_loc='w', to_loc='m')
            m_factor = self.grid.m_factors['m']
        elif loc == 'u':
            u_loc = u_phys
            v_loc = self.op.avg(self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m'), axis=0, from_loc='m', to_loc='u')
            w_loc = self.op.avg(self.op.avg(eta_dot_m_s, axis=2, from_loc='w', to_loc='m'), axis=0, from_loc='m', to_loc='u')
            m_factor = self.grid.m_factors['u']
        elif loc == 'v':
            u_loc = self.op.avg(self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m'), axis=1, from_loc='m', to_loc='v')
            v_loc = v_phys
            w_loc = self.op.avg(self.op.avg(eta_dot_m_s, axis=2, from_loc='w', to_loc='m'), axis=1, from_loc='m', to_loc='v')
            m_factor = self.grid.m_factors['v']
        elif loc == 'w':
            u_loc = self.op.avg(self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m'), axis=2, from_loc='m', to_loc='w')
            v_loc = self.op.avg(self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m'), axis=2, from_loc='m', to_loc='w')
            w_loc = eta_dot_m_s
            m_factor = self.grid.m_factors['w']

        return u_loc * m_factor, v_loc * m_factor, w_loc

    def compute_departure_indices(self, state, loc='m', iterations=1):
        u_log, v_log, w_log = self._get_logical_velocities(state['u'], state['v'], state['eta_dot'], loc)
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        
        idx_x = jnp.arange(nx + (1 if loc == 'u' else 0)) + (0.0 if loc == 'u' else 0.5)
        idx_y = jnp.arange(ny + (1 if loc == 'v' else 0)) + (0.0 if loc == 'v' else 0.5)
        idx_z = jnp.arange(nz + (1 if loc == 'w' else 0)) + (0.0 if loc == 'w' else 0.5)
        Xi, Yi, Zi = jnp.meshgrid(idx_x * self.grid.dx, idx_y * self.grid.dy, idx_z * self.grid.dz, indexing='ij')
        
        for _ in range(iterations):
            Xi_dep = Xi - self.dt * u_log
            Yi_dep = Yi - self.dt * v_log
            Zi_dep = Zi - self.dt * w_log

        offset_x = 0.0 if loc == 'u' else 0.5
        offset_y = 0.0 if loc == 'v' else 0.5
        offset_z = 0.0 if loc == 'w' else 0.5
        
        idx_x_dep = (Xi_dep / self.grid.dx) - offset_x
        idx_y_dep = (Yi_dep / self.grid.dy) - offset_y
        idx_z_dep = (Zi_dep / self.grid.dz) - offset_z
        return jnp.stack([idx_x_dep, idx_y_dep, idx_z_dep], axis=0)

    def advect(self, field, coords):
        return tensor_product_interp_3d(field, coords)


class SemiImplicitSolver3D:
    def __init__(self, physics, dt):
        self.physics, self.dt = physics, dt
        self.pi_scale = 100000.0 

    def solve(self, rhs_prime, bg_precomputed):
        rhs_scaled = {k: rhs_prime[k] * self.pi_scale if k == 'pi' else rhs_prime[k] for k in rhs_prime}

        def A_fn(state_scaled):
            state_prime = {k: state_scaled[k] / self.pi_scale if k == 'pi' else state_scaled[k] for k in state_scaled}
            L_out = self.physics.linear_operator(state_prime, bg_precomputed, self.dt)
            return {k: L_out[k] * self.pi_scale if k == 'pi' else L_out[k] for k in L_out}

        x_sol_scaled, _ = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-5, maxiter=20, restart=10)
        return {k: x_sol_scaled[k] / self.pi_scale if k == 'pi' else x_sol_scaled[k] for k in x_sol_scaled}


class SISLStepper3D:
    def __init__(self, physics, dt):
        self.physics, self.dt = physics, dt
        self.advector = SemiLagrangianAdvector3D(physics.grid, physics, dt)
        self.implicit_solver = SemiImplicitSolver3D(physics, dt)

    def step(self, state):
        if 'eta_dot' not in state: state['eta_dot'] = jnp.zeros_like(state['w'])
            
        coords_u = self.advector.compute_departure_indices(state, loc='u')
        coords_v = self.advector.compute_departure_indices(state, loc='v')
        coords_w = self.advector.compute_departure_indices(state, loc='w')
        coords_m = self.advector.compute_departure_indices(state, loc='m')

        bg_state_ref = {
            'rho': self.physics.c['p0'] / (self.physics.c['Rd'] * self.physics.theta_bg) * \
                   (self.physics.pi_bg ** (self.physics.c['cvd'] / self.physics.c['Rd'])),
            'pi': self.physics.pi_bg,
            'th_v': self.physics.theta_bg
        }
        bg_precomputed = self.physics.precompute_bg(bg_state_ref)
        state_prime_n = {
            'u': state['u'], 'v': state['v'], 'w': state['w'], 
            'pi': state['pi'] - self.physics.pi_bg, 'eta_dot': state['eta_dot']
        }
        tends_n = self.physics.get_tendencies(state_prime_n, bg_precomputed)
        
        th_v_prime_n = state['th_v'] - self.physics.theta_bg
        th_v_bg_w = self.physics.op.avg(self.physics.theta_bg, axis=2, from_loc='m', to_loc='w')
        th_v_prime_w_n = self.physics.op.avg(th_v_prime_n, axis=2, from_loc='m', to_loc='w')
        tends_n['w'] += self.physics.c['g'] * (th_v_prime_w_n / th_v_bg_w)

        u_in = state['u'] + 0.5 * self.dt * tends_n['u']
        v_in = state['v'] + 0.5 * self.dt * tends_n['v']
        w_in = state['w'] + 0.5 * self.dt * tends_n['w']
        pi_prime_in = state_prime_n['pi'] + 0.5 * self.dt * tends_n['pi']

        residual_n = bg_precomputed['dz_w_full'] * state['eta_dot'] - state['w']
        R_eta_dot = -0.5 * self.advector.advect(residual_n, coords_w)

        rhs_u = self.advector.advect(u_in, coords_u)
        rhs_v = self.advector.advect(v_in, coords_v)
        rhs_w = self.advector.advect(w_in, coords_w)
        rhs_pi_prime = self.advector.advect(pi_prime_in, coords_m)
        rho_next = self.advector.advect(state['rho'], coords_m)
        th_v_next = self.advector.advect(state['th_v'], coords_m)

        th_v_prime_next = th_v_next - self.physics.theta_bg
        th_v_prime_w_next = self.physics.op.avg(th_v_prime_next, axis=2, from_loc='m', to_loc='w')
        rhs_w += 0.5 * self.dt * (self.physics.c['g'] * (th_v_prime_w_next / th_v_bg_w))

        # CRITICAL FIX: Zero boundaries before implicit solver to prevent acoustic bleed/imprinting
        rhs_w = rhs_w.at[:, :, 0].set(0.0)
        rhs_w = rhs_w.at[:, :, -1].set(0.0)
        R_eta_dot = R_eta_dot.at[:, :, 0].set(0.0)
        R_eta_dot = R_eta_dot.at[:, :, -1].set(0.0)

        rhs_prime = {'u': rhs_u, 'v': rhs_v, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}
        state_prime_next = self.implicit_solver.solve(rhs_prime, bg_precomputed)
        
        return {
            'u': state_prime_next['u'], 'v': state_prime_next['v'], 'w': state_prime_next['w'], 
            'pi': state_prime_next['pi'] + self.physics.pi_bg,
            'rho': rho_next, 'th_v': th_v_next, 'eta_dot': state_prime_next['eta_dot']
        }