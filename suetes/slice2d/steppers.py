import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

def vmap_bicubic_interp(field, coords, periodic_x=False, true_nx=None):
    nx_arr, nz = field.shape
    nx = true_nx if true_nx is not None else nx_arr  
    x, z = coords[0], coords[1]

    z = jnp.clip(z, 0, nz - 1)
    z_idx = jnp.floor(z).astype(jnp.int32)
    dz = z - z_idx

    zm1 = jnp.maximum(z_idx - 1, 0)
    z0  = z_idx
    z1  = jnp.minimum(z_idx + 1, nz - 1)
    z2  = jnp.minimum(z_idx + 2, nz - 1)

    if periodic_x:
        x = x % nx  
        x_idx = jnp.floor(x).astype(jnp.int32)
        dx = x - x_idx
        xm1, x0, x1, x2 = (x_idx - 1) % nx, x_idx, (x_idx + 1) % nx, (x_idx + 2) % nx
    else:
        x = jnp.clip(x, 0, nx_arr - 1)
        x_idx = jnp.floor(x).astype(jnp.int32)
        dx = x - x_idx
        xm1 = jnp.maximum(x_idx - 1, 0)
        x0  = x_idx
        x1  = jnp.minimum(x_idx + 1, nx_arr - 1)
        x2  = jnp.minimum(x_idx + 2, nx_arr - 1)

    def cubic(p0, p1, p2, p3, t):
        return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3) * t**3 + \
               (p0 - 2.5*p1 + 2.0*p2 - 0.5*p3) * t**2 + \
               (-0.5*p0 + 0.5*p2) * t + p1

    row_m1 = cubic(field[xm1, zm1], field[x0, zm1], field[x1, zm1], field[x2, zm1], dx)
    row_0  = cubic(field[xm1, z0],  field[x0, z0],  field[x1, z0],  field[x2, z0],  dx)
    row_1  = cubic(field[xm1, z1],  field[x0, z1],  field[x1, z1],  field[x2, z1],  dx)
    row_2  = cubic(field[xm1, z2],  field[x0, z2],  field[x1, z2],  field[x2, z2],  dx)

    return cubic(row_m1, row_0, row_1, row_2, dz)

class SemiLagrangianAdvector:
    def __init__(self, grid, physics, dt):
        self.grid = grid
        self.physics = physics 
        self.dt = dt

    def _get_logical_velocities(self, u_phys, eta_dot, loc='m'):
        eta_dot_m_s = eta_dot * self.grid.dz 
        
        if loc == 'm':
            u_loc, w_loc = self.physics.op.avg_u_to_m(u_phys), self.physics.op.avg_w_to_m(eta_dot_m_s)
        elif loc == 'u':
            u_loc, w_loc = u_phys, self.physics.op.avg_w_to_u(eta_dot_m_s)
        elif loc == 'w':
            u_loc, w_loc = self.physics.op.avg_u_to_w(u_phys), eta_dot_m_s
        return u_loc, w_loc

    def compute_departure_indices(self, u_phys, eta_dot, loc='m', iterations=1):
        u_log, w_log = self._get_logical_velocities(u_phys, eta_dot, loc)
        
        nx, nz = self.grid.nx, self.grid.nz
        if loc == 'm':
            idx_x_arr, idx_z_arr = jnp.arange(nx) + 0.5, jnp.arange(nz) + 0.5
            offset_x, offset_z = 0.5, 0.5
        elif loc == 'u':
            idx_x_arr, idx_z_arr = jnp.arange(nx + 1), jnp.arange(nz) + 0.5
            offset_x, offset_z = 0.0, 0.5
        elif loc == 'w':
            idx_x_arr, idx_z_arr = jnp.arange(nx) + 0.5, jnp.arange(nz + 1)
            offset_x, offset_z = 0.5, 0.0

        Xi_arr, Zeta_arr = jnp.meshgrid(idx_x_arr * self.grid.dx, idx_z_arr * self.grid.dz, indexing='ij')
        Xi_dep, Zeta_dep = Xi_arr, Zeta_arr
        
        for _ in range(iterations):
            Xi_mid = 0.5 * (Xi_arr + Xi_dep)
            Zeta_mid = 0.5 * (Zeta_arr + Zeta_dep)
            
            idx_x = (Xi_mid / self.grid.dx) - offset_x
            idx_z = (Zeta_mid / self.grid.dz) - offset_z
            coords = jnp.stack([idx_x, idx_z], axis=0)
            
            # Use true_nx for correct staggered periodic wrapping
            u_mid = vmap_bicubic_interp(u_log, coords, periodic_x=self.grid.periodic_x, true_nx=nx)
            w_mid = vmap_bicubic_interp(w_log, coords, periodic_x=self.grid.periodic_x, true_nx=nx) 
            
            Xi_dep = Xi_arr - self.dt * u_mid
            Zeta_dep = Zeta_arr - self.dt * w_mid

        idx_x_dep = (Xi_dep / self.grid.dx) - offset_x
        idx_z_dep = (Zeta_dep / self.grid.dz) - offset_z
        idx_z_dep = jnp.clip(idx_z_dep, 0.0, self.grid.nz - (1 if loc in ['m', 'u'] else 0))
        
        return jnp.stack([idx_x_dep, idx_z_dep], axis=0)

    def advect(self, field, u_phys, w_phys, loc='m'):
        coords = self.compute_departure_indices(u_phys, w_phys, loc)
        return vmap_bicubic_interp(field, coords, periodic_x=self.grid.periodic_x, true_nx=self.grid.nx)

class SemiImplicitSolver:
    def __init__(self, grid, physics, dt):
        self.grid = grid
        self.physics = physics
        self.dt = dt
        self.op = physics.op
        self.pi_scale = 100000.0 

    def precompute_bg(self, bg_state):
        th_v_bg, rho_bg, pi_bg = bg_state['th_v'], bg_state['rho'], bg_state['pi']
        
        dz_m_full = self.grid.Z_w[:, 1:] - self.grid.Z_w[:, :-1]
        dz_w_full = jnp.pad(self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1], ((0,0),(1,1)), mode='edge')

        bg = {
            'th_v_at_u': self.op.avg_m_to_u(th_v_bg),
            'th_v_at_w': self.op.avg_m_to_w(th_v_bg),
            'rho_at_u': self.op.avg_m_to_u(rho_bg),
            'rho_at_w': self.op.avg_m_to_w(rho_bg),
            'dz_m_centers': self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1],
            'dz_m_full': dz_m_full,
            'dz_w_full': dz_w_full,
            'z_xi_u': self.op.diff_x_m_to_u(self.grid.Z_m),
            'z_xi_w': self.physics._get_metrics('w')['z_xi'],
            'C_pi': (self.physics.c['Rd'] / self.physics.c['cvd']) * (pi_bg / (rho_bg * th_v_bg))
        }
        bg['dz_u'] = self.op.avg_m_to_u(bg['dz_m_full'])
        return bg

    def get_tendencies(self, state_prime, bg):
        u_prime, w_prime, pi_prime, eta_dot_prime = state_prime['u'], state_prime['w'], state_prime['pi'], state_prime['eta_dot']
        
        dpi_m = pi_prime[:, 1:] - pi_prime[:, :-1]
        grad_pi_z_inner = dpi_m / bg['dz_m_centers']
        grad_pi_prime_z = jnp.pad(grad_pi_z_inner, ((0,0), (1,1)), mode='edge')
        
        dpi_dxi = self.op.diff_x_m_to_u(pi_prime)
        grad_pi_z_at_u = self.op.avg_m_to_u(self.op.avg_w_to_m(grad_pi_prime_z))
        grad_pi_prime_x = dpi_dxi - bg['z_xi_u'] * grad_pi_z_at_u

        tend_u = -self.physics.c['cp'] * bg['th_v_at_u'] * grad_pi_prime_x
        tend_w = -self.physics.c['cp'] * bg['th_v_at_w'] * grad_pi_prime_z

        flux_x_prime = u_prime * bg['rho_at_u'] * bg['th_v_at_u'] * bg['dz_u']
        div_x_prime = self.op.diff_x_u_to_m(flux_x_prime) / bg['dz_m_full']

        w_contravariant_prime = eta_dot_prime * bg['dz_w_full']
        
        flux_z_prime = w_contravariant_prime * bg['rho_at_w'] * bg['th_v_at_w']
        flux_z_prime = flux_z_prime.at[:, 0].set(0.0)
        flux_z_prime = flux_z_prime.at[:, -1].set(0.0)
        div_z_prime = (flux_z_prime[:, 1:] - flux_z_prime[:, :-1]) / bg['dz_m_full']

        tend_pi = -bg['C_pi'] * (div_x_prime + div_z_prime)
        return {'u': tend_u, 'w': tend_w, 'pi': tend_pi}

    def linear_operator(self, state_prime, bg, beta=0.65):
        tends = self.get_tendencies(state_prime, bg)
        
        # Apply off-centering (beta) to damp acoustic resonance
        L_u = state_prime['u'] - beta * self.dt * tends['u']
        L_w = (1.0 + self.dt * self.physics.tau_damp) * state_prime['w'] - beta * self.dt * tends['w']
        L_pi = state_prime['pi'] - beta * self.dt * tends['pi']
        L_eta_dot = 0.5 * (bg['dz_w_full'] * state_prime['eta_dot'] - state_prime['w'])

        # Bottom Boundary: w - u * dz/dx = 0
        u_avg_bottom = self.op.avg_u_to_m(state_prime['u'])[:, 0]
        L_w = L_w.at[:, 0].set(state_prime['w'][:, 0] - u_avg_bottom * bg['z_xi_w'][:, 0])
        L_eta_dot = L_eta_dot.at[:, 0].set(state_prime['eta_dot'][:, 0])
        
        # Top Boundary: w = 0
        L_w = L_w.at[:, -1].set(state_prime['w'][:, -1])
        L_eta_dot = L_eta_dot.at[:, -1].set(state_prime['eta_dot'][:, -1])
        
        return {'u': L_u, 'w': L_w, 'pi': L_pi, 'eta_dot': L_eta_dot}

    def solve(self, rhs_prime, bg_precomputed):
        rhs_scaled = {
            'u': rhs_prime['u'], 'w': rhs_prime['w'], 'pi': rhs_prime['pi'] * self.pi_scale, 'eta_dot': rhs_prime['eta_dot']
        }

        def A_fn(state_scaled):
            state_prime = {
                'u': state_scaled['u'], 'w': state_scaled['w'], 'pi': state_scaled['pi'] / self.pi_scale, 'eta_dot': state_scaled['eta_dot']
            }
            L_out = self.linear_operator(state_prime, bg_precomputed, beta=0.65)
            return {
                'u': L_out['u'], 'w': L_out['w'], 'pi': L_out['pi'] * self.pi_scale, 'eta_dot': L_out['eta_dot']
            }

        x_sol_scaled, _ = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-5, maxiter=20, restart=10)
        return {
            'u': x_sol_scaled['u'], 'w': x_sol_scaled['w'], 'pi': x_sol_scaled['pi'] / self.pi_scale, 'eta_dot': x_sol_scaled['eta_dot']
        }

class SISLStepper:
    def __init__(self, physics, dt, nu_ratio=0.0):  # Default to 0.0 (no artificial diffusion)
        self.physics = physics
        self.dt = dt
        self.nu_ratio = nu_ratio
        self.advector = SemiLagrangianAdvector(physics.grid, physics, dt)
        self.implicit_solver = SemiImplicitSolver(physics.grid, physics, dt)

    def integrate(self, state, t_start, num_steps, forcing, bc_fn):
        def scan_fn(curr_state, step_idx):
            t_curr = t_start + step_idx * self.dt
            next_state = self.step(curr_state, t_curr, forcing, bc_fn)
            return next_state, None
            
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))
        return final_state

    def step(self, state, t, forcing, bc_fn):
        u_n, w_n = state['u'], state['w']
        eta_dot_n = state.get('eta_dot', jnp.zeros_like(w_n))
        periodic_x = self.physics.grid.periodic_x
        nx, nz = self.physics.grid.nx, self.physics.grid.nz
        
        beta = 0.65  # Acoustic off-centering

        # 1. Compute departure points using eta_dot directly
        coords_u = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='u')
        coords_w = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='w')
        coords_m = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='m')

        # 2. --- EXPLICIT HALF (Time N) ---
        rho_bg = (self.physics.c['p0'] / (self.physics.c['Rd'] * self.physics.theta_bg)) * \
                 (self.physics.pi_bg ** (self.physics.c['cvd'] / self.physics.c['Rd']))

        bg_state_ref = {'rho': rho_bg, 'pi': self.physics.pi_bg, 'th_v': self.physics.theta_bg}
        bg_precomputed = self.implicit_solver.precompute_bg(bg_state_ref)
        
        state_prime_n = {'u': state['u'], 'w': state['w'], 'pi': state['pi'] - self.physics.pi_bg, 'eta_dot': eta_dot_n}
        tends_n = self.implicit_solver.get_tendencies(state_prime_n, bg_precomputed)
        
        th_v_prime_n = state['th_v'] - self.physics.theta_bg
        th_v_bg_w = self.physics.op.avg_m_to_w(self.physics.theta_bg)
        th_v_prime_w_n = self.physics.op.avg_m_to_w(th_v_prime_n)
        buoyancy_n = self.physics.c['g'] * (th_v_prime_w_n / th_v_bg_w)
        tends_n['w'] += buoyancy_n

        # Explicit diffusion filtering for numerical dispersion
        min_dx = min(self.physics.grid.dx, self.physics.grid.dz)
        nu4 = self.nu_ratio * (min_dx**4) / self.dt

        diff_u = self.physics.op.hyper_diff_2d(state['u'], nu4)
        diff_w = self.physics.op.hyper_diff_2d(state['w'], nu4)
        diff_th = self.physics.op.hyper_diff_2d(th_v_prime_n, nu4)

        # Apply (1 - beta) weighting and inject hyper-diffusion
        u_adv_in = state['u'] + (1.0 - beta) * self.dt * tends_n['u'] + self.dt * diff_u
        w_adv_in = state['w'] + (1.0 - beta) * self.dt * tends_n['w'] + self.dt * diff_w
        pi_prime_adv_in = state_prime_n['pi'] + (1.0 - beta) * self.dt * tends_n['pi']

        dz_w = bg_precomputed['dz_w_full']
        residual_n = dz_w * eta_dot_n - w_n
        term1 = -0.5 * vmap_bicubic_interp(residual_n, coords_w, periodic_x, true_nx=nx)
        
        X_idx_A_full = jnp.broadcast_to(jnp.arange(nx)[:, None], (nx, nz + 1))
        Z_idx_A_full = jnp.broadcast_to(jnp.arange(nz + 1)[None, :], (nx, nz + 1))
        
        coords_xD_etaA = jnp.stack([coords_w[0], Z_idx_A_full], axis=0)
        coords_xA_etaD = jnp.stack([X_idx_A_full, coords_w[1]], axis=0)
        
        z_A_eta_A = self.physics.grid.Z_w
        z_D_eta_A = vmap_bicubic_interp(self.physics.grid.Z_w, coords_xD_etaA, periodic_x, true_nx=nx)
        z_A_eta_D = vmap_bicubic_interp(self.physics.grid.Z_w, coords_xA_etaD, periodic_x, true_nx=nx)
        z_D_eta_D = vmap_bicubic_interp(self.physics.grid.Z_w, coords_w, periodic_x, true_nx=nx)
        
        term2 = -0.5 / self.dt * (z_A_eta_A - z_D_eta_A)
        term3 = -0.5 / self.dt * (z_A_eta_D - z_D_eta_D)
        R_eta_dot = term1 + term2 + term3

        # 3. --- ADVECTION ---
        rhs_u = vmap_bicubic_interp(u_adv_in, coords_u, periodic_x, true_nx=nx)
        rhs_w = vmap_bicubic_interp(w_adv_in, coords_w, periodic_x, true_nx=nx)
        rhs_pi_prime = vmap_bicubic_interp(pi_prime_adv_in, coords_m, periodic_x, true_nx=nx)
        rho_next = vmap_bicubic_interp(state['rho'], coords_m, periodic_x, true_nx=nx)
        
        # Diffuse the temperature perturbation prior to advection
        th_v_adv_in = state['th_v'] + self.dt * diff_th
        th_v_next = vmap_bicubic_interp(th_v_adv_in, coords_m, periodic_x, true_nx=nx)

        # 4. --- IMPLICIT HALF (Time N+1) ---
        th_v_prime_next = th_v_next - self.physics.theta_bg
        th_v_prime_w_next = self.physics.op.avg_m_to_w(th_v_prime_next)
        buoyancy_next = self.physics.c['g'] * (th_v_prime_w_next / th_v_bg_w)
        
        rhs_w += beta * self.dt * buoyancy_next

        rhs_w = rhs_w.at[:, 0].set(0.0)
        rhs_w = rhs_w.at[:, -1].set(0.0)
        R_eta_dot = R_eta_dot.at[:, 0].set(0.0)
        R_eta_dot = R_eta_dot.at[:, -1].set(0.0)

        rhs_prime = {'u': rhs_u, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}

        # Solve fully coupled implicit system
        # Note: We must also update the solve call to explicitly pass the beta weight
        def solve_with_beta(self, rhs_prime, bg_precomputed):
            rhs_scaled = {'u': rhs_prime['u'], 'w': rhs_prime['w'], 'pi': rhs_prime['pi'] * self.pi_scale, 'eta_dot': rhs_prime['eta_dot']}
            def A_fn(state_scaled):
                state_prime = {'u': state_scaled['u'], 'w': state_scaled['w'], 'pi': state_scaled['pi'] / self.pi_scale, 'eta_dot': state_scaled['eta_dot']}
                L_out = self.linear_operator(state_prime, bg_precomputed, beta=beta)
                return {'u': L_out['u'], 'w': L_out['w'], 'pi': L_out['pi'] * self.pi_scale, 'eta_dot': L_out['eta_dot']}
            x_sol_scaled, _ = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-5, maxiter=20, restart=10)
            return {'u': x_sol_scaled['u'], 'w': x_sol_scaled['w'], 'pi': x_sol_scaled['pi'] / self.pi_scale, 'eta_dot': x_sol_scaled['eta_dot']}
        
        # Monkey patch the local solve function to pass beta
        state_prime_next = solve_with_beta(self.implicit_solver, rhs_prime, bg_precomputed)

        pi_next = state_prime_next['pi'] + self.physics.pi_bg
        rho_next = (self.physics.c['p0'] / (self.physics.c['Rd'] * th_v_next)) * \
                   (pi_next ** (self.physics.c['cvd'] / self.physics.c['Rd']))

        state_next = {
            'u': state_prime_next['u'], 'w': state_prime_next['w'], 'pi': pi_next,
            'rho': rho_next, 'th_v': th_v_next, 'eta_dot': state_prime_next['eta_dot']
        }

        return bc_fn(state_next, forcing)