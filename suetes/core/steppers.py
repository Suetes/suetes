import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

def vmap_bilinear_interp(field, coords, periodic_x=False):
    """
    Natively vectorized bilinear interpolation.
    Replaces the jax.vmap approach to drastically reduce XLA compilation time.
    """
    nx, nz = field.shape
    x = coords[0] # No need to flatten
    z = coords[1] 

    # 1. Z-axis logic (Always wall-bounded/clipped)
    z = jnp.clip(z, 0, nz - 1)
    z0 = jnp.floor(z).astype(jnp.int32)
    z1 = jnp.minimum(z0 + 1, nz - 1)
    dz = z - z0

    # 2. X-axis logic (Periodic vs Wall-bounded)
    if periodic_x:
        x = x % nx 
        x0 = jnp.floor(x).astype(jnp.int32)
        x1 = (x0 + 1) % nx
    else:
        x = jnp.clip(x, 0, nx - 1)
        x0 = jnp.floor(x).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, nx - 1)
    
    dx = x - x0

    # 3. Direct array indexing (JAX handles this natively and optimally)
    c00 = field[x0, z0]
    c10 = field[x1, z0]
    c01 = field[x0, z1]
    c11 = field[x1, z1]

    # 4. Bilinear interpolation formula
    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    
    return c0 * (1.0 - dz) + c1 * dz

class SemiLagrangianAdvector:
    def __init__(self, grid, physics, dt):
        self.grid = grid
        self.physics = physics 
        self.dt = dt

    def _get_logical_velocities(self, u_phys, w_phys, loc='m'):
        """Converts physical (u, w) into logical contravariant velocities."""
        metrics = self.physics._get_metrics(loc)
        z_xi = metrics['z_xi']
        z_zeta = metrics['z_zeta']
        
        if loc == 'm':
            u_loc = self.physics.op.avg_u_to_m(u_phys)
            w_loc = self.physics.op.avg_w_to_m(w_phys)
        elif loc == 'u':
            u_loc = u_phys
            w_loc = self.physics.op.avg_w_to_u(w_phys)
        elif loc == 'w':
            u_loc = self.physics.op.avg_u_to_w(u_phys)
            w_loc = w_phys

        u_logical = u_loc 
        w_logical = (w_loc - u_loc * z_xi) / z_zeta
        return u_logical, w_logical

    def compute_departure_indices(self, u_phys, w_phys, loc='m', iterations=2):
        u_log, w_log = self._get_logical_velocities(u_phys, w_phys, loc)
        
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
            
            # --- USE NEW INTERPOLATOR HERE ---
            u_mid = vmap_bilinear_interp(u_log, coords, periodic_x=self.grid.periodic_x)
            w_mid = vmap_bilinear_interp(w_log, coords, periodic_x=False) 
            
            Xi_dep = Xi_arr - self.dt * u_mid
            Zeta_dep = Zeta_arr - self.dt * w_mid

        idx_x_dep = (Xi_dep / self.grid.dx) - offset_x
        idx_z_dep = (Zeta_dep / self.grid.dz) - offset_z
        idx_z_dep = jnp.clip(idx_z_dep, 0.0, self.grid.nz - (1 if loc in ['m', 'u'] else 0))
        
        return jnp.stack([idx_x_dep, idx_z_dep], axis=0)

    def advect(self, field, u_phys, w_phys, loc='m'):
        """Interpolates the field to the departure points."""
        coords = self.compute_departure_indices(u_phys, w_phys, loc)
        mode_x = 'wrap' if self.grid.periodic_x else 'nearest'
        # order=1 used due to JAX map_coordinates limitations
        return vmap_bilinear_interp(field, coords, periodic_x=self.grid.periodic_x)

class SemiImplicitSolver:
    def __init__(self, grid, physics, dt):
        self.grid = grid
        self.physics = physics
        self.dt = dt
        self.op = physics.op
        self.pi_scale = 100000.0 

    def linear_operator(self, state_prime, bg):
        """Modified to accept precomputed background metrics (bg)"""
        u_prime = state_prime['u']
        w_prime = state_prime['w']
        pi_prime = state_prime['pi']
        
        # 1. Linearized Pressure Gradients
        dpi_m = pi_prime[:, 1:] - pi_prime[:, :-1]
        grad_pi_z_inner = dpi_m / bg['dz_m_centers']
        grad_pi_prime_z = jnp.pad(grad_pi_z_inner, ((0,0), (1,1)), mode='edge')
        
        dpi_dxi = self.op.diff_x_m_to_u(pi_prime)
        grad_pi_z_at_u = self.op.avg_m_to_u(self.op.avg_w_to_m(grad_pi_prime_z))
        grad_pi_prime_x = dpi_dxi - bg['z_xi_u'] * grad_pi_z_at_u

        L_u = u_prime + self.dt * self.physics.c['cp'] * bg['th_v_at_u'] * grad_pi_prime_x
        L_w = w_prime + self.dt * self.physics.c['cp'] * bg['th_v_at_w'] * grad_pi_prime_z
        L_w = L_w.at[:, 0].set(w_prime[:, 0]) 
        L_w = L_w.at[:, -1].set(w_prime[:, -1])

        # 2. Linearized Divergence
        flux_x_prime = u_prime * bg['rho_at_u'] * bg['th_v_at_u'] * bg['dz_u']
        div_x_prime = self.op.diff_x_u_to_m(flux_x_prime) / bg['dz_m_full']

        u_prime_at_w = self.op.avg_u_to_w(u_prime)
        w_contravariant_prime = w_prime - u_prime_at_w * bg['z_xi_w']
        
        flux_z_prime = w_contravariant_prime * bg['rho_at_w'] * bg['th_v_at_w']
        flux_z_prime = flux_z_prime.at[:, 0].set(0.0)
        flux_z_prime = flux_z_prime.at[:, -1].set(0.0)
        div_z_prime = (flux_z_prime[:, 1:] - flux_z_prime[:, :-1]) / bg['dz_m_full']

        L_pi = pi_prime + self.dt * bg['C_pi'] * (div_x_prime + div_z_prime)

        return {'u': L_u, 'w': L_w, 'pi': L_pi}

    def solve(self, rhs_prime, bg_state):
        # 1. PRECOMPUTE ALL STATIC BACKGROUND DATA ONCE PER TIMESTEP
        th_v_bg, rho_bg, pi_bg = bg_state['th_v'], bg_state['rho'], bg_state['pi']
        
        bg_precomputed = {
            'th_v_at_u': self.op.avg_m_to_u(th_v_bg),
            'th_v_at_w': self.op.avg_m_to_w(th_v_bg),
            'rho_at_u': self.op.avg_m_to_u(rho_bg),
            'rho_at_w': self.op.avg_m_to_w(rho_bg),
            'dz_m_centers': self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1],
            'dz_m_full': self.grid.Z_w[:, 1:] - self.grid.Z_w[:, :-1],
            'z_xi_u': self.op.diff_x_m_to_u(self.grid.Z_m),
            'z_xi_w': self.physics._get_metrics('w')['z_xi'],
            'C_pi': (self.physics.c['Rd'] / self.physics.c['cvd']) * (pi_bg / (rho_bg * th_v_bg))
        }
        
        bg_precomputed['dz_u'] = self.op.avg_m_to_u(bg_precomputed['dz_m_full'])

        # 2. Scale RHS
        rhs_scaled = {
            'u': rhs_prime['u'],
            'w': rhs_prime['w'],
            'pi': rhs_prime['pi'] * self.pi_scale
        }

        def A_fn(state_scaled):
            state_prime = {
                'u': state_scaled['u'],
                'w': state_scaled['w'],
                'pi': state_scaled['pi'] / self.pi_scale
            }
            
            # Pass the precomputed dictionary to avoid redundant math
            L_out = self.linear_operator(state_prime, bg_precomputed)
            
            return {
                'u': L_out['u'],
                'w': L_out['w'],
                'pi': L_out['pi'] * self.pi_scale
            }

        # 3. Optimized GMRES parameters
        x_sol_scaled, _ = gmres(
            A_fn, 
            rhs_scaled, 
            x0=rhs_scaled,
            tol=1e-3,       # Relaxed tolerance (1e-5 is too strict for float32)
            maxiter=10,     # Hard cap iterations to prevent looping infinitely
            restart=10
        )
        
        return {
            'u': x_sol_scaled['u'],
            'w': x_sol_scaled['w'],
            'pi': x_sol_scaled['pi'] / self.pi_scale
        }

class SISLStepper:
    def __init__(self, physics, dt):
        self.physics = physics
        self.dt = dt
        self.advector = SemiLagrangianAdvector(physics.grid, physics, dt)
        self.implicit_solver = SemiImplicitSolver(physics.grid, physics, dt)

    def integrate(self, state, t_start, num_steps, forcing, bc_fn):
        """
        Integrates the state starting at t_start for num_steps.
        num_steps MUST be a concrete Python integer, not a JAX tracer.
        """
        def scan_fn(curr_state, step_idx):
            t_curr = t_start + step_idx * self.dt
            next_state = self.step(curr_state, t_curr, forcing, bc_fn)
            return next_state, None
            
        # num_steps is now guaranteed to be a static Python integer
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))
        
        return final_state

    def step(self, state, t, forcing, bc_fn):
        u_n, w_n = state['u'], state['w']

        coords_u = self.advector.compute_departure_indices(u_n, w_n, loc='u')
        coords_w = self.advector.compute_departure_indices(u_n, w_n, loc='w')
        coords_m = self.advector.compute_departure_indices(u_n, w_n, loc='m')

        periodic_x = self.physics.grid.periodic_x

        rhs_state = {
            'u': vmap_bilinear_interp(state['u'], coords_u, periodic_x),
            'w': vmap_bilinear_interp(state['w'], coords_w, periodic_x),
            'rho': vmap_bilinear_interp(state['rho'], coords_m, periodic_x),
            'pi': vmap_bilinear_interp(state['pi'], coords_m, periodic_x),
            'th_v': vmap_bilinear_interp(state['th_v'], coords_m, periodic_x)
        }

        # Explicit Buoyancy Force on W
        th_v_bg_w = self.physics.op.avg_m_to_w(self.physics.theta_bg)
        th_v_prime_rhs = rhs_state['th_v'] - self.physics.theta_bg
        th_v_prime_w = self.physics.op.avg_m_to_w(th_v_prime_rhs)
        
        buoyancy = self.physics.c['g'] * (th_v_prime_w / th_v_bg_w)
        rhs_state['w'] = rhs_state['w'] + self.dt * buoyancy

        # 2. SEMI-IMPLICIT CORRECTOR (Acoustics)
        # Solve for perturbations to avoid massive condition numbers
        rhs_prime = {
            'u': rhs_state['u'],
            'w': rhs_state['w'],
            'pi': rhs_state['pi'] - self.physics.pi_bg
        }

        bg_state = {
            'rho': state['rho'], 
            'pi': state['pi'],
            'th_v': state['th_v']
        }

        state_prime_next = self.implicit_solver.solve(rhs_prime, bg_state)

        pi_next = state_prime_next['pi'] + self.physics.pi_bg
        th_v_next = rhs_state['th_v']

        rho_next = (self.physics.c['p0'] / (self.physics.c['Rd'] * th_v_next)) * \
                (pi_next ** (self.physics.c['cvd'] / self.physics.c['Rd']))

        state_next = {
            'u': state_prime_next['u'],
            'w': state_prime_next['w'],
            'pi': pi_next,
            'rho': rho_next,
            'th_v': th_v_next
        }

        return bc_fn(state_next, forcing)