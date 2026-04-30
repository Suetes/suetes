import jax
from jax import vmap
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.scipy.linalg import lu_factor, lu_solve
import jax.scipy.ndimage as jnd
from .operators import tensor_product_interp_3d

class VerticalPreconditioner:
    def __init__(self, physics, dt, alpha=0.55):
        self.physics = physics
        self.dt = dt
        self.alpha = alpha

    def build_dense_matrix(self, bg_precomputed):
        nx, ny, nz = self.physics.grid.nx, self.physics.grid.ny, self.physics.grid.nz
        
        # 1. Define your 1D tridiagonal coefficients here
        # (These are placeholders - you will replace them with your actual physics)
        lower = jnp.zeros((nx, ny, nz - 1))  # Sub-diagonal
        main  = jnp.ones((nx, ny, nz))       # Main diagonal
        upper = jnp.zeros((nx, ny, nz - 1))  # Super-diagonal

        # 2. Construct the dense Nz x Nz matrix for all horizontal columns
        # Shape will be (nx, ny, nz, nz)
        i, j = jnp.meshgrid(jnp.arange(nz), jnp.arange(nz), indexing='ij')
        
        # JAX's advanced indexing/where makes building this dense matrix fast
        A_dense = jnp.where(
            i == j, main[..., i],
            jnp.where(
                i == j + 1, lower[..., j],
                jnp.where(i == j - 1, upper[..., i], 0.0)
            )
        )
        return A_dense

    def precompute_lu(self, bg_precomputed):
        """Called ONCE before GMRES to factorize the matrices."""
        A_dense = self.build_dense_matrix(bg_precomputed)
        
        # vmap over the horizontal x (axis 0) and y (axis 1) dimensions
        vmap_lu_factor = vmap(vmap(lu_factor, in_axes=0), in_axes=0)
        
        # lu_and_piv is a tuple: (LU_matrices, pivot_indices)
        self.lu_and_piv = vmap_lu_factor(A_dense)

    def __call__(self, rhs_scaled):
        """Called inside GMRES every iteration to apply M^-1."""
        rhs_pi = rhs_scaled['pi']
        
        # vmap the solve over the horizontal dimensions
        vmap_lu_solve = vmap(vmap(lu_solve, in_axes=(0, 0)), in_axes=(0, 0))
        
        # Fast preconditioned solve
        precond_pi = vmap_lu_solve(self.lu_and_piv, rhs_pi)
        
        # Placeholder for w back-substitution
        precond_w = rhs_scaled['w'] 

        return {
            'u': rhs_scaled['u'],
            'v': rhs_scaled['v'],
            'w': precond_w,
            'pi': precond_pi,
            'eta_dot': rhs_scaled['eta_dot']
        }

class SemiLagrangianAdvector3D:
    def __init__(self, grid, physics, dt):
        self.grid, self.physics, self.dt, self.op = grid, physics, dt, physics.op

    def _get_index_velocities(self, u_phys, v_phys, eta_dot, loc='m'):
        """Returns velocities in units of [array indices / second]."""
        if loc == 'm':
            u_loc = self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m')
            v_loc = self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m')
            w_loc = self.op.avg(eta_dot, axis=2, from_loc='w', to_loc='m')
            m_factor = self.grid.m_factors['m'][..., None]
        elif loc == 'u':
            u_loc = u_phys
            v_loc = self.op.avg(self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m'), axis=0, from_loc='m', to_loc='u')
            w_loc = self.op.avg(self.op.avg(eta_dot, axis=2, from_loc='w', to_loc='m'), axis=0, from_loc='m', to_loc='u')
            m_factor = self.grid.m_factors['u'][..., None]
        elif loc == 'v':
            u_loc = self.op.avg(self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m'), axis=1, from_loc='m', to_loc='v')
            v_loc = v_phys
            w_loc = self.op.avg(self.op.avg(eta_dot, axis=2, from_loc='w', to_loc='m'), axis=1, from_loc='m', to_loc='v')
            m_factor = self.grid.m_factors['v'][..., None]
        elif loc == 'w':
            u_loc = self.op.avg(self.op.avg(u_phys, axis=0, from_loc='u', to_loc='m'), axis=2, from_loc='m', to_loc='w')
            v_loc = self.op.avg(self.op.avg(v_phys, axis=1, from_loc='v', to_loc='m'), axis=2, from_loc='m', to_loc='w')
            w_loc = eta_dot
            m_factor = self.grid.m_factors['w'][..., None]

        u_idx_sec = (u_loc * m_factor) / self.grid.dx
        v_idx_sec = (v_loc * m_factor) / self.grid.dy
        w_idx_sec = w_loc  # eta_dot is already the vertical index crossing rate
        
        return u_idx_sec, v_idx_sec, w_idx_sec

    def compute_departure_indices(self, state, loc='m', iterations=2):
        u_idx_sec, v_idx_sec, w_idx_sec = self._get_index_velocities(state['u'], state['v'], state['eta_dot'], loc)
        
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        idx_x = jnp.arange(nx + (1 if loc == 'u' else 0), dtype=jnp.float64)
        idx_y = jnp.arange(ny + (1 if loc == 'v' else 0), dtype=jnp.float64)
        idx_z = jnp.arange(nz + (1 if loc == 'w' else 0), dtype=jnp.float64)
        Xi_idx, Yi_idx, Zi_idx = jnp.meshgrid(idx_x, idx_y, idx_z, indexing='ij')
        
        # Initial guess (Explicit Euler displacement)
        alpha_x = self.dt * u_idx_sec
        alpha_y = self.dt * v_idx_sec
        alpha_z = self.dt * w_idx_sec
        
        for _ in range(iterations):
            mid_coords = jnp.stack([
                Xi_idx - 0.5 * alpha_x, 
                Yi_idx - 0.5 * alpha_y, 
                Zi_idx - 0.5 * alpha_z
            ], axis=0)
            
            # Use the FAST linear advector for finding the departure points
            u_mid = self.advect_linear(u_idx_sec, mid_coords)
            v_mid = self.advect_linear(v_idx_sec, mid_coords)
            w_mid = self.advect_linear(w_idx_sec, mid_coords)
            
            alpha_x = self.dt * u_mid
            alpha_y = self.dt * v_mid
            alpha_z = self.dt * w_mid

        return jnp.stack([Xi_idx - alpha_x, Yi_idx - alpha_y, Zi_idx - alpha_z], axis=0)

    def advect_linear(self, field, coords):
        """Trilinear interpolation. No limiter needed."""
        return jnd.map_coordinates(field, coords, order=1, mode='nearest')

    def advect_cubic(self, field, coords, use_limiter=False):
        """Tricubic interpolation, used ONLY for final dynamics."""
        return tensor_product_interp_3d(field, coords, use_limiter=use_limiter)


class SemiImplicitSolver3D:
    def __init__(self, physics, dt):
        self.physics = physics
        self.dt = dt
        self.pi_scale = 100000.0

    def solve(self, rhs_prime, bg_precomputed):
        # rhs_prime['eta_dot'] is R_eta_dot, which is already a velocity [m/s]
        rhs_scaled = {k: rhs_prime[k] * self.pi_scale if k == 'pi' else rhs_prime[k] for k in rhs_prime}

        # --- 1. Setup Preconditioner ---
        preconditioner = VerticalPreconditioner(self.physics, self.dt)
        preconditioner.precompute_lu(bg_precomputed)  # Factorize ONCE

        def M_fn(state_scaled):
            return preconditioner(state_scaled)       # Fast solve inside GMRES

        def A_fn(state_scaled):
            state_prime = {
                'u': state_scaled['u'], 
                'v': state_scaled['v'], 
                'w': state_scaled['w'],
                'pi': state_scaled['pi'] / self.pi_scale,
                # Decode the solver's velocity (W_contra) back to physical eta_dot [1/s]
                'eta_dot': state_scaled['eta_dot'] / bg_precomputed['dz_w_full']
            }
            L_out = self.physics.linear_operator(state_prime, bg_precomputed, self.dt)
            
            return {
                'u': L_out['u'], 
                'v': L_out['v'], 
                'w': L_out['w'],
                'pi': L_out['pi'] * self.pi_scale,
                'eta_dot': L_out['eta_dot'] 
            }

        x_sol_scaled, _ = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-5, maxiter=10, restart=10, M=M_fn)
        
        return {
            'u': x_sol_scaled['u'], 
            'v': x_sol_scaled['v'], 
            'w': x_sol_scaled['w'],
            'pi': x_sol_scaled['pi'] / self.pi_scale,
            # Convert the final solved velocity back into the true eta_dot [1/s]
            'eta_dot': x_sol_scaled['eta_dot'] / bg_precomputed['dz_w_full']
        }

class FluxFormAdvector:
    def __init__(self, grid, dt):
        self.grid = grid
        self.dt = dt

    def advect_1d(self, scalar_1d, cfl_inter_1d):
        """Advects a 1D scalar using interface Courant numbers."""
        N = scalar_1d.shape[0]
        
        # scalar_1d acts as the logical mass in the grid cell
        M_inter = jnp.pad(jnp.cumsum(scalar_1d), (1, 0))
        
        # Use the dtype of the scalar to maintain float64 precision
        idx_inter = jnp.arange(N + 1, dtype=scalar_1d.dtype)
        idx_dep = idx_inter - cfl_inter_1d
        
        # Solid boundary condition for the advector 
        # (The Davies sponge will handle the open boundaries later)
        idx_dep = jnp.clip(idx_dep, 0.0, float(N))
        
        M_dep = jnd.map_coordinates(M_inter, [idx_dep], order=1, mode='nearest')
        return M_dep[1:] - M_dep[:-1]

    def advect_3d_split(self, field, state, bg_precomputed):
        # 1. Calculate true Courant numbers (index crossing rates)
        m_u = self.grid.m_factors['u'][..., None]
        m_v = self.grid.m_factors['v'][..., None]
        
        cfl_x = (state['u'] * m_u * self.dt) / self.grid.dx
        cfl_y = (state['v'] * m_v * self.dt) / self.grid.dy
        cfl_z = state['eta_dot'] * self.dt
        
        # --- Convert to Absolute Cell Mass ---
        m_sq = self.grid.m_factors['m'][..., None] ** 2
        cell_volumes = (self.grid.dx * self.grid.dy / m_sq) * bg_precomputed['dz_m_full']
        
        # Multiplying volumetric density (field) by volume gives absolute mass (kg)
        cell_mass = field * cell_volumes
        
        # --- X-Advection ---
        vmap_x_inner = jax.vmap(self.advect_1d, in_axes=(1, 1), out_axes=1)
        vmap_x = jax.vmap(vmap_x_inner, in_axes=(2, 2), out_axes=2)
        mass_x = vmap_x(cell_mass, cfl_x)
        
        # --- Y-Advection ---
        vmap_y_inner = jax.vmap(self.advect_1d, in_axes=(0, 0), out_axes=0)
        vmap_y = jax.vmap(vmap_y_inner, in_axes=(2, 2), out_axes=2)
        mass_y = vmap_y(mass_x, cfl_y)
        
        # --- Z-Advection ---
        vmap_z_inner = jax.vmap(self.advect_1d, in_axes=(0, 0), out_axes=0)
        vmap_z = jax.vmap(vmap_z_inner, in_axes=(1, 1), out_axes=1)
        mass_z = vmap_z(mass_y, cfl_z)
        
        # --- Convert Absolute Mass back to Volumetric Density ---
        return mass_z / cell_volumes


class SISLStepper3D:
    def __init__(self, physics, dt, tracer_keys=None):
        self.physics, self.dt = physics, dt
        self.tracer_keys = tracer_keys if tracer_keys is not None else []
        self.advector = SemiLagrangianAdvector3D(physics.grid, physics, dt)
        self.ffsl_advector = FluxFormAdvector(physics.grid, dt)
        self.implicit_solver = SemiImplicitSolver3D(physics, dt)

    def integrate(self, state, t_start, num_steps, forcing, bc_fn):
        """Wraps the step function in a JAX scan loop for fast execution."""
        def scan_fn(curr_state, step_idx):
            t_curr = t_start + step_idx * self.dt
            next_state = self.step(curr_state, t_curr, forcing, bc_fn)
            return next_state, None
            
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))
        return final_state

    def step(self, state, t, forcing, bc_fn):

        alpha = 0.55

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

        u_in = state['u'] + (1.0 - alpha) * self.dt * tends_n['u']
        v_in = state['v'] + (1.0 - alpha) * self.dt * tends_n['v']
        w_in = state['w'] + (1.0 - alpha) * self.dt * tends_n['w']
        pi_prime_in = state_prime_n['pi'] + (1.0 - alpha) * self.dt * tends_n['pi']

        # --- 3D KINEMATIC ADVECTION ---
        u_m = self.physics.op.avg(state['u'], axis=0, from_loc='u', to_loc='m')
        u_w = self.physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
        
        v_m = self.physics.op.avg(state['v'], axis=1, from_loc='v', to_loc='m')
        v_w = self.physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')

        residual_n = (
            bg_precomputed['dz_w_full'] * state['eta_dot'] + 
            u_w * self.physics.grid.z_xi_w + 
            v_w * self.physics.grid.z_eta_w - 
            state['w']
        )
        
        R_eta_dot = -(1.0 - alpha) * self.advector.advect_cubic(residual_n, coords_w)

        # UNLIMITED: Momentum and pressure waves must propagate smoothly
        rhs_u = self.advector.advect_cubic(u_in, coords_u, use_limiter=False)
        rhs_v = self.advector.advect_cubic(v_in, coords_v, use_limiter=False)
        rhs_w = self.advector.advect_cubic(w_in, coords_w, use_limiter=False)
        rhs_pi_prime = self.advector.advect_cubic(pi_prime_in, coords_m, use_limiter=False)
        
        # STRICT CONSERVATION: Advect mass with FFSL scheme
        rho_next = self.ffsl_advector.advect_3d_split(state['rho'], state, bg_precomputed)
        
        # INTENSIVE DYNAMICS: Advect virtual potential temperature with Tricubic SL
        th_v_next = self.advector.advect_cubic(state['th_v'], coords_m, use_limiter=False)
        
        # Add the buoyancy correction for w using the cleanly advected th_v
        th_v_prime_next = th_v_next - self.physics.theta_bg
        th_v_prime_w_next = self.physics.op.avg(th_v_prime_next, axis=2, from_loc='m', to_loc='w')
        rhs_w += 0.5 * self.dt * (self.physics.c['g'] * (th_v_prime_w_next / th_v_bg_w))

        # --- ZERO OUT RHS BOUNDARIES FOR KINEMATIC CONSTRAINTS ---
        rhs_w = rhs_w.at[:, :, 0].set(0.0)
        rhs_w = rhs_w.at[:, :, -1].set(0.0)
        R_eta_dot = R_eta_dot.at[:, :, 0].set(0.0)
        R_eta_dot = R_eta_dot.at[:, :, -1].set(0.0)
        
        # --- 3. IMPLICIT SOLVE ---
        rhs_prime = {'u': rhs_u, 'v': rhs_v, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}
        state_prime_next = self.implicit_solver.solve(rhs_prime, bg_precomputed)
        
        # --- 4. ASSEMBLE FINAL STATE ---
        state_next = {
            'u': state_prime_next['u'], 'v': state_prime_next['v'], 'w': state_prime_next['w'], 
            'pi': state_prime_next['pi'] + self.physics.pi_bg,
            'rho': rho_next, 
            'th_v': th_v_next, 
            'eta_dot': state_prime_next['eta_dot']
        }

        # Tracers use FFSL to strictly conserve mass
        for key in self.tracer_keys:
            if key in state:
                rho_tr_next = self.ffsl_advector.advect_3d_split(state['rho'] * state[key], state, bg_precomputed)
                state_next[key] = rho_tr_next / (rho_next + 1e-15)

        return bc_fn(state_next, forcing)