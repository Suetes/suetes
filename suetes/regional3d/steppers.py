"""
Time Integration and Advection Module.

Implements a Semi-Implicit Semi-Lagrangian (SISL) integration scheme. 
This architecture bypasses the restrictive Eulerian CFL limit by treating 
advection geometrically (trajectory tracing) and treating stiff acoustic 
and gravity wave modes implicitly via a GMRES solver.
"""

import jax
from jax import vmap
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.lax.linalg import tridiagonal_solve
import jax.scipy.ndimage as jnd

from suetes.regional3d.operators import tensor_product_interp_3d
from suetes.regional3d.physics import SimpleMicrophysics

class VerticalPreconditioner:
    r"""
    1D vertical Helmholtz equation solver for implicit preconditioning.

    By analytically eliminating the horizontal wave propagation, the linearized 
    acoustic system is reduced to a vertically implicit tridiagonal matrix:
    
    $$ -\mathcal{L}_{\pi} P_{k-1} \pi'_{k-1} + (1 - \mathcal{L}_{\pi} P_k) \pi'_k - \mathcal{L}_{\pi} P_{k+1} \pi'_{k+1} = \text{RHS} $$

    This exact vertical solve acts as a highly efficient preconditioner $M^{-1}$ 
    for the 3D GMRES solver, drastically reducing the iterations required to 
    resolve high-frequency sound waves.
    """
    def __init__(self, physics, dt, alpha=0.55):
        r"""
        Initializes the vertical preconditioner.

        Args:
            physics (Euler3D): The dynamical core physics configuration.
            dt (float): Integration time step $\Delta t$ [s].
            alpha (float, optional): Semi-implicit off-centering parameter. Defaults to 0.55.
        """
        self.physics = physics
        self.dt = dt
        self.alpha = alpha

    def precompute_banded(self, bg):
        r"""
        Precomputes the 1D tridiagonal coefficients.

        Derives the local coefficients for thermodynamic compressibility ($\mathcal{L}_\pi$) 
        and acoustic wave speed ($P$) mapped to the staggered grid.

        Args:
            bg (dict): Precomputed hydrostatic background state metrics.
        """
        dt, alpha, cp = self.dt, self.alpha, self.physics.c['cp']
        
        self.th_v_w = bg['th_v_w']
        self.dz_w_full = bg['dz_w_full']
        self.rho_w = bg['rho_w']
        self.tau_damp = self.physics.tau_damp
        
        # K_w: Acoustic wave speed / Sponge layer (defined on w-points)
        self.K_w = (alpha * dt * cp * self.th_v_w) / (self.dz_w_full * (1.0 + dt * self.tau_damp))
        
        # P: Mass-weighted acoustic propagation (defined on w-points)
        P = self.K_w * self.rho_w * self.th_v_w
        
        P = P.at[:, :, 0].set(0.0)
        P = P.at[:, :, -1].set(0.0)
        
        # L_pi: Thermodynamic compressibility (defined on mass-points)
        self.L_pi = (alpha * dt * bg['C_pi']) / bg['dz_m_full']
        
        # Tridiagonal matrix assembly
        self.lower = -self.L_pi * P[:, :, :-1]
        self.upper = -self.L_pi * P[:, :, 1:]
        self.main  = 1.0 - self.lower - self.upper

    def __call__(self, rhs_scaled):
        r"""
        Executes the vertical tridiagonal solve.

        Args:
            rhs_scaled (dict): Scaled right-hand side vectors from the GMRES solver.

        Returns:
            dict: The preconditioned state vector $\mathbf{x} = M^{-1}\mathbf{b}$.
        """
        # Unscale for physical math
        pi_scale = 100000.0
        rhs_pi_phys = rhs_scaled['pi'] / pi_scale
        rhs_w_phys = rhs_scaled['w']

        # Kinematic 3d forcing
        u_m = self.physics.op.avg(rhs_scaled['u'], axis=0, from_loc='u', to_loc='m')
        u_w = self.physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
        v_m = self.physics.op.avg(rhs_scaled['v'], axis=1, from_loc='v', to_loc='m')
        v_w = self.physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')
        
        # Apply m_w to the kinematic forcing to match euler.py
        m_w = jnp.expand_dims(self.physics.grid.m_factors['w'], axis=-1)
        kinematic_3d = m_w * (
            u_w * self.physics.grid.z_xi_w + 
            v_w * self.physics.grid.z_eta_w
        )

        # Physical preconditioner solve
        w_contra_known = (rhs_w_phys / (1.0 + self.dt * self.tau_damp)) + (rhs_scaled['eta_dot'] / self.alpha) - kinematic_3d
        
        w_tilde = self.rho_w * self.th_v_w * w_contra_known
        w_tilde = w_tilde.at[:, :, 0].set(0.0)
        w_tilde = w_tilde.at[:, :, -1].set(0.0)
        
        # Multiply by dz to undo op.diff's internal division
        div_w_tilde = self.physics.op.diff(w_tilde, axis=2, from_loc='w', to_loc='m') * self.physics.grid.dz
        rhs_helmholtz = rhs_pi_phys - self.L_pi * div_w_tilde
        
        # Fast tridiagonal solve 
        rhs_helmholtz_expanded = rhs_helmholtz[..., None]
        precond_pi_phys_expanded = tridiagonal_solve(self.lower, self.main, self.upper, rhs_helmholtz_expanded)
        precond_pi_phys = precond_pi_phys_expanded[..., 0]
        
        # Back-substitution for w
        grad_pi = self.physics.op.diff(precond_pi_phys, axis=2, from_loc='m', to_loc='w') * (self.physics.grid.dz / self.dz_w_full)
        
        alpha, dt, cp = self.alpha, self.dt, self.physics.c['cp']
        precond_w_phys = (rhs_w_phys - alpha * dt * cp * self.th_v_w * grad_pi) / (1.0 + dt * self.tau_damp)
        
        # Do not add kinematic_w. The exact RHS is the proper preconditioned value.
        precond_w_phys = precond_w_phys.at[:, :, 0].set(rhs_w_phys[:, :, 0])
        precond_w_phys = precond_w_phys.at[:, :, -1].set(rhs_w_phys[:, :, -1])

        # Back-substitution for eta_dot
        precond_eta_dot = (rhs_scaled['eta_dot'] / self.alpha) + precond_w_phys - kinematic_3d
        
        # At boundary, eta_dot is R_eta_dot * dz, because linear_operator simply evaluates W_contra
        precond_eta_dot = precond_eta_dot.at[:, :, 0].set(rhs_scaled['eta_dot'][:, :, 0] * self.dz_w_full[:, :, 0])
        precond_eta_dot = precond_eta_dot.at[:, :, -1].set(rhs_scaled['eta_dot'][:, :, -1] * self.dz_w_full[:, :, -1])

        return {
            'u': rhs_scaled['u'],
            'v': rhs_scaled['v'],
            'w': precond_w_phys,
            'pi': precond_pi_phys * pi_scale, 
            'eta_dot': precond_eta_dot
        }

class SemiLagrangianAdvector3D:
    r"""
    Computes fluid parcel trajectories and interpolates scalar quantities.

    The scheme traces the arrival point $\mathbf{x}_a$ backward in time to 
    find the departure point $\mathbf{x}_d$ by iteratively solving:
    
    $$ \mathbf{x}_d = \mathbf{x}_a - \Delta t \, \mathbf{v}(\mathbf{x}_{mid}, t_{mid}) $$
    """
    def __init__(self, grid, physics, dt):
        r"""
        Initializes the Semi-Lagrangian advector.

        Args:
            grid (RegionalGrid3D): Computational grid geometry.
            physics (Euler3D): Dynamical core configuration.
            dt (float): Integration time step $\Delta t$ [s].
        """
        self.grid, self.physics, self.dt, self.op = grid, physics, dt, physics.op

    def _get_index_velocities(self, u_phys, v_phys, eta_dot, loc='m'):
        """
        Maps physical velocities to non-dimensional index crossing rates.

        Returns velocities in units of [indices / second] scaled by local map factors.

        Args:
            u_phys (jnp.ndarray): Zonal velocity $u$ [m/s].
            v_phys (jnp.ndarray): Meridional velocity $v$ [m/s].
            eta_dot (jnp.ndarray): Vertical velocity in $\dot{\eta}$ coordinates [1/s].
            loc (str): Target grid location ('m', 'u', 'v', 'w').

        Returns:
            tuple: $(u_i, v_i, w_i, m_factor)$.
        """
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
        r"""
        Iteratively solves the implicit trajectory equation.

        Args:
            state (dict): Current prognostic state containing 3D winds.
            loc (str): Grid staggering to target ('m', 'u', 'v', 'w').
            iterations (int): Number of iterations for the midpoint trajectory solver.

        Returns:
            jnp.ndarray: Continuous departure point coordinates $\mathbf{x}_d$ [indices].
        """
        u_idx_sec, v_idx_sec, w_idx_sec = self._get_index_velocities(state['u'], state['v'], state['eta_dot'], loc)
        
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        idx_x = jnp.arange(nx + (1 if loc == 'u' else 0), dtype=jnp.float32)
        idx_y = jnp.arange(ny + (1 if loc == 'v' else 0), dtype=jnp.float32)
        idx_z = jnp.arange(nz + (1 if loc == 'w' else 0), dtype=jnp.float32)
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
            
            # Use the fast linear advector for finding the departure points
            u_mid = self.advect_linear(u_idx_sec, mid_coords)
            v_mid = self.advect_linear(v_idx_sec, mid_coords)
            w_mid = self.advect_linear(w_idx_sec, mid_coords)
            
            alpha_x = self.dt * u_mid
            alpha_y = self.dt * v_mid
            alpha_z = self.dt * w_mid

        return jnp.stack([Xi_idx - alpha_x, Yi_idx - alpha_y, Zi_idx - alpha_z], axis=0)

    def advect_linear(self, field, coords):
        """
        Evaluates a trilinear interpolation at continuous coordinates.
        Used primarily within the iterative trajectory solver.
        """
        return jnd.map_coordinates(field, coords, order=1, mode='nearest')

    def advect_cubic(self, field, coords, use_limiter=False):
        """
        Evaluates a tricubic interpolation at continuous coordinates.
        Used for the final high-order advection of prognostic fields.
        """
        return tensor_product_interp_3d(field, coords, use_limiter=use_limiter)


class SemiImplicitSolver3D:
    r"""
    GMRES solver for the linear acoustic and gravity wave matrix system $\mathcal{A}\mathbf{x} = \mathbf{b}$.

    Couples the 3D linear operator $\mathcal{L}(\mathbf{x})$ with the 
    `VerticalPreconditioner` to solve for the implicit stabilizing adjustments.
    """
    def __init__(self, physics, dt):
        r"""
        Initializes the implicit solver.

        Args:
            physics (Euler3D): The dynamical core linear operator definition.
            dt (float): Integration time step $\Delta t$ [s].
        """
        self.physics = physics
        self.dt = dt
        self.pi_scale = 100000.0

    def solve(self, rhs_prime, bg_precomputed):
        r"""
        Solves the implicit system $\mathcal{A}\mathbf{x} = \mathbf{b}$ using preconditioned GMRES.

        Args:
            rhs_prime (dict): The linear residual (forcing terms) from the explicit step.
            bg_precomputed (dict): Precomputed hydrostatic background state metrics.

        Returns:
            dict: The implicit correction vector $\mathbf{x}$.
        """
        rhs_scaled = {k: rhs_prime[k] * self.pi_scale if k == 'pi' else rhs_prime[k] for k in rhs_prime}

        # Setup preconditioner
        preconditioner = VerticalPreconditioner(self.physics, self.dt)
        # Call the banded physics pre-computation!
        preconditioner.precompute_banded(bg_precomputed)  

        def M_fn(state_scaled):
            r"""
            The Preconditioner application $M^{-1}\mathbf{b}$.

            This applies the fast vertical Helmholtz solver locally to each grid column.
            """
            return preconditioner(state_scaled)

        def A_fn(state_scaled):
            r"""
            The Linear Operator application $\mathcal{A}\mathbf{x}$.

            This computes the residual from the full 3D dynamical equations.
            """
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

        x_sol_scaled, info = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-4, maxiter=5, restart=10, M=M_fn)
        
        return {
            'u': x_sol_scaled['u'], 
            'v': x_sol_scaled['v'], 
            'w': x_sol_scaled['w'],
            'pi': x_sol_scaled['pi'] / self.pi_scale,
            # Convert the final solved velocity back into the true eta_dot [1/s]
            'eta_dot': x_sol_scaled['eta_dot'] / bg_precomputed['dz_w_full']
        }

class FluxFormAdvector:
    r"""
    Implements a Flux-Form Semi-Lagrangian (FFSL) advection scheme.

    Standard Semi-Lagrangian advection does not strictly conserve mass. 
    The FFSL scheme advects volumetric density exactly by integrating the 
    cumulative mass function across the continuous departure points:
    
    $$ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0 $$
    """
    def __init__(self, grid, dt):
        r"""
        Initializes the FFSL advector.

        Args:
            grid (RegionalGrid3D): Computational grid geometry.
            dt (float): Integration time step $\Delta t$ [s].
        """
        self.grid = grid
        self.dt = dt

    def advect_1d(self, scalar_1d, cfl_inter_1d):
        r"""
        Advects a 1D scalar column using interface Courant numbers.

        Computes the exact mass flux across cell boundaries by mapping 
        the discrete cumulative mass to the continuous departure locations.
        
        Args:
            scalar_1d (jnp.ndarray): 1D array representing the mass or tracer in each grid cell.
            cfl_inter_1d (jnp.ndarray): Interface Courant numbers for each cell face.

        Returns:
            jnp.ndarray: The updated 1D array after advection.
        """
        N = scalar_1d.shape[0]
        
        # scalar_1d acts as the logical mass in the grid cell
        M_inter = jnp.pad(jnp.cumsum(scalar_1d), (1, 0))
        
        # Use the dtype of the scalar to maintain consistency
        idx_inter = jnp.arange(N + 1, dtype=scalar_1d.dtype)
        idx_dep = idx_inter - cfl_inter_1d
        
        # Solid boundary condition for the advector (Davies sponge will handle the open boundaries later)
        idx_dep = jnp.clip(idx_dep, 0.0, float(N))
        
        M_dep = jnd.map_coordinates(M_inter, [idx_dep], order=1, mode='nearest')
        return M_dep[1:] - M_dep[:-1]

    def advect_3d_split(self, field, state, bg_precomputed):
        r"""
        Performs a 3D directional-split advection sequence.

        Evaluates the 1D advection sequentially in the $x$, $y$, and $z$ directions.
        
        Args:
            field (jnp.ndarray): Volumetric density field to advect.
            state (dict): Current dynamic state ($u, v, \eta_{\dot{t}}$).
            bg_precomputed (dict): Precomputed metrics for cell volumes.
        
        Returns:
            jnp.ndarray: The advected and strictly conserved density field.
        """
        # Calculate true Courant numbers (index crossing rates)
        m_u = self.grid.m_factors['u'][..., None]
        m_v = self.grid.m_factors['v'][..., None]
        
        cfl_x = (state['u'] * m_u * self.dt) / self.grid.dx
        cfl_y = (state['v'] * m_v * self.dt) / self.grid.dy
        cfl_z = state['eta_dot'] * self.dt
        
        # Convert to absolute cell mass
        m_sq = self.grid.m_factors['m'][..., None] ** 2
        cell_volumes = (self.grid.dx * self.grid.dy / m_sq) * bg_precomputed['dz_m_full']
        
        # Multiplying volumetric density (field) by volume gives absolute mass (kg)
        cell_mass = field * cell_volumes
        
        # X-advection
        vmap_x_inner = jax.vmap(self.advect_1d, in_axes=(1, 1), out_axes=1)
        vmap_x = jax.vmap(vmap_x_inner, in_axes=(2, 2), out_axes=2)
        mass_x = vmap_x(cell_mass, cfl_x)
        
        # Y-advection
        vmap_y_inner = jax.vmap(self.advect_1d, in_axes=(0, 0), out_axes=0)
        vmap_y = jax.vmap(vmap_y_inner, in_axes=(2, 2), out_axes=2)
        mass_y = vmap_y(mass_x, cfl_y)
        
        # Z-advection
        vmap_z_inner = jax.vmap(self.advect_1d, in_axes=(0, 0), out_axes=0)
        vmap_z = jax.vmap(vmap_z_inner, in_axes=(1, 1), out_axes=1)
        mass_z = vmap_z(mass_y, cfl_z)
        
        # Convert absolute mass back to volumetric density
        return mass_z / cell_volumes


class SISLStepper3D:
    r"""
    Coordinates the complete Semi-Implicit Semi-Lagrangian (SISL) integration cycle.
    
    Integration flow:
    1. Trajectory calculation (compute $\mathbf{x}_d$).
    2. Evaluate explicit physics and non-linear advection.
    3. Implicit GMRES solve for stiff wave modes.
    4. State assembly and a-posteriori divergence damping.
    5. Boundary condition blending and thermodynamic reconciliation.
    """
    def __init__(self, physics, dt, use_checkpointing = False):
        r"""
        Initializes the integration stepper.

        Args:
            physics (Euler3D): The dynamical core configuration.
            dt (float): Integration time step $\Delta t$ [s].
            use_checkpointing (bool): Enables JAX gradient checkpointing (rematerialization) 
                to trade re-computation for memory savings during adjoint/autodiff tasks.
        """
        self.physics, self.dt = physics, dt
        self.advector = SemiLagrangianAdvector3D(physics.grid, physics, dt)
        self.ffsl_advector = FluxFormAdvector(physics.grid, dt)
        self.implicit_solver = SemiImplicitSolver3D(physics, dt)
        self.use_checkpointing = use_checkpointing
        
        # Ask the physics suite for the active tracers
        self.tracer_keys = self.physics.physics_suite.tracer_keys if self.physics.physics_suite is not None else []

    def integrate(self, state, t_start, num_steps, forcing, bc_fn):
        """
        Executes the primary time integration loop using `jax.lax.scan`.
        
        Compiles the entire step sequence into a single, highly optimized XLA graph.
        
        Args:
            state (dict): Initial state of the atmosphere.
            t_start (float): Initial time.
            num_steps (int): Number of time steps to integrate.
            forcing (callable): Function providing external forcing fields.
            bc_fn (callable): Function applying boundary conditions.
            
        Returns:
            dict: Final state of the atmosphere after `num_steps`.
        """
        def scan_fn(curr_state, step_idx):
            t_curr = t_start + step_idx * self.dt
            next_state = self.step(curr_state, t_curr, forcing, bc_fn)
            return next_state, None
            
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))
        return final_state

    def step(self, state, t, forcing, bc_fn, ml_params=None):
        r"""
        Executes a single SISL time step.

        The core discrete equation represents an implicit discretization of the 
        momentum and continuity equations:

        $$ \begin{cases} 
        \frac{\pi^{n+1} - \pi_d}{\Delta t} + \alpha \mathcal{T}_\pi^{n+1} = (1-\alpha)\mathcal{T}_\pi^n \\ 
        \frac{\mathbf{u}^{n+1} - \mathbf{u}_d}{\Delta t} + \alpha \mathcal{T}_\mathbf{u}^{n+1} = (1-\alpha)\mathcal{T}_\mathbf{u}^n 
        \end{cases} $$

        Args:
            state (dict): Current prognostic state.
            t (float): Current simulation time [s].
            forcing (dict): External forcing conditions.
            bc_fn (callable): Boundary condition blending operator.

        Returns:
            dict: The updated prognostic state at $t + \Delta t$.
        """

        alpha = 0.55

        if 'eta_dot' not in state: state['eta_dot'] = jnp.zeros_like(state['w'])
            
        coords_u = jax.lax.stop_gradient(self.advector.compute_departure_indices(state, loc='u'))
        coords_v = jax.lax.stop_gradient(self.advector.compute_departure_indices(state, loc='v'))
        coords_w = jax.lax.stop_gradient(self.advector.compute_departure_indices(state, loc='w'))
        coords_m = jax.lax.stop_gradient(self.advector.compute_departure_indices(state, loc='m'))

        bg_state_ref = {
            'rho': self.physics.c['p0'] / (self.physics.c['Rd'] * self.physics.theta_bg) * \
                   (self.physics.pi_bg ** (self.physics.c['cvd'] / self.physics.c['Rd'])),
            'pi': self.physics.pi_bg,
            'th_v': self.physics.theta_bg
        }
        bg_precomputed = self.physics.precompute_bg(bg_state_ref)
        
        # Calculate the thermodynamic perturbation
        th_v_prime_n = state['th_v'] - self.physics.theta_bg
        
        state_prime_n = {
            'u': state['u'], 'v': state['v'], 'w': state['w'], 'th_v': state['th_v'],
            'pi': state['pi'] - self.physics.pi_bg, 'eta_dot': state['eta_dot'],
            'th_v_prime_u': self.physics.op.avg(th_v_prime_n, axis=0, from_loc='m', to_loc='u'),
            'th_v_prime_v': self.physics.op.avg(th_v_prime_n, axis=1, from_loc='m', to_loc='v'),
            'th_v_prime_w': self.physics.op.avg(th_v_prime_n, axis=2, from_loc='m', to_loc='w')
        }

        if 'theta_surf' in state: state_prime_n['theta_surf'] = state['theta_surf']
        if 'target_th_v' in state: state_prime_n['target_th_v'] = state['target_th_v']
        
        # Use checkpointing for tendencies if requested (speeds up adjoint, slows down forward)
        if self.use_checkpointing:
            cp_get_tendencies = jax.checkpoint(self.physics.get_tendencies, static_argnums=(2,))
        else:
            cp_get_tendencies = self.physics.get_tendencies
            
        # Pass ml_params as the 4th argument to the physics evaluator
        tends_n = cp_get_tendencies(state_prime_n, bg_precomputed, True, ml_params)
        
        u_in = state['u'] + (1.0 - alpha) * self.dt * tends_n['u']
        v_in = state['v'] + (1.0 - alpha) * self.dt * tends_n['v']
        w_in = state['w'] + (1.0 - alpha) * self.dt * tends_n['w']
        pi_prime_in = state_prime_n['pi'] + (1.0 - alpha) * self.dt * tends_n['pi']

        # Apply Eulerian physics tendencies to the thermodynamics before advection
        th_v_prime_in = th_v_prime_n + self.dt * tends_n['th_v']

        # Apply Eulerian tendencies to any active tracers before advection
        tracers_in = {}
        for key in self.tracer_keys:
            if key in state:
                tracers_in[key] = state[key]
                if key in tends_n:
                    tracers_in[key] += self.dt * tends_n[key]

        # 3d kinematic advection
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
        
        # Use checkpointing for advectors if requested (speeds up adjoint, slows down forward)
        if self.use_checkpointing:
            cp_advect_cubic = jax.checkpoint(self.advector.advect_cubic, static_argnums=(2,))
            cp_advect_ffsl = jax.checkpoint(self.ffsl_advector.advect_3d_split)
        else:
            cp_advect_cubic = self.advector.advect_cubic
            cp_advect_ffsl = self.ffsl_advector.advect_3d_split

        # Advect the fields
        R_eta_dot = -(1.0 - alpha) * cp_advect_cubic(residual_n, coords_w, False)

        rhs_u = cp_advect_cubic(u_in, coords_u, False)
        rhs_v = cp_advect_cubic(v_in, coords_v, False)
        rhs_w = cp_advect_cubic(w_in, coords_w, False)
        rhs_pi_prime = cp_advect_cubic(pi_prime_in, coords_m, False)
        th_v_prime_next = cp_advect_cubic(th_v_prime_in, coords_m, False)
        
        # Advect mass with checkpointed FFSL scheme
        rho_next = cp_advect_ffsl(state['rho'], state, bg_precomputed)

        # Advect virtual potential temperature
        th_v_next = th_v_prime_next + self.physics.theta_bg
        
        # Tracers use FFSL to strictly conserve mass
        tracers_next = {}
        for key in self.tracer_keys:
            if key in state:
                rho_tr_next = cp_advect_ffsl(state['rho'] * tracers_in[key], state, bg_precomputed)
                tracers_next[key] = rho_tr_next / (rho_next + 1e-15)

        # =====================================================================
        # 2. ADD BUOYANCY
        # =====================================================================
        th_v_prime_next = th_v_next - self.physics.theta_bg
        th_v_prime_w_next = self.physics.op.avg(th_v_prime_next, axis=2, from_loc='m', to_loc='w')
        rhs_w += 0.5 * self.dt * (self.physics.c['g'] * (th_v_prime_w_next / bg_precomputed['th_v_w']))

        # --- ENFORCE KINEMATIC BOUNDARY ON RHS ---
        # Bring the explicit horizontal winds to the w-points
        u_m_rhs = self.physics.op.avg(rhs_u, axis=0, from_loc='u', to_loc='m')
        u_w_rhs = self.physics.op.avg(u_m_rhs, axis=2, from_loc='m', to_loc='w')

        v_m_rhs = self.physics.op.avg(rhs_v, axis=1, from_loc='v', to_loc='m')
        v_w_rhs = self.physics.op.avg(v_m_rhs, axis=2, from_loc='m', to_loc='w')

        m_w = jnp.expand_dims(self.physics.grid.m_factors['w'], axis=-1)

        # Calculate the flow forced vertically by the explicit winds hitting the terrain
        rhs_kinematic_bottom = m_w[:, :, 0] * (
            u_w_rhs[:, :, 0] * self.physics.grid.z_xi_w[:, :, 0] + 
            v_w_rhs[:, :, 0] * self.physics.grid.z_eta_w[:, :, 0]
        )

        # Apply correct kinematic w at surface, 0.0 at the rigid lid top
        rhs_w = rhs_w.at[:, :, 0].set(rhs_kinematic_bottom)
        rhs_w = rhs_w.at[:, :, -1].set(0.0)

        # eta_dot is the cross-coordinate velocity, so 0.0 at coordinate boundaries is correct
        R_eta_dot = R_eta_dot.at[:, :, 0].set(0.0)
        R_eta_dot = R_eta_dot.at[:, :, -1].set(0.0)
        
        # =====================================================================
        # 3. IMPLICIT SOLVE
        # =====================================================================
        rhs_prime = {'u': rhs_u, 'v': rhs_v, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}
        state_prime_next = self.implicit_solver.solve(rhs_prime, bg_precomputed)
        
        # =====================================================================
        # 4. ASSEMBLE FINAL STATE
        # =====================================================================
        state_next = {
            'u': state_prime_next['u'], 
            'v': state_prime_next['v'], 
            'w': state_prime_next['w'], 
            'pi': state_prime_next['pi'] + self.physics.pi_bg,
            'rho': rho_next, 
            'th_v': th_v_next, 
            'eta_dot': state_prime_next['eta_dot']
        }

        # Load tracers
        for key in self.tracer_keys:
            if key in tracers_next:
                state_next[key] = tracers_next[key]

        # =====================================================================
        # 4b. A-POSTERIORI DIVERGENCE DAMPING
        # =====================================================================
        # Calculate the horizontal divergence of the IMPLICITLY solved winds
        du_dx = self.physics.op.diff(state_next['u'], axis=0, from_loc='u', to_loc='m') 
        dv_dy = self.physics.op.diff(state_next['v'], axis=1, from_loc='v', to_loc='m') 
        div_h_kinematic = du_dx + dv_dy

        grad_div_x = self.physics.op.diff(div_h_kinematic, axis=0, from_loc='m', to_loc='u') 
        grad_div_y = self.physics.op.diff(div_h_kinematic, axis=1, from_loc='m', to_loc='v') 

        # Apply the explicit diffusion step directly to the updated state arrays
        state_next['u'] += self.physics.nu_div * self.dt * grad_div_x
        state_next['v'] += self.physics.nu_div * self.dt * grad_div_y

        # =====================================================================
        # 4c. PHYSICAL STATE UPDATES (e.g., Saturation Adjustment)
        # =====================================================================
        if self.physics.physics_suite is not None:
            state_next = self.physics.physics_suite.apply_state_updates(state_next)

        # =====================================================================
        # 5. APPLY BOUNDARY CONDITIONS
        # =====================================================================
        state_next = bc_fn(state_next, forcing)

        # =====================================================================
        # 6. THERMODYNAMIC RECONCILIATION
        # =====================================================================
        # Because the sponge nudged th_v and pi, we MUST recalculate rho to satisfy the 
        # Equation of State, preventing a thermodynamic shock in the next step!
        cvd, Rd, p0 = self.physics.c['cvd'], self.physics.c['Rd'], self.physics.c['p0']
        state_next['rho'] = p0 / (Rd * state_next['th_v']) * (state_next['pi'] ** (cvd / Rd))

        return state_next