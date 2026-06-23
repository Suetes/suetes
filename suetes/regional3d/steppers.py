import jax
from jax import vmap
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.lax.linalg import tridiagonal_solve
import jax.scipy.ndimage as jnd

from suetes.regional3d.operators import tensor_product_interp_3d
from suetes.regional3d.euler import Euler3D


def build_dynamical_core(core_type, grid, operators, constants, initial_state, 
                         physics_suite=None, interior_mask=None, **kwargs):
    """
    Factory builder that instantiates the dynamical core and corresponding stepper.
    Safely handles default stability parameters while accepting user overrides.

    Args:
        core_type (str): 'sisl' or 'split-explicit'
        grid, operators, constants: Geometry and operator objects
        initial_state (dict): Initial atmospheric conditions
        physics_suite, interior_mask: Physical subgrid configurations
        **kwargs: Parameter overrides (e.g., dt, damp_height, nu_h_factor)
    
    Returns:
        tuple: (stepper_instance, large_time_step_dt)
    """
    core_str = core_type.lower()
    
    if core_str == "sisl":
        # Safe default parameters for standard SISL runs
        params = {
            "dt": 120.0,
            "damp_height": 9000.0,
            "max_damp": 3.0,
            "nu_div_factor": 0.1,
            "nu_h_factor": 0.1,
            "N_bv": 0.01,
            "alpha": 0.55,
        }
        # Apply any explicit user overrides passed via kwargs
        params.update(kwargs)
        
        physics = Euler3D(
            grid, operators, constants, dt=params["dt"],
            initial_era5_state=initial_state,
            damp_height=params["damp_height"], max_damp=params["max_damp"],
            nu_div_factor=params["nu_div_factor"], nu_h_factor=params["nu_h_factor"],
            N_bv=params["N_bv"],
            physics_suite=physics_suite, interior_mask=interior_mask,
        )
        stepper = SISLStepper3D(physics, params["dt"], alpha=params["alpha"])
        return stepper, params["dt"]
        
    elif core_str == "split-explicit":
        # Safe default parameters for standard Split-Explicit runs
        params = {
            "dt": 40.0,
            "ns": 3,
            "damp_height": 12000.0,  # Raised by default to absorb explicit gravity waves
            "max_damp": 3.0,
            "nu_div_factor": 0.03,   # Lowered by default due to smaller explicit dt
            "nu_h_factor": 0.03,
            "N_bv": 0.01,
            "alpha": 0.55,
        }
        params.update(kwargs)
        
        physics = Euler3D(
            grid, operators, constants, dt=params["dt"],
            initial_era5_state=initial_state,
            damp_height=params["damp_height"], max_damp=params["max_damp"],
            nu_div_factor=params["nu_div_factor"], nu_h_factor=params["nu_h_factor"],
            N_bv=params["N_bv"],
            physics_suite=physics_suite, interior_mask=interior_mask,
        )
        stepper = SplitExplicitStepper3D(physics, dt=params["dt"], ns=params["ns"], alpha=params["alpha"])
        return stepper, params["dt"]
        
    else:
        raise ValueError(f"Unknown core_type: '{core_type}'. Use 'sisl' or 'split-explicit'.")


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
    def __init__(self, physics, dt, alpha=0.55):
        r"""
        Initializes the implicit solver.

        Args:
            physics (Euler3D): The dynamical core linear operator definition.
            dt (float): Integration time step $\Delta t$ [s].
            alpha (float, optional): Semi-implicit off-centering parameter. Defaults to 0.55.
        """
        self.physics = physics
        self.dt = dt
        self.alpha = alpha
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
        preconditioner = VerticalPreconditioner(self.physics, self.dt, alpha=self.alpha)
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

        self.physics.op.use_stop_grad = False
        try:
            x_sol_scaled, info = gmres(A_fn, rhs_scaled, x0=rhs_scaled, tol=1e-4, maxiter=20, restart=20, M=M_fn)
        finally:
            self.physics.op.use_stop_grad = True
        
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
    def __init__(self, physics, dt, alpha=0.55, use_checkpointing = False):
        r"""
        Initializes the integration stepper.

        Args:
            physics (Euler3D): The dynamical core configuration.
            dt (float): Integration time step $\Delta t$ [s].
            alpha (float, optional): Semi-implicit off-centering parameter. Defaults to 0.55.
            use_checkpointing (bool): Enables JAX gradient checkpointing (rematerialization) 
                to trade re-computation for memory savings during adjoint/autodiff tasks.
        """
        self.physics, self.dt = physics, dt
        self.alpha = alpha
        self.advector = SemiLagrangianAdvector3D(physics.grid, physics, dt)
        self.ffsl_advector = FluxFormAdvector(physics.grid, dt)
        self.implicit_solver = SemiImplicitSolver3D(physics, dt, alpha=alpha)
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

        alpha = self.alpha

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
        
        u_in = state['u'] + self.dt * ((1.0 - alpha) * tends_n['u'] + alpha * tends_n.get('phys_diff_u', 0.0))
        v_in = state['v'] + self.dt * ((1.0 - alpha) * tends_n['v'] + alpha * tends_n.get('phys_diff_v', 0.0))
        w_in = state['w'] + self.dt * ((1.0 - alpha) * tends_n['w'] + alpha * tends_n.get('phys_diff_w', 0.0))
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
                tracers_next[key] = rho_tr_next / (rho_next + self.physics.c.get('eps', 1e-15))

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


class SplitExplicitStepper3D:
    r"""
    Coordinates a WRF-style Split-Explicit Runge-Kutta 3 (RK3) time integration cycle.
    
    This scheme separates the integration of slow-moving advective and physical
    modes ($\mathcal{S}$) from fast-moving acoustic and gravity waves. The slow modes 
    are advanced using a 3rd-order Runge-Kutta scheme, while the fast perturbation modes 
    ($\Phi''$) are advanced using a smaller acoustic time step $\Delta \tau$ in a 
    forward-backward explicit horizontal and implicit vertical sub-cycling loop:

    $$ \Phi^{t+\Delta t} = \text{RK3}(\Phi^t, \mathcal{S}) + \text{Acoustic}(\Phi'', \Delta \tau) $$

    By bypassing the Semi-Implicit Semi-Lagrangian (SISL) framework, this method is 
    strictly Eulerian and well-suited for high-resolution, small-scale dynamics.
    """
    def __init__(self, physics, dt, ns, alpha=0.55):
        """
        Args:
            physics (Euler3D): The dynamical core configuration.
            dt (float): Large time step for low-frequency modes [s].
            ns (int): Ratio of the RK3 time step to the acoustic time step.
            alpha (float, optional): Acoustic off-centering parameter. Defaults to 0.55.
        """
        self.physics = physics
        self.dt = dt
        self.ns = ns  
        self.alpha = alpha
        
        # Dynamically assign acoustic steps per RK3 stage to respect CFL limits
        self.ns_stage1 = max(1, round(self.ns / 3))
        self.ns_stage2 = max(1, round(self.ns / 2))
        self.ns_stage3 = self.ns
        
        self.dtau_stage1 = (self.dt / 3.0) / float(self.ns_stage1)
        self.dtau_stage2 = (self.dt / 2.0) / float(self.ns_stage2)
        self.dtau_stage3 = self.dt / float(self.ns_stage3)

    def step(self, state, t, forcing, bc_fn, ml_params=None):
        r"""
        Executes a single split-explicit RK3 time step.

        The integration is split into three RK3 stages where the slow tendencies 
        $\mathcal{S}(\Phi)$ are computed, and an inner acoustic loop advances the fast modes.
        
        $$
        \begin{align}
        &\Phi^*  = \Phi^t + \frac{\Delta t}{3} \mathcal{S}(\Phi^t), \\
        &\Phi^{**} = \Phi^t + \frac{\Delta t}{2} \mathcal{S}(\Phi^*), \\
        &\Phi^{t+\Delta t} = \Phi^t + \Delta t \mathcal{S}(\Phi^{**}).
        \end{align}
        $$
        
        Args:
            state (dict): Current prognostic state.
            t (float): Current simulation time [s].
            forcing (dict): External forcing conditions.
            bc_fn (callable): Boundary condition blending operator.
            ml_params (dict, optional): Machine learning parameterization weights. Defaults to None.

        Returns:
            dict: The updated prognostic state at $t + \Delta t$.
        """
        bg_state_ref = {
            'rho': self.physics.c['p0'] / (self.physics.c['Rd'] * self.physics.theta_bg) * \
                   (self.physics.pi_bg ** (self.physics.c['cvd'] / self.physics.c['Rd'])),
            'pi': self.physics.pi_bg,
            'th_v': self.physics.theta_bg
        }
        bg_precomputed = self.physics.precompute_bg(bg_state_ref)

        state_t = state
        
        # Diagnose the initial state boundaries before calculating slow tendencies
        state_t['eta_dot'], state_t['w'] = self._diagnose_eta_dot(state_t)

        # =====================================================================
        # PRE-COMPUTE STIFF PHYSICS AND DIFFUSION ONCE AT TIME T
        # =====================================================================
        phys_tends = {}
        diff_tends = {}
        
        # Build the perturbation state exactly as euler.py expects
        th_v_prime = state_t['th_v'] - self.physics.theta_bg
        state_prime_t = {
            'u': state_t['u'], 'v': state_t['v'], 'w': state_t['w'], 
            'th_v': state_t['th_v'], 'pi': state_t['pi'] - self.physics.pi_bg, 
            'eta_dot': state_t['eta_dot'],
            'th_v_prime_u': self.physics.op.avg(th_v_prime, axis=0, from_loc='m', to_loc='u'),
            'th_v_prime_v': self.physics.op.avg(th_v_prime, axis=1, from_loc='m', to_loc='v'),
            'th_v_prime_w': self.physics.op.avg(th_v_prime, axis=2, from_loc='m', to_loc='w')
        }
        for key in state_t:
            if key not in state_prime_t:
                state_prime_t[key] = state_t[key]

        if self.physics.physics_suite is not None:
            phys_tends = self.physics.physics_suite.get_explicit_tendencies(
                state_prime_t, bg_precomputed, 
                interior_mask=self.physics.interior_mask, 
                ml_params=ml_params
            )
            
        if self.physics.nu_h > 0.0 or self.physics.nu_div > 0.0:
            diff_tends = self.physics.diffusion.get_tendencies(state_prime_t, bg_precomputed)

        # Temporarily hide physics and diffusion from the dynamical core's RK3 sub-steps
        original_suite = self.physics.physics_suite
        original_nu_h = self.physics.nu_h
        
        self.physics.physics_suite = None
        self.physics.nu_h = 0.0

        # =====================================================================
        # RK3 - STAGE 1 (\Phi^*)
        # =====================================================================
        slow_forcings_1 = self._compute_slow_tendencies(state_t, bg_precomputed, phys_tends, diff_tends, ml_params)
        
        state_star = self._acoustic_loop(
            state_init=state_t, state_current=state_t, 
            slow_forcings=slow_forcings_1, bg=bg_precomputed, 
            dtau=self.dtau_stage1, num_steps=self.ns_stage1
        )
        state_star['eta_dot'], state_star['w'] = self._diagnose_eta_dot(state_star)

        # =====================================================================
        # RK3 - STAGE 2 (\Phi^{**})
        # =====================================================================
        slow_forcings_2 = self._compute_slow_tendencies(state_star, bg_precomputed, phys_tends, diff_tends, ml_params)
        
        state_double_star = self._acoustic_loop(
            state_init=state_t, state_current=state_star, 
            slow_forcings=slow_forcings_2, bg=bg_precomputed, 
            dtau=self.dtau_stage2, num_steps=self.ns_stage2
        )
        state_double_star['eta_dot'], state_double_star['w'] = self._diagnose_eta_dot(state_double_star)

        # =====================================================================
        # RK3 - STAGE 3 (\Phi^{t + \Delta t})
        # =====================================================================
        slow_forcings_3 = self._compute_slow_tendencies(state_double_star, bg_precomputed, phys_tends, diff_tends, ml_params)
        
        state_next = self._acoustic_loop(
            state_init=state_t, state_current=state_double_star, 
            slow_forcings=slow_forcings_3, bg=bg_precomputed, 
            dtau=self.dtau_stage3, num_steps=self.ns_stage3
        )

        # Restore the physics suite and diffusion constants
        self.physics.physics_suite = original_suite
        self.physics.nu_h = original_nu_h

        # =====================================================================
        # FINAL STATE ASSEMBLY & BOUNDARIES
        # =====================================================================

        # Physical state updates (e.g., Saturation adjustment)
        if self.physics.physics_suite is not None:
            state_next = self.physics.physics_suite.apply_state_updates(state_next)

        # Apply lateral boundary relaxation (Davies Sponge) EXACTLY ONCE
        state_next = bc_fn(state_next, forcing)
        
        # Diagnose eta_dot based on the final blended boundary winds
        state_next['eta_dot'], state_next['w'] = self._diagnose_eta_dot(state_next)

        # Final thermodynamic reconciliation (Equation of State)
        cvd, Rd, p0 = self.physics.c['cvd'], self.physics.c['Rd'], self.physics.c['p0']
        state_next['rho'] = p0 / (Rd * state_next['th_v']) * (state_next['pi'] ** (cvd / Rd))

        return state_next

    def _diagnose_eta_dot(self, state):
        r"""
        Diagnoses the contravariant vertical velocity $\dot{\eta}$ to maintain mass conservation.

        Dynamically corrects boundary vertical velocity $w$ to gracefully handle 
        unbalanced initial conditions. It enforces the physical kinematic boundary 
        condition $w_{sfc} = \mathbf{V}_h \cdot \nabla Z$ and integrates the mass flux 
        across layers to find the cross-coordinate velocity:

        $$ \dot{\eta} = \frac{1}{\Delta z_w} \left( w - m_w \left( u \frac{\partial Z}{\partial x} + v \frac{\partial Z}{\partial y} \right) \right) $$

        Args:
            state (dict): Current atmospheric state containing 3D winds.

        Returns:
            tuple: A tuple containing:
                - eta_dot (jnp.ndarray): Contravariant vertical velocity [1/s].
                - w_updated (jnp.ndarray): Vertical velocity $w$ with enforced boundaries [m/s].
        """
        u_m = self.physics.op.avg(state['u'], axis=0, from_loc='u', to_loc='m')
        u_w = self.physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
        v_m = self.physics.op.avg(state['v'], axis=1, from_loc='v', to_loc='m')
        v_w = self.physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')
        
        m_w = jnp.expand_dims(self.physics.grid.m_factors['w'], axis=-1)
        
        # Enforce physical kinematic boundary condition: flow must glide over the slope
        w_sfc = m_w[:, :, 0] * (u_w[:, :, 0] * self.physics.grid.z_xi_w[:, :, 0] + v_w[:, :, 0] * self.physics.grid.z_eta_w[:, :, 0])
        w_updated = state['w'].at[:, :, 0].set(w_sfc).at[:, :, -1].set(0.0)
        
        # Diagnose mass flux across layers
        eta_dot_dz = w_updated - m_w * (u_w * self.physics.grid.z_xi_w + v_w * self.physics.grid.z_eta_w)
        eta_dot = eta_dot_dz / self.physics.grid.dz_w_full
        
        # Strictly seal boundaries
        eta_dot = eta_dot.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        
        return eta_dot, w_updated

    def _upwind_flux_3rd_order(self, flux, field, axis):
        r"""
        Computes 3rd-order upwind biased face values for a field driven by a velocity flux.

        The interpolation is given by a 4th-order centered average plus a 3rd-order 
        upwind bias term, controlled by the sign of the local flux $U$:

        $$ q_{face} = \frac{7}{12}(q_i + q_{i-1}) - \frac{1}{12}(q_{i+1} + q_{i-2}) + \text{sgn}(U) \frac{1}{12} \left[ (q_{i+1} - q_{i-2}) - 3(q_i - q_{i-1}) \right] $$

        This includes a numerical noise-gate to preserve machine-precision symmetry 
        at flow stagnation points. The 3rd-order scheme effectively limits numerical
        dispersion while retaining desirable artificial diffusion properties to stabilize advection.

        Args:
            flux (jnp.ndarray): The velocity flux driving the transport.
            field (jnp.ndarray): The scalar or momentum field being advected.
            axis (int): The spatial dimension axis (0 for x, 1 for y, 2 for z).

        Returns:
            jnp.ndarray: The flux-weighted face values.
        """
        pad_width = [(0, 0)] * 3
        pad_width[axis] = (2, 2)
        q = jnp.pad(field, pad_width, mode='edge')
        
        s_idx = lambda start, end: tuple(
            slice(start, end) if i == axis else slice(None)
            for i in range(3)
        )
        
        q_im2 = q[s_idx(0, -3)]
        q_im1 = q[s_idx(1, -2)]
        q_i   = q[s_idx(2, -1)]
        q_ip1 = q[s_idx(3, None)]
        
        q_4th = (7.0 / 12.0) * (q_i + q_im1) - (1.0 / 12.0) * (q_ip1 + q_im2)
        q_bias = (1.0 / 12.0) * ((q_ip1 - q_im2) - 3.0 * (q_i - q_im1))
        
        # Gate out floating-point noise at stagnation points
        # and use the mathematical sign to smoothly apply the upwind bias.
        clean_flux = jnp.where(jnp.abs(flux) < self.physics.c.get('eps_l', 1e-12), 0.0, flux)
        q_face = q_4th + jnp.sign(clean_flux) * q_bias
        
        return flux * q_face

    def _compute_slow_tendencies(self, current_state, bg, phys_tends, diff_tends, ml_params):
        r"""
        Evaluates and isolates the slow-frequency RHS forcing terms $\mathcal{S}(\Phi)$.

        Computes the low-frequency advective terms using a 3rd-order upwind Eulerian 
        flux engine for momentum and thermodynamics, and appends static physical 
        parameterization and diffusion tendencies evaluated at the start of the step:

        $$ \mathcal{S}(\Phi) = - \mathbf{v} \cdot \nabla \Phi + \mathcal{F}_{physics} + \mathcal{F}_{diffusion} $$

        Args:
            current_state (dict): The intermediate state in the current RK3 stage.
            bg (dict): Precomputed hydrostatic background state metrics.
            phys_tends (dict): Cached explicit physical parameterization tendencies.
            diff_tends (dict): Cached explicit diffusion tendencies.
            ml_params (dict): Machine learning parameterization weights.

        Returns:
            dict: The total slow RHS forcing tendencies for $u, v, w$, and $\theta_v$.
        """
        th_v_prime = current_state['th_v'] - self.physics.theta_bg
        state_prime = {
            'u': current_state['u'], 'v': current_state['v'], 'w': current_state['w'], 
            'th_v': current_state['th_v'], 'pi': current_state['pi'] - self.physics.pi_bg, 
            'eta_dot': current_state.get('eta_dot', jnp.zeros_like(current_state['w'])),
            'th_v_prime_u': self.physics.op.avg(th_v_prime, axis=0, from_loc='m', to_loc='u'),
            'th_v_prime_v': self.physics.op.avg(th_v_prime, axis=1, from_loc='m', to_loc='v'),
            'th_v_prime_w': self.physics.op.avg(th_v_prime, axis=2, from_loc='m', to_loc='w')
        }

        for key in current_state:
            if key not in state_prime:
                state_prime[key] = current_state[key]
        
        # Calculate dynamical core explicit tendencies (physics/diffusion are temporarily hidden)
        tends = self.physics.get_tendencies(state_prime, bg, is_explicit=True, ml_params=ml_params)
        
        # Add the cached stiff physics and diffusion tendencies computed at the start of the timestep
        for k in ['u', 'v', 'w', 'th_v']:
            if phys_tends and k in phys_tends:
                tends[k] += phys_tends[k]
            if diff_tends and k in diff_tends:
                tends[k] += diff_tends[k]
        
        m_m = jnp.expand_dims(self.physics.grid.m_factors['m'], axis=-1)
        m_u = jnp.expand_dims(self.physics.grid.m_factors['u'], axis=-1)
        m_v = jnp.expand_dims(self.physics.grid.m_factors['v'], axis=-1)
        
        u_m = self.physics.op.avg(current_state['u'], axis=0, from_loc='u', to_loc='m')
        v_m = self.physics.op.avg(current_state['v'], axis=1, from_loc='v', to_loc='m')
        w_m = self.physics.op.avg(current_state['w'], axis=2, from_loc='w', to_loc='m')
        
        U = state_prime['u'] / m_u
        V = state_prime['v'] / m_v
        W = state_prime['eta_dot'] * bg['dz_w_full']
        
        div_U = self.physics.op.diff(U, axis=0, from_loc='u', to_loc='m')
        div_V = self.physics.op.diff(V, axis=1, from_loc='v', to_loc='m')
        div_W = self.physics.op.diff(W, axis=2, from_loc='w', to_loc='m') * (self.physics.grid.dz / bg['dz_m_full'])
        
        # --- Field 1: Virtual Potential Temperature (th_v) ---
        flux_x_th = self._upwind_flux_3rd_order(U, th_v_prime, axis=0)
        flux_y_th = self._upwind_flux_3rd_order(V, th_v_prime, axis=1)
        flux_z_th = self._upwind_flux_3rd_order(W, th_v_prime, axis=2)
        
        adv_x_th = self.physics.op.diff(flux_x_th, axis=0, from_loc='u', to_loc='m')
        adv_y_th = self.physics.op.diff(flux_y_th, axis=1, from_loc='v', to_loc='m')
        adv_z_th = self.physics.op.diff(flux_z_th, axis=2, from_loc='w', to_loc='m') * (self.physics.grid.dz / bg['dz_m_full'])
        
        tend_th_v_adv = - (m_m**2) * (adv_x_th - th_v_prime * div_U + adv_y_th - th_v_prime * div_V) - (adv_z_th - th_v_prime * div_W)
        tends['th_v'] += tend_th_v_adv
        
        # --- Field 2: Zonal Momentum (u) ---
        flux_x_u = self._upwind_flux_3rd_order(U, u_m, axis=0)
        flux_y_u = self._upwind_flux_3rd_order(V, u_m, axis=1)
        flux_z_u = self._upwind_flux_3rd_order(W, u_m, axis=2)
        
        adv_x_u = self.physics.op.diff(flux_x_u, axis=0, from_loc='u', to_loc='m')
        adv_y_u = self.physics.op.diff(flux_y_u, axis=1, from_loc='v', to_loc='m')
        adv_z_u = self.physics.op.diff(flux_z_u, axis=2, from_loc='w', to_loc='m') * (self.physics.grid.dz / bg['dz_m_full'])
        
        tend_u_adv_m = - (m_m**2) * (adv_x_u - u_m * div_U + adv_y_u - u_m * div_V) - (adv_z_u - u_m * div_W)
        tends['u'] += self.physics.op.avg(tend_u_adv_m, axis=0, from_loc='m', to_loc='u')
        
        # --- Field 3: Meridional Momentum (v) ---
        flux_x_v = self._upwind_flux_3rd_order(U, v_m, axis=0)
        flux_y_v = self._upwind_flux_3rd_order(V, v_m, axis=1)
        flux_z_v = self._upwind_flux_3rd_order(W, v_m, axis=2)
        
        adv_x_v = self.physics.op.diff(flux_x_v, axis=0, from_loc='u', to_loc='m')
        adv_y_v = self.physics.op.diff(flux_y_v, axis=1, from_loc='v', to_loc='m')
        adv_z_v = self.physics.op.diff(flux_z_v, axis=2, from_loc='w', to_loc='m') * (self.physics.grid.dz / bg['dz_m_full'])
        
        tend_v_adv_m = - (m_m**2) * (adv_x_v - v_m * div_U + adv_y_v - v_m * div_V) - (adv_z_v - v_m * div_W)
        tends['v'] += self.physics.op.avg(tend_v_adv_m, axis=1, from_loc='m', to_loc='v')
        
        # --- Field 4: Vertical Momentum (w) ---
        flux_x_w = self._upwind_flux_3rd_order(U, w_m, axis=0)
        flux_y_w = self._upwind_flux_3rd_order(V, w_m, axis=1)
        flux_z_w = self._upwind_flux_3rd_order(W, w_m, axis=2)
        
        adv_x_w = self.physics.op.diff(flux_x_w, axis=0, from_loc='u', to_loc='m')
        adv_y_w = self.physics.op.diff(flux_y_w, axis=1, from_loc='v', to_loc='m')
        adv_z_w = self.physics.op.diff(flux_z_w, axis=2, from_loc='w', to_loc='m') * (self.physics.grid.dz / bg['dz_m_full'])
        
        tend_w_adv_m = - (m_m**2) * (adv_x_w - w_m * div_U + adv_y_w - w_m * div_V) - (adv_z_w - w_m * div_W)
        tends['w'] += self.physics.op.avg(tend_w_adv_m, axis=2, from_loc='m', to_loc='w')
        
        return tends

    def _acoustic_loop(self, state_init, state_current, slow_forcings, bg, dtau, num_steps):
        r"""
        Executes the explicit forward-backward horizontal and vertically implicit
        acoustic time-split integration subloop.

        Advances the perturbation variables ($u'', v'', w'', \pi''$) over a time interval
        using a smaller acoustic time step $\Delta \tau$. Horizontal momentum is advanced 
        explicitly (forward step):

        $$ u''^{n+1} = u''^n + \Delta \tau \left( \mathcal{S}_u - c_p \theta_v \frac{\partial \pi''^n}{\partial x} \right) $$

        Vertical momentum and pressure are solved implicitly (backward step) to bypass 
        restrictive vertical stability constraints.

        Args:
            state_init (dict): The state at the beginning of the full RK3 step.
            state_current (dict): The state at the beginning of the current RK3 stage.
            slow_forcings (dict): The evaluated slow RHS tendencies $\mathcal{S}$.
            bg (dict): Precomputed hydrostatic background state metrics.
            dtau (float): The acoustic time step $\Delta \tau$ for this RK3 stage [s].
            num_steps (int): The number of acoustic steps to take in this stage.

        Returns:
            dict: The updated state after completing the acoustic integration for the stage.
        """
        u_prime_prime   = state_init['u'] - state_current['u']
        v_prime_prime   = state_init['v'] - state_current['v']
        w_prime_prime   = state_init['w'] - state_current['w']
        pi_prime_prime  = state_init['pi'] - state_current['pi']
        
        cp = self.physics.c['cp']
        m_u = jnp.expand_dims(self.physics.grid.m_factors['u'], axis=-1)
        m_v = jnp.expand_dims(self.physics.grid.m_factors['v'], axis=-1)
        m_m = jnp.expand_dims(self.physics.grid.m_factors['m'], axis=-1)

        for step in range(num_steps):
            # =================================================================
            # PART 1: ADVANCE HORIZONTAL MOMENTUM (FORWARD STEP)
            # =================================================================
            
            # Evaluate divergence damping inside the acoustic sub-step
            u_curr_total = state_current['u'] + u_prime_prime
            v_curr_total = state_current['v'] + v_prime_prime
            
            du_dx = self.physics.op.diff(u_curr_total, axis=0, from_loc='u', to_loc='m')
            dv_dy = self.physics.op.diff(v_curr_total, axis=1, from_loc='v', to_loc='m')
            div_h_kinematic = du_dx + dv_dy
            
            grad_div_x = self.physics.op.diff(div_h_kinematic, axis=0, from_loc='m', to_loc='u')
            grad_div_y = self.physics.op.diff(div_h_kinematic, axis=1, from_loc='m', to_loc='v')
            # ----------------------------------------------------------------------------

            grad_pi_pp_x = self.physics.op.diff(pi_prime_prime, axis=0, from_loc='m', to_loc='u')
            grad_pi_pp_y = self.physics.op.diff(pi_prime_prime, axis=1, from_loc='m', to_loc='v')
            
            grad_pi_pp_z_w = self.physics.op.diff(pi_prime_prime, axis=2, from_loc='m', to_loc='w') * (self.physics.grid.dz / bg['dz_w_full'])
            grad_pi_pp_z_m = self.physics.op.avg(grad_pi_pp_z_w, axis=2, from_loc='w', to_loc='m')
            
            grad_pi_pp_z_u = self.physics.op.avg(grad_pi_pp_z_m, axis=0, from_loc='m', to_loc='u')
            grad_pi_pp_z_v = self.physics.op.avg(grad_pi_pp_z_m, axis=1, from_loc='m', to_loc='v')
            
            z_xi_u = self.physics.op.diff(self.physics.grid.Z_m, axis=0, from_loc='m', to_loc='u')
            z_eta_v = self.physics.op.diff(self.physics.grid.Z_m, axis=1, from_loc='m', to_loc='v')
            
            grad_pi_x_cart = grad_pi_pp_x - z_xi_u * grad_pi_pp_z_u
            grad_pi_y_cart = grad_pi_pp_y - z_eta_v * grad_pi_pp_z_v

            # Add the divergence damping to the explicit acoustic integration
            u_prime_prime += dtau * (slow_forcings['u'] - cp * bg['th_v_u'] * grad_pi_x_cart * m_u + self.physics.nu_div * grad_div_x)
            v_prime_prime += dtau * (slow_forcings['v'] - cp * bg['th_v_v'] * grad_pi_y_cart * m_v + self.physics.nu_div * grad_div_y)

            u_prime_prime = u_prime_prime.at[0, :, :].set(state_init['u'][0, :, :] - state_current['u'][0, :, :])
            u_prime_prime = u_prime_prime.at[-1, :, :].set(state_init['u'][-1, :, :] - state_current['u'][-1, :, :])
            v_prime_prime = v_prime_prime.at[:, 0, :].set(state_init['v'][:, 0, :] - state_current['v'][:, 0, :])
            v_prime_prime = v_prime_prime.at[:, -1, :].set(state_init['v'][:, -1, :] - state_current['v'][:, -1, :])

            # =================================================================
            # PART 2: ADVANCE PRESSURE & VERTICAL VELOCITY (BACKWARD STEP)
            # =================================================================
            flux_x_pp = (u_prime_prime * bg['rho_u'] * bg['th_v_u'] * bg['dz_u']) / m_u
            flux_y_pp = (v_prime_prime * bg['rho_v'] * bg['th_v_v'] * bg['dz_v']) / m_v
            
            div_x_pp = self.physics.op.diff(flux_x_pp, axis=0, from_loc='u', to_loc='m') / bg['dz_m_full']
            div_y_pp = self.physics.op.diff(flux_y_pp, axis=1, from_loc='v', to_loc='m') / bg['dz_m_full']
            div_h_pp = m_m * (div_x_pp + div_y_pp)
            
            pi_prime_prime, w_prime_prime = self._solve_vertical_acoustic_column(
                pi_pp=pi_prime_prime, 
                w_pp=w_prime_prime, 
                u_pp=u_prime_prime,
                v_pp=v_prime_prime,
                div_h=div_h_pp, 
                slow_f_pi=slow_forcings['pi'], 
                slow_f_w=slow_forcings['w'], 
                th_v_w_bg=bg['th_v_w'],
                C_pi=bg['C_pi'],
                dz_m=bg['dz_m_full'],
                dz_w=bg['dz_w_full'],
                bg=bg,
                dtau=dtau
            )

            # Reconstruct the total w field and damp it backward-implicitly 
            # using the acoustic dtau to prevent rigid lid reflection
            w_total = state_current['w'] + w_prime_prime
            w_total_damped = w_total / (1.0 + dtau * self.physics.tau_damp)
            w_prime_prime = w_total_damped - state_current['w']

        state_updated = {
            'u': state_current['u'] + u_prime_prime,
            'v': state_current['v'] + v_prime_prime,
            'w': state_current['w'] + w_prime_prime,
            'pi': state_current['pi'] + pi_prime_prime,
            'th_v': state_init['th_v'] + dtau * num_steps * slow_forcings['th_v'],
            'rho': state_current['rho'],
            'eta_dot': state_current.get('eta_dot', jnp.zeros_like(w_prime_prime))
        }

        for key in state_current:
            if key not in state_updated:
                state_updated[key] = state_current[key]
        
        return state_updated

    def _solve_vertical_acoustic_column(self, pi_pp, w_pp, u_pp, v_pp, div_h, slow_f_pi, slow_f_w, 
                                        th_v_w_bg, C_pi, dz_m, dz_w, bg, dtau):
        r"""
        Solves the coupled 1D vertical acoustic equations implicitly.

        Eliminates the vertical velocity to form a 1D vertical Helmholtz equation for 
        the pressure increment $\delta \pi$, solved via a fast tridiagonal matrix solver:

        $$ \delta \pi - C_\pi \Delta \tau^2 \frac{\partial}{\partial z} \left( c_p \bar{\theta}_v \bar{\rho} \frac{\partial \delta \pi}{\partial z} \right) = \text{RHS}_{explicit} $$

        Where the RHS contains the step-wise explicit changes in pressure and divergence.
        By solving this system implicitly, the model safely bypasses the restrictive 
        vertical Courant-Friedrichs-Lewy (CFL) condition usually associated with tightly 
        packed vertical grids and vertically propagating sound waves.

        Args:
            pi_pp (jnp.ndarray): Current pressure perturbation $\pi''$.
            w_pp (jnp.ndarray): Current vertical velocity perturbation $w''$.
            u_pp (jnp.ndarray): Current zonal velocity perturbation $u''$.
            v_pp (jnp.ndarray): Current meridional velocity perturbation $v''$.
            div_h (jnp.ndarray): Horizontal divergence of the perturbation winds.
            slow_f_pi (jnp.ndarray): Slow forcing tendency for pressure.
            slow_f_w (jnp.ndarray): Slow forcing tendency for vertical velocity.
            th_v_w_bg (jnp.ndarray): Background virtual potential temperature at w-points.
            C_pi (jnp.ndarray): Thermodynamic compressibility profile.
            dz_m (jnp.ndarray): Vertical grid spacing at mass points.
            dz_w (jnp.ndarray): Vertical grid spacing at w-points.
            bg (dict): Precomputed background state metrics.
            dtau (float): Acoustic time step $\Delta \tau$ [s].

        Returns:
            tuple: A tuple containing:
                - pi_pp_next (jnp.ndarray): Updated pressure perturbation $\pi''$.
                - w_pp_next (jnp.ndarray): Updated vertical velocity perturbation $w''$.
        """
        alpha = self.alpha
        cp = self.physics.c['cp']
        
        u_pp_m = self.physics.op.avg(u_pp, axis=0, from_loc='u', to_loc='m')
        u_pp_w = self.physics.op.avg(u_pp_m, axis=2, from_loc='m', to_loc='w')
        v_pp_m = self.physics.op.avg(v_pp, axis=1, from_loc='v', to_loc='m')
        v_pp_w = self.physics.op.avg(v_pp_m, axis=2, from_loc='m', to_loc='w')
        m_w = jnp.expand_dims(self.physics.grid.m_factors['w'], axis=-1)

        w_sfc_pp = m_w[:, :, 0] * (u_pp_w[:, :, 0] * self.physics.grid.z_xi_w[:, :, 0] + v_pp_w[:, :, 0] * self.physics.grid.z_eta_w[:, :, 0])
        
        # Predict explicit w
        grad_pi_pp_now = self.physics.op.diff(pi_pp, axis=2, from_loc='m', to_loc='w') * (self.physics.grid.dz / dz_w)
        w_explicit_rhs = w_pp + dtau * (slow_f_w - cp * th_v_w_bg * grad_pi_pp_now)
        
        w_avg_explicit = (1.0 - alpha) * w_pp + alpha * w_explicit_rhs
        w_avg_explicit = w_avg_explicit.at[:, :, 0].set(w_sfc_pp).at[:, :, -1].set(0.0)
        
        # Predict explicit divergence
        eta_dot_avg_explicit_dz = w_avg_explicit - m_w * (u_pp_w * self.physics.grid.z_xi_w + v_pp_w * self.physics.grid.z_eta_w)
        eta_dot_avg_explicit_dz = eta_dot_avg_explicit_dz.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        
        flux_z_explicit = eta_dot_avg_explicit_dz * bg['rho_w'] * bg['th_v_w']
        div_z_explicit = (self.physics.op.diff(flux_z_explicit, axis=2, from_loc='w', to_loc='m') * self.physics.grid.dz) / dz_m

        # The RHS is the step-wise explicit change in pi, not the total pi!
        delta_pi_explicit = dtau * (slow_f_pi - C_pi * (div_h + div_z_explicit))

        # Setup Implicit Matrix
        gamma = (alpha * dtau)**2 * cp * th_v_w_bg * bg['rho_w'] * bg['th_v_w'] / dz_w
        gamma = gamma.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0) 
        
        lower = - (C_pi / dz_m) * gamma[:, :, :-1]
        upper = - (C_pi / dz_m) * gamma[:, :, 1:]
        main  = 1.0 - lower - upper
        
        # Solve for delta_pi
        rhs_helmholtz = delta_pi_explicit[..., None]
        delta_pi_expanded = tridiagonal_solve(lower, main, upper, rhs_helmholtz)
        delta_pi = delta_pi_expanded[..., 0]
        
        # Update pi and correct w
        pi_pp_next = pi_pp + delta_pi
        
        # Velocity is corrected solely by the gradient of the pressure increment!
        grad_delta_pi = self.physics.op.diff(delta_pi, axis=2, from_loc='m', to_loc='w') * (self.physics.grid.dz / dz_w)
        w_pp_next = w_explicit_rhs - alpha * dtau * cp * th_v_w_bg * grad_delta_pi
        w_pp_next = w_pp_next.at[:, :, 0].set(w_sfc_pp).at[:, :, -1].set(0.0)
        
        return pi_pp_next, w_pp_next