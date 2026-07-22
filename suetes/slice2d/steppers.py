import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

def tridiagonal_solve(a, b, c, d):
    """
    Batched Thomas algorithm for 2D (X, Z) domains.
    
    Solves the tridiagonal system $\\mathbf{A}x = d$ simultaneously for all vertical columns, 
    where the matrix equation for each cell index $k$ along the Z-axis is:
    $$ a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k $$
    
    The inversion is performed exclusively along axis 1 (the Z-dimension) using `jax.lax.scan`
    to accumulate the forward sweep coefficients and back-substitute.
    
    Parameters:
        a (jnp.ndarray): Lower diagonal coefficients, shape (nx, nz).
        b (jnp.ndarray): Main diagonal coefficients, shape (nx, nz).
        c (jnp.ndarray): Upper diagonal coefficients, shape (nx, nz).
        d (jnp.ndarray): Right-hand side (RHS) forcing vectors, shape (nx, nz).
        
    Returns:
        jnp.ndarray: The solution vector $x$, shape (nx, nz).
    """
    nx, nz = d.shape
    
    cp = jnp.zeros_like(c)
    dp = jnp.zeros_like(d)
    
    # Forward sweep
    cp = cp.at[:, 0].set(c[:, 0] / b[:, 0])
    dp = dp.at[:, 0].set(d[:, 0] / b[:, 0])
    
    def forward_step(carry, z):
        cp_prev, dp_prev = carry
        denom = b[:, z] - a[:, z] * cp_prev
        cp_curr = c[:, z] / denom
        dp_curr = (d[:, z] - a[:, z] * dp_prev) / denom
        return (cp_curr, dp_curr), (cp_curr, dp_curr)
        
    _, (cp_all, dp_all) = jax.lax.scan(forward_step, (cp[:, 0], dp[:, 0]), jnp.arange(1, nz))
    
    # Transpose the scan outputs from (nz-1, nx) to (nx, nz-1)
    cp = cp.at[:, 1:].set(cp_all.T)
    dp = dp.at[:, 1:].set(dp_all.T)
    
    # Back substitution
    x = jnp.zeros_like(d)
    x = x.at[:, -1].set(dp[:, -1])
    
    def backward_step(x_next, z):
        x_curr = dp[:, z] - cp[:, z] * x_next
        return x_curr, x_curr
        
    _, x_all = jax.lax.scan(backward_step, x[:, -1], jnp.arange(nz-2, -1, -1))
    
    # x_all is (nz-1, nx). Reverse the Z axis (axis 0), then transpose to (nx, nz-1)
    x = x.at[:, :-1].set(x_all[::-1].T)
    
    return x

class VerticalPreconditioner2D:
    r"""
    Constructs and applies a vertical Helmholtz preconditioner for the Semi-Implicit solver.
    
    By ignoring horizontal derivatives, this preconditioner analytically collapses 
    the coupled linear Euler equations into a 1D tridiagonal system along the Z-axis. 
    Substituting the vertical momentum equation into the continuity equation yields 
    a 1D Helmholtz equation for the pressure perturbation $\pi'$:
    $\pi' - \frac{d}{dz} \left( \mathcal{K} \frac{d \pi'}{dz} \right) = \mathcal{R}_{helm}$
    
    Where the acoustic coefficient $\mathcal{K}$ is defined as:
    $\mathcal{K} = \frac{(\beta \Delta t)^2 c_p \bar{\rho} \bar{\theta}_v^2 C_\pi}{1 + \Delta t \tau_{damp}}$
    
    Solving this proxy system provides an excellent initial guess for the full 2D GMRES solver, 
    massively accelerating convergence by resolving the vertically propagating sound waves analytically.
    """
    def __init__(self, physics, dt, bg_precomputed, beta=0.65, pi_scale=1000.0):
        r"""
        Initializes the preconditioner and pre-assembles the tridiagonal matrix coefficients.
        
        Parameters:
            physics (VerticalSlice): The 2D spatial physics object.
            dt (float): Time step $\Delta t$ in seconds.
            bg_precomputed (dict): Precomputed reference states and metrics.
            beta (float): Semi-implicit off-centering parameter ($0.5 \leq \beta \leq 1.0$).
            pi_scale (float, optional): Scaling factor for Exner pressure. Defaults to 1000.0.
        """
        self.physics = physics
        self.dt = dt
        self.alpha = beta * dt
        self.pi_scale = pi_scale
        
        # Extract background fields
        self.th_v_w = bg_precomputed['th_v_at_w']
        self.rho_w = bg_precomputed['rho_at_w']
        self.dz_w_full = bg_precomputed['dz_w_full']
        self.dz_m_full = bg_precomputed['dz_m_full']
        self.C_pi = bg_precomputed['C_pi']
        self.z_xi_w = bg_precomputed['z_xi_w']
        
        self.tau_damp = self.physics.tau_damp
        
        # Base Helmholtz coefficient at cell centers
        self.L_pi = self.alpha * self.C_pi / self.dz_m_full
        
        # Acoustic scaling for the pressure gradient term on W-faces
        cp = self.physics.c['cp']
        self.K_z = (self.rho_w * self.th_v_w * self.alpha * self.dt * cp * self.th_v_w) / (1.0 + self.dt * self.tau_damp)
        
        # Rigid boundaries for W
        self.K_z = self.K_z.at[:, 0].set(0.0)
        self.K_z = self.K_z.at[:, -1].set(0.0)
        
        # Tridiagonal matrix assembly
        coeff_k_plus_1 = -self.L_pi * self.K_z[:, 1:] / self.dz_w_full[:, 1:]
        coeff_k_minus_1 = -self.L_pi * self.K_z[:, :-1] / self.dz_w_full[:, :-1]
        coeff_k = 1.0 - coeff_k_plus_1 - coeff_k_minus_1
        
        self.lower = coeff_k_minus_1
        self.main = coeff_k
        self.upper = coeff_k_plus_1

    def __call__(self, rhs_scaled):
        """
        Applies the preconditioned inversion to the scaled right-hand side.
        
        Parameters:
            rhs_scaled (dict): Scaled RHS forcing vectors $\\mathcal{R}$ from the advection step.
            
        Returns:
            dict: The preconditioned state increment $[u', w', \\pi', \\dot{\\eta}']$.
        """
        # Unscale for physical math
        rhs_pi_phys = rhs_scaled['pi'] / self.pi_scale
        rhs_w_phys = rhs_scaled['w']

        # Kinematic 2D forcing
        u_m = self.physics.op.avg_u_to_m(rhs_scaled['u'])
        u_w = self.physics.op.avg_m_to_w(u_m)
        kinematic_2d = u_w * self.z_xi_w

        traj_weight = 0.5
        w_contra_known = (rhs_w_phys / (1.0 + self.dt * self.tau_damp)) + (rhs_scaled['eta_dot'] / traj_weight) - kinematic_2d
        
        w_tilde = self.rho_w * self.th_v_w * w_contra_known
        w_tilde = w_tilde.at[:, 0].set(0.0)
        w_tilde = w_tilde.at[:, -1].set(0.0)
        
        div_w_tilde = w_tilde[:, 1:] - w_tilde[:, :-1]
        rhs_helmholtz = rhs_pi_phys - self.L_pi * div_w_tilde
        
        precond_pi_phys = tridiagonal_solve(self.lower, self.main, self.upper, rhs_helmholtz)
        
        grad_pi = (precond_pi_phys[:, 1:] - precond_pi_phys[:, :-1]) / self.dz_w_full[:, 1:-1]
        grad_pi = jnp.pad(grad_pi, ((0, 0), (1, 1)), mode='edge')
        
        cp = self.physics.c['cp']
        precond_w_phys = (rhs_w_phys - self.alpha * self.dt * cp * self.th_v_w * grad_pi) / (1.0 + self.dt * self.tau_damp)
        
        precond_w_phys = precond_w_phys.at[:, 0].set(rhs_w_phys[:, 0])
        precond_w_phys = precond_w_phys.at[:, -1].set(rhs_w_phys[:, -1])

        # Divide the interior back-substitution by dz_w_full
        precond_eta_dot = ((rhs_scaled['eta_dot'] / traj_weight) + precond_w_phys - kinematic_2d) / self.dz_w_full
        
        # Override boundaries directly with the explicit RHS
        precond_eta_dot = precond_eta_dot.at[:, 0].set(rhs_scaled['eta_dot'][:, 0])
        precond_eta_dot = precond_eta_dot.at[:, -1].set(rhs_scaled['eta_dot'][:, -1])

        return {
            'u': rhs_scaled['u'],
            'w': precond_w_phys,
            'pi': precond_pi_phys * self.pi_scale, 
            'eta_dot': precond_eta_dot
        }

def _cubic_1d(p0, p1, p2, p3, t):
    """
    Standard 1D Lagrange cubic polynomial interpolation.
    
    Evaluates the cubic polynomial passing through four evenly spaced grid points.
    
    Parameters:
        p0, p1, p2, p3: Values at the surrounding grid nodes.
        t: Fractional distance ($0.0 \\leq t \\leq 1.0$) between p1 and p2.
        
    Returns:
        Interpolated value at distance $t$.
    """
    return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3) * t**3 + \
           (p0 - 2.5*p1 + 2.0*p2 - 0.5*p3) * t**2 + \
           (-0.5*p0 + 0.5*p2) * t + p1

def vmap_tensor_interp_2d(field, coords, periodic_x=False, true_nx=None, apply_limiter=False):
    """
    Batched 2D tensor-product cubic interpolation for advecting fields.
    
    Performs a cascadic 1D interpolation (X-axis first, then Z-axis) to evaluate 
    a field at the departure point $\\mathbf{x}_d$. 
    
    If `apply_limiter=True`, applies a Quasi-Monotone bounds-preserving limiter:
    $$ \\min(f_i) \\leq f_{interp}(\\mathbf{x}_d) \\leq \\max(f_i) $$
    where $f_i$ are the immediate surrounding logical neighbors.
    
    Parameters:
        field (jnp.ndarray): The 2D array to interpolate from.
        coords (tuple): Departure point coordinates $(x_d, z_d)$ in logical index space.
        periodic_x (bool): True if the X-boundary is periodic.
        true_nx (int): The true dimension of X (necessary for staggered grids).
        apply_limiter (bool): If True, clamps output to local immediate neighbors.
        
    Returns:
        jnp.ndarray: The interpolated field at the departure points.
    """
    nx_arr, nz = field.shape
    nx = true_nx if true_nx is not None else nx_arr  
    x, z = coords[0], coords[1]

    # Compute Indices and Fractional Distances
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

    # Tensor Product Interpolation (Cascadic)
    val_zm1 = _cubic_1d(field[xm1, zm1], field[x0, zm1], field[x1, zm1], field[x2, zm1], dx)
    val_z0  = _cubic_1d(field[xm1, z0],  field[x0, z0],  field[x1, z0],  field[x2, z0],  dx)
    val_z1  = _cubic_1d(field[xm1, z1],  field[x0, z1],  field[x1, z1],  field[x2, z1],  dx)
    val_z2  = _cubic_1d(field[xm1, z2],  field[x0, z2],  field[x1, z2],  field[x2, z2],  dx)

    f_interp = _cubic_1d(val_zm1, val_z0, val_z1, val_z2, dz)

    #  Quasi-Monotone Limiter
    if apply_limiter:
        neighbors = jnp.array([field[x0, z0], field[x1, z0], field[x0, z1], field[x1, z1]])
        f_min = jnp.min(neighbors, axis=0)
        f_max = jnp.max(neighbors, axis=0)
        return jnp.clip(f_interp, f_min, f_max)
    
    return f_interp

class SemiLagrangianAdvector:
    """
    Handles trajectory computations for Semi-Lagrangian advection.
    
    Calculates the departure point $\\mathbf{x}_d$ for a particle arriving at the grid   node $\\mathbf{x}_a$
    using an iterative midpoint scheme:
    $$
    \\begin{aligned}
    \\mathbf{x}_{mid} &= \\frac{\\mathbf{x}_a + \\mathbf{x}_d}{2} \\\\
    \\mathbf{x}_d &= \\mathbf{x}_a - \\Delta t \\mathbf{v}(\\mathbf{x}_{mid})
    \\end{aligned}
    $$
    """
    def __init__(self, grid, physics, dt):
        """
        Parameters:
            grid (StaggeredGrid): The 2D grid object.
            physics (VerticalSlice): The spatial physics wrapper.
            dt (float): Time step $\\Delta t$ in seconds.
        """
        self.grid = grid
        self.physics = physics 
        self.dt = dt

    def _get_logical_velocities(self, u_phys, eta_dot, loc='m'):
        """Maps physical horizontal winds to grid-relative logical speeds based on grid location."""
        eta_dot_m_s = eta_dot * self.grid.dz 
        
        if loc == 'm':
            u_loc, w_loc = self.physics.op.avg_u_to_m(u_phys), self.physics.op.avg_w_to_m(eta_dot_m_s)
        elif loc == 'u':
            u_loc, w_loc = u_phys, self.physics.op.avg_w_to_u(eta_dot_m_s)
        elif loc == 'w':
            u_loc, w_loc = self.physics.op.avg_u_to_w(u_phys), eta_dot_m_s
        return u_loc, w_loc

    def compute_departure_indices(self, u_phys, eta_dot, loc='m', iterations=1):
        """
        Iteratively computes the departure point coordinates $\\mathbf{x}_d$ for a trajectory 
        terminating at the given grid locations.
        
        Parameters:
            u_phys (jnp.ndarray): Physical X-velocity.
            eta_dot (jnp.ndarray): Contravariant Z-velocity $\\dot{\\eta}$.
            loc (str): Target staggered grid ('m', 'u', or 'w').
            iterations (int): Number of fixed-point iterations for trajectory midpoint.
            
        Returns:
            jnp.ndarray: Departure coordinates $[X_{idx}, Z_{idx}]$ in logical space.
        """
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
            
            u_mid = vmap_tensor_interp_2d(u_log, coords, periodic_x=self.grid.periodic_x, true_nx=nx)
            w_mid = vmap_tensor_interp_2d(w_log, coords, periodic_x=self.grid.periodic_x, true_nx=nx) 
            
            Xi_dep = Xi_arr - self.dt * u_mid
            Zeta_dep = Zeta_arr - self.dt * w_mid

        idx_x_dep = (Xi_dep / self.grid.dx) - offset_x
        idx_z_dep = (Zeta_dep / self.grid.dz) - offset_z
        idx_z_dep = jnp.clip(idx_z_dep, 0.0, self.grid.nz - (1 if loc in ['m', 'u'] else 0))
        
        return jnp.stack([idx_x_dep, idx_z_dep], axis=0)

    def advect(self, field, u_phys, w_phys, loc='m'):
        """One-shot helper to compute trajectories and interpolate."""
        coords = self.compute_departure_indices(u_phys, w_phys, loc)
        return vmap_tensor_interp_2d(field, coords, periodic_x=self.grid.periodic_x, true_nx=self.grid.nx)

class SemiImplicitSolver:
    """
    Wraps the JAX GMRES solver to resolve the implicit acoustic and gravity wave equations.
    
    Solves the linear system:
    $$ \mathbf{L}(\mathbf{x}^{n+1}) = \mathbf{R}_{adv}^n $$
    where $\mathbf{L}$ is the linear operator defining the fast wave dynamics, and $\mathbf{R}$ 
    is the explicit right-hand side advected to the arrival points.
    """
    def __init__(self, grid, physics, dt, solver_tol=1e-4, solver_maxiter=10, solver_restart=10):
        """
        Parameters:
            grid (StaggeredGrid): The 2D grid object.
            physics (VerticalSlice): The spatial physics operator.
            dt (float): Time step $\Delta t$ in seconds.
        """
        self.grid = grid
        self.physics = physics
        self.dt = dt
        self.pi_scale = 1000.0
        self.solver_tol = solver_tol
        self.solver_maxiter = solver_maxiter
        self.solver_restart = solver_restart

    def precompute_bg(self, bg_state):
        """
        Precalculates stationary physical backgrounds, metrics, and mappings.
        """
        th_v_bg, rho_bg, pi_bg = bg_state['th_v'], bg_state['rho'], bg_state['pi']
        
        dz_m_full = self.grid.Z_w[:, 1:] - self.grid.Z_w[:, :-1]
        dz_w_full = jnp.pad(self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1], ((0,0),(1,1)), mode='edge')

        bg = {
            'th_v_at_u': self.physics.op.avg_m_to_u(th_v_bg),
            'th_v_at_w': self.physics.op.avg_m_to_w(th_v_bg),
            'rho_at_u': self.physics.op.avg_m_to_u(rho_bg),
            'rho_at_w': self.physics.op.avg_m_to_w(rho_bg),
            'dz_m_centers': self.grid.Z_m[:, 1:] - self.grid.Z_m[:, :-1],
            'dz_m_full': dz_m_full,
            'dz_w_full': dz_w_full,
            'z_xi_u': self.physics.op.diff_x_m_to_u(self.grid.Z_m),
            'z_xi_w': self.physics._get_metrics('w')['z_xi'],
            'C_pi': (self.physics.c['Rd'] / self.physics.c['cvd']) * (pi_bg / (rho_bg * th_v_bg))
        }
        bg['dz_u'] = self.physics.op.avg_m_to_u(bg['dz_m_full'])
        return bg

    def solve(self, rhs_prime, bg_precomputed, beta=0.65, preconditioner=None, x0=None):
        """
        Executes the GMRES solve. State variables are scaled to an $O(1)$ magnitude 
        to ensure stable and well-conditioned matrix operations inside the Krylov subspace.
        
        Parameters:
            rhs_prime (dict): Advected right-hand side forcing $\mathbf{R}$.
            bg_precomputed (dict): Precomputed stationary variables.
            beta (float): Implicit off-centering parameter $\beta$.
            preconditioner (Callable): Preconditioner $\mathbf{M}^{-1}$ function to accelerate GMRES.
            x0 (dict, optional): Initial guess for the solution variables.
            
        Returns:
            dict: The implicitly solved dynamic updates $\mathbf{x}^{n+1}$.
        """
        rhs_scaled = {
            'u': rhs_prime['u'], 'w': rhs_prime['w'], 'pi': rhs_prime['pi'] * self.pi_scale, 'eta_dot': rhs_prime['eta_dot']
        }

        if x0 is not None:
            x0_scaled = {
                'u': x0['u'], 'w': x0['w'], 'pi': x0['pi'] * self.pi_scale, 'eta_dot': x0['eta_dot']
            }
        else:
            x0_scaled = rhs_scaled

        def A_fn(state_scaled):
            state_prime = {
                'u': state_scaled['u'], 'w': state_scaled['w'], 'pi': state_scaled['pi'] / self.pi_scale, 'eta_dot': state_scaled['eta_dot']
            }
            L_out = self.physics.evaluate_linear_operator(state_prime, bg_precomputed, self.dt, beta=beta)
            return {
                'u': L_out['u'], 'w': L_out['w'], 'pi': L_out['pi'] * self.pi_scale, 'eta_dot': L_out['eta_dot']
            }
        
        M_fn = preconditioner if preconditioner is not None else None

        def fwd_solve(matvec, b):
            x_sol, _ = gmres(
                matvec, b, x0=x0_scaled,
                tol=self.solver_tol, maxiter=self.solver_maxiter, restart=self.solver_restart,
                M=M_fn
            )
            return x_sol

        if M_fn is not None:
            def bwd_solve(matvec_T, b):
                M_T_raw = jax.linear_transpose(M_fn, rhs_scaled)
                M_T_fn = lambda y: M_T_raw(y)[0]
                y_sol, _ = gmres(
                    matvec_T, b, x0=None,
                    tol=self.solver_tol, maxiter=self.solver_maxiter, restart=self.solver_restart,
                    M=M_T_fn
                )
                return y_sol
        else:
            def bwd_solve(matvec_T, b):
                y_sol, _ = gmres(
                    matvec_T, b, x0=None,
                    tol=self.solver_tol, maxiter=self.solver_maxiter, restart=self.solver_restart,
                    M=None
                )
                return y_sol

        x_sol_scaled = jax.lax.custom_linear_solve(
            A_fn, rhs_scaled, solve=fwd_solve, transpose_solve=bwd_solve
        )
        
        return {
            'u': x_sol_scaled['u'], 'w': x_sol_scaled['w'], 'pi': x_sol_scaled['pi'] / self.pi_scale, 'eta_dot': x_sol_scaled['eta_dot']
        }

class SISLStepper:
    """
    The orchestrator for the 2-time-level Semi-Implicit Semi-Lagrangian (SISL) scheme.
    
    Evaluates the continuous governing equations:
    $$ \frac{D \mathbf{x}}{Dt} = \mathbf{N}(\mathbf{x}) + \mathbf{L}(\mathbf{x}) $$
    
    Using the time discretization:
    $$ \frac{\mathbf{x}^{n+1}_a - \mathbf{x}^n_d}{\Delta t} = (1-\beta) \left[ \mathbf{N}(\mathbf{x}^n_d) + \mathbf{L}(\mathbf{x}^n_d) \right] + \beta \mathbf{L}(\mathbf{x}^{n+1}_a) $$
    
    Where $\mathbf{N}$ contains slow nonlinear modes, $\mathbf{L}$ contains fast linear modes, 
    and subscripts $(a, d)$ denote the arrival and departure points respectively.
    """
    def __init__(self, physics, dt, nu_ratio=0.0, use_mass_fixer=False, tracer_keys=None, solver_tol=1e-4, solver_maxiter=10, solver_restart=10): 
        """
        Parameters:
            physics (VerticalSlice): Spatial operators and constants.
            dt (float): Time step $\Delta t$ in seconds.
            nu_ratio (float): Non-dimensional hyper-diffusion strength.
            use_mass_fixer (bool): Whether to enforce global mass conservation on tracers.
            tracer_keys (list): List of state dictionary keys denoting passive tracers.
        """
        self.physics = physics
        self.dt = dt
        self.nu_ratio = nu_ratio
        self.use_mass_fixer = use_mass_fixer
        self.tracer_keys = tracer_keys if tracer_keys is not None else []
        
        self.advector = SemiLagrangianAdvector(physics.grid, physics, dt)
        self.implicit_solver = SemiImplicitSolver(
            physics.grid, physics, dt, solver_tol=solver_tol, solver_maxiter=solver_maxiter, solver_restart=solver_restart
        )

    def _apply_mass_fixer(self, state_before, state_after):
        """
        Applies a global multiplicative mass fixer to passive tracers.
        """
        dz_m = self.physics.grid.Z_w[:, 1:] - self.physics.grid.Z_w[:, :-1]
        cell_volumes = self.physics.grid.dx * dz_m
        
        rho_before = state_before['rho']
        rho_after = state_after['rho']
        
        fixed_state = dict(state_after)
        
        for key in self.tracer_keys:
            if key in state_before and key in state_after:
                tr_before = state_before[key]
                tr_after = state_after[key]
                
                mass_before = jnp.sum(tr_before * rho_before * cell_volumes)
                mass_after = jnp.sum(tr_after * rho_after * cell_volumes)
                
                ratio = mass_before / (mass_after + 1e-15)
                fixed_state[key] = tr_after * ratio
                
        return fixed_state

    def integrate(self, state, t_start, num_steps, forcing, bc_fn):
        """Runs the time stepper iteratively using jax.lax.scan."""
        def scan_fn(curr_state, step_idx):
            t_curr = t_start + step_idx * self.dt
            next_state = self.step(curr_state, t_curr, forcing, bc_fn)
            return next_state, None
            
        final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))
        return final_state

    def step(self, state, t, forcing, bc_fn):
        """
        Performs one complete SISL integration step.
        
        Parameters:
            state (dict): Current physical state vector $\\mathbf{x}^n$.
            t (float): Current simulation time $t$.
            forcing (dict/Callable): External sources (radiation, friction, etc.).
            bc_fn (Callable): Boundary condition enforcer function.
            
        Returns:
            dict: The physical state advanced to $\\mathbf{x}^{n+1}$.
        """
        u_n, w_n = state['u'], state['w']
        eta_dot_n = state.get('eta_dot', jnp.zeros_like(w_n))
        periodic_x = self.physics.grid.periodic_x
        nx, nz = self.physics.grid.nx, self.physics.grid.nz
        beta = 0.65  

        # Compute departure points
        coords_u = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='u')
        coords_w = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='w')
        coords_m = self.advector.compute_departure_indices(u_n, eta_dot_n, loc='m')

        # EXPLICIT HALF
        rho_bg = (self.physics.c['p0'] / (self.physics.c['Rd'] * self.physics.theta_bg)) * \
                 (self.physics.pi_bg ** (self.physics.c['cvd'] / self.physics.c['Rd']))

        bg_state_ref = {'rho': rho_bg, 'pi': self.physics.pi_bg, 'th_v': self.physics.theta_bg}
        bg_precomputed = self.implicit_solver.precompute_bg(bg_state_ref)
        
        state_prime_n = {'u': state['u'], 'w': state['w'], 'pi': state['pi'] - self.physics.pi_bg, 'eta_dot': eta_dot_n}
        th_v_prime_n = state['th_v'] - self.physics.theta_bg

        # Extracted spatial tendencies and diffusions
        tends_n = self.physics.get_explicit_tendencies(state, state_prime_n, bg_precomputed)
        diff_tends = self.physics.apply_hyper_diffusion(state, th_v_prime_n, self.dt, self.nu_ratio)

        u_adv_in = state['u'] + (1.0 - beta) * self.dt * tends_n['u'] + self.dt * diff_tends['u']
        w_adv_in = state['w'] + (1.0 - beta) * self.dt * tends_n['w'] + self.dt * diff_tends['w']
        pi_prime_adv_in = state_prime_n['pi'] + (1.0 - beta) * self.dt * tends_n['pi']
        th_v_adv_in = state['th_v'] + self.dt * diff_tends['th_v']

        # Compute Explicit Kinematic Residual for R_eta_dot
        u_m_n = self.physics.op.avg_u_to_m(state['u'])
        u_w_n = self.physics.op.avg_m_to_w(u_m_n)
        
        explicit_kinematic = 0.5 * (
            bg_precomputed['dz_w_full'] * eta_dot_n 
            + u_w_n * bg_precomputed['z_xi_w'] 
            - state['w']
        )

        # ADVECTION
        rhs_u = vmap_tensor_interp_2d(u_adv_in, coords_u, periodic_x, true_nx=nx)
        rhs_w = vmap_tensor_interp_2d(w_adv_in, coords_w, periodic_x, true_nx=nx)
        rhs_pi_prime = vmap_tensor_interp_2d(pi_prime_adv_in, coords_m, periodic_x, true_nx=nx)
        th_v_next = vmap_tensor_interp_2d(th_v_adv_in, coords_m, periodic_x, true_nx=nx)
        
        # Advect the kinematic residual and apply strict boundaries
        R_eta_dot = vmap_tensor_interp_2d(explicit_kinematic, coords_w, periodic_x, true_nx=nx)
        R_eta_dot = R_eta_dot.at[:, 0].set(0.0)
        R_eta_dot = R_eta_dot.at[:, -1].set(0.0)

        # IMPLICIT HALF (Time N+1)
        th_v_prime_next = th_v_next - self.physics.theta_bg
        th_v_bg_w = self.physics.op.avg_m_to_w(self.physics.theta_bg)
        buoyancy_next = self.physics.c['g'] * (self.physics.op.avg_m_to_w(th_v_prime_next) / th_v_bg_w)
        rhs_w += beta * self.dt * buoyancy_next

        # Boundary overrides
        rhs_w = rhs_w.at[:, -1].set(0.0)

        kinematic_bottom = self.physics.op.avg_u_to_w(rhs_u)[:, 0] * bg_precomputed['z_xi_w'][:, 0]
        rhs_w = rhs_w.at[:, 0].set(kinematic_bottom)

        rhs_prime = {'u': rhs_u, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}

        # Instantiate preconditioner and solve
        precond = VerticalPreconditioner2D(self.physics, self.dt, bg_precomputed, beta, pi_scale=self.implicit_solver.pi_scale)
        state_prime_next = self.implicit_solver.solve(rhs_prime, bg_precomputed, beta=beta, preconditioner=precond, x0=state_prime_n)

        # Final state assembly
        pi_next = state_prime_next['pi'] + self.physics.pi_bg
        rho_next = (self.physics.c['p0'] / (self.physics.c['Rd'] * th_v_next)) * \
                   (pi_next ** (self.physics.c['cvd'] / self.physics.c['Rd']))

        state_next = {
            'u': state_prime_next['u'], 'w': state_prime_next['w'], 'pi': pi_next,
            'rho': rho_next, 'th_v': th_v_next, 'eta_dot': state_prime_next['eta_dot']
        }
        
        # Tracer advection (Strict Limiter Applied)
        for key in self.tracer_keys:
            if key in state:
                # Tracers are advected purely explicitly along the trajectories
                tr_next = vmap_tensor_interp_2d(state[key], coords_m, periodic_x, true_nx=nx, apply_limiter=True)
                state_next[key] = tr_next

        # Mass fixer
        if self.use_mass_fixer:
            state_next = self._apply_mass_fixer(state, state_next)

        return bc_fn(state_next, forcing)