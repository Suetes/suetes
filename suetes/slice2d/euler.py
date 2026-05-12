import jax.numpy as jnp
from suetes.slice2d.operators import CGridOperator

class VerticalSlice:
    r"""
    Defines the spatial physics, background states, and differential operators 
    for the 2D (X-Z) non-hydrostatic Euler equations on a terrain-following C-grid.
    
    This class handles the transformation of horizontal gradients from the computational 
    space $(\xi, \zeta)$ to the physical space $(x, z)$, and pre-computes the 
    hydrostatically balanced reference state.
    """
    def __init__(self, grid, constants, damp_height=20000.0, max_damp=0.5, N_bv=0.01):
        r"""
        Initializes the physical environment and the horizontally uniform background state.
        
        The background state assumes hydrostatic balance:
        $$ \frac{\partial \bar{\pi}}{\partial z} = -\frac{g}{c_p \bar{\theta}_v} $$
        
        If the Brunt-Väisälä frequency $N > 0$, an isentropic stratification is applied:
        $$ \bar{\theta}_v(z) = \theta_0 \exp\left(\frac{N^2}{g} z\right) $$
        
        Parameters:
            grid (StaggeredGrid): The terrain-following 2D grid structure.
            constants (dict): Physical constants (e.g., $g, c_p, R_d$).
            damp_height (float): Altitude in meters where the Rayleigh damping sponge begins.
            max_damp (float): Maximum damping coefficient at the model top.
            N_bv (float): Brunt-Väisälä frequency ($s^{-1}$) for the background state.
        """
        self.grid = grid
        self.c = constants
        self.op = CGridOperator(grid.dx, grid.dz, periodic_x=grid.periodic_x) 
        self.theta_0 = 300.0
        
        # Use physical heights for the horizontally uniform background state
        Z_ref_m = self.grid.Z_m
        Z_ref_w = self.grid.Z_w
        
        if N_bv > 0.0:
            self.theta_bg = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_ref_m)
            pi_factor = (self.c['g']**2) / (self.c['cp'] * self.theta_0 * N_bv**2)
            self.pi_bg = 1.0 + pi_factor * (jnp.exp(-(N_bv**2 / self.c['g']) * Z_ref_m) - 1.0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_ref_w))
        else:
            self.theta_bg = self.theta_0 * jnp.ones_like(Z_ref_m) 
            self.pi_bg = 1.0 - (self.c['g'] * Z_ref_m) / (self.c['cp'] * self.theta_0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0) * jnp.ones_like(Z_ref_w)
        
        z_w = self.grid.Z_w[:, :]
        z_top = jnp.max(z_w)
        
        # Rayleigh damping profile (tanh) to prevent wave reflection at model top
        self.tau_damp = jnp.where(
            z_w > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

        self.metrics = {
            'm': self._compute_discrete_metrics(self.grid.Z_m),
            'u': self._compute_discrete_metrics(self.grid.Z_u),
            'w': self._compute_discrete_metrics(self.grid.Z_w)
        }

    def _compute_discrete_metrics(self, Z_grid):
        r"""
        Computes the discrete grid metrics required for terrain-following transformations.
        
        Calculates $z_\xi = \frac{\partial z}{\partial x}$ and 
        $z_\zeta = \frac{\partial z}{\partial \zeta}$ for a given staggered coordinate mesh.
        """
        Z_pad_z = jnp.pad(Z_grid, ((0, 0), (1, 1)), mode='edge')
        z_zeta = (Z_pad_z[:, 2:] - Z_pad_z[:, :-2]) / (2.0 * self.grid.dz)

        if getattr(self.grid, 'periodic_x', False):
            z_xi = (jnp.roll(Z_grid, -1, axis=0) - jnp.roll(Z_grid, 1, axis=0)) / (2.0 * self.grid.dx)
        else:
            Z_pad_x = jnp.pad(Z_grid, ((1, 1), (0, 0)), mode='edge')
            z_xi = (Z_pad_x[2:, :] - Z_pad_x[:-2, :]) / (2.0 * self.grid.dx)
            
        return {'z_xi': z_xi, 'z_zeta': z_zeta}

    def _get_metrics(self, grid_type: str):
        return self.metrics[grid_type]

    def get_explicit_tendencies(self, state, state_prime, bg):
        r"""
        Computes the spatial derivatives and physical source terms for the explicit 
        half-step of the Semi-Implicit integration.
        
        Evaluates the pressure gradient forces, applying the terrain-following chain rule:
        $$ \frac{\partial \pi'}{\partial x} \bigg|_z = \frac{\partial \pi'}{\partial \xi} - z_\xi \frac{\partial \pi'}{\partial z} $$
        
        Calculates the divergence of the mass flux:
        $$ \nabla \cdot (\bar{\rho} \bar{\theta}_v \mathbf{v}) $$
        
        And adds the explicit buoyancy acceleration to the vertical momentum:
        $$ B = g \frac{\theta_v'}{\bar{\theta}_v} $$
        
        Parameters:
            state (dict): Full physical state (used for nonlinear buoyancy).
            state_prime (dict): Perturbation state vector $[u', w', \pi', \dot{\eta}']$.
            bg (dict): Precomputed background fields and grid metrics.
            
        Returns:
            dict: Explicit tendencies for $[u, w, \pi]$.
        """
        u_prime, pi_prime, eta_dot_prime = state_prime['u'], state_prime['pi'], state_prime['eta_dot']
        
        # Pressure Gradients
        dpi_m = pi_prime[:, 1:] - pi_prime[:, :-1]
        grad_pi_z_inner = dpi_m / bg['dz_m_centers']
        grad_pi_prime_z = jnp.pad(grad_pi_z_inner, ((0,0), (1,1)), mode='edge')
        
        dpi_dxi = self.op.diff_x_m_to_u(pi_prime)
        grad_pi_z_at_u = self.op.avg_m_to_u(self.op.avg_w_to_m(grad_pi_prime_z))
        grad_pi_prime_x = dpi_dxi - bg['z_xi_u'] * grad_pi_z_at_u

        tend_u = -self.c['cp'] * bg['th_v_at_u'] * grad_pi_prime_x
        tend_w = -self.c['cp'] * bg['th_v_at_w'] * grad_pi_prime_z

        # Divergence
        flux_x_prime = u_prime * bg['rho_at_u'] * bg['th_v_at_u'] * bg['dz_u']
        div_x_prime = self.op.diff_x_u_to_m(flux_x_prime) / bg['dz_m_full']

        w_contravariant_prime = eta_dot_prime * bg['dz_w_full']
        flux_z_prime = w_contravariant_prime * bg['rho_at_w'] * bg['th_v_at_w']
        flux_z_prime = flux_z_prime.at[:, 0].set(0.0)
        flux_z_prime = flux_z_prime.at[:, -1].set(0.0)
        div_z_prime = (flux_z_prime[:, 1:] - flux_z_prime[:, :-1]) / bg['dz_m_full']

        tend_pi = -bg['C_pi'] * (div_x_prime + div_z_prime)

        # Buoyancy Source Term (Only evaluate if full state is provided)
        if state is not None:
            th_v_prime_n = state['th_v'] - self.theta_bg
            th_v_bg_w = self.op.avg_m_to_w(self.theta_bg)
            th_v_prime_w_n = self.op.avg_m_to_w(th_v_prime_n)
            buoyancy = self.c['g'] * (th_v_prime_w_n / th_v_bg_w)
            tend_w += buoyancy

        return {'u': tend_u, 'w': tend_w, 'pi': tend_pi}

    def evaluate_linear_operator(self, state_prime, bg, dt, beta=0.65):
        r"""
        Defines the linearized Euler equations $\mathbf{L}(\mathbf{x})$ for the implicit GMRES solver.
        
        This operator isolates the fast-moving acoustic modes to allow for large integration 
        timesteps. It computes the residual:
        $$ \mathbf{L}(\mathbf{x}) = \mathbf{x} - \beta \Delta t f_{lin}(\mathbf{x}) $$
        
        It also enforces the rigid-lid and kinematic bottom boundary constraints. At the surface:
        $$ w = u \frac{\partial z}{\partial x} + \dot{\eta} \frac{\partial z}{\partial \eta} $$
        
        Parameters:
            state_prime (dict): Perturbation state vector for the current GMRES iteration.
            bg (dict): Precomputed background fields.
            dt (float): Timestep in seconds.
            beta (float): Implicit off-centering parameter ($0.5 \leq \beta \leq 1.0$).
            
        Returns:
            dict: The residual fields mapped by the linear operator.
        """
        # state is None because buoyancy is handled explicitly, removing it from the acoustic linear operator.
        tends = self.get_explicit_tendencies(None, state_prime, bg) 
        
        # Apply off-centering (beta) to damp acoustic resonance
        L_u = state_prime['u'] - beta * dt * tends['u']
        L_w = (1.0 + dt * self.tau_damp) * state_prime['w'] - beta * dt * tends['w']
        L_pi = state_prime['pi'] - beta * dt * tends['pi']
        
        u_m = self.op.avg_u_to_m(state_prime['u'])
        u_w = self.op.avg_m_to_w(u_m)
        L_eta_dot = 0.5 * (
            bg['dz_w_full'] * state_prime['eta_dot'] 
            + u_w * bg['z_xi_w'] 
            - state_prime['w']
        )

        # Bottom Boundary: w - u * dz/dx = 0
        L_w = L_w.at[:, 0].set(state_prime['w'][:, 0])
        L_eta_dot = L_eta_dot.at[:, 0].set(state_prime['eta_dot'][:, 0])
        
        # Top Boundary: w = 0
        L_w = L_w.at[:, -1].set(state_prime['w'][:, -1])
        L_eta_dot = L_eta_dot.at[:, -1].set(state_prime['eta_dot'][:, -1])
        
        return {'u': L_u, 'w': L_w, 'pi': L_pi, 'eta_dot': L_eta_dot}

    def apply_hyper_diffusion(self, state, th_v_prime, dt, nu_ratio):
        r"""
        Applies a fourth-order explicit hyper-diffusion filter $\nabla^4 \phi$.
        
        This operator targets grid-scale ($2\Delta x$) numerical noise caused by 
        dispersive errors in the Semi-Lagrangian advection and pressure gradient solvers, 
        without significantly damping meteorologically relevant wave structures.
        
        Parameters:
            state (dict): The current physical state vector.
            th_v_prime (jnp.ndarray): The potential temperature perturbation field.
            dt (float): Timestep in seconds.
            nu_ratio (float): Non-dimensional hyper-diffusion coefficient.
            
        Returns:
            dict: The diffusive tendencies to be explicitly added to the state.
        """
        min_dx = min(self.grid.dx, self.grid.dz)
        nu4 = nu_ratio * (min_dx**4) / dt

        diff_u = self.op.hyper_diff_2d(state['u'], nu4)
        diff_w = self.op.hyper_diff_2d(state['w'], nu4)
        diff_th = self.op.hyper_diff_2d(th_v_prime, nu4)
        
        return {'u': diff_u, 'w': diff_w, 'th_v': diff_th}