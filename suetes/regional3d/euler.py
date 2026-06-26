r"""
Dynamical Core Module.

Evaluates the right-hand side (RHS) tendencies for the 3D fully compressible, 
non-hydrostatic Euler equations. Formulated using Exner pressure ($\pi$) and 
virtual potential temperature ($\theta_v$) on a terrain-following Arakawa C-grid.
"""

import jax
import jax.numpy as jnp
from suetes.regional3d.diffusion import HyperFilter

class Euler3D:
    r"""
    3D fully compressible, non-hydrostatic Euler equation solver.

    Computes the spatial prognostic tendencies for velocity $(u, v, w)$ and 
    Exner pressure perturbation $(\pi')$. To isolate high-frequency acoustic and 
    gravity wave modes for implicit treatment, the thermodynamic fields are split 
    into a time-invariant hydrostatic background state and a prognostic perturbation:
    $$ 
    \begin{align}
    \pi &= \bar{\pi}(z) + \pi'(x, y, z, t) \\
    \theta_v &= \bar{\theta}_v(z) + \theta_v'(x, y, z, t)
    \end{align}
    $$ 
    where the background states satisfy hydrostatic balance: 
    $\frac{\partial \bar{\pi}}{\partial z} = -\frac{g}{c_p \bar{\theta}_v}$.
    """
    def __init__(self, grid, operators, constants, dt, initial_era5_state=None, 
                 N_bv=0.01, damp_height=20000.0, max_damp=0.5, 
                 nu_div_factor=0.8, nu_h_factor=0.1, physics_suite=None,
                 interior_mask=None):
        """
        Initializes the dynamical core, calculating explicit diffusion limits 
        and building the 1D thermodynamic reference state.

        Args:
            grid (RegionalGrid3D): The 3D geometry and metric tensor object.
            operators (CGridOperator3D): Spatial finite-difference operators.
            constants (dict): Physical constants (e.g., $g$, $R_d$, $c_p$).
            dt (float): Integration time step [s].
            initial_era5_state (dict, optional): 3D state used to compute the 
                horizontal-mean reference state. If None, uses an analytical profile.
            N_bv (float): Brunt-Väisälä frequency for the analytical profile [$s^{-1}$].
            damp_height (float): Altitude where the Rayleigh sponge layer begins [m].
            max_damp (float): Maximum damping coefficient at the model top.
            nu_div_factor (float): Divergence damping scale factor (0.0 to 1.0).
            nu_h_factor (float): Hyperdiffusion scale factor (0.0 to 1.0).
            physics_suite (PhysicsSuite, optional): Configured subgrid physics.
            interior_mask (dict, optional): Per-field masks in [0, 1] keyed by
                'u', 'v', 'w', 'th_v'. When set, the physics-suite tendencies
                are multiplied by these before being added to the dynamical
                RHS, which switches physics off inside the Davies sponge zone.
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.dt = dt
        
        # Compute maximum stable explicit diffusion limits dynamically
        max_nu_div = (self.grid.dx**2) / (4.0 * self.dt)
        max_nu_h = (self.grid.dx**4) / (64.0 * self.dt)
        
        # Apply tuning factors (0.0 to 1.0)
        self.nu_div = nu_div_factor * max_nu_div
        self.nu_h = nu_h_factor * max_nu_h
        
        # Physics suite injection
        self.physics_suite = physics_suite
        self.interior_mask = interior_mask

        # Use physical 3D height (Z_m)
        Z_m = self.grid.Z_m
        
        if initial_era5_state is not None:
            # Get average 1D profile
            z_1d = jnp.mean(Z_m, axis=(0, 1))
            th_v_1d = jnp.mean(initial_era5_state['th_v'], axis=(0, 1))
            
            # Interpolate the potential temperature to the 3D grid
            self.theta_bg = jnp.interp(Z_m, z_1d, th_v_1d)
            
            # Get the true horizontal-mean pressure at the domain top
            pi_anchor_top = jnp.mean(initial_era5_state['pi'][:, :, -1])
        else:
            # Analytical background for idealized test suites
            self.theta_0 = 300.0
            self.theta_bg = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_m) if N_bv > 0.0 else self.theta_0 * jnp.ones_like(Z_m)
            
            # Calculate top boundary pressure for the analytical profile
            z_max = jnp.max(Z_m)
            if N_bv > 0.0:
                pi_anchor_top = 1.0 + (self.c['g']**2 / (self.c['cp'] * self.theta_0 * N_bv**2)) * (jnp.exp(-N_bv**2 * z_max / self.c['g']) - 1.0)
            else:
                pi_anchor_top = 1.0 - (self.c['g'] / (self.c['cp'] * self.theta_0)) * z_max

        # Hydrostatic reconstruction. The background state cancels the gravitational 
        # term in the discrete w-equation, leaving zero residual.
        delta_z = Z_m[:, :, 1:] - Z_m[:, :, :-1]
        th_v_w_bg = 0.5 * (self.theta_bg[:, :, 1:] + self.theta_bg[:, :, :-1])
        
        delta_pi = -(self.c['g'] * delta_z) / (self.c['cp'] * th_v_w_bg)
        
        # Integrate pressure down from the top lid
        pi_cumsum_rev = jnp.cumsum(delta_pi[..., ::-1], axis=-1)[..., ::-1]
        
        pi_top_3d = jnp.full((self.grid.nx, self.grid.ny, 1), pi_anchor_top)
        self.pi_bg = jnp.concatenate([
            pi_top_3d - pi_cumsum_rev,
            pi_top_3d
        ], axis=2)
        
        z_w_3d = self.grid.Z_w
        z_top = self.grid.Lz 
        
        self.tau_damp = jnp.where(
            z_w_3d > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w_3d - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

        # Initialize the spatial filter
        self.diffusion = HyperFilter(self.grid, self.op, nu_h=self.nu_h, nu_v = 0.0)


    def precompute_bg(self, bg_state):
        r"""
        Maps the hydrostatic background state onto the staggered C-grid faces.

        Precomputes the horizontal and vertical 2-point averages ($\overline{\bar{\rho}}^x, \overline{\bar{\theta}}_v^z$, etc.) 
        and the thermodynamic compressibility factor $C_{\pi}$:

        $$ C_{\pi} = \frac{R_d}{c_{vd}} \frac{\bar{\pi}}{\bar{\rho} \bar{\theta}_v} $$

        Precomputing these static fields optimizes the inner linear loop executed 
        by the GMRES solver.

        Args:
            bg_state (dict): Reference background arrays (`th_v`, `rho`, `pi`) evaluated at cell centers.

        Returns:
            dict: Mapped metrics, vertical cell thicknesses ($\Delta z_m, \Delta z_w$), and compressibility factors.
        """
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
            'pi_bg': pi_bg,
            'th_v': th_v_bg
        }

    def get_tendencies(self, state_prime, bg, is_explicit=False, ml_params=None):
        r"""
        Evaluates the spatial right-hand side (RHS) tendencies for the system.

        Computes the momentum, Exner pressure, and potential temperature equations 
        in a terrain-following coordinate system.

        Horizontal momentum equations:
        $$ \frac{\partial u}{\partial t} = -c_p \theta_v \cdot m_u \left( \frac{\partial \pi'}{\partial \xi} - \frac{\partial z}{\partial \xi}\frac{\partial \pi'}{\partial z} \right) + fv + D_u $$
        $$ \frac{\partial v}{\partial t} = -c_p \theta_v \cdot m_v \left( \frac{\partial \pi'}{\partial \eta} - \frac{\partial z}{\partial \eta}\frac{\partial \pi'}{\partial z} \right) - fu + D_v $$

        Vertical momentum equation:
        $$ \frac{\partial w}{\partial t} = -c_p \theta_{v,eff} \frac{\partial \pi'}{\partial z} + g \left( \frac{\theta_v'}{\bar{\theta}_v} \right) + D_w $$

        Continuity / Exner pressure equation:
        $$ \frac{\partial \pi'}{\partial t} = - C_\pi \left[ m^2 \left( \frac{\partial}{\partial \xi}\left(\frac{\bar{\rho} \bar{\theta}_v u}{m}\right) + \frac{\partial}{\partial \eta}\left(\frac{\bar{\rho} \bar{\theta}_v v}{m}\right) \right) + \frac{\partial}{\partial z}(\bar{\rho} \bar{\theta}_v \dot{\eta}) \right] $$

        where $m$ is the map scale factor, $D_i$ are diffusion tendencies, and $\dot{\eta}$ 
        is the contravariant vertical index velocity.

        Args:
            state_prime (dict): Prognostic variable perturbations mapped to their respective staggers.
            bg (dict): Precomputed background state metrics.
            is_explicit (bool): Flag toggling non-linear terms. If False, filters out 
                buoyancy, hyperdiffusion, and subgrid physics to satisfy the strict 
                linearity required by the GMRES solver matrix assembly.

        Returns:
            dict: Evaluated discrete tendencies for $(\dot{u}, \dot{v}, \dot{w}, \dot{\pi}, \dot{\theta}_v)$.
        """
        u, v, w, pi_prime, eta_dot = state_prime['u'], state_prime['v'], state_prime['w'], state_prime['pi'], state_prime['eta_dot']
        
        th_v_u = bg['th_v_u'] + state_prime.get('th_v_prime_u', 0.0)
        th_v_v = bg['th_v_v'] + state_prime.get('th_v_prime_v', 0.0)
        th_v_w = bg['th_v_w'] + state_prime.get('th_v_prime_w', 0.0)

        # Horizontal pressure gradients (use pi_prime)
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

        # Map factors from the projection
        m_u = jnp.expand_dims(self.grid.m_factors['u'], axis=-1)
        m_v = jnp.expand_dims(self.grid.m_factors['v'], axis=-1)

        # Apply map factors to the physical gradient
        tend_u = -self.c['cp'] * th_v_u * grad_pi_x_cart * m_u
        tend_v = -self.c['cp'] * th_v_v * grad_pi_y_cart * m_v

        # Vertical pressure gradient
        grad_pi_prime_z_w = self.op.diff(pi_prime, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])

        if is_explicit:
            # Perturbation form cancels discrete gravity at rest
            th_v_prime_w = state_prime.get('th_v_prime_w', 0.0)
            buoyancy = self.c['g'] * (th_v_prime_w / bg['th_v_w'])
            
            tend_w = -self.c['cp'] * th_v_w * grad_pi_prime_z_w + buoyancy
        else:
            # The implicit solver matrix requires strict linearity
            tend_w = -self.c['cp'] * bg['th_v_w'] * grad_pi_prime_z_w

        # Coriolis
        v_at_u = self.op.avg(self.op.avg(v, axis=1, from_loc='v', to_loc='m'), axis=0, from_loc='m', to_loc='u')
        u_at_v = self.op.avg(self.op.avg(u, axis=0, from_loc='u', to_loc='m'), axis=1, from_loc='m', to_loc='v')
        
        f_u_3d, f_v_3d = jnp.expand_dims(self.grid.f_u, axis=-1), jnp.expand_dims(self.grid.f_v, axis=-1)
        tend_u += f_u_3d * v_at_u
        tend_v -= f_v_3d * u_at_v

        # Divergence
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

        m_m = jnp.expand_dims(self.grid.m_factors['m'], axis=-1)

        # Multiply the horizontal divergence sum by m (vertical divergence is unaffected by the horizontal map factor).
        tend_pi = -bg['C_pi'] * (m_m * (div_x + div_y) + div_z)

        # Calculate vertical gradient of the background state
        dth_bg_dz_w = self.op.diff(bg['th_v'], axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        dth_bg_dz_m = self.op.avg(dth_bg_dz_w, axis=2, from_loc='w', to_loc='m')
        
        # The kinematic forcing for the perturbation is the vertical transport of the background state
        w_m = self.op.avg(w, axis=2, from_loc='w', to_loc='m')
        tend_th_v = -w_m * dth_bg_dz_m

        # Diffusion and physics are only done in the explicit part
        if is_explicit:

            phys_diff_u = 0.0
            phys_diff_v = 0.0
            phys_diff_w = 0.0

            # Diffusion
            if self.nu_h > 0.0 or self.nu_div > 0.0:
                diff_tends = self.diffusion.get_tendencies(state_prime, bg_precomputed=bg)
                tend_u += diff_tends['u']
                tend_v += diff_tends['v']
                tend_w += diff_tends['w']
                tend_th_v += diff_tends['th_v']
                phys_diff_u += diff_tends['u']
                phys_diff_v += diff_tends['v']
                phys_diff_w += diff_tends['w']

            # Call physics suite
            if self.physics_suite is not None:
                phys_tends = self.physics_suite.get_explicit_tendencies(
                    state_prime, bg, interior_mask=self.interior_mask, ml_params=ml_params
                )

                # Add them to the dynamical core's right-hand side
                for k in phys_tends:
                    if k == 'u': 
                        tend_u += phys_tends['u']
                        phys_diff_u += phys_tends['u']
                    if k == 'v': 
                        tend_v += phys_tends['v']
                        phys_diff_v += phys_tends['v']
                    if k == 'w': 
                        tend_w += phys_tends['w']
                        phys_diff_w += phys_tends['w']
                    if k == 'th_v': tend_th_v += phys_tends['th_v']

            return {'u': tend_u, 'v': tend_v, 'w': tend_w, 'pi': tend_pi, 'th_v': tend_th_v, 
                    'phys_diff_u': phys_diff_u, 'phys_diff_v': phys_diff_v, 'phys_diff_w': phys_diff_w}

        return {'u': tend_u, 'v': tend_v, 'w': tend_w, 'pi': tend_pi, 'th_v': tend_th_v}

    def linear_operator(self, state_prime, bg, dt, alpha=0.55):
        r"""
        Evaluates the discrete linear operator matrix-vector product $\mathcal{L}(\mathbf{x})$.

        Used inside the implicit Krylov solver loop. Evaluates the linear residual vector:
        
        $$ \mathcal{L}(\mathbf{x}) = \mathbf{x} - \alpha \Delta t \mathcal{T}_{linear}(\mathbf{x}) $$

        where $\mathcal{T}_{linear}$ represents the purely linear acoustic and gravity 
        wave core tendencies, and $\alpha$ is the semi-implicit off-centering parameter. 
        Enforces rigid lid boundaries ($w=0, \dot{\eta}=0$) and zero-flow boundary conditions 
        along lateral limits during the solver operations.

        Args:
            state_prime (dict): Current implicit state vector guess $\mathbf{x}$.
            bg (dict): Precomputed background state metrics.
            dt (float): Integration time step $\Delta t$ [s].
            alpha (float, optional): Semi-implicit off-centering parameter. Defaults to 0.55.

        Returns:
            dict: The evaluated residual vector fields matching the state dictionary layout.
        """
        tends = self.get_tendencies(state_prime, bg, is_explicit=False)
        
        L_u = state_prime['u'] - alpha * dt * tends['u']
        L_v = state_prime['v'] - alpha * dt * tends['v']
        L_w = (1.0 + dt * self.tau_damp) * state_prime['w'] - alpha * dt * tends['w']
        L_pi = state_prime['pi'] - alpha * dt * tends['pi']
        
        # Bring horizontal winds to the W-grid
        u_m = self.op.avg(state_prime['u'], axis=0, from_loc='u', to_loc='m')
        u_w = self.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
        v_m = self.op.avg(state_prime['v'], axis=1, from_loc='v', to_loc='m')
        v_w = self.op.avg(v_m, axis=2, from_loc='m', to_loc='w')

        # Lock u and v at the lateral boundaries for GMRES
        L_u = L_u.at[0, :, :].set(state_prime['u'][0, :, :])
        L_u = L_u.at[-1, :, :].set(state_prime['u'][-1, :, :])
        L_v = L_v.at[:, 0, :].set(state_prime['v'][:, 0, :])
        L_v = L_v.at[:, -1, :].set(state_prime['v'][:, -1, :])

        # Override L_w at vertical boundaries only (this is mathematically correct)
        m_w = jnp.expand_dims(self.grid.m_factors['w'], axis=-1)

        # Fix the vertical boundary conditions (Kinematic bottom, Rigid top)
        L_w = L_w.at[:, :, 0].set(state_prime['w'][:, :, 0])
        L_w = L_w.at[:, :, -1].set(state_prime['w'][:, :, -1])

        L_eta_dot = alpha * (
            bg['dz_w_full'] * state_prime['eta_dot'] 
            + m_w * (u_w * self.grid.z_xi_w + v_w * self.grid.z_eta_w) 
            - state_prime['w']
        )
        
        # eta_dot is strictly 0 at surface and top model levels
        L_eta_dot = L_eta_dot.at[:, :, 0].set(state_prime['eta_dot'][:, :, 0])
        L_eta_dot = L_eta_dot.at[:, :, -1].set(state_prime['eta_dot'][:, :, -1])
        
        return {'u': L_u, 'v': L_v, 'w': L_w, 'pi': L_pi, 'eta_dot': L_eta_dot}