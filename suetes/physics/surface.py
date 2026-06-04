import jax.numpy as jnp

class McFarlaneSurfaceDrag:
    r"""
    Stability-dependent bulk surface drag and sensible heat flux.

    Follows McFarlane et al. (1992) GCMII, equations (2.4) and (2.5):

    $$ (C_m, C_h) = (C_{DM}, C_{DH}) \cdot (F_m(Ri_B), F_h(Ri_B)) $$

    The bulk Richardson number is formed from the lowest model mass level and
    the prescribed surface skin temperature theta_surf (Dirichlet condition
    from outside). Neutral coefficients come from the log law:

    $$ C_{DN} = (k / \ln(z_L / z_0))^2 $$

    GCMII uses the same stability function for momentum and heat.
    """

    def __init__(self, grid, operators, constants,
                 z_0=1e-4, epsilon=0.3, k_vk=0.4, theta_surf=None):
        """
        Args:
            grid (RegionalGrid3D): computational grid.
            operators (CGridOperator3D): finite-difference operators.
            constants (dict): physical constants; must contain 'g'.
            z_0 (float): surface roughness length [m]. McFarlane GCMII used
                per-class values from Wilson-Henderson-Sellers; here applied
                uniformly. Default 1e-4 m is the ocean value. Land values
                range roughly 0.01 to 1 m.
            epsilon (float): stability cutoff parameter. McFarlane: 0.3 over
                open water, 0.0 over land/ice.
            k_vk (float): von Karman constant.
            theta_surf: optional fallback if state['theta_surf'] is missing.
        """
        self.grid = grid
        self.op = operators
        self.c = constants
        self.z_0 = z_0
        self.epsilon = epsilon
        self.k_vk = k_vk
        self.theta_surf = theta_surf

    def _stability_function(self, Ri_B, A_sq):
        """McFarlane (1992) eq. (2.4). Same form for F_m and F_h in GCMII.

        Unstable branch sign convention as in McFarlaneVerticalDiffusion: the
        paper formula written with |Ri_B| produces F > 1 under Louis (1979)
        sign conventions, implemented here as an explicit + sign.
        """
        abs_Ri = jnp.abs(Ri_B)
        F_unstable = 1.0 + 10.0 * abs_Ri \
                     / (1.0 + 10.0 * jnp.sqrt(abs_Ri / (87.0 * A_sq + 1e-12)))
        # Prevents thermal runaway when the 250m deep layer encounters
        # massive temperature gradients over hot daytime land.
        F_unstable = jnp.minimum(F_unstable, 5.0)
        
        F_stable = (1.0 - 5.0 * self.epsilon * Ri_B) ** 2 \
                   / (1.0 + 10.0 * (1.0 - self.epsilon) * Ri_B)
        Ri_cutoff = 1.0 / (5.0 * self.epsilon + 1e-12)
        F = jnp.where(Ri_B < 0.0, F_unstable, F_stable)
        F = jnp.where(Ri_B > Ri_cutoff, 0.0, F)
        return F

    def get_tendencies(self, state, bg):
        u, v, th_v = state['u'], state['v'], state['th_v']

        # Mass-point wind speed throughout the column, with a soft floor
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        speed_m = jnp.sqrt(u_m ** 2 + v_m ** 2 + 1e-8)
        speed_u = self.op.avg(speed_m, axis=0, from_loc='m', to_loc='u')
        speed_v = self.op.avg(speed_m, axis=1, from_loc='m', to_loc='v')

        speed_m_surf = speed_m[:, :, 0]
        speed_u_surf = speed_u[:, :, 0]
        speed_v_surf = speed_v[:, :, 0]

        # Height of lowest mass level above the local terrain
        z_L = self.grid.Z_m[:, :, 0] - self.grid.Z_w[:, :, 0]

        # Neutral drag coefficient from log law (scalar in z_0, 2D in z_L)
        log_ratio = jnp.log(z_L / self.z_0)
        C_DN = (self.k_vk / log_ratio) ** 2

        # A^2 from eq. (2.5), used in the unstable branch denominator
        A_sq = (self.z_0 / z_L) * (self.k_vk ** 4) / (C_DN ** 2 + 1e-30)

        # Surface temperature (Dirichlet from outside)
        theta_surf = state.get('theta_surf', self.theta_surf)

        th_v_L = th_v[:, :, 0]
        speed_sq_surf = speed_m_surf ** 2

        if theta_surf is not None:
            Ri_B = self.c['g'] * (th_v_L - theta_surf) * z_L \
                   / (th_v_L * speed_sq_surf)
        else:
            Ri_B = jnp.zeros_like(z_L)

        F = self._stability_function(Ri_B, A_sq)
        C_eff = C_DN * F  # GCMII uses the same for momentum and heat

        # Promote C_eff to 3D for face averaging
        nx_m, ny_m, nz_m = u_m.shape
        C_eff_3d = jnp.broadcast_to(C_eff[:, :, None], (nx_m, ny_m, nz_m))
        C_u = self.op.avg(C_eff_3d, axis=0, from_loc='m', to_loc='u')[:, :, 0]
        C_v = self.op.avg(C_eff_3d, axis=1, from_loc='m', to_loc='v')[:, :, 0]

        # Surface momentum tendency (lowest layer only)
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        drag_u_surf = -C_u * speed_u_surf * u[:, :, 0] / dz_u_surf
        drag_v_surf = -C_v * speed_v_surf * v[:, :, 0] / dz_v_surf

        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)

        # Sensible heat flux (lowest layer only)
        if theta_surf is not None:
            dz_m_surf = bg['dz_m_full'][:, :, 0]
            shf_kin = C_eff * speed_m_surf * (theta_surf - th_v_L)
            tend_th = jnp.zeros_like(th_v).at[:, :, 0].set(shf_kin / dz_m_surf)
        else:
            tend_th = jnp.zeros_like(th_v)

        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th}


class BulkAerodynamicPBL:
    r"""
    Models the frictional deceleration of the wind and Sensible Heat Flux at the surface.

    Applies a bulk aerodynamic drag formula exclusively to the lowest model layer.
    """
    def __init__(self, grid, operators, theta_surf=None, Cd_ocean=0.001, Ch_ocean=0.001):
        self.grid = grid
        self.op = operators
        self.Cd = Cd_ocean  
        self.Ch = Ch_ocean
        self.theta_surf = theta_surf # Static surface skin temperature boundary condition

    def get_tendencies(self, state, bg):
        u, v = state['u'], state['v']
        
        # Bring horizontal winds to the mass points to calculate true wind speed
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        
        # Calculate full 3D wind speed magnitude 
        speed_m_3d = jnp.sqrt(u_m**2 + v_m**2 + 1e-8) 
        
        # Map the 3D wind speed back to the staggered faces
        speed_u_3d = self.op.avg(speed_m_3d, axis=0, from_loc='m', to_loc='u')
        speed_v_3d = self.op.avg(speed_m_3d, axis=1, from_loc='m', to_loc='v')
        
        # Extract the surface layer (level 0) for the drag calculation
        speed_u_surf = speed_u_3d[:, :, 0]
        speed_v_surf = speed_v_3d[:, :, 0]
        
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        
        drag_u_surf = -self.Cd * (speed_u_surf * u[:, :, 0]) / dz_u_surf
        drag_v_surf = -self.Cd * (speed_v_surf * v[:, :, 0]) / dz_v_surf
        
        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)
        
        # Sensible Heat Flux (SHF)
        theta_surf = state.get('theta_surf', self.theta_surf) # Extract from state
        
        if theta_surf is not None:
            speed_m_surf = speed_m_3d[:, :, 0]
            th_v_surf = state['th_v'][:, :, 0]
            dz_m_surf = bg['dz_m_full'][:, :, 0]
            
            # Positive flux warms the atmosphere (Ocean is warmer than air)
            shf_kinematic = self.Ch * speed_m_surf * (theta_surf - th_v_surf)
            heat_tend_surf = shf_kinematic / dz_m_surf
            
            tend_th_v = jnp.zeros_like(state['th_v']).at[:, :, 0].set(heat_tend_surf)
        else:
            tend_th_v = jnp.zeros_like(state['th_v'])
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v}


class BucketLSM:
    def __init__(self, grid, operators, constants, Cd_ocean=0.001, Ch_ocean=0.001, beta_land=0.2):
        self.grid = grid
        self.op = operators
        self.c = constants
        self.Cd_ocean = Cd_ocean
        self.Ch_ocean = Ch_ocean
        # Define higher roughness values for land
        self.Cd_land = 0.005 
        self.Ch_land = 0.005 
        self.beta_land = beta_land  
        self.epsilon = constants.get('epsilon', 0.622)
        self.Lv = 2.5e6

    def get_tendencies(self, state, bg):
        u, v, th_v, q = state['u'], state['v'], state['th_v'], state.get('q', jnp.zeros_like(state['th_v']))
        theta_surf = state['theta_surf']
        land_mask = state.get('land_fraction', jnp.zeros_like(theta_surf))
        
        # Spatially varying transfer coefficients (2D)
        Cd_eff = jnp.where(land_mask > 0.5, self.Cd_land, self.Cd_ocean)
        Ch_eff = jnp.where(land_mask > 0.5, self.Ch_land, self.Ch_ocean)
        
        # Calculate full 3D mass-point wind speed first
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        speed_m_3d = jnp.sqrt(u_m**2 + v_m**2 + 1e-8)
        
        # Average the 3D speed back to the staggered faces
        speed_u_3d = self.op.avg(speed_m_3d, axis=0, from_loc='m', to_loc='u')
        speed_v_3d = self.op.avg(speed_m_3d, axis=1, from_loc='m', to_loc='v')
        
        # Extract the surface layer 
        speed_u_surf = speed_u_3d[:, :, 0]
        speed_v_surf = speed_v_3d[:, :, 0]
        speed_m_surf = speed_m_3d[:, :, 0]
        
        # Use the mass-point shape, not the u-face shape
        nx_m, ny_m, nz_m = u_m.shape
        Cd_eff_3d = jnp.broadcast_to(Cd_eff[:, :, None], (nx_m, ny_m, nz_m))
        
        # Map to faces and extract the surface layer
        Cd_u = self.op.avg(Cd_eff_3d, axis=0, from_loc='m', to_loc='u')[:, :, 0]
        Cd_v = self.op.avg(Cd_eff_3d, axis=1, from_loc='m', to_loc='v')[:, :, 0]
        
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        
        # Apply the elevated land drag
        drag_u_surf = -Cd_u * (speed_u_surf * u[:, :, 0]) / dz_u_surf
        drag_v_surf = -Cd_v * (speed_v_surf * v[:, :, 0]) / dz_v_surf
        
        # Sensible Heat Flux (SHF)
        dz_m_surf = bg['dz_m_full'][:, :, 0]
        th_v_surf = th_v[:, :, 0]
        
        # Apply the elevated land heating
        shf_kinematic = Ch_eff * speed_m_surf * (theta_surf - th_v_surf)
        tend_th_v = jnp.zeros_like(th_v).at[:, :, 0].set(shf_kinematic / dz_m_surf)
        
        # Latent Heat Flux (LHF)
        p_surf = self.c['p0'] * (state['pi'][:, :, 0] ** (self.c['cp'] / self.c['Rd']))

        # Convert dry potential temperature to true absolute temperature for Tetens formula
        T_surf = theta_surf * state['pi'][:, :, 0]
        e_s_surf = 611.2 * jnp.exp(17.67 * (T_surf - 273.15) / (T_surf - 29.65))
        q_s_surf = (self.epsilon * e_s_surf) / (p_surf - (1.0 - self.epsilon) * e_s_surf)
        
        beta = jnp.where(land_mask > 0.5, self.beta_land, 1.0)
        
        lhf_kinematic = Ch_eff * speed_m_surf * beta * (q_s_surf - q[:, :, 0])
        lhf_kinematic = jnp.maximum(lhf_kinematic, 0.0) 
        tend_q = jnp.zeros_like(q).at[:, :, 0].set(lhf_kinematic / dz_m_surf)
        
        return {
            'u': jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf),
            'v': jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf),
            'th_v': tend_th_v,
            'q': tend_q
        }