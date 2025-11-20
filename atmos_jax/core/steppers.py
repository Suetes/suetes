import jax
import jax.numpy as jnp

# ==============================================================================
# BASE INTEGRATOR (Mixin)
# ==============================================================================
class IntegratorMixin:
    def integrate(self, state_init, t_start, t_end, forcing_fn, bc_fn):
        """
        Common integration loop using jax.lax.while_loop.
        """
        if not hasattr(self, 'dt'):
            raise ValueError("Integrator must have 'dt' attribute.")

        def cond_fun(carry):
            _, t = carry
            # Run while t < t_end (with epsilon for float safety)
            return t < t_end - 0.5 * self.dt

        def body_fun(carry):
            state, t = carry
            forcing = forcing_fn(t)
            # Call the specific stepper implementation
            new_state = self.step(state, t, forcing, bc_fn)
            return (new_state, t + self.dt)

        # FIX: while_loop returns ONLY the carry (state, t)
        final_state, final_t = jax.lax.while_loop(cond_fun, body_fun, (state_init, t_start))
        return final_state

# ==============================================================================
# 1. CLASSIC RK4
# ==============================================================================
class RK4(IntegratorMixin):
    def __init__(self, physics, dt):
        self.physics = physics
        self.dt = dt

    def step(self, state, t, forcing, bc_fn):
        dt = self.dt
        u_ref = forcing['u_ref'] # Extract reference velocity
        
        # Helper to update only prognostic variables
        def euler_step(s_old, tendency, factor):
            res = {'background': s_old['background']}
            for k in tendency.keys():
                res[k] = s_old[k] + factor * tendency[k]
            return res

        # k1
        s0 = bc_fn(state, u_ref)
        k1 = self.physics.compute_rhs(s0, forcing)
        
        # k2
        s1 = bc_fn(euler_step(s0, k1, 0.5*dt), u_ref)
        k2 = self.physics.compute_rhs(s1, forcing)
        
        # k3
        s2 = bc_fn(euler_step(s0, k2, 0.5*dt), u_ref)
        k3 = self.physics.compute_rhs(s2, forcing)
        
        # k4
        s3 = bc_fn(euler_step(s0, k3, dt), u_ref)
        k4 = self.physics.compute_rhs(s3, forcing)
        
        # Final Update
        final = {'background': s0['background']}
        for k in k1.keys():
            final[k] = s0[k] + (dt/6.0)*(k1[k] + 2*k2[k] + 2*k3[k] + k4[k])
            
        return bc_fn(final, u_ref)

# ==============================================================================
# 2. SSP-RK3 (Shu-Osher)
# ==============================================================================
class SSPRK3(IntegratorMixin):
    def __init__(self, physics, dt):
        self.physics = physics
        self.dt = dt

    def step(self, state, t, forcing, bc_fn):
        dt = self.dt
        u_ref = forcing['u_ref']
        u_n = bc_fn(state, u_ref)
        
        # Robust Combination Helper
        def ssp_combine(u1, c1, u2, c2, rhs, c3):
            # Result = c1*u1 + c2*u2 + c3*dt*rhs
            res = {'background': u1['background']}
            for k in rhs.keys():
                val = c1 * u1[k] + c3 * dt * rhs[k]
                if u2 is not None:
                    val += c2 * u2[k]
                res[k] = val
            return res

        # Stage 1
        L0 = self.physics.compute_rhs(u_n, forcing)
        u1 = ssp_combine(u_n, 1.0, None, 0.0, L0, 1.0)
        u1 = bc_fn(u1, u_ref)
        
        # Stage 2
        L1 = self.physics.compute_rhs(u1, forcing)
        u2 = ssp_combine(u_n, 0.75, u1, 0.25, L1, 0.25)
        u2 = bc_fn(u2, u_ref)
        
        # Stage 3
        L2 = self.physics.compute_rhs(u2, forcing)
        u_next = ssp_combine(u_n, 1.0/3.0, u2, 2.0/3.0, L2, 2.0/3.0)
        
        return bc_fn(u_next, u_ref)