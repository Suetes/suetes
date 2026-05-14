import os
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.physics import PhysicsSuite

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class NeuralEddyViscosity(nn.Module):
    """Predicts positive Horizontal and Vertical Eddy Viscosities bounded by 3D CFL limits."""
    max_nu_h: float
    max_nu_v: float
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3, 3), use_bias=False, padding='CIRCULAR')(x)
        x = nn.swish(x)
        
        x = nn.Conv(features=16, kernel_size=(3, 3, 3), use_bias=False, padding='CIRCULAR')(x)
        x = nn.swish(x)
        
        # Predict 2 channels: [nu_horizontal, nu_vertical]
        nu_raw = nn.Conv(
            features=2, 
            kernel_size=(3, 3, 3), 
            kernel_init=nn.initializers.zeros_init(),
            padding='CIRCULAR'
        )(x)
        
        # Apply independent hard bounds safely below the explicit limits
        nu_h = nn.sigmoid(nu_raw[..., 0]) * (self.max_nu_h * 0.95)
        nu_v = nn.sigmoid(nu_raw[..., 1]) * (self.max_nu_v * 0.95)
        
        return nu_h, nu_v

class LearnedDiffusion:
    """Applies strictly conservative Fickian diffusion using predicted viscosity."""
    def __init__(self, model, params, operators, grid, constants):
        self.model = model
        self.params = params
        self.op = operators
        self.grid = grid
        self.c = constants

    def get_tendencies(self, state, bg):
        u, v, w, th_v = state['u'], state['v'], state['w'], state['th_v']
        
        # 1. Calculate Galilean-Invariant Inputs (|S| and N^2)
        D11_m = self.op.diff(u, axis=0, from_loc='u', to_loc='m') / self.grid.dx
        D22_m = self.op.diff(v, axis=1, from_loc='v', to_loc='m') / self.grid.dy
        D33_m = self.op.diff(w, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])

        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        w_m = self.op.avg(w, axis=2, from_loc='w', to_loc='m')

        D12_m = self.op.avg(self.op.diff(u_m, axis=1, from_loc='m', to_loc='v') / self.grid.dy, axis=1, from_loc='v', to_loc='m') + \
                self.op.avg(self.op.diff(v_m, axis=0, from_loc='m', to_loc='u') / self.grid.dx, axis=0, from_loc='u', to_loc='m')
                
        D13_m = self.op.avg(self.op.diff(u_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full']), axis=2, from_loc='w', to_loc='m') + \
                self.op.avg(self.op.diff(w_m, axis=0, from_loc='m', to_loc='u') / self.grid.dx, axis=0, from_loc='u', to_loc='m')
                
        D23_m = self.op.avg(self.op.diff(v_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full']), axis=2, from_loc='w', to_loc='m') + \
                self.op.avg(self.op.diff(w_m, axis=1, from_loc='m', to_loc='v') / self.grid.dy, axis=1, from_loc='v', to_loc='m')

        S_mag = jnp.sqrt(2.0 * (D11_m**2 + D22_m**2 + D33_m**2) + D12_m**2 + D13_m**2 + D23_m**2 + 1e-12)
        
        dth_dz_w = self.op.diff(th_v, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
        N2_m = self.op.avg((self.c['g'] / bg['th_v_w']) * dth_dz_w, axis=2, from_loc='w', to_loc='m')

        # 2. Predict Viscosity Field
        inputs = jnp.stack([S_mag * 100.0, N2_m * 1000.0], axis=-1)
        inputs = jnp.expand_dims(inputs, axis=0) 
        nu_h_m, nu_v_m = self.model.apply(self.params, inputs)
        
        # Squeeze out the batch dimension
        nu_h_m = nu_h_m[0]
        nu_v_m = nu_v_m[0]

        # 3. Apply Strict Flux Divergence
        # Horizontal viscosity applied to x and y faces
        nu_h_u = self.op.avg(nu_h_m, axis=0, from_loc='m', to_loc='u')
        nu_h_v = self.op.avg(nu_h_m, axis=1, from_loc='m', to_loc='v')
        # Vertical viscosity applied to w faces
        nu_v_w = self.op.avg(nu_v_m, axis=2, from_loc='m', to_loc='w')

        def diffuse_scalar_on_m(phi_m):
            grad_x = self.op.diff(phi_m, axis=0, from_loc='m', to_loc='u') / self.grid.dx
            grad_y = self.op.diff(phi_m, axis=1, from_loc='m', to_loc='v') / self.grid.dy
            grad_z = self.op.diff(phi_m, axis=2, from_loc='m', to_loc='w') * (self.grid.dz / bg['dz_w_full'])
            
            div_x = self.op.diff(nu_h_u * grad_x, axis=0, from_loc='u', to_loc='m') / self.grid.dx
            div_y = self.op.diff(nu_h_v * grad_y, axis=1, from_loc='v', to_loc='m') / self.grid.dy
            div_z = self.op.diff(nu_v_w * grad_z, axis=2, from_loc='w', to_loc='m') * (self.grid.dz / bg['dz_m_full'])
            return div_x + div_y + div_z

        tend_u_m = diffuse_scalar_on_m(u_m)
        tend_v_m = diffuse_scalar_on_m(v_m)
        tend_w_m = diffuse_scalar_on_m(w_m)

        return {
            'u': self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u'),
            'v': self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v'),
            'w': self.op.avg(tend_w_m, axis=2, from_loc='m', to_loc='w')
        }

# ==========================================
# 2. SPECTRAL LOSS FUNCTION
# ==========================================
def generate_static_target_spectrum(w_spun_up, dx, dy):
    """Calculates the target k^-5/3 spectrum ONCE based on the spun-up state."""
    nx, ny, nz = w_spun_up.shape
    
    # Average the initial spectrum over all vertical levels to prevent wave-propagation noise
    energy_init_sum = jnp.zeros(nx * ny)
    for z in range(nz):
        w_hat = jnp.fft.fft2(w_spun_up[:, :, z])
        energy_init_sum += jnp.real(w_hat * jnp.conj(w_hat)).flatten()
    e_init_avg = energy_init_sum / nz

    kx = jnp.fft.fftfreq(nx, d=dx)
    ky = jnp.fft.fftfreq(ny, d=dy)
    Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
    k_mag = jnp.sqrt(Kx**2 + Ky**2).flatten()
    
    k_inertial_start = 1.0 / (15.0 * dx)  
    k_diss_start = 1.0 / (4.0 * dx)       
    
    mask_integral = (k_mag > 0.0) & (k_mag <= k_inertial_start)
    mask_inertial = (k_mag > k_inertial_start) & (k_mag <= k_diss_start)
    mask_dissipation = (k_mag > k_diss_start)
    
    anchor_energy = jnp.max(jnp.where(mask_integral, e_init_avg, 0.0))
    k_anchor = jnp.max(jnp.where(mask_integral, k_mag, 0.0))
    
    target_inertial = anchor_energy * (k_mag / (k_anchor + 1e-12))**(-5.0 / 3.0)
    anchor_diss_energy = anchor_energy * (k_diss_start / (k_anchor + 1e-12))**(-5.0 / 3.0)
    target_diss = anchor_diss_energy * (k_mag / k_diss_start)**(-3.0) 
    
    target_e = jnp.where(mask_integral, e_init_avg, 
                 jnp.where(mask_inertial, target_inertial, 
                   jnp.where(mask_dissipation, target_diss, 0.0)))
                   
    return target_e

def compute_spectral_loss_from_energy(e_flat, target_e, dx, dy, nx, ny):
    """Computes MSE using a pre-calculated time-averaged energy array."""
    kx = jnp.fft.fftfreq(nx, d=dx)
    ky = jnp.fft.fftfreq(ny, d=dy)
    Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
    k_mag = jnp.sqrt(Kx**2 + Ky**2).flatten()
    
    k_inertial_start = 1.0 / (15.0 * dx)  
    k_diss_start = 1.0 / (4.0 * dx)
    
    mask_integral = (k_mag > 0.0) & (k_mag <= k_inertial_start)
    mask_inertial = (k_mag > k_inertial_start) & (k_mag <= k_diss_start)
    mask_dissipation = (k_mag > k_diss_start)

    log_e_model = jnp.log(e_flat + 1e-12)
    log_e_target = jnp.log(target_e + 1e-12)
    
    loss_integral = jnp.sum(jnp.where(mask_integral, (log_e_model - log_e_target)**2, 0.0)) * 1.0
    loss_inertial = jnp.sum(jnp.where(mask_inertial, (log_e_model - log_e_target)**2, 0.0)) * 1.0
    loss_diss = jnp.sum(jnp.where(mask_dissipation, (log_e_model - log_e_target)**2, 0.0)) * 5.0
    
    total_valid_bins = jnp.maximum(jnp.sum(k_mag > 0), 1.0)
    return (loss_integral + loss_inertial + loss_diss) / total_valid_bins

# ==========================================
# 3. DOMAIN & INITIALIZATION
# ==========================================
def generate_initial_state(grid, physics, key):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    
    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)),
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v']
    }
    
    key_u, key_v = jax.random.split(key)
    raw_u = jax.random.normal(key_u, (nx, ny, nz))
    raw_v = jax.random.normal(key_v, (nx, ny, nz))
    
    def low_pass_filter(field, cutoff_k=4):
        f_hat = jnp.fft.fftn(field)
        kx = jnp.fft.fftfreq(nx) * nx
        ky = jnp.fft.fftfreq(ny) * ny
        kz = jnp.fft.fftfreq(nz) * nz
        Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = jnp.sqrt(Kx**2 + Ky**2 + Kz**2)
        f_hat_filtered = jnp.where(k_mag <= cutoff_k, f_hat, 0.0)
        return jnp.real(jnp.fft.ifftn(f_hat_filtered))

    pert_u = low_pass_filter(raw_u) * 5.0
    pert_v = low_pass_filter(raw_v) * 5.0

    state['u'] = state['u'].at[:-1, :, :].set(pert_u)
    state['u'] = state['u'].at[-1, :, :].set(pert_u[0, :, :]) 
    state['v'] = state['v'].at[:, :-1, :].set(pert_v)
    state['v'] = state['v'].at[:, -1, :].set(pert_v[:, 0, :])
    
    return state

# ==========================================
# 4. TRAINING ORCHESTRATION
# ==========================================
def main():
    print("[SETUP] Initializing Periodic Sandbox Domain...")
    nx, ny, nz = 64, 64, 16
    dx, dy, dz = 6000.0, 6000.0, 500.0
    dt = 30.0
    
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    
    base_physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)
    
    # Calculate grid CFL limits for 3D stability
    max_nu_h = (grid.dx**2) / (8.0 * dt)
    max_nu_v = (grid.dz**2) / (4.0 * dt)
    
    key = jax.random.PRNGKey(42)
    sgs_model = NeuralEddyViscosity(max_nu_h=max_nu_h, max_nu_v=max_nu_v)
    dummy_input = jnp.zeros((1, nx, ny, nz, 2))
    key, subkey = jax.random.split(key)
    params = sgs_model.init(subkey, dummy_input)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), 
        optax.adam(learning_rate=1e-4)   
    )
    opt_state = optimizer.init(params)
    
    def spin_up_cascade(state_in, steps=40):
        """Runs the baseline physics to develop the energy cascade before ML training."""
        stepper = SISLStepper3D(base_physics, dt, use_checkpointing=False)
        def scan_step(s, _):
            return stepper.step(s, 0.0, None, lambda x, f: x), None
        spun_up_state, _ = jax.lax.scan(scan_step, state_in, jnp.arange(steps))
        return spun_up_state

    @jax.jit
    def rollout_and_loss(current_params, current_state, target_spectrum):
        learned_sgs = LearnedDiffusion(sgs_model, current_params, op, grid, constants)
        suite = PhysicsSuite()
        suite.add_tendency_scheme(learned_sgs)
        
        physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, physics_suite=suite, nu_div_factor=0.0, nu_h_factor=0.0)
        stepper = SISLStepper3D(physics, dt, use_checkpointing=True)
        
        def scan_step(s, _):
            next_s = stepper.step(s, 0.0, None, lambda x, f: x)
            # Return next_s as both the carry and the stacked output
            return next_s, next_s['w'] 
            
        final_state, w_history = jax.lax.scan(scan_step, current_state, jnp.arange(15))
        
        # w_history has shape (15, nx, ny, nz). We want to calculate the spectrum 
        # for each of the 15 steps and average them.
        
        def get_spatial_energy(w_field):
            # Averages the spectrum over the z-axis for a single time step
            nx, ny, nz = w_field.shape
            energy_sum = jnp.zeros(nx * ny)
            for z in range(nz):
                w_hat = jnp.fft.fft2(w_field[:, :, z])
                energy_sum += jnp.real(w_hat * jnp.conj(w_hat)).flatten()
            return energy_sum / nz
            
        # vmap over the time dimension (axis 0)
        vmap_get_energy = jax.vmap(get_spatial_energy, in_axes=0)
        
        # Calculate energies for all 15 steps, then average them over time
        all_energies = vmap_get_energy(w_history)
        time_averaged_energy = jnp.mean(all_energies, axis=0)
        
        # Now pass the smooth, time-averaged energy to a slightly modified loss function
        loss = compute_spectral_loss_from_energy(time_averaged_energy, target_spectrum, dx, dy, nx, ny)
        
        return loss, final_state

    grad_fn = jax.value_and_grad(rollout_and_loss, has_aux=True)

    epochs = 200
    epochs_per_episode = 10
    print("\n[TRAINING] Starting Episodic TBPTT...")
    loss_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- NEW EPISODE SETUP ---
        if epoch % epochs_per_episode == 0:
            print(f"\n--- Starting New Episode (Resetting Domain Energy) ---")
            key, subkey = jax.random.split(key)
            raw_state = generate_initial_state(grid, base_physics, subkey)
            state = spin_up_cascade(raw_state)
            
            # Lock in the target spectrum for this specific episode
            target_spectrum = generate_static_target_spectrum(state['w'], dx, dy)
            
        # Pass the locked target spectrum into the gradient function
        (loss_val, next_state), grads = grad_fn(params, state, target_spectrum)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        state = jax.tree_util.tree_map(lambda x: x.block_until_ready(), next_state)
        
        loss_val = float(loss_val)
        loss_history.append(loss_val)
        print(f"Epoch {epoch+1:03d}/{epochs} | Spectral Loss: {loss_val:.4f} | Time: {time.time()-start_time:.1f}s")

    print("\n[COMPLETE] Saving loss history...")
    os.makedirs("suetes/plots/ml_sgs", exist_ok=True)
    plt.figure()
    plt.plot(loss_history)
    plt.title("Neural SGS Training Loss (Spectral)")
    plt.xlabel("TBPTT Chunks (15 steps each)")
    plt.ylabel("MSE against k^-5/3 target")
    plt.savefig("suetes/plots/ml_sgs/training_loss.png")

    def evaluate_benchmark(trained_params, grid, op, constants, dt):
        print("\n[BENCHMARK] Running side-by-side comparison...")
        key = jax.random.PRNGKey(777)
        eval_state_init = generate_initial_state(grid, base_physics, key)
        
        sim_steps = int(3600.0 / dt)

        def run_sim(physics_suite_obj, name):
            print(f"Running {name}...")
            phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, 
                        physics_suite=physics_suite_obj, 
                        nu_div_factor=0.5 if name=="Baseline" else 0.0, 
                        nu_h_factor=0.1 if name=="Baseline" else 0.0)
            stepper = SISLStepper3D(phys, dt, use_checkpointing=False)
            
            final_state, _ = jax.lax.scan(
                lambda s, _: (stepper.step(s, 0.0, None, lambda x, f: x), None), 
                eval_state_init, jnp.arange(sim_steps)
            )
            return final_state

        # --- Run Baseline ---
        suite_base = PhysicsSuite()
        state_base = run_sim(suite_base, "Baseline")

        # --- Run Neural SGS ---
        suite_nn = PhysicsSuite()
        learned_sgs = LearnedDiffusion(NeuralEddyViscosity(max_nu_h=max_nu_h, max_nu_v=max_nu_v), trained_params, op, grid, constants)
        suite_nn.add_tendency_scheme(learned_sgs)
        state_nn = run_sim(suite_nn, "Neural SGS")

        def get_spectrum(w_field):
            w_hat = jnp.fft.fft2(w_field[:, :, grid.nz // 2])
            energy = jnp.real(w_hat * jnp.conj(w_hat))
            kx = jnp.fft.fftfreq(grid.nx, d=grid.dx)
            ky = jnp.fft.fftfreq(grid.ny, d=grid.dy)
            Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
            k_mag = jnp.sqrt(Kx**2 + Ky**2)
            
            k_bins = jnp.linspace(0, jnp.max(k_mag), grid.nx // 2)
            digitized = jnp.digitize(k_mag.flatten(), k_bins)
            bin_energies = [jnp.mean(energy.flatten()[digitized == i]) for i in range(1, len(k_bins))]
            return k_bins[1:], bin_energies

        k_valid, e_base = get_spectrum(state_base['w'])
        _, e_nn = get_spectrum(state_nn['w'])

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        z_idx = grid.nz // 2
        
        w_base = state_base['w'][:, :, z_idx]
        vmax = float(jnp.max(jnp.abs(w_base))) * 0.8
        im1 = axs[0].imshow(w_base.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axs[0].set_title("Baseline (Heuristic Hyperdiffusion)")
        fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        w_nn = state_nn['w'][:, :, z_idx]
        im2 = axs[1].imshow(w_nn.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axs[1].set_title("Neural SGS (Learned Viscosity)")
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        
        axs[2].loglog(k_valid, e_base, 'r--', linewidth=2, label='Baseline')
        axs[2].loglog(k_valid, e_nn, 'b-', linewidth=2, label='Neural SGS')
        
        theoretical_e = e_nn[2] * (k_valid / k_valid[2])**(-5.0/3.0)
        axs[2].loglog(k_valid, theoretical_e, 'k:', label='$k^{-5/3}$ cascade')
        
        axs[2].axvline(1.0 / (2 * grid.dx), color='gray', linestyle='-', alpha=0.5, label=rf'2$\Delta x$ limit')
        axs[2].set_title("Mid-level W Power Spectrum")
        axs[2].set_xlabel("Wavenumber $k$ [$m^{-1}$]")
        axs[2].set_ylabel("Spectral power density")
        axs[2].grid(True, which="both", ls="--", alpha=0.3)
        axs[2].legend()

        plt.tight_layout()
        os.makedirs("suetes/plots/ml_sgs", exist_ok=True)
        plt.savefig("suetes/plots/ml_sgs/benchmark_comparison.png", dpi=150)
        print("\n[BENCHMARK] Saved to benchmark_comparison.png")

    evaluate_benchmark(params, grid, op, constants, dt)

if __name__ == "__main__":
    main()