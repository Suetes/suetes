#!/usr/bin/env python3
"""
Train NEUVE Coordinate via Proxy Forward Optimization (Step 1)
=============================================================================
Minimizes forward advection dispersion over aggressive steep topography
(twin-peaks mountain) by optimizing 3D NEUVE coordinate weights phi:

    L_fwd(phi) = RMSE( psi_model(X, Y, Z_phys, T; phi), psi_ref(X, Y, Z_phys, T) )

CRITICAL PHYSICAL & GEOMETRIC FIXES:
  1. High-Frequency "Jagged" & Widened Terrain Barrier:
     Introduces jagged spatial oscillations and wider cross-flow ridge extent
     so flow must advect vertically over jagged peaks where analytical coordinates fail.
  2. Cosine Decay Schedule & Best Checkpoint Tracking:
     Prevents late-stage Adam overshooting and saves the optimal weights.
  3. t_end = 900.0 s & nz=32 High-Fidelity Reference Plume Target.
"""

import os
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite
from suetes.shared.transforms import GalChenSigma, StretchedSleveSimple, NEUVECoordinate

output_dir = "output/plots/neuve"
os.makedirs(output_dir, exist_ok=True)
weights_path = "output/trained_neuve_weights.bin"

# --- 1. DOMAIN & SHEARED FLOW CONFIGURATION ---
nx, ny, nz = 32, 12, 16
dx, dy, dz = 500.0, 500.0, 375.0
dt = 3.0
t_end = 900.0
num_steps = int(t_end / dt)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

def terrain_profile(x, y):
    h0 = 2400.0
    ax, ay = 1100.0, 6000.0  # Widened along y so flow must pass over ridge barrier
    jagged_noise = 1.0 + 0.12 * jnp.cos(2 * jnp.pi * x / 2000.0)
    peak1 = h0 * jnp.exp(- ((x + 3500.0)**2 / ax**2 + y**2 / ay**2)) * jagged_noise
    h1 = 1300.0
    bx, by = 900.0, 5000.0
    peak2 = h1 * jnp.exp(- ((x - 1000.0)**2 / bx**2 + y**2 / by**2))
    return peak1 + peak2

def get_wind_u(Z_coords, n_vert=nz, dz_val=dz):
    u_sfc, u_top = 8.0, 15.0
    H_top = n_vert * dz_val
    return u_sfc + (u_top - u_sfc) * jnp.clip(Z_coords / H_top, 0.0, 1.0)

def get_wind_v(Z_coords):
    return jnp.zeros_like(Z_coords)

# Initial plume centered at (-5.0 km, 0.0 km, 3.6 km) aloft above mountain barrier
x0_km, y0_km, z0_km, A0 = -5.0, 0.0, 3.6, 10.0

# Absolute physical evaluation heights (meters above sea level)
Z_eval = jnp.linspace(1500.0, 5500.0, 16)


# --- 2. GENERATE PHYSICAL HIGH-FIDELITY REFERENCE PLUME (nz=32) ---
def compute_physical_reference_plume():
    nz_fine = 48
    dz_fine = dz / 3.0
    fine_grid = RegionalGrid3D(
        nx, ny, nz_fine, dx, dy, dz_fine,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_profile,
        transform=GalChenSigma()
    )
    op_fine = CGridOperator3D(fine_grid)
    suite = PhysicsSuite()
    suite.register_tracer('q_tr')
    
    physics_fine = Euler3D(
        fine_grid, op_fine, constants, dt=dt, N_bv=0.01, damp_height=4200.0,
        max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite
    )
    bg_ref = {
        'rho': physics_fine.c['p0'] / (physics_fine.c['Rd'] * physics_fine.theta_bg) * \
               (physics_fine.pi_bg ** (physics_fine.c['cvd'] / physics_fine.c['Rd'])),
        'pi': physics_fine.pi_bg,
        'th_v': physics_fine.theta_bg
    }
    sponge = BenchmarkSponge(nx, ny, sponge_depth=4, axes=('x', 'y'))
    
    def bc_fine(state_in, forcing=None):
        ext_state = {
            'u': get_wind_u(fine_grid.Z_u, n_vert=nz_fine, dz_val=dz_fine),
            'v': get_wind_v(fine_grid.Z_v),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi'],
            'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
        }
        return sponge.blend(state_in, ext_state)
    
    base_state_fine = {
        'u': get_wind_u(fine_grid.Z_u, n_vert=nz_fine, dz_val=dz_fine),
        'v': get_wind_v(fine_grid.Z_v),
        'w': jnp.zeros((nx, ny, nz_fine+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz_fine+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v'],
        'q_tr': jnp.zeros((nx, ny, nz_fine))
    }
    
    stepper_fine, _ = build_dynamical_core(
        core_type="split-explicit", grid=fine_grid, operators=op_fine, constants=constants,
        initial_state=base_state_fine, physics_suite=suite,
        dt=dt, ns=24, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=4200.0, max_damp=0.5, N_bv=0.01
    )
    
    X_km = fine_grid.x_m[:, None, None] / 1000.0
    Y_km = fine_grid.y_m[None, :, None] / 1000.0
    Z_km = fine_grid.Z_m / 1000.0
    q_init = A0 * jnp.exp(-((X_km - x0_km)**2 / 2.0 + (Y_km - y0_km)**2 / 2.0 + (Z_km - z0_km)**2 / 1.0))
    init_s = dict(base_state_fine)
    init_s['q_tr'] = q_init
    
    def scan_fwd(st, _):
        return stepper_fine.step(st, 0.0, None, bc_fine), None
    
    final_fine = jax.lax.scan(scan_fwd, init_s, jnp.arange(num_steps))[0]
    fine_q = final_fine['q_tr']
    
    # Interpolate from fine physical Z_m onto absolute Z_eval heights
    ref_phys = np.zeros((nx, ny, len(Z_eval)))
    for i in range(nx):
        for j in range(ny):
            ref_phys[i, j, :] = np.interp(Z_eval, fine_grid.Z_m[i, j, :], fine_q[i, j, :])
            
    return jnp.array(ref_phys)


def main():
    print("=========================================================================")
    print("STEP 1: PROXY TRAINING TASK (FORWARD ADVECTION OPTIMIZATION)")
    print("=========================================================================")

    print(f"Running high-fidelity nz=32 physical reference advection over jagged terrain (t_end={t_end} s)...")
    psi_ref_phys = compute_physical_reference_plume()
    print(f"[INFO] Reference physical plume generated. Peak signal: {float(jnp.max(psi_ref_phys)):.4e}")

    neuve_init = NEUVECoordinate(hidden_dim=32, key_seed=42)

    def make_simulation_for_params(phi_params):
        transform_op = neuve_init.with_params(phi_params)
        grid = RegionalGrid3D(
            nx, ny, nz, dx, dy, dz,
            lat_center=45.0, lon_center=0.0,
            h_func=terrain_profile,
            transform=transform_op
        )
        op = CGridOperator3D(grid)
        suite = PhysicsSuite()
        suite.register_tracer('q_tr')
        
        physics = Euler3D(
            grid, op, constants, dt=dt, N_bv=0.01, damp_height=4200.0,
            max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite
        )
        
        bg_ref = {
            'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
                   (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
            'pi': physics.pi_bg,
            'th_v': physics.theta_bg
        }
        sponge = BenchmarkSponge(nx, ny, sponge_depth=4, axes=('x', 'y'))
        
        def bc_fn(state_in, forcing=None):
            ext_state = {
                'u': get_wind_u(grid.Z_u),
                'v': get_wind_v(grid.Z_v),
                'th_v': bg_ref['th_v'],
                'rho': bg_ref['rho'],
                'pi': bg_ref['pi'],
                'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
            }
            return sponge.blend(state_in, ext_state)
        
        base_state = {
            'u': get_wind_u(grid.Z_u),
            'v': get_wind_v(grid.Z_v),
            'w': jnp.zeros((nx, ny, nz+1)),
            'pi': bg_ref['pi'],
            'eta_dot': jnp.zeros((nx, ny, nz+1)),
            'rho': bg_ref['rho'],
            'th_v': bg_ref['th_v'],
            'q_tr': jnp.zeros((nx, ny, nz))
        }
        
        stepper, _ = build_dynamical_core(
            core_type="split-explicit", grid=grid, operators=op, constants=constants,
            initial_state=base_state, physics_suite=suite,
            dt=dt, ns=12, nu_div_factor=0.0, nu_h_factor=0.0,
            damp_height=4200.0, max_damp=0.5, N_bv=0.01
        )
        stepper.use_checkpointing = True
        return grid, base_state, stepper, bc_fn

    def forward_proxy_loss(phi):
        grid, base_state, stepper, bc_fn = make_simulation_for_params(phi)
        
        X_km = grid.x_m[:, None, None] / 1000.0
        Y_km = grid.y_m[None, :, None] / 1000.0
        Z_km = grid.Z_m / 1000.0
        q_init = A0 * jnp.exp(-((X_km - x0_km)**2 / 2.0 + (Y_km - y0_km)**2 / 2.0 + (Z_km - z0_km)**2 / 1.0))
        
        init_s = dict(base_state)
        init_s['q_tr'] = q_init
        
        def scan_fwd(st, _):
            return stepper.step(st, 0.0, None, bc_fn), None
        
        final_s = jax.lax.scan(scan_fwd, init_s, jnp.arange(num_steps))[0]
        psi_model = final_s['q_tr']
        
        def interp_col(col_z, col_q):
            return jnp.interp(Z_eval, col_z, col_q)
        
        psi_model_phys = jax.vmap(jax.vmap(interp_col))(grid.Z_m, psi_model)
        
        # 1. Height Mask: Focus heavily on the bottom 5km over the mountain
        height_mask = jnp.where(grid.Z_m < 5000.0, 5.0, 1.0)
        # 2. Plume Mask: Focus where the reference plume actually exists
        plume_mask = jnp.where(psi_ref_phys > 1e-4, 10.0, 1.0)
        combined_mask = height_mask * plume_mask
        
        weighted_mse = jnp.mean(combined_mask * (psi_model_phys - psi_ref_phys)**2)
        
        # 3. Curvature Smoothness Penalty: regularize second derivatives of Z_m
        d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)
        d2Z_dz2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=2), axis=2)
        smoothness_loss = jnp.mean(d2Z_dx2**2) + jnp.mean(d2Z_dz2**2)
        
        lambda_reg = 1e-3
        total_loss = jnp.sqrt(weighted_mse) + lambda_reg * jnp.sqrt(smoothness_loss + 1e-12)
        return total_loss

    loss_and_grad_fn = jax.jit(jax.value_and_grad(forward_proxy_loss))

    n_epochs = 40
    schedule = optax.cosine_decay_schedule(
        init_value=3e-3,
        decay_steps=n_epochs,
        alpha=0.01
    )
    opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=schedule)
    )
    phi = neuve_init.params
    state = opt.init(phi)

    best_loss = float('inf')
    best_phi = phi
    loss_history = []
    t0 = time.time()

    print(f"\nOptimizing NEUVE vertical coordinate over {n_epochs} forward advection steps (t_end={t_end} s)...")
    print("-" * 75)

    for epoch in range(n_epochs):
        lval, grad = loss_and_grad_fn(phi)
        updates, state = opt.update(grad, state, phi)
        phi = optax.apply_updates(phi, updates)
        l_float = float(lval)
        loss_history.append(l_float)
        if l_float < best_loss:
            best_loss = l_float
            best_phi = phi
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{n_epochs} | Physical RMSE L_fwd(phi): {l_float:.6f} | Best: {best_loss:.6f}")
        if (epoch + 1) % 10 == 0:
            jax.clear_caches()
            print(f"  Epoch {epoch+1:02d}/{n_epochs} | Physical RMSE L_fwd(phi): {l_float:.6f} | Best: {best_loss:.6f}")

    elapsed = time.time() - t0
    print("-" * 75)
    print(f"[SUCCESS] Proxy forward optimization completed in {elapsed:.1f} seconds.")
    print(f"          Initial RMSE: {loss_history[0]:.6f} -> Best RMSE: {best_loss:.6f}")

    trained_neuve = neuve_init.with_params(best_phi)
    trained_neuve.save_weights(weights_path)
    print(f"[SUCCESS] Best NEUVE coordinate weights saved to: {weights_path}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, n_epochs + 1), loss_history, 'o-', color='#10b981', linewidth=2.5, label='Training Loss')
    ax.axhline(best_loss, color='#059669', linestyle='--', label=f'Best RMSE ({best_loss:.6f})')
    ax.set_title(f"NEUVE Proxy Training over Jagged Terrain (t={t_end}s)", fontsize=13)
    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel(r"Physical RMSE $\mathcal{L}_{fwd}(\phi)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/training_forward_loss.png", dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved training curve plot to: {output_dir}/training_forward_loss.png")


if __name__ == '__main__':
    main()
