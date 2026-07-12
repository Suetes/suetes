#!/usr/bin/env python3
"""
Train NEUVE Coordinate via Multi-Topography Proxy Forward Optimization
=============================================================================
Trains the spatially adaptive NEUVE coordinate across a FAMILY of procedurally
generated jagged mountain topographies so that it learns a generalised
terrain-conditional vertical stretching function b(y, h_norm).

Training Seeds: [101, 202, 303, 404]  (diverse jagged mountain barriers)
Test Seed:      999                    (held out — never seen during training)

Each epoch cycles through ALL training topographies and accumulates the
average forward proxy loss + curvature regulariser:

    L_train(phi) = (1/|S|) SUM_s [ RMSE(psi_model, psi_ref; terrain_s)
                                    + lambda * R_smooth(phi; terrain_s) ]
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
from suetes.shared.transforms import GalChenSigma, NEUVECoordinate

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

# Training and evaluation seed sets
TRAIN_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]
TEST_SEED = 999


# --- PARAMETRIC JAGGED TOPOGRAPHY GENERATOR ---
def parametric_terrain(x, y, params):
    """Pure JAX mathematical definition of the topography."""
    h0, cx0, ax0, ay0, h1, cx1, ax1, ay1, jagged_wavelength, jagged_amp, jagged_phase = params
    jagged = 1.0 + jagged_amp * jnp.cos(2 * jnp.pi * x / jagged_wavelength + jagged_phase)
    peak1 = h0 * jnp.exp(-((x - cx0)**2 / ax0**2 + y**2 / ay0**2)) * jagged
    peak2 = h1 * jnp.exp(-((x - cx1)**2 / ax1**2 + y**2 / ay1**2))
    return peak1 + peak2

def generate_jagged_terrain_params(seed):
    """Generates the 11 random parameters for the mountain and a callable function."""
    rng = np.random.RandomState(seed)

    params = jnp.array([
        rng.uniform(1500.0, 1900.0),      # h0
        rng.uniform(-2000.0, 2000.0),     # cx0
        rng.uniform(800.0, 1400.0),       # ax0
        rng.uniform(4500.0, 7000.0),      # ay0
        rng.uniform(300.0, 600.0),        # h1
        rng.uniform(500.0, 2500.0),       # cx1
        rng.uniform(600.0, 1200.0),       # ax1
        rng.uniform(3500.0, 6000.0),      # ay1
        rng.uniform(1200.0, 3000.0),      # jagged_wavelength
        rng.uniform(0.08, 0.18),          # jagged_amp
        rng.uniform(0.0, 2.0 * np.pi)     # jagged_phase
    ])

    def terrain_fn(x, y):
        return parametric_terrain(x, y, params)

    return params, terrain_fn


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


# --- 2. GENERATE PHYSICAL HIGH-FIDELITY REFERENCE PLUME FOR A GIVEN TERRAIN ---
def compute_physical_reference_plume(terrain_fn):
    """Compute high-resolution nz=48 GalChenSigma reference plume for a given terrain."""
    nz_fine = 48
    dz_fine = dz / 3.0
    fine_grid = RegionalGrid3D(
        nx, ny, nz_fine, dx, dy, dz_fine,
        lat_center=45.0, lon_center=0.0,
        h_func=terrain_fn,
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
    print("MULTI-TOPOGRAPHY PROXY TRAINING (ML GENERALIZATION)")
    print("=========================================================================")
    print(f"Training Seeds: {TRAIN_SEEDS}")
    print(f"Held-Out Test Seed: {TEST_SEED}")

    # Create a dummy grid just to retrieve the X, Y coordinates for precomputing heights
    dummy_grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, 
        h_func=lambda x, y: jnp.zeros_like(x), transform=GalChenSigma()
    )
    X_mesh = dummy_grid.x_m
    Y_mesh = dummy_grid.y_m

    # --- Pre-compute reference plumes and terrain arrays ---
    train_terrain_params = {}
    train_references = {}
    for seed in TRAIN_SEEDS:
        params, terrain_fn = generate_jagged_terrain_params(seed)
        print(f"  Computing nz=48 reference plume for terrain seed {seed}...")
        ref = compute_physical_reference_plume(terrain_fn)
        
        train_terrain_params[seed] = params
        train_references[seed] = ref
        print(f"    Peak signal: {float(jnp.max(ref)):.4e}")
    print(f"[INFO] All {len(TRAIN_SEEDS)} training reference plumes generated.\n")

    neuve_init = NEUVECoordinate(hidden_dim=32, key_seed=42)

    def make_simulation_for_params(phi_params, h_func):
        transform_op = neuve_init.with_params(phi_params)
        grid = RegionalGrid3D(
            nx, ny, nz, dx, dy, dz,
            lat_center=45.0, lon_center=0.0,
            h_func=h_func,  # Pass it directly
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
            dt=dt, ns=20, nu_div_factor=0.0, nu_h_factor=0.0,
            damp_height=4200.0, max_damp=0.5, N_bv=0.01
        )
        stepper.use_checkpointing = True
        return grid, base_state, stepper, bc_fn

    # --- UNIVERSAL JIT-COMPILED LOSS FUNCTION ---
    # We pass h_array and ref_array as dynamic arguments so JAX compiles this exactly ONCE.
    def universal_loss(phi, terrain_params, ref_array):
        # Dynamically build the continuous terrain function for this specific iteration
        h_func = lambda x, y: parametric_terrain(x, y, terrain_params)
        
        grid, base_state, stepper, bc_fn = make_simulation_for_params(phi, h_func)

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

        # Spatial & plume weighting mask
        height_mask = jnp.where(grid.Z_m < 5000.0, 5.0, 1.0)
        plume_mask = jnp.where(ref_array > 1e-4, 10.0, 1.0)
        combined_mask = height_mask * plume_mask

        weighted_mse = jnp.mean(combined_mask * (psi_model_phys - ref_array)**2)

        # Curvature smoothness penalty
        d2Z_dx2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=0), axis=0)
        d2Z_dz2 = jnp.gradient(jnp.gradient(grid.Z_m, axis=2), axis=2)
        smoothness_loss = jnp.mean(d2Z_dx2**2) + jnp.mean(d2Z_dz2**2)

        lambda_reg = 2e-4
        return jnp.sqrt(weighted_mse) + lambda_reg * jnp.sqrt(smoothness_loss + 1e-12)

    # Compile the value and grad function globally ONCE
    universal_val_and_grad = jax.jit(jax.value_and_grad(universal_loss))

    n_epochs = 40
    schedule = optax.cosine_decay_schedule(
        init_value=3e-3,
        decay_steps=n_epochs,
        alpha=0.001
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

    print(f"Optimizing NEUVE over {n_epochs} epochs across {len(TRAIN_SEEDS)} topographies (t_end={t_end} s)...")
    print(f"  [Strategy: Batched gradient accumulation over pre-compiled JIT graph]")
    print("-" * 75)

    for epoch in range(n_epochs):
        avg_loss = 0.0
        avg_grad = jax.tree.map(jnp.zeros_like, phi)
        
        for seed in TRAIN_SEEDS:
            # Pass the 11 terrain parameters instead of the 2D array
            t_params = train_terrain_params[seed]
            ref_array = train_references[seed]
            
            lval_s, grad_s = universal_val_and_grad(phi, t_params, ref_array)
            
            l_safe = jnp.where(jnp.isnan(lval_s), 0.0, lval_s)
            g_safe = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grad_s)
            
            avg_loss += float(l_safe)
            avg_grad = jax.tree.map(lambda a, b: a + b, avg_grad, g_safe)
            
        avg_loss /= len(TRAIN_SEEDS)
        avg_grad = jax.tree.map(lambda g: g / len(TRAIN_SEEDS), avg_grad)

        updates, state = opt.update(avg_grad, state, phi)
        phi = optax.apply_updates(phi, updates)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_phi = phi
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{n_epochs} | Avg Multi-Terrain RMSE: {avg_loss:.6f} | Best: {best_loss:.6f}")

    elapsed = time.time() - t0
    print("-" * 75)
    print(f"[SUCCESS] Multi-topography proxy optimization completed in {elapsed:.1f} seconds.")
    print(f"          Initial Avg RMSE: {loss_history[0]:.6f} -> Best Avg RMSE: {best_loss:.6f}")

    trained_neuve = neuve_init.with_params(best_phi)
    trained_neuve.save_weights(weights_path)
    print(f"[SUCCESS] Best NEUVE coordinate weights saved to: {weights_path}")

    # --- Training curve plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, n_epochs + 1), loss_history, 'o-', color='#10b981', linewidth=2.5, label='Multi-Terrain Avg Loss')
    ax.axhline(best_loss, color='#059669', linestyle='--', label=f'Best Avg RMSE ({best_loss:.6f})')
    ax.set_title(f"NEUVE Multi-Topography Training ({len(TRAIN_SEEDS)} terrains, t={t_end}s)", fontsize=13)
    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel(r"Avg Physical RMSE $\mathcal{L}_{fwd}(\phi)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/training_forward_loss.png", dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved training curve plot to: {output_dir}/training_forward_loss.png")


if __name__ == '__main__':
    main()