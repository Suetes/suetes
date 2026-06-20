import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import random
import pickle
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
import numpy as np

# suetes imports
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.preprocessing.topography import TopographyProcessor
from suetes.shared.transforms import SleveSimple
from suetes.physics.base import PhysicsSuite
from suetes.physics.turbulence import McFarlaneVerticalDiffusion
from suetes.physics.ml import MLPhysicsClosure
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.forcing import NewtonianRelaxation

DATA_DIR = 'suetes/data'

# ---------------------------------------------------------
# 1. DIFFERENTIABLE ROLLOUT
# ---------------------------------------------------------
def build_differentiable_step(stepper, sponge):
    def step_fn(curr_state, step_args):
        step_idx, nn_params, bc_state_t, local_land_frac = step_args 
        
        t_curr = step_idx * stepper.dt
        
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)
            
        dynamic_theta_surf = (
            local_land_frac * jnp.asarray(bc_state_t['theta_skt'], dtype=jnp.float64) + 
            (1.0 - local_land_frac) * jnp.asarray(bc_state_t['th_v'][:, :, 0], dtype=jnp.float64)
        )
        
        augmented_ml_params = {
            'nn_params': nn_params,
            'land_fraction': local_land_frac,
            'theta_surf': dynamic_theta_surf
        }
        
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn, ml_params=augmented_ml_params)
        
        next_state['land_fraction'] = local_land_frac
        next_state['theta_surf'] = dynamic_theta_surf
        next_state['target_th_v'] = jnp.asarray(bc_state_t['th_v'], dtype=jnp.float64)
        
        return next_state, next_state
        
    return jax.checkpoint(step_fn)


def extract_random_patch(full_states, static_fields, full_grid, h_func_full, nx_full, ny_full, patch_size=32):
    """
    Extracts physical states, land fraction, topography, and exact center coordinates 
    for the subgrid patch to ensure proper numerical metric calculation.
    """
    i_start = random.randint(15, nx_full - patch_size - 15)
    j_start = random.randint(15, ny_full - patch_size - 15)
    
    patch_states = []
    for state in full_states:
        patch_state = {}
        for k, v in state.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                px = patch_size + 1 if k == 'u' else patch_size
                py = patch_size + 1 if k == 'v' else patch_size
                
                if v.ndim == 3: 
                    patch_state[k] = v[i_start:i_start+px, j_start:j_start+py, :]
                elif v.ndim == 2: 
                    patch_state[k] = v[i_start:i_start+px, j_start:j_start+py]
                else:
                    patch_state[k] = v
            else:
                patch_state[k] = v
        patch_states.append(patch_state)
        
    patch_land = static_fields['land_fraction'][i_start:i_start+patch_size, j_start:j_start+patch_size]
    
    # ---------------------------------------------------------
    # METRIC COORDINATES AND TOPOGRAPHY SHIFT
    # ---------------------------------------------------------
    # 1. Find the physical coordinates of the patch center (in meters)
    i_center = i_start + patch_size // 2
    j_center = j_start + patch_size // 2
    
    x_center_m = float(full_grid.x_m[i_center])
    y_center_m = float(full_grid.y_m[j_center])
    
    # 2. Return a new callable that shifts the local patch (X, Y) 
    # back into the full domain's coordinate space before evaluating
    def patch_h_func(X, Y):
        return h_func_full(X + x_center_m, Y + y_center_m)
        
    # 3. Use the grid's projection to calculate the true center lat/lon
    lat_val, lon_val = full_grid.proj.get_lat_lon(x_center_m, y_center_m)
    center_lat = float(lat_val)
    center_lon = float(lon_val)
    # ---------------------------------------------------------
    
    return patch_states, patch_land, patch_h_func, center_lat, center_lon


# ---------------------------------------------------------
# 2. MAIN SCRIPT
# ---------------------------------------------------------
def main():
    print("[INIT] Loading Full Continental NAM-22 Domain...")
    
    nx_full, ny_full, nz = 310, 260, 32
    patch_size = 24
    num_training_patches = 4  
    
    dx, dy, dz = 22000.0, 22000.0, 500.0
    dt = 120.0
    sim_hours = 6
    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0}

    # Load the same NetCDF files used by the inference script
    cache_prefix = "nam22_20250718_Nx310_Ny260_dx22000"
    sl_file = os.path.join(DATA_DIR, f"{cache_prefix}_single_levels.nc")
    pl_file = os.path.join(DATA_DIR, f"{cache_prefix}_pressure_levels.nc")
    
    base_grid = RegionalGrid3D(nx_full, ny_full, nz, dx, dy, dz, 47.5, -97.0)
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"))
    h_func_full = topo_proc.process_and_blend(base_grid, sponge_depth=15, smooth_sigma=2.0)
    
    # Full domain used strictly for state generation and coordinate extraction
    full_grid = RegionalGrid3D(
        nx_full, ny_full, nz, dx, dy, dz, 47.5, -97.0, 
        h_func=h_func_full, transform=SleveSimple(scale_s=10000.0, n=1.0)
    )
    
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(full_grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    static_fields_full = bridge.process_static(raw_t0)
    
    bc_cache_path_native = os.path.join(DATA_DIR, f"{cache_prefix}_native.pkl")
    era5_states_full = bridge.build_or_load_timeseries(
        era5_proc, num_states=sim_hours+1, cache_path=bc_cache_path_native, coarsen_window=None
    )

    # Calculate Normalization Stats over the full domain
    print("[DATA] Calculating normalization stats over the full NAM-22 domain...")
    s0 = era5_states_full[0]
    
    norm_stats = {
        'mean': [
            float(jnp.mean(s0['u'])),
            float(jnp.mean(s0['v'])),
            float(jnp.mean(s0['th_v'])),
            float(jnp.mean(full_grid.Z_m))
        ],
        'std': [
            max(float(jnp.std(s0['u'])), 1e-3),
            max(float(jnp.std(s0['v'])), 1e-3),
            max(float(jnp.std(s0['th_v'])), 1e-3),
            max(float(jnp.std(full_grid.Z_m)), 1e-3)
        ]
    }

    # Extract Random Training Patches & Build Differentiable PDEs
    print(f"[DATA] Slicing {num_training_patches} random {patch_size}x{patch_size} patches for training...")
    scm_models = {}
    
    for p_idx in range(num_training_patches):
        patch_states, patch_land, patch_h_func, center_lat, center_lon = extract_random_patch(
            era5_states_full, static_fields_full, full_grid, h_func_full, nx_full, ny_full, patch_size=patch_size
        )
        
        # Grid initialized with its local coordinate transformation
        patch_grid = RegionalGrid3D(
            patch_size, patch_size, nz, dx, dy, dz, center_lat, center_lon,
            h_func=patch_h_func, transform=SleveSimple(scale_s=10000.0, n=1.0)
        )
        
        patch_ops = CGridOperator3D(patch_grid)
        patch_time_manager = TimeManager(patch_states, [float(i*3600) for i in range(sim_hours+1)], patch_grid)
        
        init_state = dict(patch_states[0])
        init_state.pop('q', None)
        init_state.pop('theta_skt', None)
        
        init_state['land_fraction'] = jnp.asarray(patch_land, dtype=jnp.float64)
        init_state['theta_surf'] = (init_state['land_fraction'] * patch_states[0]['theta_skt'] + 
                                   (1.0 - init_state['land_fraction']) * patch_states[0]['th_v'][:, :, 0])
        init_state['target_th_v'] = patch_states[0]['th_v']
        init_state = tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), init_state)
        
        patch_sponge = DaviesSponge(patch_grid, patch_ops, sponge_depth=3, dt=dt)
        
        # Build standard physics for the SCM
        z_0_field = init_state['land_fraction'] * 0.1 + (1.0 - init_state['land_fraction']) * 1e-4
        epsilon_field = init_state['land_fraction'] * 0.0 + (1.0 - init_state['land_fraction']) * 0.3
        
        physics_ml = PhysicsSuite()
        physics_ml.add_tendency_scheme(McFarlaneSurfaceDrag(
            patch_grid, patch_ops, constants, z_0=z_0_field, epsilon=epsilon_field, theta_surf=init_state['theta_surf']
        ))
        physics_ml.add_tendency_scheme(McFarlaneVerticalDiffusion(
            patch_grid, patch_ops, constants, epsilon=epsilon_field[..., None]
        ))
        physics_ml.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))
        physics_ml.add_tendency_scheme(MLPhysicsClosure(patch_ops, norm_stats))
        
        core_ml = Euler3D(
            patch_grid, patch_ops, constants, dt=dt, initial_era5_state=init_state, 
            physics_suite=physics_ml, interior_mask=patch_sponge.get_interior_mask()
        )
        
        step_fn = build_differentiable_step(SISLStepper3D(core_ml, dt), patch_sponge)
        
        scm_models[f"patch_{p_idx}"] = {
            'grid': patch_grid, 'operators': patch_ops, 'init_state': init_state,
            'time_manager': patch_time_manager, 'sponge': patch_sponge, 'step_fn': step_fn, 
            'land_fraction': init_state['land_fraction']
        }

    # Neural Network & Optimizer Initialization
    ml_closure = MLPhysicsClosure(scm_models["patch_0"]['operators'], norm_stats)
    dummy_x = jnp.ones((nz, 4))
    rng = jax.random.PRNGKey(42)
    ml_params = ml_closure.model.init(rng, dummy_x)['params']
    frozen_spinup_params = jax.tree_util.tree_map(lambda x: x, ml_params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0), 
        optax.adam(learning_rate=1e-3)   
    )
    opt_state = optimizer.init(ml_params)

    # Training Hyperparameters
    epochs = 10
    SPIN_UP_STEPS = 10  
    BPTT_STEPS = 6  

    # Prepare Data Stacks
    init_state_stack = {
        k: jnp.stack([m['init_state'][k] for m in scm_models.values()])
        for k in scm_models['patch_0']['init_state'].keys()
    }
    land_stack = jnp.stack([m['land_fraction'] for m in scm_models.values()])

    bc_forcing_list = []
    for name, m in scm_models.items():
        patch_bc_timeline = []
        for i in range(SPIN_UP_STEPS + BPTT_STEPS):
            patch_bc_timeline.append(m['time_manager'].get_forcing(i * dt))
        bc_forcing_list.append(jax.tree_util.tree_map(lambda *x: jnp.stack(x), *patch_bc_timeline))
    bc_stack = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *bc_forcing_list)

    targ_th_list, targ_u_list, targ_v_list = [], [], []
    for name, m in scm_models.items():
        targ_th_p, targ_u_p, targ_v_p = [], [], []
        for i in range(BPTT_STEPS):
            frc = m['time_manager'].get_forcing((SPIN_UP_STEPS + i + 1) * dt)
            targ_th_p.append(frc['th_v'])
            targ_u_p.append(frc['u'])
            targ_v_p.append(frc['v'])
        targ_th_list.append(jnp.stack(targ_th_p))
        targ_u_list.append(jnp.stack(targ_u_p))
        targ_v_list.append(jnp.stack(targ_v_p))
        
    scan_inputs = (
        init_state_stack, land_stack, 
        jnp.stack(targ_th_list), jnp.stack(targ_u_list), jnp.stack(targ_v_list), 
        bc_stack
    )

    # Build interleaved training strategy
    def make_patch_trainer(p_key, p_inputs):
        patch_sponge_depth = scm_models[p_key]['sponge'].depth
        patch_grid = scm_models[p_key]['grid']
        patch_ops = scm_models[p_key]['operators']
        patch_step_fn = scm_models[p_key]['step_fn']
        
        def patch_loss_fn(params, patch_data):
            p_init, p_land, p_targ_th, p_targ_u, p_targ_v, p_bc_stack = patch_data
            
            def spinup_body(carry, scan_args):
                step_idx, bc_state_t = scan_args
                next_state, _ = patch_step_fn(carry, (step_idx, frozen_spinup_params, bc_state_t, p_land))
                return next_state, None
            
            spinup_indices = jnp.arange(SPIN_UP_STEPS)
            spinup_bcs = jax.tree_util.tree_map(lambda x: x[:SPIN_UP_STEPS], p_bc_stack)
            spun_up_state, _ = jax.lax.scan(spinup_body, p_init, (spinup_indices, spinup_bcs))
            spun_up_state = jax.lax.stop_gradient(spun_up_state)

            def bptt_body(carry, scan_args):
                step_idx, bc_state_t = scan_args
                next_state, _ = patch_step_fn(carry, (step_idx, params, bc_state_t, p_land))
                
                u_m = patch_ops.avg(carry['u'], axis=0, from_loc='u', to_loc='m')
                v_m = patch_ops.avg(carry['v'], axis=1, from_loc='v', to_loc='m')
                
                # Dynamic extraction matches this specific patch's curved geometry
                X = jnp.stack([u_m, v_m, carry['th_v'], patch_grid.Z_m], axis=-1)
                X_norm = (X - jnp.array(norm_stats['mean'])) / (jnp.array(norm_stats['std']) + 1e-8)
                
                nx_, ny_, nz_, _ = X_norm.shape
                preds_flat = ml_closure.model.apply({'params': params}, X_norm.reshape((nx_ * ny_ * nz_, 4)))
                ml_preds = preds_flat.reshape((nx_, ny_, nz_, 3))

                return next_state, {'th_v': next_state['th_v'], 'u': next_state['u'], 'v': next_state['v'], 'ml_preds': ml_preds}

            bptt_indices = jnp.arange(SPIN_UP_STEPS, SPIN_UP_STEPS + BPTT_STEPS)
            bptt_bcs = jax.tree_util.tree_map(lambda x: x[SPIN_UP_STEPS:], p_bc_stack)

            final_state, pred_history = jax.lax.scan(bptt_body, spun_up_state, (bptt_indices, bptt_bcs))
            
            stencil_width = 1 
            D_safe = patch_sponge_depth + (BPTT_STEPS * stencil_width)
            
            std_th = jnp.array(norm_stats['std'][2])
            std_u = jnp.array(norm_stats['std'][0])
            std_v = jnp.array(norm_stats['std'][1])
            
            pred_th_safe = pred_history['th_v'][:, D_safe:-D_safe, D_safe:-D_safe, :]
            targ_th_safe = p_targ_th[:, D_safe:-D_safe, D_safe:-D_safe, :]
            
            pred_u_safe = pred_history['u'][:, D_safe:-D_safe, D_safe:-D_safe, :]
            targ_u_safe = p_targ_u[:, D_safe:-D_safe, D_safe:-D_safe, :]
            
            pred_v_safe = pred_history['v'][:, D_safe:-D_safe, D_safe:-D_safe, :]
            targ_v_safe = p_targ_v[:, D_safe:-D_safe, D_safe:-D_safe, :]

            loss_th = jnp.mean(jnp.abs(pred_th_safe - targ_th_safe) / std_th)
            loss_u  = jnp.mean(jnp.abs(pred_u_safe - targ_u_safe) / std_u)
            loss_v  = jnp.mean(jnp.abs(pred_v_safe - targ_v_safe) / std_v)
            
            l2_penalty = 1e-6 * jnp.mean(jnp.square(pred_history['ml_preds'][:, D_safe:-D_safe, D_safe:-D_safe, :]))
            
            return loss_th + 0.5 * (loss_u + loss_v) + l2_penalty

        patch_value_and_grad = jax.value_and_grad(patch_loss_fn)

        @jax.jit
        def train_step(params, opt_st, batch_inputs):
            def accumulate_body(carry, patch_data):
                accum_loss, accum_grads = carry
                loss, grads = patch_value_and_grad(params, patch_data)
                new_loss = accum_loss + loss
                new_grads = jax.tree_util.tree_map(lambda x, y: x + y, accum_grads, grads)
                return (new_loss, new_grads), None

            zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
            (mean_loss, mean_grads), _ = jax.lax.scan(accumulate_body, (0.0, zero_grads), batch_inputs)
            
            updates, new_opt_st = optimizer.update(mean_grads, opt_st, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_st, mean_loss
            
        return train_step

    # Pre-compile the distinct train_steps for all patches
    trainers = []
    for patch_idx in range(num_training_patches):
        p_key = f"patch_{patch_idx}"
        p_inputs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x[patch_idx], axis=0), scan_inputs)
        trainers.append((p_key, make_patch_trainer(p_key, p_inputs), p_inputs))

    # Global optimization using jax.scan
    print("\n[TRAINING] Starting global optimization...")
    for epoch in range(epochs):
        t0 = time.time()
        random.shuffle(trainers)  # Prevent catastrophic forgetting
        epoch_loss = 0.0
        
        for p_key, t_step, p_inputs in trainers:
            ml_params, opt_state, loss = t_step(ml_params, opt_state, p_inputs)
            epoch_loss += loss.block_until_ready()
            
        mean_epoch_loss = epoch_loss / num_training_patches
        print(f"  -> Epoch {epoch+1:03d} | Mean Loss: {mean_epoch_loss:.6f} | Time: {time.time()-t0:.2f}s")
        
        jax.clear_caches()

    os.makedirs("suetes/weights", exist_ok=True)
    with open("suetes/weights/universal_ml_closure.pkl", "wb") as f:
        pickle.dump({'params': ml_params, 'norm_stats': norm_stats}, f)
    print("\n[SUCCESS] Universal ML parameters saved to suetes/weights/universal_ml_closure.pkl")

if __name__ == "__main__":
    main()