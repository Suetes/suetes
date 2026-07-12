#!/usr/bin/env python3
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax

from suetes.shared.transforms import StretchedSleveSimple
from experiments.evaluate_neuve_coordinate import (
    make_core_ops,
    compute_universal_ground_truth_timeseries,
    true_params,
    guess_params,
    num_steps,
    Z_obs,
    sensor_idx_x
)

def evaluate_sleve(kappa, scale_s, universal_target_ts):
    transform = StretchedSleveSimple(stretch_kappa=kappa, scale_s=scale_s)
    grid, base_state, stepper, bc_fn = make_core_ops(transform)
    
    X_km = grid.x_m[:, None, None] / 1000.0
    Y_km = grid.y_m[None, :, None] / 1000.0
    Z_km = grid.Z_m / 1000.0
    
    try:
        def create_tracer(p):
            return p[3] * jnp.exp(-((X_km - p[0])**2 / 2.0 + (Y_km - p[1])**2 / 2.0 + (Z_km - p[2])**2 / 1.0))
        
        def scan_step(st, _):
            next_st = stepper.step(st, 0.0, None, bc_fn)
            col_q = next_st['q_tr'][sensor_idx_x]
            col_z = grid.Z_m[sensor_idx_x]
            def interp_y(z_col, q_col):
                return jnp.interp(Z_obs, z_col, q_col)
            slice_phys = jax.vmap(interp_y)(col_z, col_q)
            return next_st, slice_phys
        
        @jax.jit
        def loss_and_grad(p):
            def obj(params):
                s = dict(base_state)
                s['q_tr'] = create_tracer(params)
                _, sim_ts = jax.lax.scan(scan_step, s, jnp.arange(num_steps))
                return jnp.mean((sim_ts - universal_target_ts)**2)
            return jax.value_and_grad(obj)(p)
        
        inner_opt = optax.adam(0.25)
        opt_st = inner_opt.init(guess_params)
        p_curr = guess_params
        
        for k in range(50):
            lval, gval = loss_and_grad(p_curr)
            if jnp.isnan(lval):
                return float('nan'), float('nan')
            updates, opt_st = inner_opt.update(gval, opt_st, p_curr)
            p_curr = optax.apply_updates(p_curr, updates)
            
        pos_err = np.linalg.norm(np.array(p_curr[:3]) - np.array(true_params[:3]))
        return float(lval), float(pos_err)
    except Exception as e:
        return float('nan'), float('nan')

def main():
    kappas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    scales = [3500.0, 4000.0, 4500.0, 5000.0]
    
    print("Generating universal target time-series...")
    universal_target_ts = compute_universal_ground_truth_timeseries()
    
    print(f"\n{'kappa':<8} | {'scale_s':<8} | {'Final Loss':<12} | {'Pos Err (km)':<12}")
    print("-" * 45)
    for k in kappas:
        for s in scales:
            loss, err = evaluate_sleve(k, s, universal_target_ts)
            print(f"{k:<8.3f} | {s:<8.1f} | {loss:<12.4e} | {err:<12.4f}")

if __name__ == '__main__':
    main()
