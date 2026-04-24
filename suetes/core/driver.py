import time
import jax
import jax.numpy as jnp

class Simulation:
    """
    Manages the time integration loop, JIT compilation, and chunking.
    """
    def __init__(self, stepper, forcing_fn, bc_fn):
        self.stepper = stepper
        self.forcing_fn = forcing_fn
        self.bc_fn = bc_fn

    def run(self, state, t_start, t_end, dt, chunk_steps=500):
        """
        Runs the simulation from t_start to t_end.
        """
        t_curr = t_start
        curr_state = state
        total_steps = int((t_end - t_start) / dt)
        n_chunks = int(total_steps // chunk_steps)
        remainder = total_steps % chunk_steps
        
        chunk_dt = chunk_steps * dt

        print(f"\n[Simulation] Starting: T={t_start} -> T={t_end}")
        print(f"             Total Steps: {total_steps}")
        print(f"             Chunk Size:  {chunk_steps} steps")
        
        # 1. JIT Compile the chunk function
        # 1. JIT Compile the chunk function
        print("[Simulation] Compiling kernel...")
        t0 = time.time()
        
        @jax.jit
        def run_chunk(s, t):
            # chunk_steps is a static Python int captured from the outer scope
            return self.stepper.integrate(s, t, chunk_steps, self.forcing_fn, self.bc_fn)

        # Warmup compilation
        _ = run_chunk(curr_state, t_curr)
        print(f"[Simulation] Compilation finished in {time.time() - t0:.2f}s")

        # 2. Main Loop
        start_time = time.time()
        
        for i in range(n_chunks):
            # Execute Chunk
            curr_state = run_chunk(curr_state, t_curr)
            t_curr += chunk_dt
            
            # --- GENERIC SYNC ---
            leaves = jax.tree_util.tree_leaves(curr_state)
            if leaves:
                leaves[0].block_until_ready()
            
            # --- GENERIC LOGGING ---
            msg = f"    Progress: {t_curr:.1f}s / {t_end:.1f}s"
            if isinstance(curr_state, dict) and 'w' in curr_state:
                max_w = float(jnp.max(jnp.abs(curr_state['w'])))
                msg += f" | Max W: {max_w:.4f} m/s"
            
            print(msg)

        # 3. Remainder
        if remainder > 0:
            print(f"    Finishing remaining {remainder} steps...")
            
            # JIT compile the remainder step since its length is different from chunk_steps
            @jax.jit
            def run_remainder(s, t):
                return self.stepper.integrate(s, t, remainder, self.forcing_fn, self.bc_fn)
                
            curr_state = run_remainder(curr_state, t_curr)
            t_curr = t_end

        total_time = time.time() - start_time
        steps_per_sec = total_steps / (total_time + 1e-9)
        print(f"[Simulation] Done in {total_time:.2f}s ({steps_per_sec:.1f} steps/s)\n")
        
        return curr_state