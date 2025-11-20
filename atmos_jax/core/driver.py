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
        self._compiled_step = None

    def run(self, state, t_start, t_end, dt, chunk_steps=500):
        """
        Runs the simulation from t_start to t_end.
        chunk_steps: How many steps to run inside one JIT call (e.g., 500).
                     Controls frequency of progress prints.
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
        print("[Simulation] Compiling kernel...")
        t0 = time.time()
        
        @jax.jit
        def run_chunk(s, t):
            return self.stepper.integrate(s, t, t + chunk_dt, self.forcing_fn, self.bc_fn)

        # Warmup compilation
        _ = run_chunk(curr_state, t_curr)
        print(f"[Simulation] Compilation finished in {time.time() - t0:.2f}s")

        # 2. Main Loop
        start_time = time.time()
        
        for i in range(n_chunks):
            # Execute Chunk
            curr_state = run_chunk(curr_state, t_curr)
            t_curr += chunk_dt
            
            # Sync for timing/logging (pull scalar to CPU)
            # Assuming 'w' exists, otherwise pick first key
            key = list(curr_state.keys())[0]
            if isinstance(curr_state[key], dict): key = 'u' # Handle nested dicts if necessary
            
            # Perform a cheap blocking read to ensure GPU is done
            _ = curr_state[key].block_until_ready()
            
            # Optional: Calculate max W for status
            # (Pulling value to CPU takes a tiny bit of time, but useful for monitoring)
            if 'w' in curr_state:
                max_w = float(jnp.max(jnp.abs(curr_state['w'])))
                print(f"    Progress: {t_curr:.1f}s / {t_end:.1f}s | Max W: {max_w:.4f} m/s")
            else:
                print(f"    Progress: {t_curr:.1f}s / {t_end:.1f}s")

        # 3. Remainder (if t_end is not multiple of chunk)
        if remainder > 0:
            print(f"    Finishing remaining {remainder} steps...")
            curr_state = self.stepper.integrate(curr_state, t_curr, t_end, self.forcing_fn, self.bc_fn)
            t_curr = t_end

        total_time = time.time() - start_time
        steps_per_sec = total_steps / total_time
        print(f"[Simulation] Done in {total_time:.2f}s ({steps_per_sec:.1f} steps/s)\n")
        
        return curr_state