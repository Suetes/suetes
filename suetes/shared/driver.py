import time
import jax
import jax.numpy as jnp

class Simulation:
    """
    Manages the time integration loop, JIT compilation, and chunking.
    """
    def __init__(self, step_fn, dt):
        """
        Args:
            step_fn (callable): A function `fn(state, step_idx)` that returns 
                                `(next_state, metrics_array)`.
            dt (float): Timestep in seconds.
        """
        self.step_fn = step_fn
        self.dt = dt

    def run(self, initial_state, t_start, t_end, chunk_steps=120):
        total_steps = int((t_end - t_start) / self.dt)
        n_chunks = total_steps // chunk_steps
        remainder = total_steps % chunk_steps
        
        print(f"[SIMULATION] Starting: T={t_start} -> T={t_end}")
        print(f"[SIMULATION] Total Steps: {total_steps}")
        print(f"[SIMULATION] Chunk Size:  {chunk_steps} steps")
        
        @jax.jit(static_argnames=['n_steps'])
        def run_chunk(curr_state, start_step, n_steps):
            def scan_fn(state, step_offset):
                step_idx = start_step + step_offset
                next_state, metrics = self.step_fn(state, step_idx)
                return next_state, metrics
            
            return jax.lax.scan(scan_fn, curr_state, jnp.arange(n_steps))

        print("[SIMULATION] Compiling kernel...")
        t0 = time.time()
        _ = run_chunk(initial_state, 0, 1)
        print(f"[SIMULATION] Compilation finished in {time.time() - t0:.2f}s")
        print(f"[SIMULATION] JIT compiled, running simulation...")

        start_time = time.time()
        state = initial_state
        current_step = int(t_start / self.dt)
        
        for i in range(n_chunks):
            chunk_start = time.time()
            state, metrics = run_chunk(state, current_step, chunk_steps)
            
            leaves = jax.tree_util.tree_leaves(state)
            if leaves:
                leaves[0].block_until_ready()
            
            current_step += chunk_steps
            t_curr = current_step * self.dt
            
            max_w = float(jnp.max(jnp.abs(metrics))) if metrics is not None else 0.0
            print(f"    Progress: {t_curr:.1f}s / {t_end:.1f}s | "
                  f"Max W: {max_w:.4f} m/s | "
                  f"Chunk Wall Time: {time.time() - chunk_start:.2f}s")

        if remainder > 0:
            print(f"    Finishing remaining {remainder} steps...")
            state, _ = run_chunk(state, current_step, remainder)
            leaves = jax.tree_util.tree_leaves(state)
            if leaves:
                leaves[0].block_until_ready()

        total_time = time.time() - start_time
        print(f"[Simulation] Done in {total_time:.2f}s\n")
        
        return state