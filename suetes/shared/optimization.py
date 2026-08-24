import time
import jax
import optax

class OptaxSolver:
    def __init__(self, objective_fn, optimizer, has_aux=True):
        self.objective_fn = objective_fn
        self.optimizer = optimizer
        self.has_aux = has_aux
        # Pre-compile the gradient function
        self.grad_fn = jax.jit(jax.value_and_grad(objective_fn, has_aux=has_aux))
        
    def fit(self, init_params, total_steps, bounds_fn=None, patience=None, metric_name="Loss", maximize=False):
        """Optimize parameters and retain loss, parameter, and auxiliary histories.

        Args:
            init_params (Any): Initial parameter pytree.
            total_steps (int): Maximum number of optimization updates.
            bounds_fn (callable, optional): Function that constrains parameters
                after each update.
            patience (int, optional): Stop after this many updates without an
                improvement.
            metric_name (str): Label used in progress output.
            maximize (bool): Whether larger objective values are improvements.

        Returns:
            tuple: Final parameter pytree and an optimization-history dictionary
            containing losses, parameter pytrees, and auxiliary values.
        """
        opt_state = self.optimizer.init(init_params)
        params = init_params
        
        history = {'loss': [], 'params': [], 'aux': []}
        
        best_loss = float('inf')
        patience_counter = 0
        
        print("[OPTIMIZATION] Starting Optimization Loop...")
        for i in range(total_steps):
            start = time.time()
            
            if self.has_aux:
                (loss, aux), grads = self.grad_fn(params)
            else:
                loss, grads = self.grad_fn(params)
                aux = None
            
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Enforce any physical bounds
            if bounds_fn is not None:
                params = bounds_fn(params)
                
            history['loss'].append(float(loss))
            history['params'].append(params)
            if self.has_aux:
                history['aux'].append(aux)
            
            # Visual formatting for the print output
            display_val = -float(loss) if maximize else float(loss)
            print(f"Step {i+1:02d} | {metric_name}: {display_val:.4f} | Time: {time.time()-start:.1f}s")
            
            # --- EARLY STOPPING LOGIC ---
            if patience is not None:
                # We always want the loss to go DOWN (more negative = more energy)
                if loss < best_loss:
                    best_loss = float(loss)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n[EARLY STOPPING] Objective did not improve for {patience} steps. Halting to save compute.")
                    break
            
        return params, history
