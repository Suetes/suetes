import jax
import jax.numpy as jnp
import flax.linen as nn

class MonotonicDense(nn.Module):
    features: int
    activation: callable = nn.tanh
    
    @nn.compact
    def __call__(self, x):
        # Force positive weights to ensure monotonicity
        kernel = self.param('kernel', nn.initializers.glorot_normal(), 
                            (x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        
        # softplus ensures kernel is always positive
        y = jnp.dot(x, jax.nn.softplus(kernel)) + bias
        return self.activation(y)

class MonotonicMLP(nn.Module):
    """
    A neural network that learns a monotonic mapping.
    Used to learn the vertical decay function b(Z/H).
    """
    width: int
    depth: int

    @nn.compact
    def __call__(self, y):
        # Input y is normalized height (zeta / Lz)
        x = jnp.atleast_1d(y)
        
        for _ in range(self.depth - 1):
            x = MonotonicDense(self.width, activation=nn.tanh)(x)
            
        # Final layer linear
        kernel_out = self.param('kernel_out', nn.initializers.glorot_normal(), 
                                (x.shape[-1], 1))
        bias_out = self.param('bias_out', nn.initializers.zeros, (1,))
        
        x = jnp.dot(x, jax.nn.softplus(kernel_out)) + bias_out
        return jnp.squeeze(x)