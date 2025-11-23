import jax
import jax.numpy as jnp
from flax import linen as nn

class StandardMLP(nn.Module):
    """
    A standard Multi-Layer Perceptron.
    Used with IntegralNeuralTransform, which enforces monotonicity via integration.
    """
    width: int
    depth: int
    
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        # Final layer projects to scalar (density)
        x = nn.Dense(1)(x)
        return x

class MonotonicDense(nn.Module):
    """Dense layer with positive weights to enforce monotonicity."""
    features: int
    activation: callable = nn.tanh
    
    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', nn.initializers.glorot_normal(), 
                            (x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        # Softplus ensures positive weights
        y = jnp.dot(x, jax.nn.softplus(kernel)) + bias
        return self.activation(y)

class MonotonicMLP(nn.Module):
    """
    Network that guarantees monotonic output w.r.t input.
    Used for Direct Neural Transforms.
    """
    width: int
    depth: int

    @nn.compact
    def __call__(self, y):
        x = jnp.atleast_1d(y)
        for _ in range(self.depth - 1):
            x = MonotonicDense(self.width, activation=nn.tanh)(x)
        
        kernel_out = self.param('kernel_out', nn.initializers.glorot_normal(), 
                                (x.shape[-1], 1))
        bias_out = self.param('bias_out', nn.initializers.zeros, (1,))
        
        x = jnp.dot(x, jax.nn.softplus(kernel_out)) + bias_out
        return jnp.squeeze(x)