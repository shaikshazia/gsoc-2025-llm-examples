# flax_model_example.py

from flax import linen as nn
import jax
import jax.numpy as jnp

class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

model = SimpleMLP()
x = jnp.ones((1, 4))  # Dummy input
params = model.init(jax.random.PRNGKey(0), x)
output = model.apply(params, x)
print("Output:", output)
