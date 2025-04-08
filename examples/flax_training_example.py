# flax_training_example.py

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

# Sample data
x_data = jnp.array([1.0, 2.0, 3.0, 4.0])
y_data = 2 * x_data + 3

# Define a simple model
class LinearModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)

# Create TrainState to hold parameters and optimizer
class TrainState(train_state.TrainState):
    pass

# Loss function
def loss_fn(params, model, x, y):
    pred = model.apply(params, x)
    return jnp.mean((pred - y) ** 2)

# Training step
@jax.jit
def train_step(state, model, x, y):
    grads = jax.grad(loss_fn)(state.params, model, x, y)
    return state.apply_gradients(grads=grads)

# Initialize model
model = LinearModel()
rng = jax.random.PRNGKey(0)
params = model.init(rng, x_data.reshape(-1, 1))

# Create optimizer and train state
tx = optax.sgd(learning_rate=0.1)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
for step in range(100):
    state = train_step(state, model, x_data.reshape(-1, 1), y_data.reshape(-1, 1))
    if step % 10 == 0:
        loss = loss_fn(state.params, model, x_data.reshape(-1, 1), y_data.reshape(-1, 1))
        print(f"Step {step}: Loss = {loss:.4f}")
