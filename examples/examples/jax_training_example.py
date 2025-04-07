# jax_training_example.py

import jax
import jax.numpy as jnp

# Simple data: y = 2x + 3
x_data = jnp.array([1, 2, 3, 4])
y_data = 2 * x_data + 3

# Initialize parameters: weight and bias
params = {
    "w": jnp.array(0.0),
    "b": jnp.array(0.0),
}

# Define the model
def model(x, w, b):
    return w * x + b

# Define loss (Mean Squared Error)
def loss_fn(params, x, y):
    preds = model(x, params["w"], params["b"])
    return jnp.mean((preds - y) ** 2)

# Optimizer step using gradient descent
def update(params, x, y, lr=0.01):
    grads = jax.grad(loss_fn)(params, x, y)
    return {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }

# Training loop
for epoch in range(100):
    params = update(params, x_data, y_data)
    if epoch % 10 == 0:
        current_loss = loss_fn(params, x_data, y_data)
        print(f"Epoch {epoch}, Loss: {current_loss:.4f}")

print("\nTrained Parameters:")
print(params)
