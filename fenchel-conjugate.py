# import jax.numpy as jnp
# from jax import grad
# from jax.scipy.optimize import minimize


# # log(1 + exp(x))
# def original_f(x):
#     return jnp.log(jnp.exp(x) + 1)


# def get_conjugate(x, y, original_f, num_steps=100, learning_rate=0.01):
#     obj = lambda x: -(x * y - original_f(x))
#     grad_obj = grad(obj)
#     for _ in range(num_steps):
#         x -= learning_rate * grad_obj(x)
#     return -obj(x)


# y_value = 0.48
# x0 = 0.0
# conjugate_value = get_conjugate(y_value, x0, original_f)
# print("Conjugate at y =", y_value, "is", conjugate_value)


import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt


# Define the original function
def original_f(x):
    # return x**3 + jnp.abs(x)
    return jnp.abs(x)


# Define the conjugate function
def fenchel_conjugate(y, x0, original_f, num_steps=50, learning_rate=0.01):
    x = x0
    obj = lambda x: x * y - original_f(x)
    grad_obj = grad(obj)
    for _ in range(num_steps):
        x += learning_rate * grad_obj(x)  # gradient ascent
    return x * y - original_f(x)


# Generating x values for plotting the original function
x_values = jnp.linspace(-2, 2, 400)
original_function_values = original_f(x_values)

# Computing the conjugate for a range of y values
y_values = jnp.linspace(-2, 2, 50)
conjugate_values = jnp.array([fenchel_conjugate(y, 0.0, original_f) for y in y_values])

# Plotting both functions
plt.figure(figsize=(10, 6))
plt.plot(
    x_values,
    original_function_values,
    label="Original Function $f(x) = \log(e^x + 1)$",
    color="blue",
)
plt.plot(y_values, conjugate_values, label="Fenchel Conjugate", color="green")

plt.title("Original Function and its Fenchel Conjugate")
plt.xlabel("x values for original, y values for conjugate")
plt.ylabel("Function values")
plt.legend()
plt.grid(True)
plt.show()
