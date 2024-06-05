import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize


# log(1 + exp(x))
def original_f(x):
    return jnp.log(jnp.exp(x) + 1)


def get_conjugate(x, y, original_f, num_steps=100, learning_rate=0.01):
    obj = lambda x: -(x * y - original_f(x))
    grad_obj = grad(obj)
    for _ in range(num_steps):
        x -= learning_rate * grad_obj(x)
    return -obj(x)


y_value = 0.48
x0 = 0.0
conjugate_value = get_conjugate(y_value, x0, original_f)
print("Conjugate at y =", y_value, "is", conjugate_value)
