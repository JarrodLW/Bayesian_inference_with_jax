# 1d optimisation examples using optax

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from Regressors import GaussianProcessReg
from myFunctions import PI_acquisition

## trying to find maximum of function

example_num = 2

if example_num == 1:

    def objective(x):
        return - x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0

    initial_params = 0.65

elif example_num == 2:

    model = GaussianProcessReg(sigma=2., lengthscale=0.2, obs_noise_stdev=0.1)
    #model = GaussianProcessReg(kernel_type='Periodic', sigma=2., lengthscale=0.05, obs_noise_stdev=0.01, period=2)
    # initialising model
    X0 = jnp.arange(5).reshape((5, 1)) / 5
    y0 = jnp.array([3.7, 3.4, 3.1, 4.2, 3.6])
    model.fit(X0, y0, compute_cov=True)

    def objective(x):
        return -PI_acquisition(jnp.array([x]).reshape((1, 1)), model, 0.)

    #initial_params = 0.48
    initial_params = 0.19

# doing optimisation

def fit(x: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(x)

    @jax.jit
    def step(x, opt_state):
        loss_value, grads = jax.value_and_grad(objective)(x)
        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, opt_state, loss_value

    for i in range(1000):
        x, opt_state, loss_value = step(x, opt_state)
        if i % 100 == 0:
            print(f'step {i}, loss: {loss_value}')

    return x


optimizer = optax.adam(learning_rate=1e-2)
x_opt = fit(initial_params, optimizer)

print("x opt: " + str(x_opt))

x_vals = jnp.arange(0, 1, 1 / 50)
y_vals = jax.vmap(objective)(x_vals)
plt.plot(np.asarray(x_vals), np.asarray(y_vals))



