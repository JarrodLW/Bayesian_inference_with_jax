# 1d optimisation examples using optax

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from Regressors import *
from AcquisitionFuncs import *
from Algorithms import *

## trying to find maximum of function

example_num = 3

if example_num == 1:

    def objective(x):
        return - x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0

    initial_params = 0.65
    optimizer = optax.adam(learning_rate=1e-2)

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

elif example_num == 2:

    model = GaussianProcessReg(sigma=2., lengthscale=0.2, obs_noise_stdev=0.1)
    #model = GaussianProcessReg(kernel_type='Periodic', sigma=2., lengthscale=0.05, obs_noise_stdev=0.01, period=2)
    # initialising model
    X0 = jnp.arange(5).reshape((5, 1)) / 5
    y0 = jnp.array([3.7, 3.4, 3.1, 4.2, 3.6])
    model.fit(X0, y0, compute_cov=True)

    initial_params = 0.48
    optimizer = optax.adam(learning_rate=1e-2)
    acq_alg = OptaxAcqAlgBuilder(optimizer)
    acq_func = acq_func_builder('PI', margin=0.01)
    #x_opt = acq_alg(acq_func, model, initial_params)
    x_opt = acq_alg(acq_func, model)

    # def acq_objective(x):
    #     return - acq_func(jnp.array([x]).reshape((1, model.domain_dim)), model)

    def acq_objective(x):
        return - PI_acquisition(jnp.array([x]).reshape((1, 1)), model, 0.)

    print("x opt: " + str(x_opt))
    x_vals = jnp.arange(0, 1, 1 / 50)
    y_vals = jax.vmap(acq_objective)(x_vals)
    plt.plot(np.asarray(x_vals), np.asarray(y_vals))

## running BayesOpt workflow - example adapted from "Single example" file
elif example_num == 3:

    # is ravel the correct thing to be using?
    def objective(x):
        return jnp.ravel(x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0)

    def quadratic(x, a, b, c):
        return jnp.ravel(a * x ** 2 + b * x + c)

    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
                               prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})
    # model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)
    #model = GaussianProcessReg(kernel_type='Periodic', sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, period=2)

    # defining acquisition function and algorithm etc
    optimizer = optax.adam(learning_rate=1e-2)
    acq_alg = OptaxAcqAlgBuilder(optimizer)
    acq_func = acq_func_builder('PI', margin=0.01)

    # initialising model
    num_iters = 3
    # initialising
    X0 = jnp.asarray([0.2, 0.3, 0.8]).reshape((3, model.domain_dim))
    y0 = objective(X0)
    model.fit(X0, y0, compute_cov=True)

    # optimisation
    X, y, surrogate_data = opt_routine(acq_func, model, num_iters, X0, y0, objective, return_surrogates=False,
                                       acq_alg=acq_alg, dynamic_plot=True) #TODO: X0, y0 incorporated into model?

    x_vals = jnp.arange(0, 1, 1 / 50)
    y_vals = jax.vmap(objective)(x_vals)
    plt.figure()
    plt.plot(np.asarray(x_vals), np.asarray(y_vals))
