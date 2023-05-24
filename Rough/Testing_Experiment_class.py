import jax
import jax.numpy as jnp
from Experiment import Experiment
from AcquisitionFuncs import *
from Algorithms import opt_routine, log_marg_likelihood
import matplotlib.pyplot as plt
from matplotlib import cm
import optax
import numpy as np

# def quadratic(x, a, b, c):
#     return jnp.ravel(a * x ** 2 + b * x + c)

clueless_user_mode = True
slightly_less_clueless_user_mode = True

def objective(x):
    return jnp.ravel(x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0)

X0 = jnp.asarray(list(np.linspace(0., 1., num=10))).reshape((10, 1))
y0 = objective(X0)

exp = Experiment(X0, y0)

if clueless_user_mode == True:
    # initialising experiment. By default it will use an RBF kernel with hyperparameters
    # fixed by maximum likelihood estimation.
    # Query: is there simply a closed formula for this estimate?

    exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.01})
    print(exp._model.kernel_hyperparam_kwargs)

    # checking that optimal params have been correctly identified
    num = 100
    sigmas = [10**x for x in np.linspace(-5, 0, num=num)]
    lengthscales = [10**x for x in np.linspace(-5, 0, num=num)]
    sig, len = np.meshgrid(sigmas, lengthscales)

    Z = np.zeros((num, num))

    def log_marg_likelihood_wrapper(Xsamples, ysamples, sigma, lengthscale):
        kernel_hyperparam_kwargs = {'sigma': sigma, 'lengthscale': lengthscale}
        log_prob = log_marg_likelihood(Xsamples, ysamples, kernel_type='RBF', kernel_hyperparam_kwargs=kernel_hyperparam_kwargs,
                                obs_noise_stdev=1e-3)
        return log_prob

    for i, sigma in enumerate(sigmas):
        for j, lengthscale in enumerate(lengthscales):
            Z[i, j] = jnp.exp(log_marg_likelihood_wrapper(X0, y0, sigma, lengthscale))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(sig, len, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()

    print(exp._model.kernel_hyperparam_kwargs)

    np.amax(Z)
    np.argmax(Z)
    print('sigma: ' + str(sigmas[np.argmax(Z)//num]))
    print('lengthscale: ' + str(lengthscales[np.argmax(Z)%num]))

    # Bayes Opt

    model = exp._model
    num_iters = 6
    acq_func = acq_func_builder('PI', margin=0.01)

    X, y, surrogate_data = opt_routine(acq_func, model, num_iters, objective, return_surrogates=False,
                                           dynamic_plot=True)

if slightly_less_clueless_user_mode == True:
    # we'll again use an RBF kernel but we'll specify the sigma, meaning that only the lengthscale has to be determined
    # by ML estimation.

    # might as well turn off ML_est flag, because it estimates will need to be recomputed at next step anyway
    # it's not optimising the period correctly
    # exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparam_kwargs={'sigma': 0.5, 'lengthscale': 0.01})

    exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5})


X, y, surrogate_data = opt_routine(acq_func, model, num_iters, objective, return_surrogates=False,
                                           dynamic_plot=True)

## Some examples

# note: if mle is called

# TODO: use sympy to convert user's string input into a prior mean function of the right form
def quadratic(x, a=0.5, b=0, c=0):
    return jnp.ravel(a * x ** 2 + b * x + c)

# no user input, default RBF kernel with parameters inferred by mle
exp = Experiment(X0, y0)

# no user input, but mel flag turned off: default RBF kernel but parameters initialised as None
exp = Experiment(X0, y0, mle=False)

# user input from outset: default RBF kernel, all parameters provided
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.01})

# user input from outset: default RBF kernel, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5})

# user input from outset: specified kernel, non-default prior, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5})

# user input from outset: non-default prior mean, default prior, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5}, prior_mean=quadratic)

# user resetting all hyper-parameters
exp = Experiment(X0, y0)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = 0.05
exp.model.fit(X0, y0)  # need to re-fit model, this is not automatic

# user resetting only some of the hyper-parameters
exp = Experiment(X0, y0)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = None
exp.maximum_likelihood_estimation()  # need to estimate remaining hyper-param, this is not automatic

# user initialising experiment but turning off initial mle
exp = Experiment(X0, y0, mle=False)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = None
exp.maximum_likelihood_estimation()  # need to estimate remaining hyper-param, this is not automatic

# user resetting prior mean function
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.05, 'period': 0.1})
exp.model.prior_mean = quadratic
exp.maximum_likelihood_estimation()  # this does nothing, since all parameters fixed
exp.model.fit(X0, y0)  # need to re-fit model, this is not automatic
