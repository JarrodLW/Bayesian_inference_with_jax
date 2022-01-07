import jax
import jax.numpy as jnp
from Experiment import Experiment
from AcquisitionFuncs import *
from Algorithms import opt_routine, log_marg_likelihood
import matplotlib.pyplot as plt
from matplotlib import cm
import optax
import numpy as np


def quadratic(x, a=0.5, b=0, c=0):
    return jnp.ravel(a * x ** 2 + b * x + c)

def objective(x):
    return jnp.ravel(x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0)


X0 = jnp.asarray(list(np.linspace(0., 1., num=10))).reshape((10, 1))
y0 = objective(X0)

# 1. no user input, default RBF kernel with parameters inferred by mle
exp = Experiment(X0, y0)

# 2. no user input, but mel flag turned off: default RBF kernel but parameters initialised as None
exp = Experiment(X0, y0, mle=False)
exp.model.fit(X0, y0)  #

# 3. user input from outset: default RBF kernel, all parameters provided
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.01})

# 4. user input from outset: periodic kernel, all parameters provided
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.01, 'period': 0.1})

# 5. user input from outset: default RBF kernel, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5})

# 6. user input from outset: specified kernel, non-default prior, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5})

# 7. same as above but with Matern kernel
exp = Experiment(X0, y0, kernel_type='Matern', kernel_hyperparams={'sigma': 0.5})

# 8. user input from outset: non-default prior mean, default prior, not all parameters provided (mle called)
exp = Experiment(X0, y0, kernel_hyperparams={'sigma': 0.5}, prior_mean=quadratic)

# 9. user resetting all hyper-parameters
exp = Experiment(X0, y0)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = 0.05
exp.model.fit(X0, y0)  # need to re-fit the model, this is not automatic

# 10. user resetting only some of the hyper-parameters
exp = Experiment(X0, y0)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = None
exp.maximum_likelihood_estimation()  # need to estimate remaining hyper-param, this is not automatic

# 11. user initialising experiment but turning off initial mle
exp = Experiment(X0, y0, mle=False)
exp.model.kernel.sigma = 0.3
exp.model.kernel.lengthscale = None
exp.maximum_likelihood_estimation()  # need to estimate remaining hyper-param, this is not automatic

# 12. user resetting prior mean function - broken :/
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.5, 'lengthscale': 0.05, 'period': 0.2})
exp.model.prior_mean = quadratic
exp.maximum_likelihood_estimation()  # just for illustration; this does nothing since all parameters are fixed
exp.model.fit(X0, y0)  # need to re-fit model, this is not automatic

# 13. no user input, default RBF kernel with parameters inferred by mle, with subsequent Bayesian optimisation outside of
# experiment class
exp = Experiment(X0, y0)
num_iters = 5
acq_func = acq_func_builder('PI', margin=0.01)
X, y, surrogate_data = opt_routine(acq_func, exp.model, num_iters, objective, return_surrogates=False, dynamic_plot=True)

# 14. no user input, default RBF kernel with parameters inferred by mle, with subsequent Bayesian optimisation inside of
# experiment class
exp = Experiment(X0, y0)

# 15. user setting up model via the experiment class, using mle, and performing subsequent Bayesian optimisation,
# default 'PI' acquisition function optimised by random search (also default)

# 16. user setting up model via the experiment class, using mle, and performing subsequent Bayesian optimisation,
# acquisition function specified, optimised by specified optax optimiser

# 17. user setting up model via the experiment class, using mle, and performing subsequent Bayesian optimisation,
# acquisition function specified, optimised by specified optax optimiser, followed by new mle estimation and re-fitting
