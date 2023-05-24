from Regressors import *
from AcquisitionFuncs import *
from Algorithms import *
from numpy.random import normal


# defining objective
def objective(x, noise=0.05):
    noise = normal(loc=0, scale=noise)
    return np.ndarray.flatten((x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise)


# defining the prior
def quadratic(x, a, b, c):
    return np.ndarray.flatten(a * x ** 2 + b * x + c)


# defining model
# model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
#                                    prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})

model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)

# defining acquisition function
pi = acq_func_builder('PI', margin=0.01)
ei = acq_func_builder('EI', margin=0.01)
ucb = acq_func_builder('UCB', std_weight=1.)
acq_func = 0.1*pi + 0.5*ei + 0.02*ucb

# setting up optimisation
# setting hyper-parameters
num_iters = 5
domain_dim = 1
# initialising
X0 = np.asarray(0.2).reshape((1, domain_dim))
y0 = objective(X0)
model.fit(X0, y0, compute_cov=True)

# optimisation
X, y, surrogate_data = opt_routine(acq_func, model, num_iters, X0, y0, objective, return_surrogates=True, dynamic_plot=True)
