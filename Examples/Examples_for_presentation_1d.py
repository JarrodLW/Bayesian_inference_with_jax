# Created 28/10/2021
# Runs regression and optimisation examples for choice of prior mean function, acquisition function and kernel

from Regressors import *
from AcquisitionFuncs import *
from Algorithms import *
import matplotlib.pyplot as plt
import time
from numpy.random import normal


# defining objective
def objective(x, noise=0.05):
    noise = normal(loc=0, scale=noise)
    return np.ndarray.flatten((x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise)


acq_types = ['PI', 'EI', 'UCB', 'PI-EI']
kernels = ['RBF', 'Periodic']
prior_mean_funcs = ['None', 'linear', 'quadratic']

# acq_types = ['PI']
# kernels = ['RBF']
# prior_mean_funcs = ['quadratic']

for acq_type in acq_types:
    for kernel_type in kernels:
        for prior_mean_func in prior_mean_funcs:

            if prior_mean_func == 'None':

                if kernel_type == 'RBF':
                    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)

                elif kernel_type == 'Periodic':
                    model = GaussianProcessReg(kernel_type='Periodic', sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, period=2)

            elif prior_mean_func == 'linear':

                def linear(x, a, b):
                    return np.ndarray.flatten(a * x + b)

                if kernel_type=='RBF':
                    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=linear,
                                               prior_mean_kwargs={'a': 0.5, 'b': 0})

                elif kernel_type=='Periodic':
                    model = GaussianProcessReg(kernel_type='Periodic', sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, period=2,
                                               prior_mean=linear, prior_mean_kwargs={'a': 0.5, 'b': 0})

            elif prior_mean_func == 'quadratic':

                def quadratic(x, a, b, c):
                    return np.ndarray.flatten(a * x ** 2 + b * x + c)

                if kernel_type == 'RBF':
                    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
                                               prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})

                elif kernel_type == 'Periodic':
                    model = GaussianProcessReg(kernel_type='Periodic', sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, period=2,
                                               prior_mean=quadratic, prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})

            if acq_type == 'PI':
                acq_func = acq_func_builder('PI', margin=0.01)

            elif acq_type == 'EI':
                acq_func = acq_func_builder('EI', margin=0.01)

            elif acq_type == 'UCB':
                acq_func = acq_func_builder('UCB', std_weight=1)

            elif acq_type == 'PI-EI':
                pi = acq_func_builder('PI', margin=0.01)
                ei = acq_func_builder('EI', margin=0.01)
                acq_func = 0.1 * pi + 0.5 * ei

            num_iters = 6
            domain_dim = 1
            # initialising
            X0 = np.asarray(0.47).reshape((1, domain_dim))
            y0 = objective(X0)
            model.fit(X0, y0, compute_cov=True)

            X, y, surrogate_data = opt_routine(acq_func, model, num_iters, X0, y0, objective, return_surrogates=True)

            surrogate_means = surrogate_data['means']
            surrogate_stds = surrogate_data['stds']
            test_points = np.asarray(np.arange(0, 1, 1 / 1000)).reshape((1000, domain_dim))
            y_vals_no_noise = objective(test_points, noise=0.)

            f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
            for i in range(num_iters):
                #plt.clf()
                axarr[i//3, i%3].scatter(np.ndarray.flatten(X)[:i+1], y[:i+1])
                axarr[i//3, i%3].plot(np.ndarray.flatten(test_points), surrogate_means[i])
                axarr[i//3, i%3].plot(np.ndarray.flatten(test_points), y_vals_no_noise)
                axarr[i//3, i%3].fill_between(np.ndarray.flatten(test_points), surrogate_means[i] - surrogate_stds[i],
                                 surrogate_means[i] + surrogate_stds[i], alpha=0.4)
                axarr[i//3, i%3].set_title("t=" + str(i))

            plt.tight_layout()
            plt.savefig("/Users/jlw31/Desktop/Some results/" + acq_type + "/" + kernel_type + "/" +
                        prior_mean_func + "/Results.png")


