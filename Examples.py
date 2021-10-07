# Created 06/10/2021
# Runs regression and optimisation examples for choice of prior mean function, ...

from Regressors import *
from Utils import plot
from myFunctions import *
from myAlgorithms import opt_acquisition
import matplotlib.pyplot as plt
import time
from numpy.random import normal


gaussian_reg_example = False
optimisation = True
num_initial_samples = 3
acq_type = 'PI'  # only needed for optimisation example

prior_mean_func = 'quadratic'

if prior_mean_func is None:
    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)

elif prior_mean_func == 'linear':

    def linear(x, a, b):
        return a * x + b

    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=linear,
                               prior_mean_kwargs={'a': 0.5, 'b': 0})

elif prior_mean_func == 'quadratic':

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
                               prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})


def objective(x, noise=0.01):
    noise = normal(loc=0, scale=noise)
    return (x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise


# initialising model
X0 = np.random.random(num_initial_samples)
y0 = np.asarray([objective(x) for x in X0])
ix = np.argmax(y0)

X = X0
y = y0

model.fit(X, y, compute_cov=True)

if gaussian_reg_example: # TODO: move plotting functionality etc into Utils

    iters = 30
    num_test_points = 1000
    predictions_mu = np.zeros((iters, num_test_points))
    predictions_std_squared = np.zeros((iters, num_test_points))
    test_points = np.asarray(np.arange(0, 1, 1 / num_test_points))
    x_vals = list(X)
    y_vals = list(y)
    y_vals_no_noise = list(objective(test_points, noise=0.))

    for i in range(iters):
        x_val = np.random.random(1)
        # x_vals += [x_val]
        # X = np.asarray(x_vals)
        y_val = objective(x_val)
        model.fit(np.asarray([x_val]), np.asarray([y_val]))

        x_vals += [x_val]
        y_vals += [y_val]

        mu, covs = model.predict(test_points)
        stds = np.sqrt(np.diagonal(covs))

        plt.clf()
        plt.ylim(-0.2, 1.0)
        plt.scatter(x_vals, y_vals)
        plt.plot(test_points, mu)
        plt.plot(test_points, y_vals_no_noise)
        plt.fill_between(test_points, mu - stds, mu + stds, alpha=0.4)
        plt.pause(1e-17)
        time.sleep(0.5)

elif optimisation:

    num_iters = 5
    num_samples = 2000

    if acq_type == 'EI' or 'PI':
        margin = 0.01

    for i in range(num_iters):
        # select the next point to sample
        x = opt_acquisition(acq_type, model, margin, num_samples)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(np.asarray([x]))
        print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # update the model
        model.fit(np.asarray([x]), np.asarray([actual]))

        plt.clf()
        plt.ylim(-0.1, 1.)
        plot(model.X, model.y, model, objective)
        plt.pause(1e-17)
        time.sleep(2.)
        # plt.show()

        # print("Iter " + str(i) + " successful")

    print('First best guess: x=%.3f, y=%.3f' % (X[ix], y[ix]))

    ix = np.argmax(model.y)
    print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))
