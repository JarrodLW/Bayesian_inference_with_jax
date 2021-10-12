# Created 06/10/2021
# Runs regression and optimisation examples for choice of prior mean function, acquisition function
import numpy as np

from Regressors import *
from Utils import plot
from myFunctions import *
from myAlgorithms import opt_acquisition
import matplotlib.pyplot as plt
import time
from numpy.random import normal


gaussian_reg_example = False
optimisation = True
acq_type = 'PI'
num_initial_samples = 20

prior_mean_func = None

if prior_mean_func is None:
    model = GaussianProcessReg(kernel_type='Periodic', domain_dim=2, sigma=0.05, lengthscale=0.1,
                               obs_noise_stdev=0.01, period=4)


def objective(x, noise=0.05):
    noise = normal(loc=0, scale=noise)
    return np.ndarray.flatten((1 - 5*(x[:, 0] - 0.5) ** 2 - 5*(x[:, 1] - 0.5)**2) * np.cos(3 * np.pi * (x[:, 0]-0.5)) ** 6.0
                              * np.cos(3 * np.pi * (x[:, 1]-0.5)) ** 4.0 + noise)


# initialising model
domain_dim = 2
X0 = np.random.random((num_initial_samples, domain_dim))
y0 = objective(X0)
ix = np.argmax(y0)

model.fit(X0, y0, compute_cov=True)

if gaussian_reg_example: # TODO: move plotting functionality etc into Utils

    iters = 500
    sqrt_num_test_points = 100
    num_test_points = sqrt_num_test_points**2
    predictions_mu = np.zeros((iters, num_test_points))
    predictions_std_squared = np.zeros((iters, num_test_points))
    #test_points = np.random.random((num_test_points, domain_dim))
    x_0 = np.sort(np.random.random(sqrt_num_test_points))
    x_1 = np.sort(np.random.random(sqrt_num_test_points))
    test_points = np.transpose([np.tile(x_0, sqrt_num_test_points), np.repeat(x_1, sqrt_num_test_points)])
    x_0, x_1 = np.meshgrid(x_0, x_1)
    x_vals = X0
    y_vals = y0
    y_vals_no_noise = objective(test_points, noise=0.)

    for i in range(iters):
        x_coords = np.random.random((1, domain_dim))
        # x_vals += [x_val]
        # X = np.asarray(x_vals)
        y_val = objective(x_coords)
        model.fit(x_coords, y_val)

        x_vals = np.append(x_vals, x_coords, axis=0)
        y_vals = np.append(y_vals, y_val)


        # plt.clf()
        # plt.ylim(-0.2, 1.0)
        # plt.scatter(np.ndarray.flatten(x_vals), y_vals)
        # plt.plot(np.ndarray.flatten(test_points), mu)
        # plt.plot(np.ndarray.flatten(test_points), y_vals_no_noise)
        # plt.fill_between(np.ndarray.flatten(test_points), mu - stds, mu + stds, alpha=0.4)
        # plt.pause(1e-17)
        # time.sleep(0.5)

    mu, covs = model.predict(test_points)
    stds = np.sqrt(np.diagonal(covs))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_0, x_1, mu.reshape((sqrt_num_test_points, sqrt_num_test_points)), alpha=1.0)#, color='b')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_0, x_1, y_vals_no_noise.reshape((sqrt_num_test_points, sqrt_num_test_points)), alpha=0.8, color='y')

elif optimisation:

    iters = 10
    num_samples = 2000
    sqrt_num_test_points = 100
    num_test_points = sqrt_num_test_points ** 2
    predictions_mu = np.zeros((iters, num_test_points))
    predictions_std_squared = np.zeros((iters, num_test_points))
    x_0 = np.sort(np.random.random(sqrt_num_test_points))
    x_1 = np.sort(np.random.random(sqrt_num_test_points))
    test_points = np.transpose([np.tile(x_0, sqrt_num_test_points), np.repeat(x_1, sqrt_num_test_points)])
    x_0, x_1 = np.meshgrid(x_0, x_1)
    y_vals_no_noise = np.ndarray.flatten(objective(test_points, noise=0.))

    x_vals = X0
    y_vals = y0

    if acq_type == 'EI' or 'PI':
        margin = 0.01

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_0, x_1, y_vals_no_noise.reshape((sqrt_num_test_points, sqrt_num_test_points)), alpha=0.5,
                    color='y')

    for i in range(iters):
        # select the next point to sample
        x = opt_acquisition(acq_type, model, num_samples, margin=0.01)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(x)
        #print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # update the model
        model.fit(x, actual)

        x_vals = np.append(x_vals, x, axis=0)
        y_vals = np.append(y_vals, actual)

        ax.scatter(x_vals[:, 0], x_vals[:, 1], y_vals, color='k')
        #time.sleep(2.)

    print('First best guess: x=' + str(X0[ix]) + ', y=%.3f' % y0[ix])

    ix_new = np.argmax(model.y)
    print('Best result: x=' + str(model.X[ix_new]) + ', y=%.3f' % model.y[ix_new])


