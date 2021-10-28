# Created

import numpy as np
import matplotlib.pyplot as plt
import time
from myFunctions import PI_acquisition, EI_acquisition, UCB_acquisition


def opt_acquisition(acq_func, model, num_samples): #, std_weight=1., margin=None):  # TODO allow for multiple points to be kept
    # TODO allow for differing domain geometries
    domain_dim = model.domain_dim
    # random search, generate random samples
    Xsamples = np.random.random((num_samples, domain_dim))
    # calculate the acquisition function for each sample
    # if acq_type=='PI':
    #     scores = PI_acquisition(margin, Xsamples, model)
    #
    # elif acq_type=='EI':
    #     scores = EI_acquisition(margin, Xsamples, model)
    #
    # elif acq_type=='UCB':
    #     scores = UCB_acquisition(std_weight, Xsamples, model)

    scores = acq_func(Xsamples, model)

    # locate the index of the largest scores
    ix = np.argmax(scores)

    print("Best score " + str(np.amax(scores)))

    return Xsamples[ix].reshape(1, domain_dim)


def opt_routine(acq_func, model, num_iters, X0, y0, objective, num_samples=1000, return_surrogates=False,
                dynamic_plot=False): #TODO: refactor. This is a mess. Also, plotting functionality will only work in 1d

    x_vals = X0
    y_vals = y0
    ix = np.argmax(y0)

    test_points = np.asarray(np.arange(0, 1, 1 / 1000)).reshape((1000, x_vals.shape[1]))  # TODO: remove hard-coding

    if return_surrogates:
        surrogate_means = np.zeros((num_iters, 1000))
        surrogate_stds = np.zeros((num_iters, 1000))

    for i in range(num_iters):
        # select the next point to sample
        x = opt_acquisition(acq_func, model, num_samples)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(x)
        print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # update the model
        model.fit(x, actual)

        x_vals = np.append(x_vals, x, axis=0)
        y_vals = np.append(y_vals, actual)

        mu, covs = model.predict(test_points) # TODO: needn't compute these if not returning surrogate or plotting
        stds = np.sqrt(np.diagonal(covs))

        if return_surrogates:
            surrogate_means[i, :] = mu
            surrogate_stds[i, :] = stds

        if dynamic_plot:
            plt.clf()
            plt.scatter(np.ndarray.flatten(x_vals), y_vals)
            plt.plot(np.ndarray.flatten(test_points), mu)
            plt.fill_between(np.ndarray.flatten(test_points), mu - stds, mu + stds, alpha=0.4)
            plt.pause(1e-17)
            time.sleep(2.)

    if return_surrogates:
        surrogate_data = {'means': surrogate_means, 'stds': surrogate_stds}

    else:
        surrogate_data = None

    print('First best guess: x=%.3f, y=%.3f' % (X0[ix], y0[ix]))

    ix = np.argmax(model.y)
    print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))

    return x_vals, y_vals, surrogate_data


