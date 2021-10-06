# Created 06/10/2021
# includes acquisition functions and kernels...

import numpy as np
from scipy.stats import norm


def PI_acquisition(margin, Xsamples, model):
    # isn't there a closed-form solution for the optimal sampling point?
    # calculate the best surrogate score found so far
    #yhat, _ = model.predict(model.X)
    #best = np.amax(yhat)  # is this the correct value to be using? Highest surrogate value versus highest observed...
    best = np.amax(model.y)
    best_plus_margin = best + margin
    # calculate mean and stdev via surrogate function
    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))
    # calculate the probability of improvement
    probs = norm.cdf((mu - best_plus_margin) / (std + 1E-20))

    return probs


def EI_acquisition(margin, Xsamples, model):

    best = np.amax(model.y)
    best_plus_margin = best + margin
    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))
    Z = (mu - best_plus_margin) / (std + 1E-20)
    scores = ((mu - best_plus_margin)*norm.cdf(Z) + std*norm.pdf(Z))*(std > 0)  # see Mockus and Mockus

    return scores


def UCB_acquisition(std_weight, Xsamples, model):

    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))
    scores = mu + std_weight*std

    return scores


class RBF():

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale):
        self.stdev = stdev
        self.lengthscale = lengthscale

    def __call__(self, X1, X2):
        # computes the matrix of covariances of sample points X1 against sample points X2. Each of X1, X2 is a 1d numpy array

        squared_dists = -2 * np.outer(X1, X2) + X1[:, None] ** 2 + X2 ** 2
        covs = self.stdev ** 2 * np.exp(-0.5*squared_dists / self.lengthscale ** 2)

        return covs


