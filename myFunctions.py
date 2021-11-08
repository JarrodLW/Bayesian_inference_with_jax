# Created 06/10/2021
# includes acquisition functions and kernels...

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance
from scipy.special import gamma, kv


# acquisition functions

class base_func:

    # class structure for algebra of acquisition functions

    def __init__(self, function):
        self.function = function

    def __call__(self, x: np.array, model):
        print("base func call :" + str(type(model)))
        return self.function(x, model)

    def __add__(self, g):
        return base_func(lambda x, model: self.function(x, model) + g(x, model))

    def __rmul__(self, lam):
        return base_func(lambda x, model: lam*self.function(x, model))


def PI_acquisition(Xsamples, model, margin):
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


def EI_acquisition(Xsamples, model, margin):

    best = np.amax(model.y)
    best_plus_margin = best + margin
    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))
    Z = (mu - best_plus_margin) / (std + 1E-20)
    scores = ((mu - best_plus_margin)*norm.cdf(Z) + std*norm.pdf(Z))*(std > 0)  # see Mockus and Mockus

    return scores


def UCB_acquisition(Xsamples, model, std_weight):

    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))
    scores = mu + std_weight*std

    return scores


def acq_func_builder(method, *args, **kwargs):

    if method=='PI':
        f = lambda x, model: PI_acquisition(x, model, *args, **kwargs)
        return base_func(f)

    elif method == 'EI':
        f = lambda x, model: EI_acquisition(x, model, *args, **kwargs)
        return base_func(f)

    elif method=='UCB':
        f = lambda x, model: UCB_acquisition(x, model, *args, **kwargs)
        return base_func(f)

# kernels


class kernels():

    def __init__(self, cov_func):
        self.cov_func = cov_func

    def __call__(self, x1: np.array, x2: np.array):
        return self.cov_func(x1, x2)

    def __add__(self, g):
        return kernels(lambda x1, x2: self.cov_func(x1, x2) + g(x1, x2))

    def __rmul__(self, lam):
        return kernels(lambda x1, x2: lam*self.cov_func(x1, x2))

    def __mul__(self, g):
        return kernels(lambda x1, x2: self.cov_func(x1, x2)*g(x1, x2))


class RBF(kernels):

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale, dist='euclidean'):

        def cov_func(x1, x2):
            return stdev ** 2 * np.exp(-0.5*distance.cdist(x1, x2, dist)**2 / lengthscale ** 2)

        super().__init__(cov_func)


class Periodic(kernels):

    # \sigma^2\exp(-2\sin^2(\pi\Vert x - x'\Vert/p)/l^2)

    def __init__(self, stdev, lengthscale, period, dist='euclidean'):

        def cov_func(x1, x2):
            covs = stdev ** 2 * np.exp(-2*np.sin(np.pi*distance.cdist(x1, x2, dist)/period)**2
                                        / lengthscale ** 2)
            return covs

        super().__init__(cov_func)


class Matern(kernels):

    # \sigma^2*(2**(1-nu)/Gamma(nu))*(sqrt(2*nu)*\Vert x - x'\Vert/l)**nu*K_nu(sqrt(2*nu)*\Vert x - x'\Vert/l)
    # nu is the "order"
    # K_nu is the modified Bessel function of the second kind

    def __init__(self, stdev, lengthscale, order, dist='euclidean'):

        def cov_func(x1, x2):
            rescaled_dist = np.sqrt(2 * order) * distance.cdist(x1, x2, dist) / lengthscale
            rescaled_dist = np.maximum(1.e-8, rescaled_dist)
            covs = stdev ** 2 * (2 ** (1 - order) / gamma(order)) * (rescaled_dist ** order) \
                   * kv(order, rescaled_dist)
            return covs

        super().__init__(cov_func)

