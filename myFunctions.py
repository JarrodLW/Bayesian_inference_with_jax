# Created 06/10/2021
# includes acquisition functions and kernels...

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance
from scipy.special import gamma, kv


# acquisition functions

class base_func:

    # class structure for algebra of functions

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

class RBF():

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale, dist='euclidean'):

        self.stdev = stdev
        self.lengthscale = lengthscale
        self.dist = dist

    def __call__(self, X1, X2):

        covs = self.stdev ** 2 * np.exp(-0.5*distance.cdist(X1, X2, self.dist)**2 / self.lengthscale ** 2)

        return covs


class Periodic():

    # \sigma^2\exp(-2\sin^2(\pi\Vert x - x'\Vert/p)/l^2)

    def __init__(self, stdev, lengthscale, period, dist='euclidean'):

        self.stdev = stdev
        self.lengthscale = lengthscale
        self.period = period
        self.dist = dist

    def __call__(self, X1, X2):

        covs = self.stdev ** 2 * np.exp(-2*np.sin(np.pi*distance.cdist(X1, X2, self.dist)/self.period)**2
                                        / self.lengthscale ** 2)

        return covs


class Matern():

    # \sigma^2*(2**(1-nu)/Gamma(nu))*(sqrt(2*nu)*\Vert x - x'\Vert/l)**nu*K_nu(sqrt(2*nu)*\Vert x - x'\Vert/l)
    # nu is the "order"
    # K_nu is the modified Bessel function of the second kind

    def __init__(self, stdev, lengthscale, order, dist='euclidean'):

        self.stdev = stdev
        self.lengthscale = lengthscale
        self.order = order
        self.dist = dist

    def __call__(self, X1, X2): #TODO: get rid of hack used to deal with infinities of Bessel func.

        rescaled_dist = np.sqrt(2*self.order)*distance.cdist(X1, X2, self.dist)/self.lengthscale
        print(rescaled_dist)
        covs = self.stdev ** 2 * (2**(1-self.order)/gamma(self.order))*(rescaled_dist**self.order)\
               *np.nan_to_num(kv(self.order, rescaled_dist))

        return covs


