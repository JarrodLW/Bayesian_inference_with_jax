# Created 06/10/2021
# includes acquisition functions and kernels...

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


# acquisition functions

class BaseFunc:
    # class structure for algebra of acquisition functions

    def __init__(self, function):
        self.function = function

    def __call__(self, x: jnp.array, model):
        print("base func call :" + str(type(model)))
        return self.function(x, model)

    def __add__(self, g):
        return BaseFunc(lambda x, model: self.function(x, model) + g(x, model))

    def __rmul__(self, lam):
        return BaseFunc(lambda x, model: lam * self.function(x, model))


def PI_acquisition(Xsamples, model, margin):
    # isn't there a closed-form solution for the optimal sampling point?
    # calculate the best surrogate score found so far
    #yhat, _ = model.predict(model.X)
    #best = np.amax(yhat)  # is this the correct value to be using? Highest surrogate value versus highest observed...

    best = jnp.amax(model.y)
    best_plus_margin = best + margin
    # calculate mean and stdev via surrogate function
    mu, covs = model.predict(Xsamples)
    std = jnp.sqrt(jnp.diag(covs))

    if Xsamples.shape[0] == 1: # must be a better way than this!
        mu = mu[0]
        std = std[0]

    # calculate the probability of improvement
    probs = norm.cdf((mu - best_plus_margin) / (std + 1E-20))

    return probs


def EI_acquisition(Xsamples, model, margin):

    best = np.amax(model.y)
    best_plus_margin = best + margin
    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))

    if Xsamples.shape[0] == 1: # must be a better way than this!
        mu = mu[0]
        std = std[0]

    Z = (mu - best_plus_margin) / (std + 1E-20)
    scores = ((mu - best_plus_margin)*norm.cdf(Z) + std*norm.pdf(Z))*(std > 0)  # see Mockus and Mockus

    return scores


def UCB_acquisition(Xsamples, model, std_weight):

    mu, covs = model.predict(Xsamples)
    std = np.sqrt(np.diag(covs))

    if Xsamples.shape[0] == 1: # must be a better way than this!
        mu = mu[0]
        std = std[0]

    scores = mu + std_weight*std

    return scores


def acq_func_builder(method, *args, **kwargs):

    if method=='PI':
        f = lambda x, model: PI_acquisition(x, model, *args, **kwargs)
        return BaseFunc(f)

    elif method == 'EI':
        f = lambda x, model: EI_acquisition(x, model, *args, **kwargs)
        return BaseFunc(f)

    elif method=='UCB':
        f = lambda x, model: UCB_acquisition(x, model, *args, **kwargs)
        return BaseFunc(f)


