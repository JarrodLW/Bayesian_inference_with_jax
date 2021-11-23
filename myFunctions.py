# Created 06/10/2021
# includes acquisition functions and kernels...

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.special import gamma, kv


# acquisition functions

class base_func:
    # class structure for algebra of acquisition functions

    def __init__(self, function):
        self.function = function

    def __call__(self, x: jnp.array, model):
        print("base func call :" + str(type(model)))
        return self.function(x, model)

    def __add__(self, g):
        return base_func(lambda x, model: self.function(x, model) + g(x, model))

    def __rmul__(self, lam):
        return base_func(lambda x, model: lam*self.function(x, model))


# def current_max(model, observed=True):
#     # returns either the highest observed value so far, or the


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


def rescaled_sq_pair_dists(x1, x2, lengthscale, dist='euclidean'):
    # lengthscale: either a scalar or list of same dimension as domain
    #TODO: add error when lengthscale has length not equal to base dimension?
    # Note: there some to be larger errors than I'd expect
    # TODO: assert error when x1 and x2 has mismatched .shape[1]

    # height = x1.shape[0]
    # width = x2.shape[0]
    #dm = jnp.zeros((height, width))

    # rescaling
    x1 = x1/lengthscale
    x2 = x2/lengthscale

    if dist == 'euclidean': # sometimes returns small negative number, so take max with zero
        squared_dm = jnp.maximum(0, jnp.linalg.norm(x1, axis=1)[:, None]**2 +
                            jnp.linalg.norm(x2, axis=1)**2 - 2*jnp.matmul(x1, x2.T))

    else:
        raise NotImplementedError("Distance function not implemented")

    return squared_dm


class RBF(kernels):

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale, dist='euclidean'):

        def cov_func(x1, x2):
            return stdev ** 2 * jnp.exp(-0.5*rescaled_sq_pair_dists(x1, x2, lengthscale, dist))

        super().__init__(cov_func)


class Periodic(kernels):

    # \sigma^2\exp(-2\sin^2(\pi\Vert x - x'\Vert/p)/l^2)

    def __init__(self, stdev, lengthscale, period, dist='euclidean'):

        def cov_func(x1, x2):
            covs = stdev ** 2 * jnp.exp(-2*jnp.sin(np.pi*jnp.sqrt(rescaled_sq_pair_dists(x1, x2, lengthscale, dist))/
                                                   period)/lengthscale**2)
            return covs

        super().__init__(cov_func)


class Matern(kernels):

    # \sigma^2*(2**(1-nu)/Gamma(nu))*(sqrt(2*nu)*\Vert x - x'\Vert/l)**nu*K_nu(sqrt(2*nu)*\Vert x - x'\Vert/l)
    # nu is the "order"
    # K_nu is the modified Bessel function of the second kind
    #TODO: re-implement with jax

    def __init__(self, stdev, lengthscale, order, dist='euclidean'):

        def cov_func(x1, x2):
            rescaled_dist = jnp.sqrt(2 * order) * jnp.sqrt(rescaled_sq_pair_dists(x1, x2, lengthscale, dist))
            rescaled_dist = jnp.maximum(1.e-8, rescaled_dist) # How does this interact with jax grad?
            covs = stdev ** 2 * (2 ** (1 - order) / gamma(order)) * (rescaled_dist ** order) \
                   * kv(order, rescaled_dist)
            return covs

        super().__init__(cov_func)

