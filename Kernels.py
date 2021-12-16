# Created 06/10/2021
# includes acquisition functions and kernels...

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma, kv


class Kernels():

    def __init__(self, cov_func):
        self.cov_func = cov_func

    def __call__(self, x1: np.array, x2: np.array):
        return self.cov_func(x1, x2)

    def __add__(self, g):
        return Kernels(lambda x1, x2: self.cov_func(x1, x2) + g(x1, x2))

    def __rmul__(self, lam):
        return Kernels(lambda x1, x2: lam * self.cov_func(x1, x2))

    def __mul__(self, g):
        return Kernels(lambda x1, x2: self.cov_func(x1, x2) * g(x1, x2))


def rescaled_sq_pair_dists(x1, x2, lengthscale=1., dist='euclidean'):
    # lengthscale: either a scalar or list of same dimension as domain
    # TODO: add error when lengthscale has length not equal to base dimension?
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


class RBF(Kernels):

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale, dist='euclidean'):

        def cov_func(x1, x2):
            return stdev ** 2 * jnp.exp(-0.5*rescaled_sq_pair_dists(x1, x2, lengthscale, dist))

        super().__init__(cov_func)


class Periodic(Kernels):
    # TODO: finish implementation of gradient
    # \sigma^2\exp(-2\sin^2(\pi\Vert x - x'\Vert/p)/l^2)
    # jax isn't able to compute the gradient at zero, despite being well-defined, so we have to code this up by hand

    def __init__(self, stdev, lengthscale, period, dist='euclidean'): # only supports isotropic lengthscale
        # TODO: assert error if you try to pass lengthscale list?

        #@jax.custom_jvp
        def cov_func(x1, x2):
            covs = stdev ** 2 * jnp.exp(-2*jnp.sin(np.pi*jnp.sqrt(rescaled_sq_pair_dists(x1, x2, dist=dist))/
                                                   period)**2/lengthscale**2)
            return covs

        # @cov_func.defjvp
        # def cov_func_jvp(primals, tangents):
        #     x1, x2 = primals
        #     x1_dot, x2_dot = tangents
        #     primal_out = cov_func(x1, x2)
        #     # grad_milestone = - (2*np.pi/(period*lengthscale**2))*jnp.matmul(jnp.sin(2*np.pi*(x1 - x2)/period), primal_out.T)
        #     # print(grad_milestone.shape)
        #     # tangent_out = grad_milestone*x1_dot + grad_milestone*x2_dot
        #     return primal_out, tangent_out

        super().__init__(cov_func)


class Matern(Kernels):

    # \sigma^2*(2**(1-nu)/Gamma(nu))*(sqrt(2*nu)*\Vert x - x'\Vert/l)**nu*K_nu(sqrt(2*nu)*\Vert x - x'\Vert/l)
    # nu is the "order"
    # K_nu is the modified Bessel function of the second kind
    #TODO: re-implement with jax

    def __init__(self, stdev, lengthscale, order, dist='euclidean'):

        def cov_func(x1, x2):
            rescaled_dist = jnp.sqrt(2 * order) * jnp.sqrt(rescaled_sq_pair_dists(x1, x2, lengthscale, dist))
            rescaled_dist = jnp.maximum(1.e-8, rescaled_dist)  # How does this interact with jax grad?
            covs = stdev ** 2 * (2 ** (1 - order) / gamma(order)) * (rescaled_dist ** order) \
                   * kv(order, rescaled_dist)
            return covs

        super().__init__(cov_func)
