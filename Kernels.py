# Created 06/10/2021
# includes acquisition functions and kernels...

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma, kv


class Kernels():

    ''' Instances of this class may be multiplied and linear combinations taken,
    allowing for bespoke kernels to be easily built. '''

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

    # Note: scipy has a distance implementation, but I don't think there's an analogue in jax yet, hence this
    # implementation.
    # lengthscale: either a scalar or list of same dimension as domain
    # TODO: add error when lengthscale has length not equal to base dimension?
    # TODO: assert error when x1 and x2 has mismatched .shape[1]

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

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l^2)
    _sigma = None
    _lengthscale = None

    def __init__(self, sigma=None, lengthscale=None, dist='euclidean'):

        self.sigma = sigma
        self.lengthscale = lengthscale
        self.dist = dist
        self.make_func()

    def make_func(self):

        if self.sigma is not None and self.lengthscale is not None:

            def cov_func(x1, x2):
                return self.sigma ** 2 * jnp.exp(-0.5 * rescaled_sq_pair_dists(x1, x2, self.lengthscale, self.dist))
            self.cov_func = cov_func
        else:
            self.cov_func = None

    @property
    def hyperparams(self):
        return {'sigma': self.sigma, 'lengthscale': self.lengthscale}

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        # TODO: this is incompatible with jax, in the same way as the assert statements in model fit/predict methods --fix this.
        # if value is not None:
        #     print(value)
        #     if value < 0:
        #         raise ValueError("Negative standard deviation not possible")
        self._sigma = value
        self.make_func()

    @property
    def lengthscale(self):
        return self._lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        # if value is not None:
        #     if value <= 0:
        #         raise ValueError("Non-positive lengthscale not possible")
        self._lengthscale = value
        self.make_func()


class Periodic(Kernels):
    # TODO: finish implementation of gradient
    # \sigma^2\exp(-2\sin^2(\pi\Vert x - x'\Vert/p)/l^2)
    # jax isn't able to compute the gradient at zero, despite being well-defined, so we have to code the Jacobian up
    # by hand. See https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html.

    _sigma = None
    _lengthscale = None
    _period = None

    def __init__(self, sigma=None, lengthscale=None, period=None, dist='euclidean'):  # only supports isotropic lengthscale
        # TODO: assert error if you try to pass lengthscale list?

        self.sigma = sigma
        self.lengthscale = lengthscale
        self.period = period
        self.dist = dist
        self.make_func()

    def make_func(self):

        if self.sigma is not None and self.lengthscale is not None and self.period is not None:

            def cov_func(x1, x2):
                return self.sigma ** 2 * jnp.exp(-2 * jnp.sin(np.pi * jnp.sqrt(rescaled_sq_pair_dists(x1, x2, dist=self.dist)) /
                                                         self.period) ** 2 / self.lengthscale ** 2)
            self.cov_func = cov_func
        else:
            self.cov_func = None

    @property
    def hyperparams(self):
        return {'sigma': self.sigma, 'lengthscale': self.lengthscale, 'period': self.period}

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        # if value is not None:
        #     print(value)
        #     if value < 0:
        #         raise ValueError("Negative standard deviation not possible")
        self._sigma = value
        self.make_func()

    @property
    def lengthscale(self):
        return self._lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        # if value is not None:
        #     if value <= 0:
        #         raise ValueError("Non-positive lengthscale not possible")
        self._lengthscale = value
        self.make_func()

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        # if value is not None:
        #     if value <= 0:
        #         raise ValueError("Non-positive lengthscale not possible")
        self._period = value
        self.make_func()

    # @cov_func.defjvp
    # def cov_func_jvp(primals, tangents):
    #     x1, x2 = primals
    #     x1_dot, x2_dot = tangents
    #     primal_out = cov_func(x1, x2)
    #     # grad_milestone = - (2*np.pi/(period*lengthscale**2))*jnp.matmul(jnp.sin(2*np.pi*(x1 - x2)/period), primal_out.T)
    #     # print(grad_milestone.shape)
    #     # tangent_out = grad_milestone*x1_dot + grad_milestone*x2_dot
    #     return primal_out, tangent_out


class Matern(Kernels):

    # \sigma^2*(2**(1-nu)/Gamma(nu))*(sqrt(2*nu)*\Vert x - x'\Vert/l)**nu*K_nu(sqrt(2*nu)*\Vert x - x'\Vert/l)
    # nu is the "order"
    # K_nu is the modified Bessel function of the second kind

    _sigma = None
    _lengthscale = None
    _order = None

    # TODO: add gradient/Jacobian implementation for orders 1.5 and 2.5 and error statement for order 0.5 (not diff'able)
    def __init__(self, sigma=None, lengthscale=None, order=None, dist='euclidean'):

        self.sigma = sigma
        self.lengthscale = lengthscale

        if order is None:
            order = '3/2'
            print("Setting order to default of 3/2")

        self.order = order
        self.dist = dist
        self.make_func()

    def make_func(self):

        if self.sigma is not None and self.lengthscale is not None and self.order is not None:

            if self.order == '1/2':
                def cov_func(x1, x2):
                    d = jnp.sqrt(rescaled_sq_pair_dists(x1, x2, self.lengthscale, self.dist))
                    return self.sigma ** 2 * jnp.exp(-d / self.lengthscale) # check this expression!

            elif self.order == '3/2':
                def cov_func(x1, x2):
                    d = jnp.sqrt(rescaled_sq_pair_dists(x1, x2, self.lengthscale, self.dist))
                    return self.sigma ** 2 * (1 + jnp.sqrt(3) * d / self.lengthscale) * jnp.exp(-jnp.sqrt(3) * d / self.lengthscale)

            elif self.order == '5/2':
                def cov_func(x1, x2):
                    d = jnp.sqrt(rescaled_sq_pair_dists(x1, x2, self.lengthscale, self.dist))
                    return self.sigma ** 2 * (1 + jnp.sqrt(5) * d / self.lengthscale + 5 * d ** 2 / (3 * self.lengthscale ** 2)) * \
                           jnp.exp(-jnp.sqrt(5) * d / self.lengthscale)

            else:
                print("Matern kernel of order " + str(self.order) + " not implemented")
                return

            self.cov_func = cov_func
        else:
            self.cov_func = None

    @property
    def hyperparams(self):
        return {'sigma': self.sigma, 'lengthscale': self.lengthscale, 'order': self.order}

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        # if value is not None:
        #     print(value)
        #     if value < 0:
        #         raise ValueError("Negative standard deviation not possible")
        self._sigma = value
        self.make_func()

    @property
    def lengthscale(self):
        return self._lengthscale

    @lengthscale.setter
    def lengthscale(self, value):
        # if value is not None:
        #     if value <= 0:
        #         raise ValueError("Non-positive lengthscale not possible")
        self._lengthscale = value
        self.make_func()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        # if value is not None:
        #     if value <= 0:
        #         raise ValueError("Non-positive lengthscale not possible")
        self._order = value
        self.make_func()

    # general
    # def cov_func(x1, x2):
    #     rescaled_dist = jnp.sqrt(2 * order) * jnp.sqrt(rescaled_sq_pair_dists(x1, x2, lengthscale, dist))
    #     rescaled_dist = jnp.maximum(1.e-8, rescaled_dist)  # How does this interact with jax grad?
    #     covs = stdev ** 2 * (2 ** (1 - order) / gamma(order)) * (rescaled_dist ** order) \
    #            * kv(order, rescaled_dist)
    #     return covs

