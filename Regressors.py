# Created 06/10/2021

#import numpy as np
import jax.lax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, solve_triangular
from Kernels import RBF, Periodic, Matern
from time import time
from jax.lax import stop_gradient as sg
import copy


class GaussianProcessReg:
    ''' Instance is a Gaussian process model with prescribed prior, with fit and predict methods.'''
    # obs noise just big enough to have a regularising effect when solving linear algebraic systems

    def __init__(self, kernel_type='RBF', domain_dim=1, kernel_hyperparam_kwargs=None, obs_noise_stdev=1e-6,
                prior_mean=None, prior_mean_kwargs=None):  # TODO: rename obs_noise_stdev and sigma

        self.mu = None
        self.std = None
        self.covs = None
        self._hyperparams_cache = None
        self.y = None
        self.X = None
        self.obs_noise_stdev = obs_noise_stdev
        self.L = None
        self.domain_dim = domain_dim
        self.alpha = None
        self.log_marg_likelihood = None
        self.kernel_type = kernel_type

        if prior_mean is None:
            prior_mean = lambda x: 0

        if kernel_hyperparam_kwargs is None:
            kernel_hyperparam_kwargs = {}

        self.prior_mean = prior_mean

        if prior_mean_kwargs is not None:
            self.prior_mean_kwargs = prior_mean_kwargs
        else:
            self.prior_mean_kwargs = {}

        if kernel_type == 'RBF':
            self.kernel = RBF(**kernel_hyperparam_kwargs)

        elif kernel_type == 'Periodic':
            self.kernel = Periodic(**kernel_hyperparam_kwargs)

        elif kernel_type == 'Matern':
            self.kernel = Matern(**kernel_hyperparam_kwargs)

    def fit(self, Xsamples, ysamples, verbose=True):
        # Note to self: fit is probably bad terminology ---we don't actually "fit" anything but simply
        # compute certain quantities that are needed for prediction and mle etc.

        if any([val is None for val in self.kernel.hyperparams.values()]):
            print("Cannot fit as kernel hyper-parameters have yet to be specified.")
            return

        if verbose:
            print("Fitting GP to data")

        # check that the hyperparams of the kernel haven't been changed; else recompute cov matrix from scratch.
        if self._hyperparams_cache != self.kernel.hyperparams:
            if verbose:
                print("Kernel hyper-parameters have been modified. Recomputing cov matrix. ")
            self.covs = None

        if self.covs is None:
            # this is the case of initial fitting, in which case the samples are all the data that it has access to
            self.covs = self.kernel(Xsamples, Xsamples)
            self.X = Xsamples
            self.y = ysamples
            self._hyperparams_cache = copy.deepcopy(self.kernel.hyperparams)

        else:
            # cross covariances
            test_train_covs = self.kernel(self.X, Xsamples)

            # broadcast covariances
            k = self.kernel(Xsamples, Xsamples)
            self.covs = jnp.block([[self.covs, test_train_covs], [test_train_covs.T, k]])

            # update x and y vectors
            self.X = jnp.concatenate((self.X, Xsamples), axis=0)
            self.y = jnp.concatenate((self.y, ysamples), axis=0)

        # perform Cholesky factorisation of noise-shifted covariance matrix
        covs_plus_noise = self.covs + self.obs_noise_stdev**2*jnp.identity(self.covs.shape[0])
        self.L = cholesky(covs_plus_noise, lower=True)

        # TODO: this assert interacts badly with jax gradient, fix this.
        # assert sg((jnp.sqrt(jnp.sum(jnp.square(jnp.matmul(sg(self.L), sg(self.L.T)) - sg(covs_plus_noise))))
        #       /jnp.sqrt(jnp.sum(jnp.square(sg(covs_plus_noise))))) < 1e-6), "factorisation error too large"

        # computing log marginal likelihood
        y_shifted = self.y - self.prior_mean(self.X, **self.prior_mean_kwargs)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y_shifted, lower=True),
                                 lower=False)  # following nomenclature in Rasmussen & Williams

        # TODO: this assert interacts badly with jax gradient, fix this.
        # assert (jnp.sqrt(jnp.sum(jnp.square(jnp.matmul(self.L, jnp.matmul(self.L.T, self.alpha))
        #                                        - y_shifted))) /
        #            (jnp.sqrt(jnp.sum(jnp.square(y_shifted))) + 1e-7)) < 1e-6, "matrix inversion error too large"

        self.log_marg_likelihood = - (1/2)*jnp.dot(self.y, self.alpha) - jnp.sum(jnp.diag(self.L)) \
                                   - (Xsamples.shape[0]/2)*jnp.log(2*jnp.pi)

    def predict(self, Xsamples):

        if any([val is None for val in self.kernel.hyperparams.values()]):
            print("Cannot predict as kernel hyper-parameters have yet to be specified.")
            return

        # should I be saving the mu and std to memory?
        test_train_covs = self.kernel(self.X, Xsamples)
        pred_mu = jnp.matmul(test_train_covs.T, self.alpha)
        pred_mu += self.prior_mean(Xsamples, **self.prior_mean_kwargs)
        k = self.kernel(Xsamples, Xsamples)

        v = solve_triangular(self.L, test_train_covs, lower=True)
        pred_covs = k - jnp.matmul(v.T, v)

        return pred_mu, pred_covs
