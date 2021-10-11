# Created 06/10/2021

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from myFunctions import RBF, Periodic


class GaussianProcessReg():
    # instance is a Gaussian process model with prescribed prior, with fit and predict methods.

    def __init__(self, kernel_type='RBF', domain_dim=1, sigma=1., obs_noise_stdev=0.1, lengthscale=1.0, period=None,
                 prior_mean=None, prior_mean_kwargs=None): #TODO: rename obs_noise_stdev and sigma

        self.mu = None
        self.std = None
        self.covs = None
        self.y = None
        self.X = None
        self.obs_noise_stdev = obs_noise_stdev
        self.L = None
        self.domain_dim = domain_dim

        if prior_mean is None:
            prior_mean = lambda x: 0

        self.prior_mean = prior_mean

        if prior_mean_kwargs is not None:
            self.prior_mean_kwargs = prior_mean_kwargs
        else:
            self.prior_mean_kwargs = {}

        if kernel_type == 'RBF':
            self.kernel = RBF(sigma, lengthscale)

        elif kernel_type == 'Periodic':
            self.kernel = Periodic(sigma, lengthscale, period)

    def fit(self, Xsamples, ysamples, compute_cov=False):

        if compute_cov:
            self.covs = self.kernel(Xsamples, Xsamples)
            self.X = Xsamples
            self.y = ysamples

        else:
            # cross covariances
            test_train_covs = self.kernel(self.X, Xsamples)

            # broadcast covariances
            k = self.kernel(Xsamples, Xsamples)
            self.covs = np.block([[self.covs, test_train_covs],
                                  [test_train_covs.T, k]])

            # update x and y vectors
            self.X = np.concatenate((self.X, Xsamples), axis=0)
            self.y = np.concatenate((self.y, ysamples), axis=0)

        # perform Cholesky factorisation of noise-shifted covariance matrix
        covs_plus_noise = self.covs + self.obs_noise_stdev**2*np.identity(self.covs.shape[0])
        self.L = cholesky(covs_plus_noise, lower=True)

        # TODO throw up error if it failed to to factorise to sufficient accuracy
        print("failure to factorise " + str(np.sqrt(np.sum(np.square(np.matmul(self.L, self.L.T) - covs_plus_noise)))
              /np.sqrt(np.sum(np.square(covs_plus_noise)))))

    def predict(self, Xsamples):  # TODO generalise this to allow for multiple sampling points
        # should I be saving the mu and std to memory?

        test_train_covs = self.kernel(self.X, Xsamples)

        y_shifted = self.y - self.prior_mean(self.X, **self.prior_mean_kwargs)
        alpha = solve_triangular(self.L.T, solve_triangular(self.L, y_shifted, lower=True), lower=False)  # following nomenclature in Rasmussen

        pred_mu = np.matmul(test_train_covs.T, alpha)
        pred_mu += self.prior_mean(Xsamples, **self.prior_mean_kwargs)
        pred_mu = np.ndarray.flatten(pred_mu)

        k = self.kernel(Xsamples, Xsamples)
        v = solve_triangular(self.L, test_train_covs, lower=True)
        pred_covs = k - np.matmul(v.T, v)
        #pred_std = np.sqrt(np.abs(np.diag(pred_covs)))

        # TODO: throw up error if inversion wasn't successful
        print("failure to invert " +
              str(np.sqrt(np.sum(np.square(np.matmul(self.L, np.matmul(self.L.T, alpha)) - y_shifted)))/np.sqrt(np.sum(np.square(y_shifted)))))

        return pred_mu, pred_covs

