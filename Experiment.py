from Regressors import GaussianProcessReg
from Algorithms import ML_for_hyperparams
import optax
import jax.numpy as jnp


class Experiment():

    ''' By default, initialisation of this class builds an RBF GP with parameters estimated by maximum likelihood.
    If the user intends to specify a model (either complete or incomplete) then the 'ML_est' flag can be turned to
    False. Data must be provided. '''

    def __init__(self, Xsamples=None, ysamples=None, obs_noise_stdev=1e-2, ML_est=True):
        # the noise level is chosen as small as possible while maintaining numerical stability

        assert Xsamples is not None, "No input data points provided."
        assert ysamples is not None, "No output data provided."
        assert Xsamples.shape[0] == ysamples.shape[0], "Number of input and output data-points not equal."

        self.Xsamples = Xsamples
        self.ysamples = ysamples
        self._model = None
        self.ML_optimizer = optax.adam(learning_rate=1e-2)  # hard-coded for now.
        # we set the prior mean to be the mean of the observed values
        mean = jnp.average(ysamples)
        # this is a bit of a hack, but the prior mean is expected to be a function, so...
        default_prior_func = (lambda x: mean)

        if ML_est:
            hyperparam_dict = {'sigma': None, 'lengthscale': None}
            optimal_param_dict, _ = ML_for_hyperparams(Xsamples, ysamples, self.ML_optimizer,
                                                       hyperparam_dict=hyperparam_dict,
                                                       obs_noise_stdev=obs_noise_stdev, prior_mean=default_prior_func,
                                                       prior_mean_kwargs=None, iters=5000, num_restarts=5)

            print("Kernel parameters estimated by maximum-likelihood: " + str(optimal_param_dict))

            self._model = GaussianProcessReg(kernel_hyperparam_kwargs=optimal_param_dict,
                                            obs_noise_stdev=obs_noise_stdev)

            # fit to data
            self._model.fit(Xsamples, ysamples, compute_cov=True)

        else:
            print("Model not built")

    # @property
    # def model(self, kernel_type='RBF', kernel_hyperparam_kwargs=None, obs_noise_stdev=1e-6):  # more parameters?
    #
    #     ''' This will instantiate and fit a new model if either a non-RBF kernel has been specified, or if the user
    #     has prescribed the values of some model hyper-parameters. To prescribe (some subset) of the relevant hyper-
    #     parameters, populate the 'kernel_hyperparam_kwargs' with the relevant
    #
    #     Kernel choices:
    #
    #         'RBF',          hyper-parameters: 'sigma', 'lengthscale',
    #         'Periodic',     hyper-parameters: 'sigma', 'lengthscale', 'periodic'
    #         'Matern',       hyper-parameters: 'sigma', 'lengthscale', 'order'
    #
    #     '''
    #
    #     #TODO: checks that prescribed hyperparams satisfy appropriate conditions e.g. positivity
    #
    #     if kernel_hyperparam_kwargs is None:
    #         kernel_hyperparam_kwargs = {}
    #
    #     sigma = kernel_hyperparam_kwargs.get('sigma')
    #     lengthscale = kernel_hyperparam_kwargs.get('lengthscale')
    #     period = kernel_hyperparam_kwargs.get('period')
    #     order = kernel_hyperparam_kwargs.get('order')
    #
    #     # build the dictionary of hyper-parameters
    #     hyperparam_dict = {}
    #     hyperparam_dict['sigma'] = sigma
    #     hyperparam_dict['lengthscale'] = lengthscale
    #
    #     if kernel_type == 'RBF':
    #         if sigma is None:
    #             if lengthscale is None:
    #                 print("Default model not updated as no model hyper-parameters have been specified")
    #                 return
    #
    #     elif kernel_type == 'Periodic':
    #         hyperparam_dict['period'] = period
    #
    #     elif kernel_type == 'Matern':
    #         hyperparam_dict['order'] = order
    #
    #     # run maximum likelihood optimisation to estimate non-prescribed kernel hyper-parameters
    #     updated_hyperparam_dict, _ = ML_for_hyperparams(self.Xsamples, self.ysamples, self.ML_optimizer,
    #                                            kernel_type=kernel_type, obs_noise_stdev=1e-6, prior_mean=None,
    #                                            prior_mean_kwargs=None, iters=1000, num_restarts=5)
    #
    #     self._model = GaussianProcessReg(kernel_hyperparam_kwargs=updated_hyperparam_dict,
    #                                      obs_noise_stdev=obs_noise_stdev)
    #
    #     # now we fit the new model to the data
    #
