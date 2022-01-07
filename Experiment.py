import copy

from Regressors import GaussianProcessReg
from Algorithms import ML_for_hyperparams, opt_routine
from AcquisitionFuncs import acq_func_builder
import optax
import jax.numpy as jnp

class Experiment():

    ''' By default, initialisation of this class builds an RBF GP with parameters estimated by maximum likelihood.
    If the user intends to specify a model (either complete or incomplete) then the 'ML_est' flag can be turned to
    False. Data (Xsamples, ysamples) must be provided. By default, the prior function is set to be the constant
    function, the constant being the mean of the ysamples. '''

    def __init__(self, Xsamples=None, ysamples=None, kernel_type='RBF', kernel_hyperparams=None,
                 obs_noise_stdev=1e-2, prior_mean=None, mle=True, ML_optimizer=None, objective=None):

        if kernel_hyperparams is None:
            kernel_hyperparams = {}

        assert Xsamples is not None, "No input data points provided."
        assert ysamples is not None, "No output data provided."
        assert Xsamples.shape[0] == ysamples.shape[0], "Number of input and output data-points not equal."

        self.Xsamples = Xsamples
        self.ysamples = ysamples

        # we set the prior mean to be the mean of the observed values, as default
        mean = jnp.average(ysamples)
        if prior_mean is None:
            prior_mean = (lambda x: mean)

        if ML_optimizer is None:
            ML_optimizer = optax.adam(learning_rate=1e-1)

        self.ML_optimizer = ML_optimizer
        self.objective = objective

        # initialize a GP model
        self.model = GaussianProcessReg(kernel_type=kernel_type, domain_dim=Xsamples.shape[1],
                                        kernel_hyperparam_kwargs=kernel_hyperparams,
                                        obs_noise_stdev=obs_noise_stdev,
                                        prior_mean=prior_mean)

        if mle:
            self.maximum_likelihood_estimation()
        else:
            print("mle flag off, so no maximum-likelihood estimation of kernel hyper-parameters is being performed.")

    def maximum_likelihood_estimation(self):
        ''' This will run maximum likelihood estimation on all those 'hyperparam_dict' keys with None values and
        define/fit a new model based on these hyperparameters. '''

        # if all keys populated, do nothing
        if None not in self.model.kernel.hyperparams.values():
            print("All hyper-parameters specified; nothing to be estimated.")
            return

        if self.model.kernel_type == 'Matern':
            if self.model.kernel.order is None:
                print("Optimisation of Matern kernel order not currently supported; only discrete orders implemented. "
                      "Please specify order.")
                return

        optimal_param_dict, _ = ML_for_hyperparams(self.Xsamples, self.ysamples, self.ML_optimizer,
                                                   kernel_type=self.model.kernel_type,
                                                   hyperparam_dict=self.model.kernel.hyperparams,
                                                   obs_noise_stdev=self.model.obs_noise_stdev,
                                                   prior_mean=self.model.prior_mean,
                                                   prior_mean_kwargs=None)

        # now build the model with the identified kernel hyperparameters
        self.model = GaussianProcessReg(kernel_type=self.model.kernel_type, domain_dim=self.model.domain_dim,
                                        kernel_hyperparam_kwargs=optimal_param_dict,
                                         obs_noise_stdev=self.model.obs_noise_stdev, prior_mean=self.model.prior_mean)

        # re-fit to data
        self.model.fit(self.Xsamples, self.ysamples)

    def run_bayes_opt(self, num_iters=1, acq_func=None, dynamic_plot=False, acq_alg=None):
        # runs Bayesian optimisation to propose next x coordinate at which to run experiment
        # TODO: add condition to opt_routine that if only 1 iteration being called, give option not to evaluate the objective but simply "request" it

        if self.objective is None:
            print("No objective function provided. Can't perform Bayesian optimisation. ")
            return

        if acq_func is None:
            acq_func = acq_func_builder('PI', margin=0.01)

        if acq_alg is None:
            opt_routine(acq_func, self.model, num_iters, self.objective, dynamic_plot=dynamic_plot)

        else:
            opt_routine(acq_func, self.model, num_iters, self.objective, acq_alg=acq_alg, dynamic_plot=dynamic_plot)



