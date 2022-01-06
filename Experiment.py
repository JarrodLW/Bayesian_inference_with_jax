from Regressors import GaussianProcessReg
from Algorithms import ML_for_hyperparams
import optax
import jax.numpy as jnp

class Experiment():

    ''' By default, initialisation of this class builds an RBF GP with parameters estimated by maximum likelihood.
    If the user intends to specify a model (either complete or incomplete) then the 'ML_est' flag can be turned to
    False. Data (Xsamples, ysamples) must be provided. By default, the prior function is set to be the constant
    function, the constant being the mean of the ysamples. '''

    def __init__(self, Xsamples=None, ysamples=None, kernel_type='RBF', kernel_hyperparams=None,
                 obs_noise_stdev=1e-2, prior_mean=None, mle=True, ML_optimizer = None):

        if kernel_hyperparams is None:
            kernel_hyperparams = {}

        assert Xsamples is not None, "No input data points provided."
        assert ysamples is not None, "No output data provided."
        assert Xsamples.shape[0] == ysamples.shape[0], "Number of input and output data-points not equal."

        self.Xsamples = Xsamples
        self.ysamples = ysamples

        if ML_optimizer is None:
            ML_optimizer = optax.adam(learning_rate=1e-1)

        self.ML_optimizer = ML_optimizer
        # we set the prior mean to be the mean of the observed values
        mean = jnp.average(ysamples)
        # this is a bit of a hack, but the prior mean is expected to be a function, so...
        if prior_mean is None:
            prior_mean = (lambda x: mean)

        self.model = GaussianProcessReg(kernel_type=kernel_type, domain_dim=Xsamples.shape[1],
                                        kernel_hyperparam_kwargs=kernel_hyperparams,
                                        obs_noise_stdev=obs_noise_stdev,
                                        prior_mean=prior_mean)

        if mle:
            self.maximum_likelihood_estimation()

    def maximum_likelihood_estimation(self):
        ''' This will run maximum likelihood estimation on all those 'hyperparam_dict' keys with None values and
        define/fit a new model based on these hyperparameters. '''

        # if all keys populated, do nothing
        if None not in self.model.kernel.hyperparams.values():
            print("All hyper-parameters specified; nothing to be estimated.")
            return

        #print(self.model.kernel.hyperparams)
        optimal_param_dict, _ = ML_for_hyperparams(self.Xsamples, self.ysamples, self.ML_optimizer,
                                                   hyperparam_dict=self.model.kernel.hyperparams,
                                                   obs_noise_stdev=self.model.obs_noise_stdev,
                                                   prior_mean=self.model.prior_mean,
                                                   prior_mean_kwargs=None)

        # now build the model with the identified kernel hyperparameters
        self.model = GaussianProcessReg(kernel_hyperparam_kwargs=optimal_param_dict,
                                         obs_noise_stdev=self.model.obs_noise_stdev)

        # re-fit to data
        self.model.fit(self.Xsamples, self.ysamples, compute_cov=True)



