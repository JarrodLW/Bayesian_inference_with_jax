# Created

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax
from Regressors import GaussianProcessReg


class OptaxAcqAlgBuilder:
    # More generally, could pass a more complicated optimisation schedule?
    def __init__(self, optimizer, iters=1000):
        self.optimizer = optimizer
        self.iters = iters

    def __call__(self, acq_func, model, init=None):
        ## builds optax gradient-based algorithm for optimisation of acquisition function
        # acq_func(x, model) -> \mathbb{R}, a pure function,
        # model: a regressor class instance,
        # optimizer: an optax optimizer e.g. optax.adam(learning_rate=1e-2)
        # init: where to initialise the optimisation
        # TODO: assert error if init inconsistent with model dimension
        # TODO: incorporate other initialisation strategies

        # random initialisation if no init given. Use jax random instead?
        if init is None:
            init = jnp.array(np.random.random(model.domain_dim)).reshape((1, model.domain_dim))

        # we take the negative of the acquisition function since optax algs designed to minimise rather than maximise
        def acq_objective(x):
            return - acq_func(jnp.array([x]).reshape((1, model.domain_dim)), model)
            # this can probably be avoided by using partial derivatives instead. Do I need to "build" everytime?

        def optimization(x: optax.Params) -> optax.Params:
            opt_state = self.optimizer.init(x)

            @jax.jit
            def step(x, opt_state):
                loss_value, grads = jax.value_and_grad(acq_objective)(x)
                updates, opt_state = self.optimizer.update(grads, opt_state, x)
                x = optax.apply_updates(x, updates)
                return x, opt_state, loss_value

            for i in range(self.iters):
                x, opt_state, loss_value = step(x, opt_state)
                # if i % 100 == 0:
                #     print(f'step {i}, loss: {loss_value}')
                # TODO: generalise these conditions -- these checks are for a unit cube [0, 1]^n.
                if jnp.any(x < 0) or jnp.any(x > 1):
                    break

            return x, loss_value

        x_opt, final_loss = optimization(init)
        return x_opt, final_loss


def random_acq(acq_func, model, num_samples=None):
    # TODO allow for batching
    # TODO allow for differing domain geometries

    domain_dim = model.domain_dim

    if num_samples is None:
        num_samples = 100**domain_dim

    # random search, generate random samples
    Xsamples = np.random.random((num_samples, domain_dim))
    scores = acq_func(Xsamples, model)
    # locate the index of the largest scores
    ix = jnp.argmax(scores)
    #print("Best score " + str(np.amax(scores)))
    return Xsamples[ix].reshape(1, domain_dim), - scores[ix] # any point returning scores?


# def opt_routine(acq_func, model, num_iters, X0, y0, objective, acq_alg=random_acq,
#                 return_surrogates=False, return_acq_func_vals=False, dynamic_plot=False):
def opt_routine(acq_func, model, num_iters, objective, acq_alg=random_acq,
                return_surrogates=False, return_acq_func_vals=False, dynamic_plot=False):
    #TODO: refactor. This is a mess. Also, plotting functionality will only work in 1d

    x_vals = X0 = model.X
    y_vals = y0 = model.y
    ix = jnp.argmax(y_vals)

    # only hold on to surrogates and/or plot dynamically if in dimension 1
    # TODO: add dynamic function plotting for dimension equal to 2
    if model.domain_dim != 1:
        return_surrogates = False
        dynamic_plot = False

    if return_surrogates or dynamic_plot: #TODO should be using jax?
        surrogate_means = np.zeros((num_iters, 1000))
        surrogate_stds = np.zeros((num_iters, 1000))
        acq_func_vals = np.zeros((num_iters, 1000))
        #test_points = jnp.asarray(np.arange(0, 1, 1 / 1000)).reshape((1000, x_vals.shape[1]))
        test_points = jnp.asarray(sorted(np.random.random((1000, x_vals.shape[1]))))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(num_iters):
        # select the next point to sample
        # random restarts. Should this be moved into optimizer?

        num_restarts = 5  # remove hard-coded restart num

        # TODO change this to keep only those x which are an improvement. I don't need to be storing everything.
        x_cands = jnp.zeros((num_restarts, model.domain_dim), dtype=jnp.float32)
        x_cand_losses = jnp.zeros(num_restarts, dtype=jnp.float32)
        for j in range(num_restarts):
            x, final_loss = acq_alg(acq_func, model)
            x_cands = x_cands.at[j, :].set(jnp.ravel(x))
            x_cand_losses = x_cand_losses.at[j].set(final_loss)

        ind_best = jnp.argmin(x_cand_losses)
        x = x_cands[jnp.argmin(x_cand_losses), :].reshape((1, model.domain_dim))
        final_loss = x_cand_losses[ind_best]

        if return_surrogates or dynamic_plot:
            mu, covs = model.predict(test_points)
            stds = jnp.sqrt(jnp.diagonal(covs))
            surrogate_means[i, :] = mu
            surrogate_stds[i, :] = stds
            acq_func_val = acq_func(test_points, model)
            acq_func_vals[i, :] = acq_func_val # not using or returning this currently...

        if dynamic_plot:
            #plt.clf()
            ax1.cla()
            ax2.cla()
            ax1.set_xlim(-0.1, 1.1)
            ax1.set_ylim(-0.2, 1.1)
            ax1.scatter(jnp.ravel(x_vals), y_vals)
            ax1.plot(jnp.ravel(test_points), mu)
            ax1.fill_between(jnp.ravel(test_points), mu - stds, mu + stds, alpha=0.4)
            ax2.plot(jnp.ravel(test_points), acq_func_val)
            ax2.scatter(x, - final_loss) # we take the minus since the loss is the negative of the acquisition
            #plt.plot(jnp.ravel(test_points), acq_func_val)
            plt.pause(1e-17)
            #time.sleep(2.)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(x)
        print(f">x={str(x[0])}, f()={est[0]:.3f}, actual={actual[0]:.3f}")
        # update the model
        model.fit(x, actual)

        x_vals = jnp.append(x_vals, x, axis=0)
        y_vals = jnp.append(y_vals, actual)

        print("iter "+str(i+1)+" successful")

    if return_surrogates:
        surrogate_data = {'means': surrogate_means, 'stds': surrogate_stds}
    else:
        surrogate_data = None

    #print('First best guess: x=%.3f, y=%.3f' % (X0[ix], y0[ix]))
    print(f"First best guess: x={str(X0[ix])}, y={y0[ix]:.3f}")
    ix = np.argmax(model.y)
    #print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))
    print(f"First best guess: x={str(model.X[ix])}, y={model.y[ix]:.3f}")

    return x_vals, y_vals, surrogate_data

# learning hyperparameters

def log_marg_likelihood(Xsamples, ysamples, kernel_type='RBF', kernel_hyperparam_kwargs=None, obs_noise_stdev=1e-2,
                                prior_mean=None, prior_mean_kwargs=None):

    model = GaussianProcessReg(kernel_type=kernel_type, kernel_hyperparam_kwargs=kernel_hyperparam_kwargs,
                               obs_noise_stdev=obs_noise_stdev,
                               prior_mean=prior_mean, prior_mean_kwargs=prior_mean_kwargs)
    model.fit(Xsamples, ysamples, compute_cov=True)
    # log_prob = jnp.nan_to_num(model.log_marg_likelihood)
    log_prob = model.log_marg_likelihood

    return log_prob


def ML_for_hyperparams(Xsamples, ysamples, optimizer,
                       kernel_type='RBF', hyperparam_dict=None, obs_noise_stdev=1e-2, prior_mean=None,
                       prior_mean_kwargs=None, iters=5000, num_restarts=5):

    ''' Returns a maximum marginal likelihood estimate for the kernel hyper-parameters, in the form of a dictionary.
    The dictionary object hyperparam_dict should only contain the hyper-parameters pertaining to the kernel type;
    # this is already taken care of in the "model" method of the Experiment class. '''

    # TODO: add in checks that hyperparam_dict contains only the keys relevant for the given kernel?
    hyper_to_be_updated = [k for k in hyperparam_dict.keys() if hyperparam_dict[k] is None]
    print("Optimising for kernel hyperparameters: " + ', '.join(hyper_to_be_updated))

    def acq_objective(x):
        # x is a vector. Zeroth entry corresponds to ln(sigma), first entry to ln(lengthscale) ---we've remapped onto
        # positive reals using the exponential.

        assert x.shape[0] == len(hyper_to_be_updated), "Variable dimension inconsistent with number of parameters " \
                                                       "to be estimated"

        # building the variable dictionary of hyper-parameters
        variable_hyperparam_dict = {}
        for key in hyperparam_dict.keys():
            if hyperparam_dict[key] is not None:
                variable_hyperparam_dict[key] = hyperparam_dict[key]
        # inserting the x's into the hype-param dictionary
        for j, key in enumerate(hyper_to_be_updated):
            variable_hyperparam_dict[key] = jnp.exp(x[j])

        return - log_marg_likelihood(Xsamples, ysamples, kernel_type=kernel_type,
                                     kernel_hyperparam_kwargs=variable_hyperparam_dict,
                                     obs_noise_stdev=obs_noise_stdev,
                                     prior_mean=prior_mean, prior_mean_kwargs=prior_mean_kwargs)

    def optimization(x: optax.Params) -> optax.Params:
        opt_state = optimizer.init(x)

        @jax.jit
        def step(x, opt_state):
            loss_value, grads = jax.value_and_grad(acq_objective)(x)
            updates, opt_state = optimizer.update(grads, opt_state, x)
            x = optax.apply_updates(x, updates)
            return x, opt_state, loss_value

        for i in range(iters):
            x, opt_state, loss_value = step(x, opt_state)

            if i%100 == 0:
                print('Loss at iter ' + str(i) + ': ' + str(loss_value))

        return x, loss_value

    # TODO change this to keep only those x which are an improvement
    x_cands = jnp.zeros((num_restarts, len(hyper_to_be_updated)), dtype=jnp.float32)
    x_cand_losses = jnp.zeros(num_restarts, dtype=jnp.float32)
    for i in range(num_restarts):
        init = jnp.array(jnp.log(np.random.random(len(hyper_to_be_updated))))
        x_opt, final_loss = optimization(init)
        x_cands = x_cands.at[i, :].set(jnp.ravel(x_opt))
        x_cand_losses = x_cand_losses.at[i].set(final_loss)

    ind_best = jnp.argmin(x_cand_losses)
    optimal_params = jnp.exp(x_cands[jnp.argmin(x_cand_losses), :])
    final_loss = x_cand_losses[ind_best]

    # now populate the dictionary of hyperparameters
    updated_hyperparam_dict = {}
    # the fixed values remain the same
    for key in hyperparam_dict.keys():
        if hyperparam_dict[key] is not None:
            updated_hyperparam_dict[key] = hyperparam_dict[key]
    for j, key in enumerate(hyper_to_be_updated):
        updated_hyperparam_dict[key] = optimal_params[j]

    return updated_hyperparam_dict, final_loss

