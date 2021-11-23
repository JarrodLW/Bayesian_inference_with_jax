# Created

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import optax
import jax


class optax_acq_alg_builder:
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
                if jnp.any(x < 0) or jnp.any(x > 1):
                    break

            return x, loss_value

        x_opt, final_loss = optimization(init)
        return x_opt, final_loss


def random_acq(acq_func, model, num_samples=1000): #, std_weight=1., margin=None):  # TODO allow for multiple points to be kept
    # TODO allow for differing domain geometries
    domain_dim = model.domain_dim
    # random search, generate random samples
    Xsamples = np.random.random((num_samples, domain_dim))
    scores = acq_func(Xsamples, model)
    # locate the index of the largest scores
    ix = jnp.argmax(scores)
    print("Best score " + str(np.amax(scores)))
    return Xsamples[ix].reshape(1, domain_dim)


def opt_routine(acq_func, model, num_iters, X0, y0, objective, acq_alg=random_acq,
                return_surrogates=False, return_acq_func_vals=False, dynamic_plot=False):
    #TODO: refactor. This is a mess. Also, plotting functionality will only work in 1d

    x_vals = X0
    y_vals = y0
    ix = jnp.argmax(y0)

    if return_surrogates or dynamic_plot: #TODO should be using jax?
        surrogate_means = np.zeros((num_iters, 1000))
        surrogate_stds = np.zeros((num_iters, 1000))
        acq_func_vals = np.zeros((num_iters, 1000))
        test_points = jnp.asarray(np.arange(0, 1, 1 / 1000)).reshape((1000, x_vals.shape[1]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(num_iters):
        # select the next point to sample
        # random restarts. Should this be moved into optimizer?
        num_restarts = 10  # remove hard-coded restart num

        x_cands = jnp.zeros((num_restarts, model.domain_dim), dtype=jnp.float32)
        x_cand_losses = jnp.zeros(num_restarts, dtype=jnp.float32)
        for j in range(num_restarts):
            x, final_loss = acq_alg(acq_func, model)
            x_cands = x_cands.at[j, :].set(jnp.ravel(x))
            x_cand_losses = x_cand_losses.at[j].set(final_loss)

        ind_best = jnp.argmin(x_cand_losses)
        print("index of best: "+str(ind_best))
        x = x_cands[jnp.argmin(x_cand_losses), :].reshape((1, model.domain_dim))
        final_loss = jnp.amin(x_cand_losses)

        if return_surrogates or dynamic_plot:
            mu, covs = model.predict(test_points)  # TODO: needn't compute these if not returning surrogate or plotting
            stds = np.sqrt(np.diagonal(covs))
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
            ax2.scatter(x, -final_loss) # we take the minus since the loss is the negative of the acquisition
            #plt.plot(jnp.ravel(test_points), acq_func_val)
            #plt.pause(1e-17)
            #time.sleep(2.)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(x)
        print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # update the model
        model.fit(x, actual)

        x_vals = jnp.append(x_vals, x, axis=0)
        y_vals = jnp.append(y_vals, actual)

        print("iter "+str(i)+" successful")

    if return_surrogates:
        surrogate_data = {'means': surrogate_means, 'stds': surrogate_stds}
    else:
        surrogate_data = None


    print('First best guess: x=%.3f, y=%.3f' % (X0[ix], y0[ix]))
    ix = np.argmax(model.y)
    print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))

    return x_vals, y_vals, surrogate_data
