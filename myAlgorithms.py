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
            #init = jnp.array([0.5]*model.domain_dim).reshape((1, 1))

        # print("init: "+str(init))
        # print(init)

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
                return_surrogates=False, dynamic_plot=False):
    #TODO: refactor. This is a mess. Also, plotting functionality will only work in 1d

    x_vals = X0
    y_vals = y0
    ix = jnp.argmax(y0)

    if return_surrogates:
        surrogate_means = np.zeros((num_iters, 1000))
        surrogate_stds = np.zeros((num_iters, 1000))

    for i in range(num_iters):
        # select the next point to sample
        print(acq_func(jnp.array([0.5]).reshape((1, 1)), model))
        #print(model)
        # random restarts:
        #final_loss = 0.
        #for j in range(20): # remove hard-coded restart num
        x, final_loss = acq_alg(acq_func, model)

        # sample the point
        actual = objective(x)
        # summarize the finding
        est, _ = model.predict(x)
        print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
        # update the model
        model.fit(x, actual)

        x_vals = jnp.append(x_vals, x, axis=0)
        y_vals = jnp.append(y_vals, actual)

        if return_surrogates:
            test_points = np.asarray(np.arange(0, 1, 1 / 1000)).reshape((1000, x_vals.shape[1]))
            mu, covs = model.predict(test_points)  # TODO: needn't compute these if not returning surrogate or plotting
            stds = np.sqrt(np.diagonal(covs))
            surrogate_means[i, :] = mu
            surrogate_stds[i, :] = stds

        if dynamic_plot:
            plt.clf()
            plt.scatter(np.ndarray.flatten(x_vals), y_vals)
            plt.plot(np.ndarray.flatten(test_points), mu)
            plt.fill_between(np.ndarray.flatten(test_points), mu - stds, mu + stds, alpha=0.4)
            plt.pause(1e-17)
            time.sleep(2.)

        print("iter "+str(i)+" successful")

    if return_surrogates:
        surrogate_data = {'means': surrogate_means, 'stds': surrogate_stds}
    else:
        surrogate_data = None

    print('First best guess: x=%.3f, y=%.3f' % (X0[ix], y0[ix]))
    ix = np.argmax(model.y)
    print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))

    return x_vals, y_vals, surrogate_data
