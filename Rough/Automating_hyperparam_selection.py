from Regressors import *
from jax import grad
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import optax
import jax


def quadratic(x, a, b, c):
    return jnp.ravel(a * x ** 2 + b * x + c)


def log_marg_likelihood_wrapper(Xsamples, ysamples, sigma, lengthscale):

    model = GaussianProcessReg(sigma=sigma, lengthscale=lengthscale, obs_noise_stdev=0.001,
                               prior_mean=quadratic, prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})
    model.fit(Xsamples, ysamples, compute_cov=True)
    log_prob = model.log_marg_likelihood # + (Xsamples.shape[0]/2)*jnp.log(2*jnp.pi)

    return log_prob

def objective(x):
    return jnp.ravel(x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0)

X0 = jnp.linspace(0., 1., num=10).reshape((10, 1))
y0 = objective(X0)

# checking we can differentiate
log_marg_likelihood_wrapper(X0, y0, 0.1, 0.05)
grad(log_marg_likelihood_wrapper, argnums=0)(X0, y0, 0.1, 0.05)
grad(log_marg_likelihood_wrapper, argnums=1)(X0, y0, 0.1, 0.05)
grad(log_marg_likelihood_wrapper, argnums=2)(X0, y0, 0.1, 0.05)
grad(log_marg_likelihood_wrapper, argnums=3)(X0, y0, 0.1, 0.05)

# finding best parameters by brute force search
sigmas = [10**x for x in np.linspace(-5, 0, num=50)]
lengthscales = [10**x for x in np.linspace(-5, 0, num=50)]
sig, len = np.meshgrid(sigmas, lengthscales)

Z = np.zeros((50, 50))
for i, sigma in enumerate(sigmas):
    for j, lengthscale in enumerate(lengthscales):
        Z[i, j] = jnp.exp(log_marg_likelihood_wrapper(X0, y0, sigma, lengthscale))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(sig, len, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

np.amax(Z)
np.argmax(Z)
print(sigmas[np.argmax(Z)//50])
print(lengthscales[np.argmax(Z)%50])

# find optimal hyperparameters using optax optimiser

# TODO: shares some of the same functionality as OptaxAcqAlgBuilder. Refactor?
# TODO: use stopping criteria
# TODO: needs to be flexible enough to take different sets of hyperparams, depending on kernel choice


def ML_for_hyperparams(Xsamples, ysamples, optimizer, iters=1000, num_restarts=5):
    # this returns a maximum marginal likelihood estimator for the model parameters

    def acq_objective(x):
        # x is a vector. Zeroth entry corresponds to sigma, first entry to lengthscale
        return - log_marg_likelihood_wrapper(Xsamples, ysamples, x[0], x[1])

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

            # TODO: extend these conditions?
            # none of the hyperparameters can be negative
            if jnp.any(x < 0):
                break

        return x, loss_value

    # TODO change this to keep only those x which are an improvement
    x_cands = jnp.zeros((num_restarts, 2), dtype=jnp.float32)
    x_cand_losses = jnp.zeros(num_restarts, dtype=jnp.float32)
    for i in range(num_restarts):
        init = jnp.array(np.random.random(2))
        x_opt, final_loss = optimization(init)
        x_cands = x_cands.at[i, :].set(jnp.ravel(x_opt))
        x_cand_losses = x_cand_losses.at[i].set(final_loss)

    ind_best = jnp.argmin(x_cand_losses)
    x_opt = x_cands[jnp.argmin(x_cand_losses), :]
    final_loss = x_cand_losses[ind_best]

    sig_opt, lengthscale_opt = x_opt

    return sig_opt, lengthscale_opt, final_loss


optimizer = optax.adam(learning_rate=1e-2)
sig_opt, lengthscale_opt, final_loss = ML_for_hyperparams(X0, y0, optimizer, iters=5000)

