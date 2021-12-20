import jax
import jax.numpy as jnp
from Experiment import Experiment
from AcquisitionFuncs import *
from Algorithms import opt_routine, log_marg_likelihood
import matplotlib.pyplot as plt
from matplotlib import cm
import optax

# def quadratic(x, a, b, c):
#     return jnp.ravel(a * x ** 2 + b * x + c)


def objective(x):
    return jnp.ravel(x ** 2 * jnp.sin(5 * jnp.pi * x) ** 6.0)


#X0 = jnp.asarray(list(np.linspace(0., 1., num=150))).reshape((150, 1))
#X0 = jnp.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).reshape((7, 1))
X0 = jnp.asarray(list(np.linspace(0., 1., num=10))).reshape((10, 1))
#X0 = jnp.asarray([0.2, 0.35, 0.5, 0.65, 0.8]).reshape((5, 1))
#X0 = jnp.asarray([0.3, 0.4, 0.7]).reshape((3, 1))
#X0 = jnp.asarray([0.2, 0.3, 0.8]).reshape((3, 1))
y0 = objective(X0)

exp = Experiment(X0, y0)

# checking that optimal params have been correctly identified
num = 100
sigmas = [10**x for x in np.linspace(-5, 0, num=num)]
lengthscales = [10**x for x in np.linspace(-5, 0, num=num)]
sig, len = np.meshgrid(sigmas, lengthscales)

Z = np.zeros((num, num))

def log_marg_likelihood_wrapper(Xsamples, ysamples, sigma, lengthscale):
    kernel_hyperparam_kwargs = {'sigma': sigma, 'lengthscale': lengthscale}
    log_prob = log_marg_likelihood(Xsamples, ysamples, kernel_type='RBF', kernel_hyperparam_kwargs=kernel_hyperparam_kwargs,
                            obs_noise_stdev=1e-3)
    return log_prob


for i, sigma in enumerate(sigmas):
    for j, lengthscale in enumerate(lengthscales):
        Z[i, j] = jnp.exp(log_marg_likelihood_wrapper(X0, y0, sigma, lengthscale))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(sig, len, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

print(exp._model.kernel_hyperparam_kwargs)

np.amax(Z)
np.argmax(Z)
print('sigma: ' + str(sigmas[np.argmax(Z)//num]))
print('lengthscale: ' + str(lengthscales[np.argmax(Z)%num]))

# Bayes Opt

model = exp._model
num_iters = 3
acq_func = acq_func_builder('PI', margin=0.01)

X, y, surrogate_data = opt_routine(acq_func, model, num_iters, objective, return_surrogates=False,
                                       dynamic_plot=True)  # TODO: X0, y0 incorporated into model?
