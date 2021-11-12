import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from myFunctions import PI_acquisition, RBF, acq_func_builder, rescaled_pairwise_dists
from Regressors import GaussianProcessReg
from myAlgorithms import opt_acquisition
from scipy.spatial.distance import cdist
from time import time
from jax.scipy.stats import norm
import matplotlib.pyplot as plt


def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)


grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
# prints 0.4199743

print(vmap(tanh)(jnp.array([[0.], [1.], [0.5]])))
print(jit(vmap(tanh))(jnp.array([[0.], [1.], [0.5]])))

# differentiating a non-differentiable function?

def func1(x):

  if x>0:
    return x**2
  elif x<= 0:
    return x

grad_func1 = grad(func1)
print(grad_func1(0.1))
print(grad_func1(-0.1))
print(grad_func1(0.))

grad_PI_acquisition = grad(PI_acquisition)

model = GaussianProcessReg(sigma=0.2, lengthscale=0.2, obs_noise_stdev=0.1)
# initialising model
X0 = jnp.arange(5).reshape((5, 1))/5
y0 = jnp.array([0.6, 2.4, 3.1, 4.2, 3.6])
model.fit(X0, y0, compute_cov=True)

Xsamples = jnp.arange(50).reshape((50, 1))/50

mu, covs = model.predict(Xsamples)

plt.plot(Xsamples, mu)
plt.fill_between(np.ndarray.flatten(np.asarray(Xsamples)), mu - np.sqrt(np.diag(covs)), mu + np.sqrt(np.diag(covs)), alpha=0.4)

print(PI_acquisition(Xsamples, model, 0.))

margin = 0.01
kwargs = {'margin': margin}

acq_type = 'PI'
num_samples = 1000
acq_func = acq_func_builder(acq_type, **kwargs)
x = opt_acquisition(acq_func, model, num_samples)

grad(PI_acquisition)(np.asarray([1.0]).reshape((1, 1)), model, 0.)


# rough
mu, covs = model.predict(Xsamples)

x1_np = np.arange(10).reshape(2, 5)/10
x2_np = np.arange(15).reshape(3, 5)/15
x1_jnp = jnp.arange(10).reshape(2, 5)/10
x2_jnp = jnp.arange(15).reshape(3, 5)/15

t0 = time()
cdist(x1_np, x2_np)
print(time() - t0)

t0 = time()
print(rescaled_pairwise_dists(x1_jnp, x2_jnp, lengthscale=1., dist='euclidean'))
print(time() - t0)

### scalar PI acquisition

def scalar_PI_acquisition(x, model, margin):
  # isn't there a closed-form solution for the optimal sampling point?
  # calculate the best surrogate score found so far
  # yhat, _ = model.predict(model.X)
  # best = np.amax(yhat)  # is this the correct value to be using? Highest surrogate value versus highest observed...

  best = jnp.amax(model.y)
  best_plus_margin = best + margin
  # calculate mean and stdev via surrogate function
  mu, covs = model.predict(x)
  std = jnp.sqrt(jnp.diag(covs))
  # calculate the probability of improvement
  prob = norm.cdf((mu - best_plus_margin) / (std + 1E-20))

  return prob[0]


scalar_PI_acquisition(jnp.array([0.25]).reshape((1, 1)), model, 0.)
grad(scalar_PI_acquisition)(jnp.array([0.25]).reshape((1, 1)), model, 0.)
