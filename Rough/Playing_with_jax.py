import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from myFunctions import PI_acquisition, RBF, Periodic, acq_func_builder, rescaled_sq_pair_dists
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

print(PI_acquisition(jnp.asarray(1.0).reshape((1, 1)), model, 0.))
grad(PI_acquisition, argnums=0)(jnp.asarray([1.0]).reshape((1, 1)), model, 0.)


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
print(rescaled_sq_pair_dists(x1_jnp, x2_jnp, lengthscale=1., dist='euclidean'))
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
  #mu, covs = model.predict(x)
  #std = jnp.sqrt(jnp.diag(covs))
  mu = 16.7*x
  std = 0.2*x**2
  # calculate the probability of improvement
  prob = norm.cdf((mu - best_plus_margin) / (std + 1E-20))

  return prob

dx_vals = 10.**(-np.arange(6, 10))[::-1]
x_vals = [0.25] + [0.25 + dx for dx in dx_vals]
y_vals = [scalar_PI_acquisition(x, model, 0.) for x in x_vals]

gradient = grad(scalar_PI_acquisition, argnums=0)(0.25, model, 0.)
y_vals_linear = [scalar_PI_acquisition(0.25, model, 0.) + (x - 0.25)*gradient for x in x_vals]

plt.plot(x_vals, y_vals, color='b', label='true')
plt.plot(x_vals, y_vals_linear, color='r', label='linear approx')
plt.legend()


## testing pairwise distance function

x1 = jnp.arange(30).reshape((5, 6))/30
x2 = jnp.arange(18).reshape((3, 6))/18

rescaled_sq_pair_dists(x1, x1, 0.05, dist='euclidean')

## differentiating a kernel

sigma = 0.1
lengthscale = 0.05
RBF_kernel = RBF(sigma, lengthscale)
Periodic_kernel = Periodic(sigma, lengthscale, jnp.pi/3)

x1 = jnp.arange(30).reshape((5, 6))/30
x2 = jnp.arange(18).reshape((3, 6))/18
RBF_kernel(x1, x2)


def distance_component(X1, X2):
    dist = rescaled_sq_pair_dists(X1, X2, lengthscale, dist='euclidean')[0, 1]
    return dist

def RBF_kernel_component(X1, X2):
    dist = RBF_kernel(X1, X2)[0, 1]
    return dist

def Periodic_kernel_component(X1, X2):
    dist = Periodic_kernel(X1, X2)[0, 1]
    return dist

print(rescaled_sq_pair_dists(x1, x1, 0.05))

print(distance_component(x1, x1))

grad(distance_component, argnums=0)(x1, x2)

grad(RBF_kernel_component, argnums=0)(x1, x2)

grad(Periodic_kernel_component, argnums=0)(x1, x2)




#

def oscillatory_func(x):

    return x*jnp.sin(jnp.abs(x))

grad(oscillatory_func)(0.2)

grad(oscillatory_func)(-0.2)

grad(oscillatory_func)(0.)

