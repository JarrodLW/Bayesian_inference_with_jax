import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from myFunctions import PI_acquisition, RBF, acq_func_builder
from Regressors import GaussianProcessReg
from myAlgorithms import opt_acquisition


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

model = GaussianProcessReg(sigma=0.1, lengthscale=0.2, obs_noise_stdev=0.01)
# initialising model
X0 = jnp.asarray([0., 0.1, 0.2, 0.3]).reshape((4, 1))
y0 = jnp.asarray([2.2, 3.6, 1.5, 1.])
model.fit(X0, y0, compute_cov=True)

Xsamples = jnp.arange(10).reshape((10, 1))/10

print(PI_acquisition(Xsamples, model, 0.))

margin = 0.01
kwargs = {'margin': margin}

acq_type = 'PI'
num_samples = 1000
acq_func = acq_func_builder(acq_type, **kwargs)
x = opt_acquisition(acq_func, model, num_samples)

grad_PI_acquisition(np.asarray([1.0]).reshape((1, 1)), model, 0.)


# rough
mu, covs = model.predict(Xsamples)
