import numpy as np
from Rough.BayesOptFromScratch_Cholesky import *

# objective function
def objective(x, noise=0.01):
    noise = norm(loc=0, scale=noise)
    return (x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise


# sample the domain sparsely with noise as initialisation
X0 = np.random.random(1)
y0 = np.asarray([objective(x) for x in X0])
ix = np.argmax(y0)

X = X0
y = y0

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


# define the model
# model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=linear,
#                            prior_mean_kwargs={'a': 0.5, 'b': 0})
model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
                           prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})
# model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.001)
# fit the model
model.fit(X, y, compute_cov=True)

# just fitting and plotting the surrogate

iters = 30
#x_vals = list(X0)
num_test_points = 1000
predictions_mu = np.zeros((iters, num_test_points))
predictions_std_squared = np.zeros((iters, num_test_points))
test_points = np.asarray(np.arange(0, 1, 1/num_test_points))
x_vals = list(X)
y_vals = list(y)
y_vals_no_noise = list(objective(test_points, noise=0.))

for i in range(iters):

    x_val = np.random.random(1)
    #x_vals += [x_val]
    #X = np.asarray(x_vals)
    y_val = objective(x_val)
    model.fit(np.asarray([x_val]), np.asarray([y_val]))

    x_vals += [x_val]
    y_vals += [y_val]

    mu, covs = model.predict(test_points)
    stds = np.sqrt(np.diagonal(covs))

    plt.clf()
    plt.ylim(-0.2, 1.0)
    plt.scatter(x_vals, y_vals)
    plt.plot(test_points, mu)
    plt.plot(test_points, y_vals_no_noise)
    plt.fill_between(test_points, mu - stds, mu + stds, alpha=0.4)
    plt.pause(1e-17)
    time.sleep(0.5)
