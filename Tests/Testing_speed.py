from Regressors import *
from myFunctions import *
from myAlgorithms import *
from numpy.random import normal


domain_dims = [1, 2, 3, 5, 10]

domain_dim = 5

# defining objective
def objective(x, noise=0.05):
    noise = normal(loc=0, scale=noise)
    return np.ndarray.flatten((x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise)


# defining model
model = GaussianProcessReg(sigma=0.1, domain_dim=domain_dim, lengthscale=0.05, obs_noise_stdev=0.01)

# defining acquisition function
pi = acq_func_builder('PI', margin=0.01)
ei = acq_func_builder('EI', margin=0.01)
ucb = acq_func_builder('UCB', std_weight=1.)
acq_func = 0.1*pi + 0.5*ei + 0.02*ucb

# setting up optimisation
# setting hyper-parameters
num_iters = 5
# initialising
X0 = np.asarray([1.]*domain_dim).reshape((1, domain_dim))
y0 = objective(X0)
model.fit(X0, y0, compute_cov=True)

# optimisation
X, y, surrogate_data = opt_routine(acq_func, model, num_iters, X0, y0, objective)
