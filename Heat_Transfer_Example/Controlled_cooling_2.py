from Experiment import Experiment
from AcquisitionFuncs import *
import scipy.io as io
import jax.numpy as jnp
import numpy as np
import os
from time import time

comm_file_path = "Heat_Transfer_Example/value_communicating_file_2.mat"


def objective(X: jnp.array):
    # TODO: re-implement this using a matlab engine API.

    # remap X onto Fourier coeffs; domain for BayesOpt code is [0, 1], phase domain is [0, 2*pi]
    phases = 2*np.pi*X
    io.savemat(comm_file_path, {'x_vals': X, 'phases': phases})  # saving phases
    # running matlab script to solve IBVP
    os.system('/Applications/MATLAB_R2021b.app/bin/matlab -batch "run(\'Heat_Transfer_Example/cooling_problem_2.m\');"')
    mat = io.loadmat(comm_file_path)  # grabbing objective values
    obs = jnp.asarray(mat.get("obs")).reshape(X.shape[0])  # grabbing resulting objective vals

    return obs


t0 = time()
X0 = jnp.asarray([[0.1], [0.3], [0.6], [0.8], [0.9]])
#X0 = jnp.asarray(list(np.linspace(0., 1., num=25))).reshape((25, 1))
#X0 = jnp.asarray(list(np.linspace(0., 1., num=5))).reshape((5, 1))
objective(X0)
results = io.loadmat(comm_file_path)
t1 = time()
print("Time elapsed "+str(t1 - t0))

max_t0 = results.get("max_t0")[0, 0]
y0 = jnp.asarray(results.get("obs")).reshape(X0.shape[0])
#y0 = jnp.asarray([1.3347493, 1.5273362, 1.6475159, 1.4136025, 1.3393945])
x_vals = jnp.asarray(results.get("x_vals"))
phases = jnp.asarray(results.get("phases"))

# print(y0)
#
# y_vals = np.zeros(5)
# for i in range(5):
#     y = objective(X0[i].reshape((1, 1)))
#     y_vals[i] = y
#
# print(y_vals)


## doing Bayes opt
# instantiating the model and fitting a RBF GP to the data using maximum-likelihood-estimation
# exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.05, 'lengthscale': 0.2, 'period': 1.},
#                  objective=objective)
# exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'sigma': 0.05, 'period': 2.},
#                  objective=objective)
exp = Experiment(X0, y0, kernel_type='Periodic', kernel_hyperparams={'period': 2.}, objective=objective)
#acq_func = acq_func_builder('UCB', std_weight=4.)
exp.run_bayes_opt(num_iters=2, dynamic_plot=True)
exp.run_bayes_opt(num_iters=3, dynamic_plot=True)
