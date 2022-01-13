from Experiment import Experiment
import scipy.io as io
import jax.numpy as jnp
import numpy as np
import os
from time import time

comm_file_path = "Heat_Transfer_Example/value_communicating_file.mat"


def objective(X: jnp.array):
    # TODO: re-implement this using a matlab engine API.

    ''' X is an array encoding instances of Fourier coefficients defining the boundary function, g.

        X.shape[0]: the number of instances,
        X.shape[1]: the number of Fourier coefficients (i.e. the dimension of the BayesOpt problem). Columns
        alternate between cosine coefficients and sine coefficients.

    This function runs the matlab script to solve the Dirichlet IBVP for each instance of Fourier coefficients
    and returns the value of the observable of each of the corresponding solution
    at a designated time. The observable value, in addition to the fourier coefficients and x values are
    also stored in the specified .mat file. '''

    # remap X onto Fourier coeffs; domain for BayesOpt code is [0, 1]^N, coeffs domain is [-1/sqrt(2)*N, 1/sqrt(2)*N]^N
    N = X.shape[1]
    Fourier_coeffs = (X - 0.5)*(np.sqrt(2)/N)
    cos_coeffs = Fourier_coeffs[:, ::2]  # the even rows will be the coefficients of the cosine terms
    sin_coeffs = Fourier_coeffs[:, 1::2]  # the odd rows will be the coefficients of the sine terms

    io.savemat(comm_file_path, {'x_vals': X, 'cos_coeffs': cos_coeffs, 'sin_coeffs': sin_coeffs})  # saving Fourier coeffs etc
    # running matlab script to solve IBVP
    os.system('/Applications/MATLAB_R2021b.app/bin/matlab -batch "run(\'Heat_Transfer_Example/cooling_problem.m\');"')
    mat = io.loadmat(comm_file_path)  # grabbing objective values
    obs = jnp.asarray(mat.get("obs")).reshape(X.shape[0])  # grabbing resulting objective vals

    return obs


t0 = time()
#X0 = jnp.asarray([[0.5, 0.5, 0.5, 0.5], [0.1, -0.1, 0.2, -0.3]])
#X0 = jnp.asarray([[0.], [0.1], [0.3], [0.8], [1.]])
X0 = jnp.asarray(list(np.linspace(0., 1., num=25))).reshape((25, 1))
objective(X0)
results = io.loadmat(comm_file_path)
t1 = time()
print("Time elapsed "+str(t1 - t0))

#Dirichlet_energy_t0 = results.get("Dirichlet_energy_t0")[0, 0]
max_t0 = results.get("max_t0")[0, 0]
y0 = jnp.asarray(results.get("obs")).reshape(X0.shape[0])
#y0 = jnp.asarray(results.get("Dirichlet_energy_diffs")).reshape(X0.shape[0])
x_vals = jnp.asarray(results.get("x_vals"))
cos_coeffs = jnp.asarray(results.get("cos_coeffs"))
sin_coeffs = jnp.asarray(results.get("sin_coeffs"))

## doing Bayes opt
# instantiating the model and fitting a RBF GP to the data using maximum-likelihood-estimation
exp = Experiment(X0, y0, objective=objective)
# running Bayesian Optimisation with "probability of improvement" (PI) acquisition func, maximised by random search
exp.run_bayes_opt(num_iters=2, dynamic_plot=True)
