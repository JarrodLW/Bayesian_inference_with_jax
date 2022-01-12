import Experiment
import scipy.io as io
import jax.numpy as jnp
import numpy as np
import os
from time import time

save_path = "/Heat_Transfer_Example/value_communicating_file.mat"


def matlab_heat_eq_solver(X: jnp.array, save_path: str):
    # TODO: re-implement this using a matlab engine API.
    ''' X is an array encoding instances of Fourier coefficients defining the boundary function, g.

        X.shape[0]: the number of instances,
        X.shape[1]: the number of Fourier coefficients (i.e. the dimension of the BayesOpt problem). Columns
        alternate between cos coefficients and sin coefficients.

    This function runs the matlab script, solving the IBVP for each instance of Fourier coefficients and returns the
    Dirichlet energies of each of the corresponding solution at a designated time. '''

    # remap X onto Fourier coeffs; domain for BayesOpt code is [0, 1]^N, coeffs domain is [-1/sqrt(2)*N, 1/sqrt(2)*N]^N
    N = X.shape[1]
    Fourier_coeffs = (X - 0.5)*(np.sqrt(2)/N)
    cos_coeffs = Fourier_coeffs[:, ::2]
    sin_coeffs = Fourier_coeffs[:, 1::2]

    io.savemat(save_path, {'x_vals': X, 'cos_coeffs': cos_coeffs, 'sin_coeffs': sin_coeffs})  # saving Fourier coeffs etc
    os.system('/Applications/MATLAB_R2021b.app/bin/matlab -batch "run(\'Heat_Transfer_Example/cooling_problem.m\');"')  # running matlab script to solve IBVP
    mat = io.loadmat(save_path)  # grabbing objective values
    Dirichlet_energies = jnp.asarray(mat.get("Dirichlet_energies")).reshape(X.shape[0])  # grabbing resulting objective vals

    return Dirichlet_energies


t0 = time()
X0 = jnp.asarray([[0.5, 0.5, 0.5, 0.5], [0.1, -0.1, 0.2, -0.3]])
matlab_heat_eq_solver(X0, save_path)
results = io.loadmat(save_path)
t1 = time()
print("Time elapsed "+str(t1 - t0))

Dirichlet_energy_t0 = results.get("Dirichlet_energy_t0")[0, 0]
y0 = jnp.asarray(results.get("Dirichlet_energies")).reshape(X0.shape[0])
x_vals = jnp.asarray(results.get("x_vals"))
cos_coeffs = jnp.asarray(results.get("cos_coeffs"))
sin_coeffs = jnp.asarray(results.get("sin_coeffs"))

# doing Bayes opt
exp = Experiment(X0, y0)
