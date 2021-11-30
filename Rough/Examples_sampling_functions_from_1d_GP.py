# Created 02/11/2021
# Sampling from GP with mean zero and given kernel

import numpy as np
from numpy.random import multivariate_normal as normal
#from scipy.stats import multivariate_normal as normal
from AcquisitionFuncs import RBF, Periodic, Matern
import matplotlib.pyplot as plt

kernel_type='RBF'

num_funcs = 3
num_test_points = 1000
test_points = np.asarray(np.arange(0, 1, 1 / num_test_points))
lengthscale = 0.01
stdev = 0.1

if kernel_type=='RBF':
    kernel = RBF(stdev, lengthscale)
    test_points_alt = test_points.reshape((num_test_points, 1))
    cov_matrix = kernel(test_points_alt, test_points_alt)

func_evals = normal(np.zeros(num_test_points), cov_matrix, size=num_funcs)

plt.plot(test_points, np.zeros(num_test_points), color='blue')
plt.fill_between(test_points, - stdev, stdev, alpha=0.4, color='green')
for i in range(num_funcs):
    plt.plot(test_points, func_evals[i, :])