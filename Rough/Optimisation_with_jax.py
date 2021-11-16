from Regressors import *
from Utils import plot
from myFunctions import *
from myAlgorithms import opt_acquisition
import matplotlib.pyplot as plt
import time
from numpy.random import normal


num_initial_samples = 1
acq_type = 'PI'  # only needed for optimisation example
prior_mean_func = None
kernel_type = 'Matern'


model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)