import numpy as np
import scipy as sp
from time import time

class RBF():

    # \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

    def __init__(self, stdev, lengthscale):
        self.stdev = stdev
        self.lengthscale = lengthscale

    def __call__(self, X1, X2):
        # computes the matrix of covariances of sample points X1 against sample points X2. Each of X1, X2 is a 1d numpy array

        squared_dists = -2 * np.outer(X1, X2) + X1[:, None] ** 2 + X2 ** 2
        covs = self.stdev ** 2 * np.exp(-squared_dists / 2 * self.lengthscale ** 2)

        return covs

n = 100000
x = np.asarray(np.arange(0, 1, 1/100))

kernel = RBF(0.1, 1)

A = kernel(x, x)

A += 0.001 * np.identity(100)

t0 = time()
L = sp.linalg.cholesky(A, lower=True)
dt = time() - t0
print("Cholesky factorisation in: " + str(dt))

np.sqrt(np.sum(np.square(np.matmul(L, L.T) - A)))/np.sqrt(np.sum(np.square(A)))

##

b = np.random.random(100)

c = sp.linalg.solve_triangular(L, b, lower=True)
a = sp.linalg.solve_triangular(L.T, c)

np.sqrt(np.sum(np.square(np.matmul(L, c) - b)))/np.sqrt(np.sum(np.square(b)))
np.sqrt(np.sum(np.square(np.matmul(L.T, a) - c)))/np.sqrt(np.sum(np.square(c)))
np.sqrt(np.sum(np.square(np.matmul(L, np.matmul(L.T, a)) - b)))/np.sqrt(np.sum(np.square(b)))

