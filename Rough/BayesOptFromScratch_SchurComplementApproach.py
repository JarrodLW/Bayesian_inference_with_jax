# Created 08/09/2021. Mirroring the "BasicBayesOptScikitLearn" file but implementing the
# Gaussian process regressor from scratch.
# Supported acquisition functions: 'PI'
# The formulae can be found in Rasmussen and Williams

import numpy as np
from numpy.random import normal
from scipy.stats import norm
import matplotlib.pyplot as plt


# objective function
def objective(x, noise=0.):
	noise = normal(loc=0, scale=noise)
	return (x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise

# TODO adapt entire script to allow for multi-dimensional inputs

## defining kernel

class RBF():

	# \sigma^2\exp(-\Vert x - x'\Vert^2/2l**2)

	def __init__(self, stdev, lengthscale):

		self.stdev = stdev
		self.lengthscale = lengthscale

	def __call__(self, X1, X2):

		# computes the matrix of covariances of sample points X1 against sample points X2. Each of X1, X2 is a 1d numpy array

		squared_dists = -2*np.outer(X1, X2) + X1[:, None]**2 + X2**2
		covs = self.stdev**2*np.exp(-squared_dists/2*self.lengthscale**2)

		return covs

## defining surrogate

# def surrogate(model, x):
# 	# Returns the mean and std of the model (Gaussian process) at point x
# 	# catch any warning generated when making a prediction
# 	with catch_warnings():
# 		# ignore generated warnings
# 		simplefilter("ignore")
# 		return model.predict(x)

## The model (Gaussian process regressor)

class GaussianProcessReg():
	# TODO: there's no need to save covariance matrix, I think. Only need sensitivity.
	# instance is a Gaussian process model with mean-zero prior, with fit and predict methods.

	def __init__(self, kernel_type='RBF', sigma=0.1, lengthscale=10, obs_noise_stdev=0.1):

		self.mu = None
		self.std = None
		#self.covs = None
		self.sensitivity = None
		self.y = None
		self.X = None
		self.obs_noise_stdev = obs_noise_stdev

		if kernel_type=='RBF':
			self.kernel = RBF(sigma, lengthscale)

	def fit(self, Xsamples, ysamples, compute_cov=False):

		num_samples = Xsamples.shape[0]

		if compute_cov:
			covs = self.kernel(Xsamples, Xsamples)
			self.covs = covs # just for testing. Get rid of this eventually
			covs += self.obs_noise_stdev ** 2 * np.identity(num_samples)
			self.sensitivity = np.linalg.inv(covs)
			self.X = Xsamples
			self.y = ysamples

		else:
			# recompute cross covariances
			test_train_covs = self.kernel(self.X, Xsamples)

			print("Shape of new covs matrix: " + str(test_train_covs.shape))

			# compute sample covariances
			k = self.kernel(Xsamples, Xsamples)
			k += self.obs_noise_stdev ** 2*np.identity(num_samples)

			covs = np.zeros((self.covs.shape[0] + num_samples, self.covs.shape[0] + num_samples))
			covs[:-num_samples, :-num_samples] = self.covs
			covs[-num_samples, :-num_samples] = np.ndarray.flatten(test_train_covs)
			covs[:-num_samples, -num_samples] = np.ndarray.flatten(test_train_covs)
			covs[-num_samples:, -num_samples:] = k
			self.covs = covs

			# update sensitivity matrix using the Schur complement method
			schur_compl = k - np.matmul(test_train_covs.T, np.matmul(self.sensitivity, test_train_covs))
			schur_compl_inv = np.linalg.inv(schur_compl)
			block00 = self.sensitivity + np.matmul(self.sensitivity,
												   np.matmul(test_train_covs,
															 np.matmul(schur_compl_inv,
																	   np.matmul(test_train_covs.T, self.sensitivity))))

			block01 = - np.matmul(self.sensitivity, np.matmul(test_train_covs, schur_compl_inv))
			block10 = block01.T
			block11 = schur_compl_inv

			self.sensitivity = np.block([[block00, block01], [block10, block11]])

			# checking inversion of matrix
			covs_plus_noise = self.covs + self.obs_noise_stdev ** 2*np.identity(self.covs.shape[0])
			error = np.sum(np.square(np.matmul(self.sensitivity, covs_plus_noise) - np.identity(covs_plus_noise.shape[0])))
			print("error in inversion: "+str(error))

			# updating y-vector
			y_new = np.zeros(self.X.shape[0] + num_samples)
			y_new[:-num_samples] = self.y
			y_new[-num_samples:] = ysamples
			self.y = y_new

			# update x-sample vector
			X_new = np.zeros(self.X.shape[0] + num_samples)
			X_new[:-num_samples] = self.X
			X_new[-num_samples:] = Xsamples
			self.X = X_new


	def predict(self, Xsamples): #TODO generalise this to allow for multiple sampling points
		# should I be saving the mu and std to memory?

		test_train_covs = self.kernel(self.X, Xsamples)

		# print("Shape of sensitivity matrix: " + str(self.sensitivity.shape))
		# print("Shape of new covs matrix: " + str(test_train_covs.shape))

		if self.sensitivity is None:
			raise ValueError("Model not yet trained")

		# print("test-train cov shape: "+ str(self.test_train_covs.shape))
		# print("covs shape: " + str(noise_corrected_covs_inv.shape))

		pred_mu = np.matmul(test_train_covs.T, np.matmul(self.sensitivity, self.y))
		k = self.kernel(Xsamples, Xsamples)
		pred_std = k - np.matmul(test_train_covs.T, np.matmul(self.sensitivity, test_train_covs))

		return pred_mu, pred_std

## defining acquisition functions

def PI_acquisition(margin, Xsamples, model):
	# isn't there a closed-form solution for the optimal sampling point?
	# calculate the best surrogate score found so far
	yhat, _ = model.predict(model.X)
	best = np.amax(yhat) # is this the correct value to be using? Highest surrogate value versus highest observed...
	best_plus_margin = best + margin
	# calculate mean and stdev via surrogate function
	mu, std = model.predict(Xsamples)
	# calculate the probability of improvement
	probs = norm.cdf((mu - best_plus_margin) / (np.diag(std) + 1E-20))

	return probs

##  optimize the acquisition function

def opt_acquisition(model, margin, num_samples): #TODO allow for multiple points to be kept
	# random search, generate random samples
	Xsamples = np.random.random(num_samples)
	# calculate the acquisition function for each sample
	scores = PI_acquisition(margin, Xsamples, model)
	# locate the index of the largest scores
	ix = np.argmax(scores)

	return Xsamples[ix]

# sample the domain sparsely with noise as initialisation
X0 = np.random.random(5)
y0 = np.asarray([objective(x) for x in X0])
ix = np.argmax(y0)

X = X0
y = y0

# define the model
model = GaussianProcessReg(obs_noise_stdev=0.01)
# fit the model
model.fit(X, y, compute_cov=True)

## optimisation process
num_iters = 100
num_samples = 10
margin = 0.

for i in range(num_iters):
	# select the next point to sample
	x = opt_acquisition(model, margin, num_samples)

	print("Next x: " + str(x))

	# sample the point
	actual = objective(x)
	# summarize the finding
	#est, _ = model.predict(x)
	#print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# update the model
	model.fit(np.asarray([x]), np.asarray([actual]))

	print(model.predict(model.X)[0])

	print("Iter " + str(i) + " successful")

print('First best guess: x=%.3f, y=%.3f' % (X[ix], y[ix]))

ix = np.argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))

def plot(X, y, model):
	# scatter plot of inputs and real objective function
	plt.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = np.asarray(np.arange(0, 1, 0.001))
	#ysamples = []
	#yreals = []

	#for xsample in Xsamples:
	ysamples, sample_stds = model.predict(Xsamples)
	#ysamples.append(ysample)

	yreals = objective(Xsamples, noise=0.)
	#yreals.append(yreal)

	plt.plot(Xsamples, ysamples)
	plt.plot(Xsamples, yreals)
	# show the plot
	plt.show()

	return np.diag(sample_stds)

#plot(X0, y0, model)

# Xsamples = np.asarray(np.arange(0, 1, 0.001))
# y_actual = objective(Xsamples)
#
# fig, ax = plt.subplots()
# ax.plot(Xsamples, y_actual)
# ax.scatter(X[5:], y[5:])
#
# for i in range(num_iters):
#     ax.annotate(str(i), (X[5+i], y[5+i]))

plot(X, y, model)
