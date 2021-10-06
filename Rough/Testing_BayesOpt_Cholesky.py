import numpy as np
from Rough.BayesOptFromScratch_Cholesky import *

# objective function
def objective(x, noise=0.01):
    noise = normal(loc=0, scale=noise)
    return (x ** 2 * np.sin(5 * np.pi * x) ** 6.0) + noise

acq_type='PI'

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


X0 = np.random.random(1)
#X0 = np.arange(0, 1, 5)
y0 = np.asarray([objective(x) for x in X0])
ix = np.argmax(y0)

X = X0
y = y0

# define the model
model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=quadratic,
                           prior_mean_kwargs={'a': 0.5, 'b': 0, 'c': 0})
# model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01, prior_mean=linear,
#                            prior_mean_kwargs={'a': 0.5, 'b': 0})
#model = GaussianProcessReg(sigma=0.1, lengthscale=0.05, obs_noise_stdev=0.01)
# fit the model
model.fit(X, y, compute_cov=True)

num_iters = 10
num_samples = 2000

if acq_type=='EI' or 'PI':
    margin = 0.01

elif acq_type=='UCB':
    std_weight = 0.5

def plot(X, y, model):
    # scatter plot of inputs and real objective function
    plt.scatter(X, y)
    # line plot of surrogate function across domain
    Xsamples = np.asarray(np.arange(0, 1, 0.001))
    # ysamples = []
    # yreals = []

    # for xsample in Xsamples:
    ysamples, sample_covs = model.predict(Xsamples)
    # ysamples.append(ysample)

    sample_stds = np.sqrt(np.diag(sample_covs))

    yreals = objective(Xsamples, noise=0.)
    # yreals.append(yreal)

    plt.plot(Xsamples, ysamples)
    plt.plot(Xsamples, yreals)
    plt.fill_between(Xsamples, ysamples - sample_stds, ysamples + sample_stds, alpha=0.4)
    # show the plot
    plt.show()

    return np.diag(sample_stds)

for i in range(num_iters):
    # select the next point to sample
    x = opt_acquisition(acq_type, model, margin, num_samples)

    # sample the point
    actual = objective(x)
    # summarize the finding
    est, _ = model.predict(np.asarray([x]))
    print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
    # update the model
    model.fit(np.asarray([x]), np.asarray([actual]))

    plt.clf()
    plt.ylim(-0.1, 1.)
    plot(model.X, model.y, model)
    plt.pause(1e-17)
    time.sleep(2.)
    #plt.show()

    #print("Iter " + str(i) + " successful")

print('First best guess: x=%.3f, y=%.3f' % (X[ix], y[ix]))

ix = np.argmax(model.y)
print('Best Result: x=%.3f, y=%.3f' % (model.X[ix], model.y[ix]))


# plot(X0, y0, model)

# Xsamples = np.asarray(np.arange(0, 1, 0.001))
# y_actual = objective(Xsamples)
#
# fig, ax = plt.subplots()
# ax.plot(Xsamples, y_actual)
# ax.scatter(X[5:], y[5:])
#
# for i in range(num_iters):
#     ax.annotate(str(i), (X[5+i], y[5+i]))

# plt.figure()
# plot(model.X, model.y, model)

#plt.scatter(model.X, model.y)

