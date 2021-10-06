# Created 06/10/2021
# To include: plotting functionality ...

import matplotlib.pyplot as plt
import numpy as np


def plot(X, y, model, objective):
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
