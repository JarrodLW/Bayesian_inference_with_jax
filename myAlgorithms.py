# Created

import numpy as np
from myFunctions import PI_acquisition, EI_acquisition, UCB_acquisition


def opt_acquisition(acq_type, model, num_samples, std_weight=1., margin=None):  # TODO allow for multiple points to be kept
    # TODO allow for differing domain geometries
    domain_dim = model.domain_dim
    # random search, generate random samples
    Xsamples = np.random.random((num_samples, domain_dim))
    # calculate the acquisition function for each sample

    if acq_type=='PI':
        scores = PI_acquisition(margin, Xsamples, model)

    elif acq_type=='EI':
        scores = EI_acquisition(margin, Xsamples, model)

    elif acq_type=='UCB':
        scores = UCB_acquisition(std_weight, Xsamples, model)

    # locate the index of the largest scores
    ix = np.argmax(scores)

    print("Best score " + str(np.amax(scores)))

    return Xsamples[ix].reshape(1, domain_dim)
