# Created

import numpy as np
from myFunctions import PI_acquisition, EI_acquisition, UCB_acquisition


def opt_acquisition(acq_type, model, margin, num_samples):  # TODO allow for multiple points to be kept
    # TODO allow for differing domain geometries
    domain_dim = model.domain_dim
    # random search, generate random samples
    Xsamples = np.random.random((num_samples, domain_dim))
    # calculate the acquisition function for each sample

    if acq_type=='PI':
        scores = PI_acquisition(margin, Xsamples, model)

    elif acq_type=='EI':
        scores = EI_acquisition(margin, Xsamples, model)

    #print("Scores computed successfully")

    # locate the index of the largest scores
    ix = np.argmax(scores)

    print("Best score " + str(np.amax(scores)))

    # print("index: " + str(ix))
    # print("Xsamples shape: " + str(Xsamples.shape))
    # print("new point shape: " + str(Xsamples[ix].shape))

    return Xsamples[ix].reshape(1, domain_dim)
