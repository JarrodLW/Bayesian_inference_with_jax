# Created

import numpy as np
from myFunctions import PI_acquisition, EI_acquisition, UCB_acquisition


def opt_acquisition(acq_type, model, margin, num_samples):  # TODO allow for multiple points to be kept
    # random search, generate random samples
    Xsamples = np.random.random(num_samples)
    # calculate the acquisition function for each sample

    if acq_type=='PI':
        scores = PI_acquisition(margin, Xsamples, model)

    elif acq_type=='EI':
        scores = EI_acquisition(margin, Xsamples, model)

    # locate the index of the largest scores
    ix = np.argmax(scores)

    print("Best score " + str(np.amax(scores)))

    return Xsamples[ix]
