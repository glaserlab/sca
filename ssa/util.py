############ Import packages

import numpy as np
import numpy.random as npr
import copy
import time

import torch



############ Utilities

def get_sample_weights(Y,eps=.1):

    """
    Function for getting weights for each sample (So different time points can be upweighted/downweighted in the cost function)
    The weights are inversely related to the norm of the activity across all neurons

    Parameters
    ----------
    Y: neural data
        numpy 2d array of shape [n_time,n_output_neurons]
    eps: a small offset that limits the maximal sample weight of a time point (in the scenario there is zero activity)
        scalar

    Returns
    -------
    The sample weights - an array of shape [n_time,1]

    """


    tmp=1/(np.sqrt(np.sum(Y**2,axis=1))+eps)
    tmp2=tmp/np.mean(tmp)
    return tmp2[:,None]


def torchify(array_list):

    """
    Function that turns a list of arrays into a list of torch tensors.

    Parameters
    ----------
    array_list: a list of numpy arrays

    Returns
    -------
    a list of torch tensors (corresponding to the original arrays)

    """

    return [torch.FloatTensor(array_entry) for array_entry in array_list]
