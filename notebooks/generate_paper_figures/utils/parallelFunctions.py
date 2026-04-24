# import modules
from sca.models import SCA, WeightedPCA
from sca.util import get_sample_weights
import numpy as np


''' 
function that samples neurons (with replacement) and runs SCA and weighted PCA using the default parameters for both

as above, the only reason this function exists is as a workaround for a known bug. 

inputs: 
R_est: number of dimensions
X: CT x N matrix of trial averaged rates 
trainMask: CT x 1 mask of times we want to use for analyses

note: X should already be pre-processed (downsampled in time, soft-normalized, mean-subtracted) 
'''
def bootstrapNeurons_SCA_PCA(R_est,X,trainMask):

    ### redraw our neurons ###

    # number of neurons
    numN = X.shape[1]

    # random index of neurons
    randIdx = np.random.randint(0, numN, size=numN)

    # grab our random neurons
    X_sample = X[:,randIdx]


    ### calculate sample weights ###
    sw = get_sample_weights(X_sample)

    ### run sca ###

    sca = SCA(n_components=R_est, orth=False)
    sca.fit(X=X_sample[trainMask, :], sample_weight=sw[trainMask])
    sca_latents = sca.transform(X_sample)

    ### run pca ###
    wpca = WeightedPCA(n_components=R_est)
    wpca.fit(X=X_sample[trainMask, :], sample_weight=sw[trainMask])
    pca_latents = wpca.transform(X_sample)

    # return latents
    return sca_latents, pca_latents