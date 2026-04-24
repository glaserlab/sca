### Import stuff

import numpy as np
import itertools
import sklearn


''''
given a tensor of latents, and three time vectors, calculate the fractional occupancy of this dimension
during preparation, execution, and posture epochs. 

# inputs: proj: T x C x K tensor of latents
          prepTimes: vector of times corresponding to prep. epoch/epochs 
          moveTimes/postTimes: the same for execution and posture periods
          projTimes: vector of times of length T that will be used to index into proj 
'''

def calculateEpochOccupancy(proj,prepTimes,moveTimes,postTimes,projTimes):

    # add all the times to a list for convenience
    times = [prepTimes, moveTimes, postTimes]

    # initialize a K x 3 matrix to hold results
    timeL, numConds, numDims = proj.shape
    fractOcc = np.zeros((numDims,3))

    # calculate the total occupancy for each dimension
    totalOcc = np.sum(np.var(proj, axis=1), axis=0)

    # cycle through each dimension and each epoch
    for dd in range(numDims):
        for ee in range(3):

            # grab the projections from the times that we care about
            tempTimes = np.isin(projTimes,times[ee])
            tempLat = proj[tempTimes,:,dd]

            # calculate cross-condition variance
            fractOcc[dd, ee] = np.sum(np.var(tempLat, axis=1), axis=0) / totalOcc[dd]

    # return output
    return fractOcc


''''
Given a matrix of fractional occupancies, calculate the sum of the absolute difference between fractional occupancies

# inputs: occMat: K x p matrix of occupancies, where K is number of dimensions, and p is number of epochs
(may or may not be normalized to sum to 1 in each dimension) 

'''

def calculateOccDispersion(occMat):

    # get number of dimensions and number of epochs
    numDims, numEpochs = occMat.shape

    # initialize a vector to hold results
    dimDispersion = np.zeros(numDims)

    # make a list of all the pairwise differences we need to calculate
    pairs = list(itertools.combinations(list(range(numEpochs)), 2))

    # cycle through dimensions
    for dd in range(numDims):

        # grab the occupancies for this dimension
        tempOcc = occMat[dd, :]

        # initialize the sum for this dimension
        dimSum = 0

        # cycle through difference
        for ii in range(len(pairs)):
            dimSum += abs(tempOcc[pairs[ii][0]] - tempOcc[pairs[ii][1]])

        # add to matrix
        dimDispersion[dd] = np.copy(dimSum)

    return dimDispersion


'''

given two CT x N matrix of rates and a number of dimensions calculate the alignment index 
input: 
X1/2 - CT x N matrix of rates (assumed to be preprocessed, e.g., mean subtracted/normalized) 
numDims - scalar corresponding to the number of dimensions you want to use to calculate alignment index 

I'm making the arbitrary choice to calculate the alignment index by projecting data from X2 into the space defined by X1.
'''
def calculateAI(X1,X2,numDims):

    # calculate the covariance matrix of X2
    X2_center = X2 - np.mean(X2, axis=0)
    C2 = (X2_center.T @ X2_center) / (X2_center.shape[0]-1)

    # run pca on X1 and X2
    pca1 = sklearn.decomposition.PCA(n_components=numDims)
    pca1.fit(X1)

    pca2 = sklearn.decomposition.PCA(n_components=numDims)
    pca2.fit(X2)

    # grab the pcs from X1 (spits out a k x N matrix by default)
    W = pca1.components_.T

    # calculate the variance explained by the top 'nDims' PCs of X2
    totVar = sum(pca2.explained_variance_)

    # calculate alignment index
    AI = np.trace((W.T @ C2 @ W)) / totVar

    return AI

# TODO: test test test
'''

calculate the alignment index that can be expected given the dimensionality of a given data matrix
input: 
X - CT x N matrix of rates (assumed to be preprocessed, e.g., mean subtracted/normalized) 
numDims - scalar corresponding to the number of dimensions you want to use to calculate alignment index 
numReps: number of times to repeat analysis 

copying the procedure used in Elsayed et al. (2016)
'''
def calculateChanceAI(X,numDims,numReps = 1000):

    # calculate covariance of data
    C = np.cov(X.T)

    # calculate the right eigenvectors and eigenvalues of covariance matrix
    W,U = np.linalg.eig(C)

    # make our eigenvalues a matrix
    W = np.diag(W)

    # calculate how much to weight each direction in the original
    V = U @ np.sqrt(W)

    # number of neurons
    numN = X.shape[1]

    # initialize a vector to hold results
    AI = np.zeros(numReps)

    # cycle through repetitions
    for ii in range(numReps):

        # draw dimensions from a standard normal distribution
        rand1 = np.random.randn(numN, numDims)

        # weight our random dimensions and normalize
        v1 = V @ rand1
        v1 = v1 / np.linalg.norm(v1, ord=2)

        # calculate the first 'numDims' singular vectors of our new space
        U1 = np.linalg.svd(v1)[0][:, :numDims]

        # do all of that again to get a second space
        rand2 = np.random.randn(numN, numDims)

        v2 = V @ rand2
        v2 = v2 / np.linalg.norm(v2, ord=2)

        # calculate the first 'numDims' singular vectors of our new space
        U2 = np.linalg.svd(v2)[0][:, :numDims]

        # calculate alignment index of these two spaces
        AI[ii] = np.copy(np.trace(U2.T @ U1 @ U1.T @ U2) / numDims)

    # return distribution
    return AI







