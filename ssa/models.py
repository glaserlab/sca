############ Import packages

import numpy as np
import numpy.random as npr
import copy
import time

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import TruncatedSVD

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

import geotorch

from ssa.util import torchify

from tqdm import tqdm

############ Functions for initialization

def weighted_pca(X,R,sample_weight=1):
    """
    Run weighted PCA w/ R dimensions

    Parameters
    ----------
    X: neural data
        numpy 2d array of shape [n_time,n_neurons]
    R: dimensionality (rank) of PCA projection
        scalar
    sample_weight: weighting of each sample
        numpy 2d array of shape [n_time,1]

    Returns
    -------
    U, U^T
    U is size [n_neurons, R], and projects the data down into an R-dimensional space
    U^T projects back up to the high dimensional space, and can be though of like "V" in SSA
    """

    Xw=X*sample_weight
    svd=TruncatedSVD(R)
    svd.fit(Xw)
    return svd.components_.T, svd.components_


def weighted_rrr(X,Y,R,sample_weight=1,ridge=.1):
    """
    Run weighted reduced rank regression w/ R dimensions

    Parameters
    ----------
    X: neural data input
        numpy 2d array of shape [n_time,n_input_neurons]
    Y: neural data output
        numpy 2d array of shape [n_time,n_output_neurons]
    R: dimensionality (rank) of PCA projection
        scalar
    sample_weight: weighting of each sample
        numpy 2d array of shape [n_time,1]
    ridge: regularization amount for ridge regression
        scalar

    Returns
    -------
    beta,U,V,b

    beta are the weights of the reduced rank regression
    beta can be written as beta = UV, where U is size [n_input_neurons, R] and V is size [R, n_output_neurons]
    U and V can serve as initializations for SSA (for SSA's U and V variables)
    b is the offset of the reduced rank regression

    """



    if ridge==0:
       lr=LinearRegression()
    else:
       lr=Ridge(alpha=ridge)

    lr.fit(X,Y,sample_weight=sample_weight.flatten())

    beta0=lr.coef_.T
    b_est=lr.intercept_
    svd_comp , __ = weighted_pca(X@beta0,R,sample_weight)
    beta=beta0@svd_comp@svd_comp.T
    return beta, beta@svd_comp, svd_comp.T, b_est





############ Model fitting

# Note - I would like to credit the pytorch tutorial, that the formatting of my functions is similar to:
# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb


class LowROrth(nn.Module):
    """
    Class for SSA model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowROrth, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b
        geotorch.orthogonal(self.fc2,"weight") #Make V orthogonal

    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output



class Sphere(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)
    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)

class LowRNorm(nn.Module):
    """
    Class for SSA (with unit norm, but not orthogonal) model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowRNorm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias

        self.fc2 = P.register_parametrization(nn.Linear(self.hidden_size, self.output_size), "weight", Sphere(dim=0))
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b


    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output





def my_loss(output, target, latent, lam_sparse, sample_weight):

    """
    Loss function

    Parameters
    ----------
    output: the predictions
        torch 2d tensor of size [n_time, output_size]
    target: ground truth output
        torch 2d tensor of size [n_time, output_size]
    latent: low dimensional representations
        torch 2d tensor of size [n_time, hidden_size]
    lam_sparse: sparsity penalty weight
        scalar
    sample_weight: weighting of each sample
        torch 2d tensor of size [n_time, 1]


    Returns
    -------
    loss: the value of the cost function, a scalar
    """

    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent))
    return loss



def my_loss_norm(output, target, latent, V, lam_sparse, lam_orthog, sample_weight):

    """
    Loss function when using orthogonality penalty instead of constraint

    Parameters
    ----------
    output: the predictions
        torch 2d tensor of size [n_time, output_size]
    target: ground truth output
        torch 2d tensor of size [n_time, output_size]
    latent: low dimensional representations
        torch 2d tensor of size [n_time, hidden_size]
    lam_sparse: sparsity penalty weight
        scalar
    lam_orthog: orthogonality regularization weight
        scalar
    sample_weight: weighting of each sample
        torch 2d tensor of size [n_time, 1]


    Returns
    -------
    loss: the value of the cost function, a scalar
    """

    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent)) + lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1]))**2
    return loss


def fit_ssa(X,Y=None,R=None,sample_weight=None,lam_sparse=None,lr=0.001,n_epochs=3000,orth=True,lam_orthog=None,scheduler_params_input=dict()):

    """
    Wrapper function for fitting the SSA model

    Parameters
    ----------
    X: the input neural data (required)
        numpy 2d array of size [n_time, n_input_neurons]
    Y: the output neural data (optional)
        numpy 2d array of size [n_time, n_output_neurons]
        This does not need to be included if doing dim. reduction on one neural population
    R: dimensionality of the latent (required)
        scalar
    sample_weight: weighting of each sample (optional)
        numpy 2d array of size [n_time, 1]
        If this argument is not used, will default to no weighting
    lam_sparse: sparsity penalty weight (optional)
        scalar
        Will default so that the initial sparsity penalty (based on PCA or RRR initialization) is 10% of the reconstruction error
    lr: learning rate (optional)
        scalar
        Will default to 0.001
    n_epochs: number of training epochs (optional)
        scalar
        Will default to 3000
    orth: whether to constrain the V matrix to be orthogonal (optional)
        boolean
        Default is True
    lam_orthog: penalty weight for V matrix deviating from orthogonality, to be used if orth=False
        scalar
        Will default so that the orthogonality penalty would be 10% of the PCA/RRR squared error if all off-diag values of V.T@V were 0.1

    Returns
    -------
    model: the pytorch model that was fit
    latent: the low dimensional representation
        2d torch tensor of size [n_time, R_est]
    y_pred: the output predictions
        2d torch tensor of size [n_time, n_output_neurons]
    """



    #Require the dimensionality
    if R is None:
        raise Exception("Error: you must include a value for R, the rank")

    #If scale_array is None, don't scale weight function (just make scale_array) an array of ones
    if sample_weight is None:
        sample_weight=np.ones([X.shape[0],1])

    #Include input scheduler params
    scheduler_params={'use_scheduler': True, 'factor': .5, 'min_lr': 5e-4, 'patience': 100, 'threshold':1e-6, 'threshold_mode':'rel'}
    for key in scheduler_params_input.keys():
        scheduler_params[key]=scheduler_params_input[key]


    #Initialize with PCA if we only have X. Otherwise initialize with RRR
    if Y is None:
        b_est=np.zeros(X.shape[1])
        U_est,V_est = weighted_pca(X,R,sample_weight)
    else:
        __,U_est,V_est, b_est = weighted_rrr(X,Y,R,sample_weight)

    #Set default lam_sparse:
    #It is set so that the initial sparsity penalty (based on PCA or RRR initialization) is 10% of the reconstruction error
    if lam_sparse is None:
        if Y is None:
            pca_latent = X@U_est
            pca_recon=pca_latent@V_est
            lam_sparse = .1*np.sum((X-pca_recon)**2)/np.sum(np.abs(pca_latent))
            print('Using lam_sparse= ', lam_sparse)
        else:
            rrr_latent = X@U_est
            rrr_recon=rrr_latent@V_est
            lam_sparse = .1*np.sum((Y-rrr_recon)**2)/np.sum(np.abs(rrr_latent))
            print('Using lam_sparse= ', lam_sparse)

    #Set default lam_orthog:
    #It is set so that the orthogonality penalty would be 10% of the PCA/RRR squared error if all off-diag values of V.T@V were 0.1
    if orth is False:
        if lam_orthog is None:
            if Y is None:
                pca_recon=X@U_est@V_est
                lam_orthog = .1*np.sum((X-pca_recon)**2)/np.sum(R*(R-1)*.01)
                print('Using lam_orthog= ', lam_orthog)
            else:
                rrr_recon=X@U_est@V_est
                lam_orthog = .1*np.sum((Y-rrr_recon)**2)/np.sum(R*(R-1)*.01)
                print('Using lam_orthog= ', lam_orthog)

    #To make the rest generic, we will predict Y from X, where Y=X in the scenario that Y has not been input
    if Y is None:
        Y=X

    # move to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    #Declare the model and optimizer
    if orth:
        model = LowROrth(X.shape[1], Y.shape[1], R, U_est.T, b_est).to(device)
    else:
        model = LowRNorm(X.shape[1], Y.shape[1], R, U_est.T, b_est).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    #Initialize V in the model
    model.fc2.weight = torch.tensor(V_est.T, dtype=torch.float).to(device)

    #Create torch tensors of our variables
    [X_torch,Y_torch,sample_weight_torch] = torchify([X,Y,sample_weight])

    # scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=schedule_factor, min_lr=5e-4, threshold=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, patience=scheduler_params['patience'], factor=scheduler_params['factor'], min_lr=scheduler_params['min_lr'], threshold=scheduler_params['threshold'], threshold_mode=scheduler_params['threshold_mode'])

    #Get initial model loss before training
    model.eval()
    latent, y_pred = model(X_torch)
    if orth:
        before_train = my_loss(y_pred, Y_torch, latent, lam_sparse, sample_weight_torch)
    else:
        before_train = my_loss_norm(y_pred, Y_torch, latent, model.fc2.weight, lam_sparse, lam_orthog, sample_weight_torch)

    #Fit the model!

    # t1=time.time()
    losses=np.zeros(n_epochs+1) #Save loss at each training epoch
    losses[0]=before_train.item()

    model.train()
    for epoch in tqdm(range(n_epochs), position=0, leave=True):
        optimizer.zero_grad()
        # Forward pass
        latent, y_pred = model(X_torch)
        # Compute Loss
        if orth:
            loss = my_loss(y_pred, Y_torch, latent, lam_sparse, sample_weight_torch)
        else:
            loss = my_loss_norm(y_pred, Y_torch, latent, model.fc2.weight, lam_sparse, lam_orthog, sample_weight_torch)
        losses[epoch+1]=loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler_params['use_scheduler']:
            scheduler.step(loss.item())
    # print('time',time.time()-t1)

    return model,latent,y_pred,losses
