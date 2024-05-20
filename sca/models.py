############ Import packages

import numpy as np
import numpy.random as npr
import copy
import time

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import orth as orthog
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

import geotorch

from sca.util import torchify
from sca.loss_funcs import my_loss, my_loss_norm
from sca.architectures import LowROrth, LowRNorm

from tqdm import tqdm



############ Functions for initialization

class WeightedPCA(object):   ### ADD attributes section at beginning


    """
    Class for Weighted PCA

    Parameters
    ----------
    n_components: dimensionality (rank) of PCA projection
        scalar


    Attributes
    ----------
    components_: the model loadings
        numpy 2d array of size [n_components,n_neurons]
    params: This contains the model parameters in the format/notation used by SCA
        dictionary
    params['U']:
        numpy 2d array of size [n_neurons,n_components]
    params['V']:
        numpy 2d array of size [n_components,n_neurons]
    """

    def __init__(self,n_components=None):

         self.n_components = n_components


    def fit(self,X,sample_weight=1):

        """
        Fit weighted PCA model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_neurons]

        sample_weight: weighting of each sample
            numpy 2d array of shape [n_time,1]


        Returns
        ----------
        self : the instance itself
            object
        """

        Xw=X*sample_weight
        svd=TruncatedSVD(self.n_components)
        svd.fit(Xw)
        self.components_ = svd.components_

        self.params={}
        self.params['U']=svd.components_.T
        self.params['V']=svd.components_


        return self

        # return svd.components_.T, svd.components_

    def transform(self,X):

        """
        Get latents (low-dimensional representation) of neural data X from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_neurons]

        Returns
        -------
        Latents (low-dimensional representation) of input data
            numpy 2d array of shape [n_time,n_components]
        """

        return X@self.params['U']


    def fit_transform(self,X,sample_weight=1):

        """
        Fit weighted Pca model and then get latents (low-dimensional representation) of data X from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_neurons]

        sample_weight: weighting of each sample
            numpy 2d array of shape [n_time,1]

        Returns
        -------
        Latents (low-dimensional representation) of input data
            numpy 2d array of shape [n_time,n_components]
        """

        self.fit(X,sample_weight)
        return self.transform(X)


    def reconstruct(self,X):

        """
        Get reconstructed neural data from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_neurons]

        Returns
        -------
        Xhat: reconstructed neural data
            numpy 2d array of shape [n_time,n_neurons]
        """

        Xhat = X@self.params['U']@self.params['V']

        return Xhat



class WeightedRRR(object):  ### DOUBLE CHECK THIS NEW VERSION!!! (that it gives same results as old version)

    """
    Class for weighted reduced rank regression


    Parameters
    ----------
    n_components: dimensionality (rank) of bottleneck in regression
        scalar
    ridge: regularization amount for ridge regression
        scalar

    Attributes
    ----------

    params: This contains the model parameters in the format/notation used by SCA
        dictionary
    params['U']:
        numpy 2d array of size [n_neurons,n_components]
    params['V']:
        numpy 2d array of size [n_components,n_neurons]

    beta - the weights of the reduced rank regression
    beta can be written as beta = UV, where U is size [n_input_neurons, R] and V is size [R, n_output_neurons]
    b_est - the offset of the reduced rank regression

    """

    def __init__(self,n_components=None,ridge=.1):

         self.n_components = n_components
         self.ridge = ridge


    def fit(self,X,Y,sample_weight=1):

        """
        Fit weighted RRR model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_input_neurons]

        Y: neural data output
            numpy 2d array of shape [n_time,n_output_neurons]

        sample_weight: weighting of each sample
            numpy 2d array of shape [n_time,1]


        Returns
        ----------
        self : the instance itself
            object
        """


        if self.ridge==0:
           lr=LinearRegression()
        else:
           lr=Ridge(alpha=self.ridge)

        lr.fit(X,Y,sample_weight=sample_weight.flatten())

        beta0=lr.coef_.T
        b_est=lr.intercept_
        # svd_comp , __ = weighted_pca(X@beta0,R,sample_weight)
        wpca=WPCA(self.n_components)
        wpca.fit(X@beta0,sample_weight)
        beta=beta0@wpca.params['U']@wpca.params['V']

        self.beta = beta
        self.b_est = b_est
        self.params={}
        self.params['U']=beta@wpca.params['U']
        self.params['V']=wpca.params['V']


        # svd_comp , __ = weighted_pca(X@beta0,R,sample_weight)
        # beta=beta0@svd_comp@svd_comp.T
        return beta, beta@svd_comp, svd_comp.T, b_est


    def transform(self,X):

        """
        Get latents (low-dimensional representation) of neural data X from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_input_neurons]

        Returns
        -------
        Latents (low-dimensional representation) of input data
            numpy 2d array of shape [n_time,n_components]
        """

        return X@self.params['U']


    def fit_transform(self,X,Y,sample_weight=1):

        """
        Fit weighted RRR model and then get latents (low-dimensional representation) of data X from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_input_neurons]

        Y: neural data output
            numpy 2d array of shape [n_time,n_output_neurons]

        sample_weight: weighting of each sample
            numpy 2d array of shape [n_time,1]

        Returns
        -------
        Latents (low-dimensional representation) of input data
            numpy 2d array of shape [n_time,n_components]
        """

        self.fit(X,Y,sample_weight)
        return self.transform(X)


    def reconstruct(self,X):

        """
        Get predicted output neural data from fit model

        Parameters
        ----------
        X: neural data
            numpy 2d array of shape [n_time,n_input_neurons]

        Returns
        -------
        Yhat: reconstructed output neural data
            numpy 2d array of shape [n_time,n_output_neurons]
        """

        return X@self.params['U']@self.params['V']



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
    U and V can serve as initializations for SCA (for SCA's U and V variables)
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





class SCA(object):

    """
    Class for Sparse Component Analysis Model


    Parameters
    ----------
    n_components: dimensionality of the latent (required)
        scalar
    lam_sparse: sparsity penalty weight (optional)
        scalar
        Will default during model fitting so that the initial sparsity penalty (based on PCA or RRR initialization) is 10% of the reconstruction error
    lr: learning rate (optional)
        scalar
        Will default to 0.001
    n_epochs: number of training epochs (optional)
        scalar
        Will default to 3000
    orth: whether to constrain the V matrix to be strictly orthogonal (optional)
        boolean
        Default is False
    lam_orthog: penalty weight for V matrix deviating from orthogonality, to be used if orth=False
        scalar
        Will default in model fitting so that the orthogonality penalty would be 10% of the PCA/RRR squared error if all off-diag values of V.T@V were 0.1
    init: initialization scheme (optional)
        string
        For single population, can be 'pca' or 'rand', and defaults to 'pca'.
        For two-population, can be 'rrr' or 'rand', and defaults to 'rrr'


        Attributes
        -------
        model: the pytorch model that was fit
        losses: the model loss for each epoch
            list of length n_epochs
        explained_squared_activity: The amount of squared neural activity that each latent explains
            array of length n_components

        reconstruction_loss: the reconstruction loss term in the cost function (the weighted sum squared error)
            scalar
        r2_score: the r2 value of the model fit. Neurons are weighted by their amount of variance, and sample-weighting is used
            scalar
        params: This contains the relevant model parameters, described below
            dictionary
        params['U']:
            numpy 2d array of size [n_input_neurons,n_components]
        params['b_u']:
            numpy 1d array of size [n_components]
        params['V']:
            numpy 2d array of size [n_components,n_output_neurons]
        params['b_v']:
            numpy 2d array of size [n_output_neurons]


    """


    def __init__(self,n_components=None,lam_sparse=None,lr=None,n_epochs=3000,orth=False,lam_orthog=None,init=None,scheduler_params_input=dict()):

         self.n_components = n_components
         self.lam_sparse=lam_sparse
         self.lam_orthog=lam_orthog
         self.lr=lr
         self.n_epochs=n_epochs
         self.orth=orth
         self.init=init
         self.scheduler_params_input=scheduler_params_input


    def fit_transform(self,X,Y=None,sample_weight=None):


        """
        Fit SCA model and get latents (low-dimensional representation) of neural data X

        Parameters
        ----------
        X: the input neural data (required)
            numpy 2d array of size [n_time, n_input_neurons]
        Y: the output neural data (optional)
            numpy 2d array of size [n_time, n_output_neurons]
            This does not need to be included if doing dim. reduction on one neural population
        sample_weight: weighting of each sample (optional)
            numpy 2d array of size [n_time, 1]
            If this argument is not used, will default to no weighting

        Returns
        -------
        latent: the low dimensional representation
            2d torch tensor of size [n_time, n_components]
        """


        #Require the dimensionality
        if self.n_components is None:
            raise Exception("Error: you must include a value for n_components, the number of low-dimensional components")

        #If sample_weight is None, don't scale weight function (just make sample_weight) an array of ones
        if sample_weight is None:
            sample_weight=np.ones([X.shape[0],1])

        #Include input scheduler params
        self.scheduler_params={'use_scheduler': True, 'factor': .5, 'min_lr': 5e-4, 'patience': 100, 'threshold':1e-6, 'threshold_mode':'rel'}
        for key in self.scheduler_params_input.keys():
            self.scheduler_params[key]=self.scheduler_params_input[key]


        #Initialize weights
        #Default, if we only have X, is initializing with PCA, and if we have Y, initializing with RRR
        #Initializing randomly is also an option

        #Get weights when initializing with PCA and RRR
        if Y is None:
            b_est=np.zeros(X.shape[1])
            # U_est_pca,V_est_pca = weighted_pca(X,self.n_components,sample_weight)
            wpca=WeightedPCA(self.n_components)
            wpca.fit(X,sample_weight)
            U_est_pca,V_est_pca = wpca.params['U'], wpca.params['V']
        else:
            __,U_est_rrr,V_est_rrr, b_est = weighted_rrr(X,Y,self.n_components,sample_weight)

        #Initialize weights (either with PCA/RRR weights from above, or randomly)
        if Y is None:
            if self.init is None or self.init=='pca':
                U_est=np.copy(U_est_pca)
                V_est=np.copy(V_est_pca)
            elif self.init=='rand':
                U_est = orthog(npr.randn(X.shape[1],R))
                V_est=U_est.T
            else:
                raise Exception("Invalid initialization: options are 'pca' or 'rand' ")
        else:
            if self.init is None or self.init=='rrr':
                U_est=np.copy(U_est_rrr)
                V_est=np.copy(V_est_rrr)
            elif self.init=='rand':
                U_est = orthog(npr.randn(X.shape[1],R))
                V_est = npr.randn(R,Y.shape[1])
            else:
                raise Exception("Invalid initialization: options are 'rrr' or 'rand' ")


        #Set default learning rate:
        #.001 if initializing weights w/ PCA/RRR and .01 if random weights
        if self.lr is None:
            if self.init=='rand':
                self.lr=.01
            else:
                self.lr=.001

        #Set default lam_sparse:
        #It is set so that the initial sparsity penalty (based on PCA or RRR initialization) is 10% of the reconstruction error
        if self.lam_sparse is None:
            if Y is None:
                pca_latent = X@U_est_pca
                pca_recon=pca_latent@V_est_pca
                self.lam_sparse = .1*np.sum((X-pca_recon)**2)/np.sum(np.abs(pca_latent))
                print('Using lam_sparse= ', self.lam_sparse)
            else:
                rrr_latent = X@U_est_rrr
                rrr_recon=rrr_latent@V_est_rrr
                self.lam_sparse = .1*np.sum((Y-rrr_recon)**2)/np.sum(np.abs(rrr_latent))
                print('Using lam_sparse= ', self.lam_sparse)

        #Set default lam_orthog:
        #It is set so that the orthogonality penalty would be 10% of the PCA/RRR squared error if all off-diag values of V.T@V were 0.1
        if self.orth is False:
            if self.lam_orthog is None:
                if self.n_components==1:
                    self.lam_orthog=0
                else:
                    if Y is None:
                        pca_recon=X@U_est@V_est
                        self.lam_orthog = .1*np.sum((X-pca_recon)**2)/np.sum(self.n_components*(self.n_components-1)*.01)
                    else:
                        rrr_recon=X@U_est@V_est
                        self.lam_orthog = .1*np.sum((Y-rrr_recon)**2)/np.sum(self.n_components*(self.n_components-1)*.01)
                print('Using lam_orthog= ', self.lam_orthog)

        #To make the rest generic, we will predict Y from X, where Y=X in the scenario that Y has not been input
        if Y is None:
            Y=X

        # move to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device", device)

        #Declare the model and optimizer
        if self.orth:
            model = LowROrth(X.shape[1], Y.shape[1], self.n_components, U_est.T, b_est).to(device)
        else:
            model = LowRNorm(X.shape[1], Y.shape[1], self.n_components, U_est.T, b_est).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)

        #Initialize V in the model
        model.fc2.weight = torch.tensor(V_est.T, dtype=torch.float).to(device)

        #Create torch tensors of our variables
        [X_torch,Y_torch,sample_weight_torch] = torchify([X,Y,sample_weight])

        # scheduler = ReduceLROnPlateau(optimizer, patience=100, factor=schedule_factor, min_lr=5e-4, threshold=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.scheduler_params['patience'], factor=self.scheduler_params['factor'], min_lr=self.scheduler_params['min_lr'], threshold=self.scheduler_params['threshold'], threshold_mode=self.scheduler_params['threshold_mode'])

        #Get initial model loss before training
        model.eval()
        latent, y_pred = model(X_torch)
        if self.orth:
            before_train = my_loss(y_pred, Y_torch, latent, self.lam_sparse, sample_weight_torch)
        else:
            before_train = my_loss_norm(y_pred, Y_torch, latent, model.fc2.weight, self.lam_sparse, self.lam_orthog, sample_weight_torch)

        #Fit the model!

        # t1=time.time()
        losses=np.zeros(self.n_epochs+1) #Save loss at each training epoch
        losses[0]=before_train.item()

        model.train()
        for epoch in tqdm(range(self.n_epochs), position=0, leave=True):
            optimizer.zero_grad()
            # Forward pass
            latent, y_pred = model(X_torch)
            # Compute Loss
            if self.orth:
                loss = my_loss(y_pred, Y_torch, latent, self.lam_sparse, sample_weight_torch)
            else:
                loss = my_loss_norm(y_pred, Y_torch, latent, model.fc2.weight, self.lam_sparse, self.lam_orthog, sample_weight_torch)
            losses[epoch+1]=loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            if self.scheduler_params['use_scheduler']:
                scheduler.step(loss.item())
        # print('time',time.time()-t1)

        # Include attributes as part of self
        self.model=model
        self.losses=losses

        self.params={}
        self.params['U']=model.fc1.weight.detach().cpu().numpy().T
        self.params['b_u']=model.fc1.bias.detach().cpu().numpy()
        self.params['V']=model.fc2.weight.detach().cpu().numpy().T
        self.params['b_v']=model.fc2.bias.detach().cpu().numpy()

        self.r2_score=r2_score(Y,y_pred.detach().cpu().numpy(),sample_weight=sample_weight,multioutput='variance_weighted')
        self.reconstruction_loss=np.sum((sample_weight*(y_pred.detach().cpu().numpy() - Y))**2)

        #Explained squared activity
        sq_activity=[np.sum((latent[:,i:i+1].detach().cpu().numpy()@model.fc2.weight[:,i:i+1].detach().cpu().numpy().T)**2) for i in range(self.n_components)]
        self.explained_squared_activity = np.array(sq_activity)

        return latent.detach().cpu().numpy()




    def fit(self,X,Y=None,sample_weight=None):

        """
        Fit SCA model of neural data X (and Y if finding shared subspace)

        Parameters
        ----------
        X: the input neural data (required)
            numpy 2d array of size [n_time, n_input_neurons]
        Y: the output neural data (optional)
            numpy 2d array of size [n_time, n_output_neurons]
            This does not need to be included if doing dim. reduction on one neural population
        sample_weight: weighting of each sample (optional)
            numpy 2d array of size [n_time, 1]
            If this argument is not used, will default to no weighting

        Returns
        -------
        self : the instance itself
            object
        """


        latent=self.fit_transform(X,Y,sample_weight)
        return self


    def transform(self,X):

        """
        Get latents (low-dimensional representation) of neural data X from fit model

        Parameters
        ----------
        X: the input neural data (required)
            numpy 2d array of size [n_time, n_input_neurons]


        Returns
        -------
        latent: the low dimensional representation
            2d torch tensor of size [n_time, n_components]

        """

        [X_torch] = torchify([X])
        latent, y_pred = self.model(X_torch)
        return latent.detach().cpu().numpy()



    def reconstruct(self,X):

        """
        Get reconstructed neural data from fit model

        Parameters
        ----------
        X: the input neural data (required)
            numpy 2d array of size [n_time, n_input_neurons]


        Returns
        -------
        Xhat: reconstructed neural data
            numpy 2d array of shape [n_time,n_neurons]
        """


        [X_torch] = torchify([X])
        latent, y_pred = self.model(X_torch)
        return y_pred.detach().cpu().numpy()
