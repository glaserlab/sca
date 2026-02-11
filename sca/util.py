############ Import packages

import numpy as np
import numpy.random as npr
import copy
import time

import torch
from sklearn.metrics import r2_score



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


def sort_dims(sca, X, rule="activity"):
    """
    Reorder SCA dimensions by activity or timing.
    
    Parameters
    ----------
    sca : SCA object
        Fitted SCA model with params containing U and V matrices
    X : array-like
        Neural activity data. Shape depends on the sorting rule:
        - For "activity" and "time" rules: shape [n_time, n_neurons]
        - For "condition-time" rule: shape [n_time_per_condition, n_conditions, n_neurons]
          (typically created by reshaping concatenated data as np.reshape(data, (-1, n_conditions, n_neurons), order='F'))
    rule : str, default="activity"
        Sorting rule:
        - "activity": 
          Order by total activity (sum of squared latents over time) in descending order.
          Requires X of shape [n_time, n_neurons].
          Most active dimensions first.
          
        - "time": 
          Order by peak activity timing in ascending order.
          Requires X of shape [n_time, n_neurons].
          Earliest peaks in the timeseries first.
          
        - "condition-time":
          Order by time of peak across-condition variance in ascending order.
          Requires X of shape [n_time_per_condition, n_conditions, n_neurons].
          This computes variance across conditions at each timepoint, then finds when
          each dimension shows maximum across-condition variance.
          Useful for multi-condition experiments where task structure matters.
    
    Returns
    -------
    sca_new : SCA object
        New SCA object with reordered dimensions (U and V matrices reordered)
    sort_indices : array
        Indices of the original dimensions in the new order (length n_components)
    """
    
    if rule not in ["activity", "time", "condition-time"]:
        raise ValueError(f"rule must be 'activity', 'time', or 'condition-time', got '{rule}'")
       
    # Validate that required parameters exist
    if not hasattr(sca, 'params'):
        raise ValueError(f"Model must have a 'params' attribute, sort_dims does not support models without params including SCANonlinear.")
    required_params = ['U', 'V']
    missing_params = [param for param in required_params if param not in sca.params]
    if missing_params:
        raise ValueError(f"Model params must contain {required_params}. Missing: {missing_params}.")
    
    if rule == "condition-time":
        # X should be 3D: [n_time_per_condition, n_conditions, n_neurons]
        if X.ndim != 3:
            raise ValueError(f"For rule='condition-time', X must be 3D with shape [n_time_per_condition, n_conditions, n_neurons], got shape {X.shape}")
        
        # Reshape to 2D for latent computation: [n_time_per_condition * n_conditions, n_neurons]
        n_time_per_cond, n_conds, n_neurons = X.shape
        X_flat = X.reshape(-1, n_neurons)
        
        # Compute latents
        latents = X_flat @ sca.params['U']  # shape: [n_time_per_condition * n_conditions, n_components]
        
        # Reshape back to 3D: [n_time_per_condition, n_conditions, n_components]
        latents_3d = latents.reshape(n_time_per_cond, n_conds, -1)
        
        # Calculate across-condition variance at each timepoint
        # np.var along axis 1 gives variance across conditions
        cond_variance = np.var(latents_3d, axis=1)  # shape: [n_time_per_condition, n_components]
        
        # Find timepoint with maximum across-condition variance for each dimension
        peak_times = np.argmax(cond_variance, axis=0)  # shape: [n_components]
        sort_indices = np.argsort(peak_times)  # ascending order
        
    else:
        # For "activity" and "time" rules, X should be 2D: [n_time, n_neurons]
        if X.ndim != 2:
            raise ValueError(f"For rule='{rule}', X must be 2D with shape [n_time, n_neurons], got shape {X.shape}")
        
        # Compute latents from the data using params directly
        latents = X @ sca.params['U']  # shape: [n_time, n_components]
        
        if rule == "activity":
            # Compute the magnitude of activity for each dimension
            # Activity = sum of squared latent values over time
            activity = np.sum(latents ** 2, axis=0)
            sort_indices = np.argsort(activity)[::-1]  # descending order
            
        elif rule == "time":
            # Find the time of peak activity for each dimension
            peak_times = np.argmax(np.abs(latents), axis=0)  # Get time index of max absolute value
            sort_indices = np.argsort(peak_times)  # ascending order
    
    # Create a new SCA object with reordered dimensions
    sca_new = copy.deepcopy(sca)
 
    # Reorder U matrix: [n_neurons, n_components] -> sort along axis 1
    sca_new.params['U'] = sca_new.params['U'][:, sort_indices]
    # Reorder V matrix: [n_components, n_neurons] -> sort along axis 0
    sca_new.params['V'] = sca_new.params['V'][sort_indices, :]
    # Reorder b_u bias terms
    if 'b_u' in sca_new.params:
        sca_new.params['b_u'] = sca_new.params['b_u'][sort_indices]
    
    # Override transform method to use reordered params directly
    def new_transform(X_input):
        return X_input @ sca_new.params['U']
    sca_new.transform = new_transform

    return sca_new, sort_indices


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
    # use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return [torch.tensor(array_entry, dtype=torch.float).to(device) for array_entry in array_list]


def get_accuracy(self,X,sample_weight=None):

    Xhat = self.reconstruct(X)
    if sample_weight is None:
        r2=r2_score(X,Xhat,multioutput='variance_weighted')
        reconstruction_loss=np.sum(((Xhat - X))**2)
    else:
        r2=r2_score(X,Xhat,sample_weight=sample_weight,multioutput='variance_weighted')
        reconstruction_loss=np.sum((sample_weight*(Xhat - X))**2)

    return [r2,reconstruction_loss]
