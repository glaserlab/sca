import torch

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

    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent)) + lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1], device=V.device))**2
    # print('term1:',torch.sum((sample_weight*(output - target))**2))
    # print('term2:',lam_sparse*torch.sum(torch.abs(latent)))
    # print('term3:',lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1], device=V.device))**2)
    return loss




def my_loss_norm_nonlin(output, target, latent, V, V2, lam_sparse, lam_orthog, sample_weight):

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
    V_size_ratio=V.shape[1]**2/V2.shape[1]**2
    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent)) + .5*lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1], device=V.device))**2 + .5*V_size_ratio*lam_orthog*torch.norm(V2.T@V2-torch.eye(V2.shape[1], device=V2.device))**2
    # loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent)) + lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1], device=V.device))**2 + lam_orthog*torch.norm(V2.T@V2-torch.eye(V2.shape[1], device=V2.device))**2
    return loss
