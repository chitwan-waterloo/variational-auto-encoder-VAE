import torch
import torch.nn as nn

def loss_function(recon_x, x, mu, logvar):
    """
    Calculate the VAE loss function, which is a combination of Mean Squared Error (MSE) and
    Kullback-Leibler Divergence (KLD) terms.

    Parameters:
    - recon_x (torch.Tensor): Reconstructed input from the VAE decoder.
    - x (torch.Tensor): Original input data.
    - mu (torch.Tensor): Mean of the latent space.
    - logvar (torch.Tensor): Log variance of the latent space.

    Returns:
    - loss (torch.Tensor): Total loss combining MSE and KLD terms.
    """
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
