import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(x, recon_x, mu, logvar):
    Recon = reconstruction(x, recon_x)
    KLD = KL_divergence(mu, logvar)
    return Recon + KLD

def reconstruction(x, recon_x):
    return F.mse_loss(recon_x, x)

def KL_divergence(mu, logvar):
    KLD = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD *= -0.5
    return torch.sum(KLD)
