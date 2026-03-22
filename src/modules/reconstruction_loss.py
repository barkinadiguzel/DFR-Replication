import torch

def reconstruction_loss(f, f_hat):
    return torch.mean((f - f_hat) ** 2)
