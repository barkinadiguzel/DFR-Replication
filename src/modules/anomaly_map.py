import torch

def compute_anomaly_map(f, f_hat, use_residual=True):
    if use_residual:
        r = f - f_hat  
    else:
        r = f_hat  
    anomaly_map = torch.sum(r ** 2, dim=1, keepdim=True)
    return anomaly_map
