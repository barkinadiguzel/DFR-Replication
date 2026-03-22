import torch


def compute_anomaly_map(f, f_hat):
    return torch.sum((f - f_hat) ** 2, dim=1, keepdim=True)
