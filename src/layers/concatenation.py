import torch


def concatenate(features):
    return torch.cat(features, dim=1)
