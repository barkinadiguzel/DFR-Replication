import torch.nn.functional as F


def aggregate(feature, stride=1):
    return F.avg_pool2d(feature, kernel_size=3, stride=stride, padding=1)
