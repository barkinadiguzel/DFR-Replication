import torch.nn.functional as F


def resize_align(feature, target_size):
    return F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
