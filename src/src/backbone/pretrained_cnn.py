import torch.nn as nn
from torchvision import models


def get_pretrained_cnn(name="vgg19"):
    if name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    else:
        raise NotImplementedError(f"{name} not supported")

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model
