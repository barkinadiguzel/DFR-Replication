import torch.nn as nn
from .pretrained_cnn import get_pretrained_cnn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name="vgg19", layers=None):
        super().__init__()
        self.model = get_pretrained_cnn(backbone_name)
        self.layers = layers if layers else [3, 8, 17, 26]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features  
