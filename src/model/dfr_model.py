import torch.nn as nn

from src.backbone.feature_extractor import FeatureExtractor
from src.modules.multi_scale_generator import MultiScaleFeatureGenerator
from src.modules.feature_autoencoder import FeatureAutoencoder
from src.modules.anomaly_map import compute_anomaly_map


class DFRModel(nn.Module):
    def __init__(self, backbone="vgg19", layers=None, stride=1, latent_dim=256, use_residual=True):
        super().__init__()

        self.use_residual = use_residual  

        # CNN
        self.feature_extractor = FeatureExtractor(backbone, layers)

        # Multi-scale generator
        self.generator = MultiScaleFeatureGenerator(stride)

        # Autoencoder
        self.autoencoder = FeatureAutoencoder(latent_dim)

    def forward(self, x):
        features = self.feature_extractor(x)

        f = self.generator(features)
      
        f_hat = self.autoencoder(f)

        anomaly_map = compute_anomaly_map(f, f_hat, use_residual=self.use_residual)

        return anomaly_map
