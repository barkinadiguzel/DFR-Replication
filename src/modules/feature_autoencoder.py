import torch.nn as nn


class FeatureAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.built = False

    def build(self, in_channels):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.latent_dim, kernel_size=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_dim, in_channels, kernel_size=1)
        )

        self.built = True

    def forward(self, x):
        if not self.built:
            self.build(x.shape[1])

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat 
