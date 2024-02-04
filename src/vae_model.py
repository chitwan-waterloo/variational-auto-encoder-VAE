import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        """
        Variational Autoencoder (VAE) class.

        Parameters:
        - latent_dim (int, optional): Dimensionality of the latent space (default: 2).

        Attributes:
        - latent_dim (int): Dimensionality of the latent space.

        Architecture:
        - Encoder: Convolutional neural network with two convolutional layers, followed by
          flattening and fully connected layers to output mean and log variance of the latent space.
        - Decoder: Fully connected layers followed by reshaping and transposed convolutional layers
          to reconstruct the input.

        Reparameterization Trick:
        - Gaussian reparameterization trick is applied during encoding.

        Forward Method:
        - Takes an input tensor and passes it through the encoder and decoder.
        - Returns the reconstructed input, mean, and log variance of the latent space.

        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim  # Add this line

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Two outputs for mean and log variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick for Gaussian distributions.

        Parameters:
        - mu (torch.Tensor): Mean of the distribution.
        - logvar (torch.Tensor): Log variance of the distribution.

        Returns:
        - z (torch.Tensor): Reparameterized sample from the distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - x_recon (torch.Tensor): Reconstructed input.
        - mu (torch.Tensor): Mean of the latent space.
        - logvar (torch.Tensor): Log variance of the latent space.
        """
        # Encode
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]  # Use self.latent_dim
        logvar = mu_logvar[:, self.latent_dim:]  # Use self.latent_dim
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
