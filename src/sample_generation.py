import torch
import os
import matplotlib.pyplot as plt

def generate_samples(model, output_dir, latent_dim, n_samples=100):
    """
    Generate and save sample images using the trained VAE model.

    Parameters:
    - model: Trained Variational Autoencoder (VAE) model.
    - output_dir (str): Directory to save the generated sample images.
    - latent_dim (int): Dimensionality of the latent space.
    - n_samples (int, optional): Number of samples to generate (default: 100).

    Returns:
    - None
    """
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        samples = model.decoder(z).cpu().numpy()

    for i in range(n_samples):
        plt.imshow(samples[i, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{i + 1}.pdf'))
        plt.close()
