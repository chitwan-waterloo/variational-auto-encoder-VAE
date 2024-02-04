# main.py
import os
import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from src.vae_model import VAE
from src.loss_function import loss_function
from src.plotting import plot_loss
from src.sample_generation import generate_samples
from src.training import train_vae

def main():
    parser = argparse.ArgumentParser(description='Variational Auto-encoder for Even Numbers')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory for result files')
    parser.add_argument('-n', '--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of the latent space')

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    # Load and preprocess data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use correct mean and std for MNIST
    ])

    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # Initialize VAE model
    vae = VAE(latent_dim=args.latent_dim)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Train VAE
    train_vae(vae, dataloader, args.epochs, args.output_dir, device, args.verbose)

    # Generate and save sample images
    generate_samples(vae, args.output_dir, latent_dim=args.latent_dim, n_samples=args.num_samples)

if __name__ == '__main__':
    main()
