import torch.optim as optim
from src.loss_function import loss_function
from src.plotting import plot_loss

def train_vae(model, dataloader, epochs, output_dir, device, verbose=False):
    """
    Train the Variational Autoencoder (VAE) model.

    Parameters:
    - model: Variational Autoencoder (VAE) model to be trained.
    - dataloader: DataLoader for the training data.
    - epochs (int): Number of training epochs.
    - output_dir (str): Directory to save training result files.
    - device: Device to run the training on (e.g., 'cuda' or 'cpu').
    - verbose (bool, optional): If True, print training loss for each epoch (default: False).

    Returns:
    - None
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in dataloader:
            batch = data[0].to(device)  # Assuming the data is a tuple (input, target)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)  # Assuming loss_function is imported
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss / len(dataloader.dataset))

        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]}')

    plot_loss(losses, output_dir)  # Assuming plot_loss is imported
