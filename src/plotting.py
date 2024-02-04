import os
import matplotlib.pyplot as plt

def plot_loss(losses, output_dir):
    """
    Plot and save the training loss over epochs.

    Parameters:
    - losses (list): List of training losses.
    - output_dir (str): Directory to save the loss plot.

    Returns:
    - None
    """
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'loss.pdf'))
    plt.close()
