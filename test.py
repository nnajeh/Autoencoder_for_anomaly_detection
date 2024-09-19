
from libraries import *
from model import *
from dataset import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score




net_new = Net().to(device)
net_new.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
net_new.eval()



def normalize(img):
    # Clip values to be within [0, 1] if they are outside this range
    return np.clip(img, 0, 1)



import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow_grid(img, title=''):
    img = torchvision.utils.make_grid(img.cpu().detach())  # Create a grid of images
    img_numpy = img.numpy()

    # Check the range of img_numpy
    print(f'Image data range: {img_numpy.min()} to {img_numpy.max()}')

    # Normalize if needed (assuming image is in [0, 255] range)
    img_numpy = img_numpy / 255.0  # Uncomment this if your data is in [0, 255] range

    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))  # Convert (C, H, W) to (H, W, C)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_reconstructions(autoencoder, dataloader, num_images=5):
    autoencoder.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            
            # Get the model's reconstruction
            reconstructed = autoencoder(inputs)
            
            # Normalize images before displaying if necessary
            inputs = normalize(inputs.cpu().numpy())
            reconstructed = normalize(reconstructed.cpu().numpy())

            # Show a grid of original images
            print("Original Images:")
            imshow_grid(torch.tensor(inputs), 'Original Images')
            
            # Show a grid of reconstructed images
            print("Reconstructed Images:")
            imshow_grid(torch.tensor(reconstructed), 'Reconstructed Images')
            
            break  # Show only one batch

# Example usage
# Assuming testloader is a DataLoader object for the test set
visualize_reconstructions(net_new, testloader)





def check_image_range(images, label=''):
    if isinstance(images, torch.Tensor):
        images_np = images.cpu().numpy()
    elif isinstance(images, np.ndarray):
        images_np = images
    else:
        raise TypeError("Unsupported type for images: must be torch.Tensor or numpy.ndarray")
    print(f'{label} - Min: {images_np.min()}, Max: {images_np.max()}')






import torch

# Assuming `net_test`, `criterion`, `device`, and `testloader` are defined elsewhere

def test_autoencoder(testloader=None, print_report=False):
    """Run the autoencoder on the test set and provide the loss vs the label."""

    # Lists to store the losses and labels for each batch and each data point
    losses, labels, batches_losses = [], [], []
    img_list, gt_list = [], []  # For storing input images and ground truth labels
    
    # No need to store the gradients for inference
    with torch.no_grad():
        # Iterate over the data in batches
        for i, batch_data in enumerate(testloader, 0):
            # Get the inputs and labels
            inputs, batch_labels = batch_data
            inputs = inputs.to(device)

            # Forward pass through the autoencoder
            outputs = net_new(inputs)

            # Compute the loss for the entire batch
            batch_loss = criterion(outputs, inputs)
            batches_losses.append(batch_loss)

            # Compute loss for each individual image in the batch and store results
            for j, label in enumerate(batch_labels):
                # Compute the loss for individual data points
                loss = criterion(outputs[j], inputs[j])
                
                # Store the loss and label
                losses.append(loss)
                labels.append(label)

            # Convert images and labels to numpy arrays and store
            img_list.extend(inputs.cpu().detach().numpy())
            gt_list.extend(batch_labels.cpu().detach().numpy())

    # Convert lists to tensors
    losses: torch.Tensor = torch.stack(losses).cpu()
    labels: torch.Tensor = torch.tensor(labels).cpu()
    batches_losses: torch.Tensor = torch.stack(batches_losses).cpu()

    # Calculate average loss
    average_loss: torch.Tensor = torch.mean(losses)

    # Print the average loss if required
    if print_report:
        print(f'Finished running the autoencoder with average loss {average_loss.item():.3f}')

    # Return the losses and labels, and optionally other data
    return losses, labels, batches_losses, average_loss, img_list, gt_list

# Example usage:
losses, labels, batches_losses, average_loss, img_list, gt_list = test_autoencoder(testloader, print_report=True)





mean_loss = torch.mean(losses)
std_loss = torch.std(losses)
print(f'Mean Reconstruction Loss: {mean_loss.item():.4f}')
print(f'Standard Deviation of Reconstruction Loss: {std_loss.item():.4f}')





# Assuming 'labels' contains ground truth (0 for normal, 1 for anomaly)
# and 'losses' serves as the anomaly score.
labels_np = labels.numpy()
losses_np = losses.numpy()

roc_auc = roc_auc_score(labels_np, losses_np)
print(f'ROC-AUC Score: {roc_auc:.4f}')





# Define a threshold for anomaly detection
threshold = mean_loss + 2 * std_loss  # Example threshold

# Binary classification based on the threshold
predictions = (losses > threshold).numpy()
ground_truth = labels.numpy()

precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
