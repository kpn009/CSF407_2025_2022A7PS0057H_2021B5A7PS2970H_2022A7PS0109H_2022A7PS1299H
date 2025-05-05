import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from collections import Counter

def create_mnist_subsets(root_dir='./src/Dataset', save_dir='./src/Dataset'):
    """
    Download MNIST dataset and create balanced and imbalanced subsets
    
    Args:
        root_dir (str): Directory to download MNIST dataset
        save_dir (str): Directory to save the subsets
        
    Returns:
        tuple: (balanced_subset, imbalanced_subset)
    """
    # Create directories if they don't exist
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Download MNIST dataset
    mnist_train = datasets.MNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Get all indices by digit class
    digit_indices = {i: [] for i in range(10)}
    
    for idx, (_, label) in enumerate(mnist_train):
        digit_indices[label].append(idx)
    
    # Create balanced subset (1000 samples per digit class = 10,000 total)
    samples_per_class = 1000
    balanced_indices = []
    
    for digit in range(10):
        # Sample from each digit class
        class_indices = random.sample(digit_indices[digit], samples_per_class)
        balanced_indices.extend(class_indices)
    
    # Shuffle balanced indices
    random.shuffle(balanced_indices)
    
    # Create imbalanced subset
    # We'll reduce samples for digits 0-4 to 50% of the balanced amount
    imbalanced_indices = []
    
    for digit in range(10):
        if digit < 5:  # Digits 0-4 have fewer samples
            samples = samples_per_class // 2
        else:  # Digits 5-9 have normal samples
            samples = samples_per_class
        
        class_indices = random.sample(digit_indices[digit], samples)
        imbalanced_indices.extend(class_indices)
    
    # Shuffle imbalanced indices
    random.shuffle(imbalanced_indices)
    
    # Create subsets
    balanced_subset = Subset(mnist_train, balanced_indices)
    imbalanced_subset = Subset(mnist_train, imbalanced_indices)
    
    # Analyze and print distribution of balanced subset
    balanced_counter = Counter([mnist_train[idx][1] for idx in balanced_indices])
    
    # Analyze and print distribution of imbalanced subset
    imbalanced_counter = Counter([mnist_train[idx][1] for idx in imbalanced_indices])
    
    # Save the balanced and imbalanced subset indices
    balanced_path = os.path.join(save_dir, "mnist_balanced_indices.pt")
    imbalanced_path = os.path.join(save_dir, "mnist_imbalanced_indices.pt")
    
    torch.save(balanced_indices, balanced_path)
    torch.save(imbalanced_indices, imbalanced_path)
    
    # Extract and save the actual data for easier access later
    balanced_images = []
    balanced_labels = []
    imbalanced_images = []
    imbalanced_labels = []
    
    # Extract balanced data
    for idx in balanced_indices:
        image, label = mnist_train[idx]
        balanced_images.append(image)
        balanced_labels.append(label)
    
    # Extract imbalanced data
    for idx in imbalanced_indices:
        image, label = mnist_train[idx]
        imbalanced_images.append(image)
        imbalanced_labels.append(label)
    
    # Save actual data
    balanced_data_path = os.path.join(save_dir, "mnist_balanced.pt")
    imbalanced_data_path = os.path.join(save_dir, "mnist_imbalanced.pt")
    
    torch.save({
        "images": balanced_images,
        "labels": balanced_labels
    }, balanced_data_path)
    
    torch.save({
        "images": imbalanced_images,
        "labels": imbalanced_labels
    }, imbalanced_data_path)
    
    print(f"\nBalanced dataset saved to {balanced_data_path}")
    print(f"Imbalanced dataset saved to {imbalanced_data_path}")
    
    return balanced_subset, imbalanced_subset

def apply_transforms_to_dataset(input_path, output_path, transform_type="basic"):
    """
    Apply transformations to a dataset and save the augmented dataset
    
    Args:
        input_path (str): Path to the dataset
        output_path (str): Path to save the augmented dataset
        transform_type (str): Type of transformations to apply
                             "basic" - normalization only
                             "moderate" - normalization + minor rotation/translation
                             "heavy" - normalization + stronger augmentations
    """
    # Load the dataset
    data = torch.load(input_path)
    images = data["images"]
    labels = data["labels"]
    
    # Define transformation based on the specified type
    if transform_type == "basic":
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    elif transform_type == "moderate":
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.RandomRotation(10),  # Rotation by up to 10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Small translations
        ])
    elif transform_type == "heavy":
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
            transforms.RandomRotation(15),  # Rotation by up to 15 degrees
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),  # Translation + scaling
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5)  # Perspective transform
        ])
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")
    
    # Apply transformations
    augmented_images = []
    
    for image in images:
        # Apply the transformations
        augmented_image = transform(image)
        augmented_images.append(augmented_image)
    
    # Save augmented dataset
    torch.save({
        "images": augmented_images,
        "labels": labels
    }, output_path)
    
    print(f"Augmented dataset saved to {output_path}")

def visualize_dataset_samples(dataset_path, num_samples=10):
    """
    Visualize samples from a dataset
    
    Args:
        dataset_path (str): Path to the dataset
        num_samples (int): Number of samples to visualize
    """
    data = torch.load(dataset_path)
    images = data["images"]
    labels = data["labels"]
    
    # Randomly select samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    # Create a grid for visualization
    rows = min(2, num_samples)
    cols = (num_samples + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Flatten the axes array for easy indexing if there's more than 1 row
    if rows > 1:
        axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = images[idx].squeeze().numpy()  # Remove channel dimension
        label = labels[idx]
        
        # Handle both single and multi-row plots
        if num_samples > 1:
            ax = axes[i]
        else:
            ax = axes
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set up directories
    root_dir = './src/Dataset'
    save_dir = './src/Dataset'
    
    # Create balanced and imbalanced subsets
    print("Creating balanced and imbalanced MNIST subsets...")
    balanced_subset, imbalanced_subset = create_mnist_subsets(root_dir, save_dir)
    
    # Apply different transformations to both subsets
    print("\nApplying transformations to balanced subset...")
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_balanced.pt"),
        os.path.join(save_dir, "mnist_balanced_augmented_basic.pt"),
        transform_type="basic"
    )
    
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_balanced.pt"),
        os.path.join(save_dir, "mnist_balanced_augmented_moderate.pt"),
        transform_type="moderate"
    )
    
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_balanced.pt"),
        os.path.join(save_dir, "mnist_balanced_augmented_heavy.pt"),
        transform_type="heavy"
    )
    
    print("\nApplying transformations to imbalanced subset...")
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_imbalanced.pt"),
        os.path.join(save_dir, "mnist_imbalanced_augmented_basic.pt"),
        transform_type="basic"
    )
    
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_imbalanced.pt"),
        os.path.join(save_dir, "mnist_imbalanced_augmented_moderate.pt"),
        transform_type="moderate"
    )
    
    apply_transforms_to_dataset(
        os.path.join(save_dir, "mnist_imbalanced.pt"),
        os.path.join(save_dir, "mnist_imbalanced_augmented_heavy.pt"),
        transform_type="heavy"
    )
    
    print("\nMNIST datasets created and saved successfully!")

if __name__ == "__main__":
    main()