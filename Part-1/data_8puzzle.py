import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import random
from PIL import Image
import json

class EightPuzzleDataset(Dataset):
    """
    Custom Dataset for 8-Puzzle states using MNIST digits
    """
    def __init__(self, root_dir='./src/Dataset', train=True, transform=None, download=True):
        """
        Args:
            root_dir (string): Directory to store the dataset
            train (bool): If True, uses training set, else test set
            transform (callable, optional): Optional transform to be applied on samples
            download (bool): If True, downloads the dataset from the internet
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # Create directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)
        
        # Download MNIST dataset
        self.mnist_dataset = datasets.MNIST(
            root=root_dir,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        # Dictionary to store indices of each digit (0-9)
        self.digit_indices = self._get_digit_indices()
        
        # Generate 8-puzzle states
        self.puzzle_states, self.puzzle_labels = self._generate_puzzle_states(num_samples=20000)
        
    def _get_digit_indices(self):
        """
        Create a dictionary with keys as digits (0-9) and values as lists of indices
        where these digits occur in the MNIST dataset
        """
        digit_indices = {i: [] for i in range(10)}
        
        for idx, (_, label) in enumerate(self.mnist_dataset):
            digit_indices[label].append(idx)
            
        return digit_indices
    
    def _generate_puzzle_states(self, num_samples=20000):
        """
        Generate 8-puzzle states using MNIST digits
        
        Args:
            num_samples (int): Number of puzzle states to generate
            
        Returns:
            tuple: (puzzle_states, puzzle_labels)
                puzzle_states: List of puzzles where each puzzle is a 3x3 grid of MNIST digit images
                puzzle_labels: List of puzzles where each puzzle is a 3x3 grid of digit labels
        """
        puzzle_states = []
        puzzle_labels = []
        
        print(f"Generating {num_samples} 8-puzzle states...")
        
        for _ in range(num_samples):
            # Generate a random valid 8-puzzle state
            # A valid state is a permutation of [0,1,2,3,4,5,6,7,8] (0 represents the empty space)
            # For our dataset, we'll use 9 as the empty space to make it visually distinct
            digits = list(range(1, 9)) + [0]  # 1-8 and 0 (empty space)
            random.shuffle(digits)
            
            # Create 3x3 grid for puzzle state and labels
            puzzle_state = torch.zeros((9, 28, 28))  # 9 positions, each 28x28 MNIST image
            puzzle_label = torch.tensor(digits)  # Digit labels for each position
            
            # Fill the puzzle state with MNIST digit images
            for i, digit in enumerate(digits):
                if digit == 0:  # Empty space
                    # Use a blank image (all zeros) for empty space
                    puzzle_state[i] = torch.zeros((28, 28))
                else:
                    # Randomly select an image for this digit from MNIST
                    idx = random.choice(self.digit_indices[digit])
                    img, _ = self.mnist_dataset[idx]
                    puzzle_state[i] = img.squeeze()  # Remove channel dimension
            
            puzzle_states.append(puzzle_state)
            puzzle_labels.append(puzzle_label)
        
        print(f"Generated {len(puzzle_states)} 8-puzzle states")
        return puzzle_states, puzzle_labels
    
    def __len__(self):
        return len(self.puzzle_states)
    
    def __getitem__(self, idx):
        puzzle_state = self.puzzle_states[idx]
        puzzle_label = self.puzzle_labels[idx]
        
        if self.transform:
            # Apply transform to each digit image in the puzzle
            transformed_state = torch.zeros_like(puzzle_state)
            for i in range(9):
                img = puzzle_state[i].unsqueeze(0)  # Add channel dimension for transforms
                transformed_state[i] = self.transform(img).squeeze()
            puzzle_state = transformed_state
        
        return puzzle_state, puzzle_label
    
    def visualize_puzzle(self, idx):
        """
        Visualize a puzzle state at index idx
        
        Args:
            idx (int): Index of puzzle to visualize
        """
        puzzle_state, puzzle_label = self[idx]
        
        fig, ax = plt.subplots(3, 3, figsize=(8, 8))
        
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                ax[i, j].imshow(puzzle_state[pos].numpy(), cmap='gray')
                ax[i, j].set_title(f"Label: {puzzle_label[pos].item()}")
                ax[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def save_dataset(self, file_path):
        """
        Save the dataset to a file
        
        Args:
            file_path (str): Path to save the dataset
        """
        torch.save({
            'puzzle_states': self.puzzle_states,
            'puzzle_labels': self.puzzle_labels
        }, file_path)
        print(f"Dataset saved to {file_path}")
        
    @staticmethod
    def load_dataset(file_path, transform=None):
        """
        Load a dataset from a file
        
        Args:
            file_path (str): Path to load the dataset from
            transform (callable, optional): Optional transform to be applied on samples
            
        Returns:
            EightPuzzleDataset: Loaded dataset
        """
        data = torch.load(file_path)
        dataset = EightPuzzleDataset(download=False)
        dataset.puzzle_states = data['puzzle_states']
        dataset.puzzle_labels = data['puzzle_labels']
        dataset.transform = transform
        print(f"Dataset loaded from {file_path} with {len(dataset)} samples")
        return dataset

def create_balanced_imbalanced_subsets(dataset, save_dir='./src/Dataset'):
    """
    Create balanced and imbalanced subsets of the dataset
    
    Args:
        dataset (EightPuzzleDataset): Dataset to create subsets from
        save_dir (str): Directory to save the subsets
        
    Returns:
        tuple: (balanced_dataset, imbalanced_dataset)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get positions with each digit for all puzzles
    digit_positions = {d: [] for d in range(9)}  # 0-8 (including empty space)
    
    for idx, (_, label) in enumerate(dataset):
        for pos, digit in enumerate(label):
            digit = digit.item()
            digit_positions[digit].append((idx, pos))
    
    # Print distribution of digits
    print("Original digit distribution:")
    for digit, positions in digit_positions.items():
        print(f"Digit {digit}: {len(positions)} positions")
    
    # Determine subset size (let's make it 25% of original dataset)
    subset_size = len(dataset) // 4
    
    # Create balanced subset
    balanced_indices = set()
    digits_per_class = subset_size // 9  # Equal number for each digit
    
    for digit in range(9):
        # Randomly select indices for this digit
        positions = random.sample(digit_positions[digit], min(digits_per_class, len(digit_positions[digit])))
        for idx, _ in positions:
            balanced_indices.add(idx)
    
    # Convert to list for Subset creation
    balanced_indices = list(balanced_indices)
    
    # Create imbalanced subset (reduce samples for digits 1, 3, 5, 7 by 50%)
    imbalanced_indices = set()
    digits_reduced = [1, 3, 5, 7]  # Digits to have fewer samples
    
    for digit in range(9):
        if digit in digits_reduced:
            # 50% samples for these digits
            samples = digits_per_class // 2
        else:
            # Full samples for other digits
            samples = digits_per_class
        
        positions = random.sample(digit_positions[digit], min(samples, len(digit_positions[digit])))
        for idx, _ in positions:
            imbalanced_indices.add(idx)
    
    # Convert to list for Subset creation
    imbalanced_indices = list(imbalanced_indices)
    
    # Create subsets
    balanced_subset = Subset(dataset, balanced_indices)
    imbalanced_subset = Subset(dataset, imbalanced_indices)
    
    # Save subsets
    balanced_path = os.path.join(save_dir, "8puzzle_balanced.pt")
    imbalanced_path = os.path.join(save_dir, "8puzzle_imbalanced.pt")
    
    # Extract the data from subsets
    balanced_states = [dataset.puzzle_states[i] for i in balanced_indices]
    balanced_labels = [dataset.puzzle_labels[i] for i in balanced_indices]
    
    imbalanced_states = [dataset.puzzle_states[i] for i in imbalanced_indices]
    imbalanced_labels = [dataset.puzzle_labels[i] for i in imbalanced_indices]
    
    # Save the extracted data
    torch.save({
        'puzzle_states': balanced_states,
        'puzzle_labels': balanced_labels
    }, balanced_path)
    
    torch.save({
        'puzzle_states': imbalanced_states,
        'puzzle_labels': imbalanced_labels
    }, imbalanced_path)
    
    print(f"Balanced subset saved with {len(balanced_indices)} samples")
    print(f"Imbalanced subset saved with {len(imbalanced_indices)} samples")
    
    # Analyze digit distribution in subsets
    balanced_digit_counts = {d: 0 for d in range(9)}
    imbalanced_digit_counts = {d: 0 for d in range(9)}
    
    for idx in balanced_indices:
        for digit in dataset.puzzle_labels[idx]:
            balanced_digit_counts[digit.item()] += 1
            
    for idx in imbalanced_indices:
        for digit in dataset.puzzle_labels[idx]:
            imbalanced_digit_counts[digit.item()] += 1
    
    print("\nBalanced subset digit distribution:")
    for digit, count in balanced_digit_counts.items():
        print(f"Digit {digit}: {count} occurrences")
    
    print("\nImbalanced subset digit distribution:")
    for digit, count in imbalanced_digit_counts.items():
        print(f"Digit {digit}: {count} occurrences")
    
    return balanced_subset, imbalanced_subset
    
def apply_transforms(dataset_path, output_path):
    """
    Apply transformations to dataset and save the augmented dataset
    
    Args:
        dataset_path (str): Path to the dataset
        output_path (str): Path to save the augmented dataset
    """
    # Load the dataset
    data = torch.load(dataset_path)
    puzzle_states = data['puzzle_states']
    puzzle_labels = data['puzzle_labels']
    
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Apply transformations
    augmented_states = []
    
    for puzzle_state in puzzle_states:
        transformed_state = torch.zeros_like(puzzle_state)
        for i in range(9):
            # Skip transforming empty cells (digit 0)
            if torch.sum(puzzle_state[i]) > 0:
                img = puzzle_state[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                transformed_state[i] = transform(img).squeeze()
            else:
                transformed_state[i] = puzzle_state[i]
        augmented_states.append(transformed_state)
    
    # Save augmented dataset
    torch.save({
        'puzzle_states': augmented_states,
        'puzzle_labels': puzzle_labels
    }, output_path)
    
    print(f"Augmented dataset saved to {output_path}")

def visualize_dataset_samples(dataset_path, num_samples=5):
    """
    Visualize samples from a dataset
    
    Args:
        dataset_path (str): Path to the dataset
        num_samples (int): Number of samples to visualize
    """
    data = torch.load(dataset_path)
    puzzle_states = data['puzzle_states']
    puzzle_labels = data['puzzle_labels']
    
    # Randomly select samples
    indices = random.sample(range(len(puzzle_states)), min(num_samples, len(puzzle_states)))
    
    for idx in indices:
        puzzle_state = puzzle_states[idx]
        puzzle_label = puzzle_labels[idx]
        
        fig, ax = plt.subplots(3, 3, figsize=(8, 8))
        fig.suptitle(f"Puzzle Sample {idx}")
        
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                ax[i, j].imshow(puzzle_state[pos].numpy(), cmap='gray')
                ax[i, j].set_title(f"Label: {puzzle_label[pos].item()}")
                ax[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create dataset directory
    os.makedirs('../Dataset', exist_ok=True)
    
    # Create 8-puzzle dataset
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    dataset = EightPuzzleDataset(root_dir='../Dataset', transform=transform)
    
    # Visualize a few samples
    print("Visualizing 3 samples from the original dataset:")
    for i in range(3):
        dataset.visualize_puzzle(random.randint(0, len(dataset)-1))
    
    # Save full dataset
    dataset.save_dataset('../Dataset/8puzzle_full.pt')
    
    # Create balanced and imbalanced subsets
    balanced_subset, imbalanced_subset = create_balanced_imbalanced_subsets(dataset)
    
    # Apply transforms to the subsets
    apply_transforms('../Dataset/8puzzle_balanced.pt', '../Dataset/8puzzle_balanced_augmented.pt')
    apply_transforms('../Dataset/8puzzle_imbalanced.pt', '../Dataset/8puzzle_imbalanced_augmented.pt')
    
    # Visualize samples from augmented datasets
    print("\nVisualization of balanced augmented dataset:")
    visualize_dataset_samples('../Dataset/8puzzle_balanced_augmented.pt', num_samples=2)
    
    print("\nVisualization of imbalanced augmented dataset:")
    visualize_dataset_samples('../Dataset/8puzzle_imbalanced_augmented.pt', num_samples=2)