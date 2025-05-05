import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import random
from PIL import Image

class EightPuzzleDataset(Dataset):
    def __init__(self, root_dir='./src/Dataset', train=True, transform=None, download=True, generate_84x84=True):
        """
        Args:
            root_dir (string): Directory to store the dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.generate_84x84 = generate_84x84
        
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
        """
        puzzle_states = []
        puzzle_labels = []
        
        print(f"Generating {num_samples} 8-puzzle states...")
        
        for _ in range(num_samples):
            digits = list(range(1, 9)) + [0]  # 1-8 and 0 (empty space)
            random.shuffle(digits)
            
            if self.generate_84x84:
                # Create a single 84x84 image
                puzzle_state = torch.zeros((84, 84))
                
                # Fill the puzzle state with MNIST digit images
                for i in range(3):
                    for j in range(3):
                        pos = i * 3 + j
                        digit = digits[pos]
                        
                        # Start position for this tile in the 84x84 image
                        row_start = i * 28
                        col_start = j * 28
                        
                        if digit == 0:  # Empty space
                            # Use a blank image (all zeros) for empty space
                            puzzle_state[row_start:row_start+28, col_start:col_start+28] = torch.zeros((28, 28))
                        else:
                            # Randomly select an image for this digit from MNIST
                            idx = random.choice(self.digit_indices[digit])
                            img, _ = self.mnist_dataset[idx]
                            puzzle_state[row_start:row_start+28, col_start:col_start+28] = img.squeeze()
            else:
                # Create 3x3 grid for puzzle state (original approach)
                puzzle_state = torch.zeros((9, 28, 28))  # 9 positions, each 28x28 MNIST image
                
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
            puzzle_labels.append(torch.tensor(digits))
        
        print(f"Generated {len(puzzle_states)} 8-puzzle states")
        return puzzle_states, puzzle_labels
    
    def __len__(self):
        return len(self.puzzle_states)
    
    def __getitem__(self, idx):
        puzzle_state = self.puzzle_states[idx]
        puzzle_label = self.puzzle_labels[idx]
        
        if not self.generate_84x84 and self.transform:
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
        
        if self.generate_84x84:
            # For 84x84 format, split the image into 9 tiles for visualization
            fig, ax = plt.subplots(3, 3, figsize=(8, 8))
            
            for i in range(3):
                for j in range(3):
                    pos = i * 3 + j
                    # Extract the tile
                    row_start = i * 28
                    col_start = j * 28
                    tile = puzzle_state[row_start:row_start+28, col_start:col_start+28]
                    
                    ax[i, j].imshow(tile.numpy(), cmap='gray')
                    ax[i, j].set_title(f"Label: {puzzle_label[pos].item()}")
                    ax[i, j].axis('off')
        else:
            # Original format with 9 separate images
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
        
        # Determine if this is an 84x84 format dataset by checking the shape of the first item
        first_item = dataset.puzzle_states[0]
        if isinstance(first_item, torch.Tensor) and first_item.shape == (84, 84):
            dataset.generate_84x84 = True
        else:
            dataset.generate_84x84 = False
            
        print(f"Dataset loaded from {file_path} with {len(dataset)} samples")
        return dataset

def preprocess_puzzle_image(image):
    """
    Preprocess a puzzle image by splitting it into 9 individual 28x28 images
    
    Args:
        image (torch.Tensor): Input puzzle image of shape (84, 84)
        
    Returns:
        torch.Tensor: Processed images of shape (9, 28, 28)
    """
    # Check if image is already in the right format (9, 28, 28)
    if isinstance(image, torch.Tensor) and image.shape == (9, 28, 28):
        return image
    
    # If image is not a tensor or not 84x84, raise an error
    if not isinstance(image, torch.Tensor) or image.shape != (84, 84):
        raise ValueError(f"Expected tensor of shape (84, 84), got {image.shape if isinstance(image, torch.Tensor) else type(image)}")
    
    # Create a tensor to hold the 9 individual images
    processed_images = torch.zeros((9, 28, 28))
    
    # Split the image into 9 individual images
    for i in range(3):
        for j in range(3):
            pos = i * 3 + j
            row_start = i * 28
            col_start = j * 28
            processed_images[pos] = image[row_start:row_start+28, col_start:col_start+28]
    
    return processed_images

def preprocess_batch(batch):
    """
    Preprocess a batch of puzzle images
    
    Args:
        batch (torch.Tensor): Batch of puzzle images. Can be either:
            - (batch_size, 84, 84) for 84x84 images
            - (batch_size, 9, 28, 28) for already processed images
            
    Returns:
        torch.Tensor: Processed batch of shape (batch_size, 9, 28, 28)
    """
    # Check if batch is already in the right format
    if isinstance(batch, torch.Tensor) and len(batch.shape) == 4 and batch.shape[1:] == (9, 28, 28):
        return batch
    
    # Process each image in the batch
    if len(batch.shape) == 3 and batch.shape[1:] == (84, 84):
        batch_size = batch.shape[0]
        processed_batch = torch.zeros((batch_size, 9, 28, 28))
        
        for i in range(batch_size):
            processed_batch[i] = preprocess_puzzle_image(batch[i])
            
        return processed_batch
    else:
        raise ValueError(f"Expected tensor of shape (batch_size, 84, 84) or (batch_size, 9, 28, 28), got {batch.shape}")

def create_balanced_imbalanced_subsets(dataset, save_dir='./src/Dataset'):
    """
    Create balanced and imbalanced subsets of the dataset based on positional patterns
    
    Args:
        dataset (EightPuzzleDataset): Dataset to create subsets from
        save_dir (str): Directory to save the subsets
        
    Returns:
        tuple: (balanced_dataset, imbalanced_dataset)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze positional patterns in the dataset
    position_digit_counts = {}
    for pos in range(9):
        position_digit_counts[pos] = {digit: 0 for digit in range(9)}  # 0-8 digits
    
    # Count occurrences of each digit at each position
    for _, label in dataset:
        for pos, digit in enumerate(label):
            digit = digit.item()
            position_digit_counts[pos][digit] += 1
    
    # Identify positions with strong digit biases (for imbalanced dataset)
    # Calculate standard deviation of digit counts at each position
    position_stdevs = {}
    for pos in range(9):
        counts = list(position_digit_counts[pos].values())
        position_stdevs[pos] = np.std(counts)
    
    # Sort positions by standard deviation (higher std = more imbalanced)
    sorted_positions = sorted(position_stdevs.items(), key=lambda x: x[1], reverse=True)
    
    # Choose the top 4 positions with highest standard deviation for creating imbalanced dataset
    imbalanced_positions = [pos for pos, _ in sorted_positions[:4]]
    print(f"\nPositions with highest bias (to be used for imbalanced dataset): {imbalanced_positions}")
    
    # Subset size (25% of original dataset)
    subset_size = len(dataset) // 4
    
    # Create balanced subset - we'll randomly select puzzles that have more uniform distribution
    # across positions, avoiding strong positional biases
    balanced_indices = []
    imbalanced_indices = []
    
    # For each puzzle, calculate a "balance score" - lower means more balanced
    balance_scores = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        # Calculate how typical this puzzle's digit placements are compared to overall distribution
        score = 0
        for pos, digit in enumerate(label):
            digit = digit.item()
            # Higher score means this position-digit combination is more common
            score += position_digit_counts[pos][digit] / sum(position_digit_counts[pos].values())
        balance_scores.append((idx, score))
    
    # Sort puzzles by balance score
    balance_scores.sort(key=lambda x: x[1])
    
    # Take subset_size puzzles with lowest balance scores for balanced dataset
    balanced_indices = [idx for idx, _ in balance_scores[:subset_size]]
    
    # For imbalanced dataset, focus on puzzles with strong positional patterns
    # Calculate imbalance scores - higher means more imbalanced for the target positions
    imbalance_scores = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        score = 0
        for pos in imbalanced_positions:
            digit = label[pos].item()
            # Higher score means this position-digit combination is more common in imbalanced positions
            pos_total = sum(position_digit_counts[pos].values())
            score += position_digit_counts[pos][digit] / pos_total if pos_total > 0 else 0
        imbalance_scores.append((idx, score))
    
    # Take subset_size puzzles with highest imbalance scores
    imbalance_scores.sort(key=lambda x: x[1], reverse=True)
    imbalanced_indices = [idx for idx, _ in imbalance_scores[:subset_size]]
    
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
    
    # Analyze position-digit distribution in subsets
    balanced_position_counts = {}
    imbalanced_position_counts = {}
    for pos in range(9):
        balanced_position_counts[pos] = {digit: 0 for digit in range(9)}  # 0-8 digits
        imbalanced_position_counts[pos] = {digit: 0 for digit in range(9)}  # 0-8 digits
    
    # Count balanced subset
    for idx in balanced_indices:
        for pos, digit in enumerate(dataset.puzzle_labels[idx]):
            balanced_position_counts[pos][digit.item()] += 1
            
    # Count imbalanced subset
    for idx in imbalanced_indices:
        for pos, digit in enumerate(dataset.puzzle_labels[idx]):
            imbalanced_position_counts[pos][digit.item()] += 1
    
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
    
    # Define transformations - keep the same but apply them after preprocessing
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Apply transformations
    augmented_states = []
    
    for puzzle_state in puzzle_states:
        # First preprocess to get 9 separate images if needed
        if puzzle_state.shape == (84, 84):
            processed_state = preprocess_puzzle_image(puzzle_state)
        else:
            processed_state = puzzle_state
        
        # Now apply transforms to each 28x28 image
        transformed_state = torch.zeros_like(processed_state)
        for i in range(9):
            # Skip transforming empty cells (digit 0)
            if torch.sum(processed_state[i]) > 0:
                img = processed_state[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                transformed_state[i] = transform(img).squeeze()
            else:
                transformed_state[i] = processed_state[i]
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
        
        # If it's an 84x84 image, preprocess it first
        if puzzle_state.shape == (84, 84):
            puzzle_state = preprocess_puzzle_image(puzzle_state)
            
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