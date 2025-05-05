import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from torchvision import transforms

# Import our custom modules
from data_8puzzle import EightPuzzleDataset, create_balanced_imbalanced_subsets, apply_transforms, visualize_dataset_samples
from model_8puzzle import EightPuzzleModel
from trainer_8puzzle import EightPuzzleTrainer
from preprocess import preprocess_puzzle_image
from real_image import load_and_preprocess_real_image
import mnist_data

def setup_environment():
    """Set up the environment for reproducibility"""
    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Create directories
    os.makedirs('./src/Dataset', exist_ok=True)
    os.makedirs('./src/Results/models', exist_ok=True)
    os.makedirs('./src/Results/plots', exist_ok=True)

def preprocess_dataset(dataset_path, output_path=None):
    """
    Preprocess a dataset by converting 84x84 images to 9x28x28 format
    
    Args:
        dataset_path (str): Path to the dataset
        output_path (str, optional): Path to save the preprocessed dataset
        
    Returns:
        dict: Preprocessed dataset with 'puzzle_states' and 'puzzle_labels'
    """
    # Load the dataset
    data = torch.load(dataset_path)
    puzzle_states = data['puzzle_states']
    puzzle_labels = data['puzzle_labels']
    
    preprocessed_states = []
    
    # Process each puzzle state
    for state in puzzle_states:
        # Check if the state is already in 9x28x28 format
        if state.shape == (9, 28, 28):
            preprocessed_states.append(state)
        elif state.shape == (84, 84):
            # Preprocess the 84x84 image into 9x28x28
            preprocessed_state = preprocess_puzzle_image(state)
            preprocessed_states.append(preprocessed_state)
        else:
            raise ValueError(f"Unexpected puzzle state shape: {state.shape}")
    
    # Create the preprocessed dataset
    preprocessed_data = {
        'puzzle_states': preprocessed_states,
        'puzzle_labels': puzzle_labels
    }
    
    # Save the preprocessed dataset if an output path is provided
    if output_path:
        torch.save(preprocessed_data, output_path)
        print(f"Preprocessed dataset saved to {output_path}")
    
    return preprocessed_data

def predict_from_image_demo(model_path, image_path):
    """
    Simple function to load a model and predict digits from an image
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the input image
        
    Returns:
        torch.Tensor: Predicted digits for the puzzle
    """
    print(f"\n=== Processing image: {image_path} ===")
    
    # Create trainer with the same config used for training
    config_path = 'config.json'
    trainer = EightPuzzleTrainer(config_path)
    
    # Load the model
    trainer.load_model(model_path)
    
    # Load and preprocess the image with visualization
    print("Preprocessing image...")
    raw_image = load_and_preprocess_real_image(image_path, visualize=True)
    
    # Print shape of preprocessed image tiles
    print(f"Preprocessed image shape: {raw_image.shape}")
    
    # Now predict using the processed image
    print("Predicting puzzle state...")
    predictions = trainer.predict(raw_image)
    
    # Display the predictions
    if predictions is not None:
        # Display as a grid
        plt.figure(figsize=(8, 8))
        
        # Create a grid
        for i in range(3):
            for j in range(3):
                # Get the predicted digit
                digit = int(predictions[i * 3 + j])
                
                # Create a subplot
                plt.subplot(3, 3, i*3 + j + 1)
                
                # Display the digit with a border
                plt.text(0.5, 0.5, str(digit), fontsize=50, ha='center', va='center')
                plt.gca().add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                                fill=False, edgecolor='black', linewidth=2))
                
                # Remove ticks
                plt.xticks([])
                plt.yticks([])
                
                # Add grid
                plt.grid(False)
        
        plt.tight_layout()
        plt.suptitle("Predicted 8-Puzzle State", fontsize=16)
        plt.savefig('puzzle_prediction.png')
        plt.show()
        
        # Print as text grid
        print("\nPredicted 8-Puzzle State:")
        print("-" * 13)
        for i in range(3):
            print("| ", end="")
            for j in range(3):
                print(f"{int(predictions[i * 3 + j])} | ", end="")
            print("\n" + "-" * 13)
        
        # Print in flat format for easy reference
        print(f"\nPrediction as flat array: {[int(x) for x in predictions]}")
    
    return predictions

def generate_datasets(train_ratio=0.8):
    """
    Generate 8-puzzle datasets with train and test splits
    
    Args:
        train_ratio (float): Ratio of data to use for training (default: 0.8)
    """
    print("\n=== Generating 8-Puzzle Datasets with Train/Test Splits ===")
    
    # Create base transform
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Create dataset
    print("Creating main dataset...")
    full_dataset = EightPuzzleDataset(root_dir='./src/Dataset', transform=transform, generate_84x84=True)
    
    # Save full dataset
    full_dataset.save_dataset('./src/Dataset/8puzzle_full.pt')
    
    # Visualize some samples
    print("\nVisualization of original dataset samples:")
    for i in range(2):
        full_dataset.visualize_puzzle(random.randint(0, len(full_dataset)-1))
    
    # Create balanced and imbalanced subsets
    print("\nCreating balanced and imbalanced subsets...")
    balanced_subset, imbalanced_subset = create_balanced_imbalanced_subsets(full_dataset)
    
    # Apply transformations to create augmented datasets
    print("\nCreating Augmented Balanced and Imbalanced datasets...")
    apply_transforms('./src/Dataset/8puzzle_balanced.pt', './src/Dataset/8puzzle_balanced_augmented.pt')
    apply_transforms('./src/Dataset/8puzzle_imbalanced.pt', './src/Dataset/8puzzle_imbalanced_augmented.pt')
    
    # Create train/test splits for each dataset
    print("\nCreating train/test splits for all datasets...")
    
    # Function to split and save datasets
    def split_and_save_dataset(dataset_path, base_path):
        # Load the dataset
        data = torch.load(dataset_path)
        puzzle_states = data['puzzle_states']
        puzzle_labels = data['puzzle_labels']
        
        # Calculate split point
        dataset_size = len(puzzle_states)
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size
        
        # Create indices for splitting
        indices = list(range(dataset_size))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create train and test datasets
        train_puzzles = [puzzle_states[i] for i in train_indices]
        train_labels = [puzzle_labels[i] for i in train_indices]
        test_puzzles = [puzzle_states[i] for i in test_indices]
        test_labels = [puzzle_labels[i] for i in test_indices]
        
        # Preprocess train and test datasets
        train_path = f"{base_path}_train.pt"
        test_path = f"{base_path}_test.pt"
        
        # Save preprocessed train data
        torch.save({
            'puzzle_states': train_puzzles,
            'puzzle_labels': train_labels
        }, train_path)
        print(f"Saved train split ({train_size} samples) to {train_path}")
        
        # Save preprocessed test data
        torch.save({
            'puzzle_states': test_puzzles,
            'puzzle_labels': test_labels
        }, test_path)
        print(f"Saved test split ({test_size} samples) to {test_path}")
        
        return train_path, test_path
    
    # Split and save the full dataset
    full_train_path, full_test_path = split_and_save_dataset(
        './src/Dataset/8puzzle_full.pt', './src/Dataset/8puzzle_full'
    )
    
    # Split and save balanced dataset
    balanced_train_path, balanced_test_path = split_and_save_dataset(
        './src/Dataset/8puzzle_balanced.pt', './src/Dataset/8puzzle_balanced'
    )
    
    # Split and save imbalanced dataset
    imbalanced_train_path, imbalanced_test_path = split_and_save_dataset(
        './src/Dataset/8puzzle_imbalanced.pt', './src/Dataset/8puzzle_imbalanced'
    )
    
    # Split and save augmented datasets
    balanced_aug_train_path, balanced_aug_test_path = split_and_save_dataset(
        './src/Dataset/8puzzle_balanced_augmented.pt', './src/Dataset/8puzzle_balanced_augmented'
    )
    
    imbalanced_aug_train_path, imbalanced_aug_test_path = split_and_save_dataset(
        './src/Dataset/8puzzle_imbalanced_augmented.pt', './src/Dataset/8puzzle_imbalanced_augmented'
    )

def train_and_evaluate_model():
    """Train and evaluate the 8-puzzle model"""
    print("\n=== Training and Evaluating 8-Puzzle Model ===")
    
    # Create trainer
    trainer = EightPuzzleTrainer('config.json')
    
    # Preprocess datasets before loading
    print("Preprocessing datasets...")
    preprocess_dataset('./src/Dataset/8puzzle_full_train.pt', './src/Dataset/8puzzle_full_train_preprocessed.pt')
    preprocess_dataset('./src/Dataset/8puzzle_full_test.pt', './src/Dataset/8puzzle_full_test_preprocessed.pt')
    
    # Load preprocessed data
    trainer.load_data(
        train_path='./src/Dataset/8puzzle_full_train_preprocessed.pt',
        val_path=None,  # Will be split from training data
        test_path='./src/Dataset/8puzzle_full_test_preprocessed.pt'
    )
    
    # Train model
    print("\nTraining model...")
    history = trainer.train()
    
    # Save model
    trainer.save_model('./src/Results/models/8puzzle_model_on_full_dataset.pt')
    
    # Test model
    print("\nTesting model...")
    test_loss, test_accuracy, test_precision, test_recall = trainer.test()
    
    # Plot training history
    trainer.plot_training_history('./src/Results/plots/training_history.png')
    
    return history, test_loss, test_accuracy, test_precision, test_recall

def cross_test_datasets():
    """Perform cross-testing between balanced and imbalanced datasets"""
    print("\n=== Cross-Testing Between Balanced and Imbalanced Datasets ===")
    
    # Create trainer
    trainer = EightPuzzleTrainer('config.json')
    
    # Preprocess the datasets
    print("Preprocessing datasets for cross-testing...")
    preprocess_dataset('./src/Dataset/8puzzle_balanced_train.pt', './src/Dataset/8puzzle_balanced_train_preprocessed.pt')
    preprocess_dataset('./src/Dataset/8puzzle_balanced_test.pt', './src/Dataset/8puzzle_balanced_test_preprocessed.pt')
    preprocess_dataset('./src/Dataset/8puzzle_imbalanced_train.pt', './src/Dataset/8puzzle_imbalanced_train_preprocessed.pt')
    preprocess_dataset('./src/Dataset/8puzzle_imbalanced_test.pt', './src/Dataset/8puzzle_imbalanced_test_preprocessed.pt')
    
    # Perform cross-testing
    results = trainer.cross_test(
        balanced_train_path='./src/Dataset/8puzzle_balanced_train_preprocessed.pt',
        balanced_test_path='./src/Dataset/8puzzle_balanced_test_preprocessed.pt',
        imbalanced_train_path='./src/Dataset/8puzzle_imbalanced_train_preprocessed.pt',
        imbalanced_test_path='./src/Dataset/8puzzle_imbalanced_test_preprocessed.pt'
    )
    
    print("\nCross-testing results:")
    print(f"Balanced → Imbalanced: Accuracy = {results['balanced_to_imbalanced']['accuracy']:.4f}")
    print(f"Imbalanced → Balanced: Accuracy = {results['imbalanced_to_balanced']['accuracy']:.4f}")
    
    return results

def main():
    """Main function to run the complete workflow"""
    # Setup environment
    setup_environment()
    
    mnist_data.main()
    
    # Generate datasets
    generate_datasets()
    
    # Train and evaluate model
    history, test_loss, test_accuracy, test_precision, test_recall = train_and_evaluate_model()
    
    # Cross-test datasets
    cross_test_results = cross_test_datasets()
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Cross-test Balanced → Imbalanced Accuracy: {cross_test_results['balanced_to_imbalanced']['accuracy']:.4f}")
    print(f"Cross-test Imbalanced → Balanced Accuracy: {cross_test_results['imbalanced_to_balanced']['accuracy']:.4f}")

if __name__ == "__main__":
    main()