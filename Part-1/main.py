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

def predict_from_image_demo(model_path, image_path):
    """
    Simple function to load a model and predict digits from an image
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the input image
    """
    # Create trainer with the same config used for training
    config_path = 'config.json'
    trainer = EightPuzzleTrainer(config_path)
    
    # Load the model
    trainer.load_model(model_path)
    
    # Predict from image
    predictions = trainer.predict_from_image(image_path)
    
    # Display the predictions
    if predictions is not None:
        # Display as a grid
        plt.figure(figsize=(8, 8))
        
        # Create a grid
        for i in range(3):
            for j in range(3):
                # Get the predicted digit
                digit = int(predictions[i, j])
                
                # Create a subplot
                plt.subplot(3, 3, i*3 + j + 1)
                
                # Display the digit
                plt.text(0.5, 0.5, str(digit), fontsize=50, ha='center', va='center')
                
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
                print(f"{int(predictions[i, j])} | ", end="")
            print("\n" + "-" * 13)
    
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
    full_dataset = EightPuzzleDataset(root_dir='./src/Dataset', transform=transform)
    
    # Save full dataset
    full_dataset.save_dataset('./src/Dataset/8puzzle_full.pt')
    
    # Visualize some samples
    print("\nVisualization of original dataset samples:")
    for i in range(2):
        full_dataset.visualize_puzzle(random.randint(0, len(full_dataset)-1))
    
    # Create balanced and imbalanced subsets
    print("\nCreating balanced and imbalanced subsets...")
    balanced_subset, imbalanced_subset = create_balanced_imbalanced_subsets(full_dataset)
    
    # Create balanced and imbalanced subsets
    print("\nCreating Augmented Balanced and Imbalanced datasets...")
    apply_transforms('./src/Dataset/8puzzle_balanced.pt', './src/Dataset/8puzzle_balanced_augmented.pt')
    apply_transforms('./src/Dataset/8puzzle_imbalanced.pt', './src/Dataset/8puzzle_imbalanced_augmented.pt')
    
    print("\nVisualization of balanced augmented dataset:")
    visualize_dataset_samples('./src/Dataset/8puzzle_balanced_augmented.pt', num_samples=1)
    
    print("\nVisualization of imbalanced augmented dataset:")
    visualize_dataset_samples('./src/Dataset/8puzzle_imbalanced_augmented.pt', num_samples=1)
    
    # Create train/test splits for each dataset
    print("\nCreating train/test splits for all datasets...")
    
    # Function to split and save datasets
    def split_and_save_dataset(dataset, base_path):
        # Calculate split point
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        
        # Random split
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Save splits
        train_path = f"{base_path}_train.pt"
        test_path = f"{base_path}_test.pt"
        
        # Extract and save train data
        train_puzzles = torch.stack([dataset[i][0] for i in train_dataset.indices])
        train_labels = torch.stack([dataset[i][1] for i in train_dataset.indices])
        torch.save({'puzzle_states': train_puzzles, 'puzzle_labels': train_labels}, train_path)
        print(f"Saved train split ({train_size} samples) to {train_path}")
        
        # Extract and save test data
        test_puzzles = torch.stack([dataset[i][0] for i in test_dataset.indices])
        test_labels = torch.stack([dataset[i][1] for i in test_dataset.indices])
        torch.save({'puzzle_states': test_puzzles, 'puzzle_labels': test_labels}, test_path)
        print(f"Saved test split ({test_size} samples) to {test_path}")
        
        return train_path, test_path
    
    # Split and save the full dataset
    full_train_path, full_test_path = split_and_save_dataset(
        full_dataset, './src/Dataset/8puzzle_full'
    )
    
    # Split and save balanced dataset
    balanced_train_path, balanced_test_path = split_and_save_dataset(
        balanced_subset, './src/Dataset/8puzzle_balanced'
    )
    
    # Split and save imbalanced dataset
    imbalanced_train_path, imbalanced_test_path = split_and_save_dataset(
        imbalanced_subset, './src/Dataset/8puzzle_imbalanced'
    )
    
    # # Apply transforms to the train and test splits
    # print("\nApplying transforms to train/test splits...")
    
    # # Function to apply transforms and save
    # def apply_transforms_to_split(input_path, output_path):
    #     apply_transforms(input_path, output_path)
    #     print(f"Applied transforms to {input_path} and saved to {output_path}")
    #     return output_path
    
    # # Apply transforms to balanced train/test
    # balanced_train_aug_path = apply_transforms_to_split(
    #     balanced_train_path, './src/Dataset/8puzzle_balanced_augmented_train.pt'
    # )
    # balanced_test_aug_path = apply_transforms_to_split(
    #     balanced_test_path, './src/Dataset/8puzzle_balanced_augmented_test.pt'
    # )
    
    # # Apply transforms to imbalanced train/test
    # imbalanced_train_aug_path = apply_transforms_to_split(
    #     imbalanced_train_path, './src/Dataset/8puzzle_imbalanced_augmented_train.pt'
    # )
    # imbalanced_test_aug_path = apply_transforms_to_split(
    #     imbalanced_test_path, './src/Dataset/8puzzle_imbalanced_augmented_test.pt'
    # )
    
    # # Visualize samples from augmented datasets
    # print("\nVisualization of balanced augmented train dataset:")
    # visualize_dataset_samples(balanced_train_aug_path, num_samples=1)
    
    # print("\nVisualization of balanced augmented test dataset:")
    # visualize_dataset_samples(balanced_test_aug_path, num_samples=1)
    
    # print("\nVisualization of imbalanced augmented train dataset:")
    # visualize_dataset_samples(imbalanced_train_aug_path, num_samples=1)
    
    # print("\nVisualization of imbalanced augmented test dataset:")
    # visualize_dataset_samples(imbalanced_test_aug_path, num_samples=1)

def train_and_evaluate_model():
    """Train and evaluate the 8-puzzle model"""
    print("\n=== Training and Evaluating 8-Puzzle Model ===")
    
    # Create trainer
    trainer = EightPuzzleTrainer('config.json')
    
    # Load data
    trainer.load_data(
        train_path='./src/Dataset/8puzzle_full_train.pt',
        val_path=None,  # Will be split from training data
        test_path='./src/Dataset/8puzzle_full_test.pt'
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
    
    # Perform cross-testing
    results = trainer.cross_test(
        balanced_train_path='./src/Dataset/8puzzle_balanced_train.pt',
        balanced_test_path='./src/Dataset/8puzzle_balanced_test.pt',
        imbalanced_train_path='./src/Dataset/8puzzle_imbalanced_train.pt',
        imbalanced_test_path='./src/Dataset/8puzzle_imbalanced_test.pt'
    )
    
    print("\nCross-testing results:")
    print(f"Balanced → Imbalanced: Accuracy = {results['balanced_to_imbalanced']['accuracy']:.4f}")
    print(f"Imbalanced → Balanced: Accuracy = {results['imbalanced_to_balanced']['accuracy']:.4f}")
    
    return results

def main():
    """Main function to run the complete workflow"""
    # Setup environment
    setup_environment()
    
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
    
    # print("\nTesting prediction from image...")
    # model_path = 'models/8puzzle_model.pt'  # Use your saved model path
    # image_path = 'image.png'           # Path to your test image
    
    # # Run prediction
    # prediction_result = predict_from_image_demo(model_path, image_path)
    
    # print("Prediction complete!")

if __name__ == "__main__":
    main()