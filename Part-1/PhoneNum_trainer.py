import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our model and data modules
try:
    from PhoneNum_model import PhoneNumberMLP, load_model_from_config, save_model
    from PhoneNum_data import get_data_loaders
except ImportError:
    # Try alternative import paths
    from PhoneNumModel import PhoneNumberMLP, load_model_from_config, save_model
    from PhoneNumData import get_data_loaders

class PhoneNumberTrainer:
    """
    Trainer class for the phone number recognition model
    """
    def __init__(self, config_path, device=None):
        """
        Initialize the trainer with configuration
        
        Args:
            config_path: Path to the configuration JSON file
            device: Device to use for training (cpu or cuda)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model, _ = load_model_from_config(config_path)
        self.model = self.model.to(self.device)
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['learning_rate'], 
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # Loss function - CrossEntropyLoss for each digit
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # History for tracking metrics
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_accuracy': [], 
            'val_accuracy': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': []
        }
        
        # Set paths for saving results
        self.results_dir = self.config['paths']['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Best validation loss for model checkpointing
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training set
            
        Returns:
            avg_loss: Average loss for the epoch
            metrics: Dictionary of performance metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # For calculating metrics
        all_preds = []
        all_targets = []
        
        for images, targets in tqdm(train_loader, desc="Training"):
            # Move data to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss for each digit position
            loss = 0
            batch_preds = []
            
            for i, output in enumerate(outputs):
                digit_target = targets[:, i]
                loss += self.criterion(output, digit_target)
                
                # Get predictions
                _, preds = torch.max(output, 1)
                batch_preds.append(preds.cpu().numpy())
            
            # Average loss across all digits
            loss = loss / len(outputs)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Collect predictions and targets for metrics
            batch_preds = np.array(batch_preds).T  # Shape: [batch_size, num_digits]
            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Reshape to calculate metrics across all digits
        all_preds_flat = all_preds.flatten()
        all_targets_flat = all_targets.flatten()
        
        accuracy = accuracy_score(all_targets_flat, all_preds_flat)
        precision = precision_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        recall = recall_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation set
            
        Returns:
            avg_loss: Average loss for validation
            metrics: Dictionary of performance metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # For calculating metrics
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss for each digit position
                loss = 0
                batch_preds = []
                
                for i, output in enumerate(outputs):
                    digit_target = targets[:, i]
                    loss += self.criterion(output, digit_target)
                    
                    # Get predictions
                    _, preds = torch.max(output, 1)
                    batch_preds.append(preds.cpu().numpy())
                
                # Average loss across all digits
                loss = loss / len(outputs)
                
                # Track loss
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Collect predictions and targets for metrics
                batch_preds = np.array(batch_preds).T  # Shape: [batch_size, num_digits]
                all_preds.append(batch_preds)
                all_targets.append(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Reshape to calculate metrics across all digits
        all_preds_flat = all_preds.flatten()
        all_targets_flat = all_targets.flatten()
        
        accuracy = accuracy_score(all_targets_flat, all_preds_flat)
        precision = precision_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        recall = recall_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, metrics
    
    def train(self, data_loaders):
        """
        Train the model
        
        Args:
            data_loaders: Dictionary of DataLoaders for training and validation
        """
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Train Precision: {train_metrics['precision']:.4f}, Val Precision: {val_metrics['precision']:.4f}")
            print(f"Train Recall: {train_metrics['recall']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_save_path = os.path.join(self.results_dir, f"best_model.pth")
                save_model(self.model, model_save_path, self.config)
                print(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                model_save_path = os.path.join(self.results_dir, f"checkpoint_epoch_{epoch+1}.pth")
                save_model(self.model, model_save_path, self.config)
        
        # Save final model
        model_save_path = os.path.join(self.results_dir, "final_model.pth")
        save_model(self.model, model_save_path, self.config)
        
        # Plot and save training history
        self.plot_training_history()
    
    def test(self, test_loader, dataset_name="test"):
        """
        Test the model on a test set
        
        Args:
            test_loader: DataLoader for test set
            dataset_name: Name of the dataset for logging
            
        Returns:
            metrics: Dictionary of performance metrics
        """
        print(f"\nTesting on {dataset_name} dataset")
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        # For calculating metrics
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Testing {dataset_name}"):
                # Move data to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss for each digit position
                loss = 0
                batch_preds = []
                
                for i, output in enumerate(outputs):
                    digit_target = targets[:, i]
                    loss += self.criterion(output, digit_target)
                    
                    # Get predictions
                    _, preds = torch.max(output, 1)
                    batch_preds.append(preds.cpu().numpy())
                
                # Average loss across all digits
                loss = loss / len(outputs)
                
                # Track loss
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Collect predictions and targets for metrics
                batch_preds = np.array(batch_preds).T  # Shape: [batch_size, num_digits]
                all_preds.append(batch_preds)
                all_targets.append(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Reshape to calculate metrics across all digits
        all_preds_flat = all_preds.flatten()
        all_targets_flat = all_targets.flatten()
        
        accuracy = accuracy_score(all_targets_flat, all_preds_flat)
        precision = precision_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        recall = recall_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot and save training history metrics
        """
        # Create figures directory
        figures_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'loss.png'))
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_accuracy'], label='Training Accuracy')
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'accuracy.png'))
        plt.close()
        
        # Plot precision
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_precision'], label='Training Precision')
        plt.plot(self.history['val_precision'], label='Validation Precision')
        plt.title('Precision Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'precision.png'))
        plt.close()
        
        # Plot recall
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_recall'], label='Training Recall')
        plt.plot(self.history['val_recall'], label='Validation Recall')
        plt.title('Recall Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'recall.png'))
        plt.close()
    
    def cross_dataset_evaluation(self, loaders):
        """
        Evaluate model trained on one dataset type on another dataset type
        
        Args:
            loaders: Dictionary containing cross-dataset loaders
        """
        # Evaluate balanced-trained model on imbalanced test data
        balanced_on_imbalanced = self.test(
            loaders['cross']['balanced_train_imbalanced_test'],
            dataset_name="balanced_train_imbalanced_test"
        )
        
        # Evaluate imbalanced-trained model on balanced test data
        imbalanced_on_balanced = self.test(
            loaders['cross']['imbalanced_train_balanced_test'],
            dataset_name="imbalanced_train_balanced_test"
        )
        
        # Save results for comparison
        cross_metrics = {
            'balanced_on_imbalanced': balanced_on_imbalanced,
            'imbalanced_on_balanced': imbalanced_on_balanced
        }
        
        # Save metrics to JSON
        with open(os.path.join(self.results_dir, 'cross_dataset_metrics.json'), 'w') as f:
            json.dump(cross_metrics, f, indent=4)
        
        # Plot comparison
        metrics = ['accuracy', 'precision', 'recall']
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            
            # Set bar positions
            x = np.array([0, 1])
            width = 0.35
            
            # Plot bars
            plt.bar(x[0], balanced_on_imbalanced[metric], width, label='Balanced → Imbalanced')
            plt.bar(x[1], imbalanced_on_balanced[metric], width, label='Imbalanced → Balanced')
            
            # Add labels and title
            plt.xlabel('Training → Testing')
            plt.ylabel(metric.capitalize())
            plt.title(f'Cross-Dataset {metric.capitalize()} Comparison')
            plt.xticks(x, ['Balanced→Imbalanced', 'Imbalanced→Balanced'])
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(self.results_dir, 'figures', f'cross_{metric}.png'))
            plt.close()
        
        return cross_metrics


def main():
    """
    Main function to run the training process
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Phone Number Recognition Model")
    parser.add_argument('--config', type=str, default='../src/config/phone_config.json', 
                      help='Path to configuration file')
    parser.add_argument('--balanced_data', type=str, default=None, 
                      help='Path to balanced dataset file')
    parser.add_argument('--imbalanced_data', type=str, default=None, 
                      help='Path to imbalanced dataset file')
    parser.add_argument('--create_new_data', action='store_true', 
                      help='Whether to create new datasets')
    parser.add_argument('--dataset_type', type=str, choices=['balanced', 'imbalanced'], default='balanced',
                      help='Type of dataset to train on')
    
    args = parser.parse_args()
    
    # Ensure config directory exists
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    
    # Default configuration with all required parameters
    default_config = {
        "model": {
            "input_size": 1 * 28 * 280,
            "hidden_sizes": [512, 256],
            "num_digits": 10,
            "num_classes": 10,
            "dropout_rate": 0.3
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 30,
            "checkpoint_interval": 5
        },
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "num_samples": 20000,
            "save_dir": "../src/Dataset/",
            "balanced_path": None,
            "imbalanced_path": None
        },
        "paths": {
            "results_dir": f"../src/results/phone_{args.dataset_type}"
        }
    }
    
    # Load or create config file
    if not os.path.exists(args.config):
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default configuration at {args.config}")
    else:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Ensure all required sections and parameters exist
        for section in ['model', 'training', 'data', 'paths']:
            if section not in config:
                config[section] = default_config[section]
            else:
                # Update missing parameters in each section
                for key, value in default_config[section].items():
                    if key not in config[section]:
                        config[section][key] = value
        
        # Save updated config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
    
    # Default data paths if not provided
    if args.balanced_data is None:
        args.balanced_data = os.path.join("../src/Dataset/", "phone_balanced.pt")
    if args.imbalanced_data is None:
        args.imbalanced_data = os.path.join("../src/Dataset/", "phone_imbalanced.pt")
    
    # Load data loaders
    try:
        data_loaders, balanced_path, imbalanced_path = get_data_loaders(
            balanced_path=args.balanced_data if not args.create_new_data else None,
            imbalanced_path=args.imbalanced_data if not args.create_new_data else None,
            create_new=args.create_new_data,
            batch_size=config['training']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            num_samples=config['data']['num_samples'],
            save_dir=config['data']['save_dir']
        )
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        print("Creating new datasets...")
        data_loaders, balanced_path, imbalanced_path = get_data_loaders(
            create_new=True,
            batch_size=config['training']['batch_size'],
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            num_samples=config['data']['num_samples'],
            save_dir=config['data']['save_dir']
        )
    
    # Update config with data paths
    config['data']['balanced_path'] = balanced_path
    config['data']['imbalanced_path'] = imbalanced_path
    
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize trainer
    trainer = PhoneNumberTrainer(args.config)
    
    # Train model
    trainer.train(data_loaders[args.dataset_type])
    
    # Test on test set
    test_metrics = trainer.test(data_loaders[args.dataset_type]['test'])
    
    # Cross-dataset evaluation
    cross_metrics = trainer.cross_dataset_evaluation(data_loaders)
    
    print("Training and evaluation complete!")
    return test_metrics, cross_metrics


if __name__ == "__main__":
    main()