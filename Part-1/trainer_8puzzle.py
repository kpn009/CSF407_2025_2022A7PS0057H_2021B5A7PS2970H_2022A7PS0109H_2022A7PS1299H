import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# Import custom modules
from model_8puzzle import EightPuzzleModel
from data_8puzzle import EightPuzzleDataset

class CustomDataset(Dataset):
    """
    Custom Dataset for loading saved puzzle states
    """
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to the data file
            transform (callable, optional): Optional transform to be applied on samples
        """
        data = torch.load(data_path)
        self.puzzle_states = data['puzzle_states']
        self.puzzle_labels = data['puzzle_labels']
        self.transform = transform
    
    def __len__(self):
        return len(self.puzzle_states)
    
    def __getitem__(self, idx):
        puzzle_state = self.puzzle_states[idx]
        puzzle_label = self.puzzle_labels[idx]
        
        if self.transform:
            transformed_state = torch.zeros_like(puzzle_state)
            for i in range(9):
                img = puzzle_state[i].unsqueeze(0)  # Add channel dimension for transforms
                transformed_state[i] = self.transform(img).squeeze()
            puzzle_state = transformed_state
        
        return puzzle_state, puzzle_label

class EightPuzzleTrainer:
    """
    Trainer for the 8-Puzzle model
    """
    def __init__(self, config_path):
        """
        Initialize the trainer with parameters from a configuration file
        
        Args:
            config_path (str): Path to the configuration file
        """
        
        # Store the config path
        self.config_path = config_path
    
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = EightPuzzleModel(config_path).to(self.device)
        
        # Initialize optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
        
        # Initialize learning rate scheduler if specified
        if 'scheduler' in self.config:
            scheduler_config = self.config['scheduler']
            if scheduler_config['type'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            elif scheduler_config['type'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config['T_max']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        
        # Print configuration
        print(f"Trainer initialized with configuration:")
        print(f"  Batch size: {self.config['data']['batch_size']}")
        print(f"  Learning rate: {optimizer_config['learning_rate']}")
        print(f"  Optimizer: {optimizer_config['type']}")
        
    def load_data(self, train_path, val_path=None, test_path=None):
        """
        Load training, validation, and test data
        
        Args:
            train_path (str): Path to the training data
            val_path (str, optional): Path to the validation data
            test_path (str, optional): Path to the test data
        """
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
        
        # Load training data
        train_dataset = CustomDataset(train_path, transform)
        
        # If validation path is not specified, split training data
        if val_path is None:
            val_size = int(len(train_dataset) * 0.2)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            val_dataset = CustomDataset(val_path, transform)
        
        # Load test data if specified
        if test_path is not None:
            test_dataset = CustomDataset(test_path, transform)
        else:
            test_dataset = None
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data'].get('num_workers', 0)
            )
        else:
            self.test_loader = None
        
        print(f"Data loaded:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        if test_dataset is not None:
            print(f"  Test samples: {len(test_dataset)}")
    
    def train_epoch(self):
        """
        Train the model for one epoch
        
        Returns:
            tuple: (mean_loss, accuracy, precision, recall)
        """
        self.model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Calculate loss for each position
            loss = 0
            for i in range(9):
                loss += self.criterion(outputs[i], target[:, i])
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            preds = self.model.predict(data)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        mean_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return mean_loss, accuracy, precision, recall
        
    def validate(self, data_loader=None):
        """
        Validate the model
        
        Args:
            data_loader (DataLoader, optional): Data loader to use for validation
            
        Returns:
            tuple: (mean_loss, accuracy, precision, recall)
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        # Use validation loader if not specified
        if data_loader is None:
            data_loader = self.val_loader
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Validating")):
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss for each position
                loss = 0
                for i in range(9):
                    loss += self.criterion(outputs[i], target[:, i])
                
                # Update total loss
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                preds = self.model.predict(data)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        mean_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return mean_loss, accuracy, precision, recall
    
    def train(self, num_epochs=None, early_stopping=None):
        """
        Train the model
        
        Args:
            num_epochs (int, optional): Number of epochs to train
            early_stopping (int, optional): Number of epochs without improvement to stop training
            
        Returns:
            dict: Dictionary containing training history
        """
        # Use config if not specified
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        if early_stopping is None and 'early_stopping' in self.config['training']:
            early_stopping = self.config['training']['early_stopping']
        
        print(f"Training for {num_epochs} epochs...")
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        # Start training
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_accuracy, train_precision, train_recall = self.train_epoch()
            val_loss, val_accuracy, val_precision, val_recall = self.validate()
            
            # Update learning rate scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Learning rate: {current_lr:.6f}")
            
            # Print metrics
            print(f"  Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}, precision: {train_precision:.4f}, recall: {train_recall:.4f}")
            print(f"  Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}, precision: {val_precision:.4f}, recall: {val_recall:.4f}")
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.train_precisions.append(train_precision)
            self.val_precisions.append(val_precision)
            self.train_recalls.append(train_recall)
            self.val_recalls.append(val_recall)
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = self.model.state_dict()
                print(f"  New best model with validation loss: {best_val_loss:.4f}")
            elif early_stopping and epoch - best_epoch >= early_stopping:
                print(f"Early stopping triggered. No improvement for {early_stopping} epochs.")
                break
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        
        # Load best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'train_precisions': self.train_precisions,
            'val_precisions': self.val_precisions,
            'train_recalls': self.train_recalls,
            'val_recalls': self.val_recalls,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }
    
    def test(self, test_loader=None):
        """
        Test the model
        
        Args:
            test_loader (DataLoader, optional): Data loader to use for testing
            
        Returns:
            tuple: (loss, accuracy, precision, recall)
        """
        # Use test loader if not specified
        if test_loader is None:
            if self.test_loader is not None:
                test_loader = self.test_loader
            else:
                print("No test loader available. Using validation loader.")
                test_loader = self.val_loader
        
        print("Testing model...")
        loss, accuracy, precision, recall = self.validate(test_loader)
        
        print(f"Test results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return loss, accuracy, precision, recall
    
    def save_model(self, path):
        """
        Save the model
        
        Args:
            path (str): Path to save the model
        """
        self.model.save_model(path)
    
    def load_model(self, path):
        """
        Load a model
        
        Args:
            path (str): Path to load the model from
        """
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str, optional): Path to save the plots
        """
        # Create directory if it doesn't exist
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Train')
        plt.plot(self.val_accuracies, label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot precision
        plt.subplot(2, 2, 3)
        plt.plot(self.train_precisions, label='Train')
        plt.plot(self.val_precisions, label='Validation')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(self.train_recalls, label='Train')
        plt.plot(self.val_recalls, label='Validation')
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Training history plots saved to {save_path}")
        
        plt.show()
    
    def cross_test(self, balanced_train_path, balanced_test_path, imbalanced_train_path, imbalanced_test_path):
        """
        Train on balanced dataset and test on imbalanced dataset, and vice versa
        
        Args:
            balanced_path (str): Path to balanced dataset
            imbalanced_path (str): Path to imbalanced dataset
            
        Returns:
            dict: Dictionary containing test results
        """
        results = {}
        
        # Reset model and metrics
        self.model = EightPuzzleModel(self.config_path).to(self.device)
        
        # Re-initialize optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        
        # Reset metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        
        # Train on balanced, test on imbalanced
        print("\n=== Training on balanced dataset, testing on imbalanced dataset ===")
        self.load_data(balanced_train_path, test_path=imbalanced_test_path)
        history_balanced = self.train()
        loss_b2i, accuracy_b2i, precision_b2i, recall_b2i = self.test()
        
        self.plot_training_history('./src/Results/plots/tb2ti_metrics.png')
        
        # Save model
        os.makedirs('./src/Results/models', exist_ok=True)
        self.save_model('./src/Results/models/model_tb2ti.pt')
        
        # Reset model and metrics
        self.model = EightPuzzleModel(self.config_path).to(self.device)
        
        # Re-initialize optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=optimizer_config.get('momentum', 0),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        
        # Reset metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        
        # Train on imbalanced, test on balanced
        print("\n=== Training on imbalanced dataset, testing on balanced dataset ===")
        self.load_data(imbalanced_train_path, test_path=balanced_test_path)
        history_imbalanced = self.train()
        loss_i2b, accuracy_i2b, precision_i2b, recall_i2b = self.test()
        
        self.plot_training_history('./src/Results/plots/ti2tb_metrics.png')
        
        # Save model
        self.save_model('./src/Results/models/model_ti2tb.pt')
        
        # Save results
        results = {
            'balanced_to_imbalanced': {
                'loss': loss_b2i,
                'accuracy': accuracy_b2i,
                'precision': precision_b2i,
                'recall': recall_b2i,
                'history': history_balanced
            },
            'imbalanced_to_balanced': {
                'loss': loss_i2b,
                'accuracy': accuracy_i2b,
                'precision': precision_i2b,
                'recall': recall_i2b,
                'history': history_imbalanced
            }
        }
        return results

# Example usage
if __name__ == "__main__":
    # Create a sample configuration file
    config = {
        "model_config": "config.json",
        "model": {
            "input_size": 28*28,
            "hidden_layers": [256, 128],
            "output_size": 9,  # 9 different digits (0-8)
            "dropout_rate": 0.2
        },
        "optimizer": {
            "type": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5
        },
        "scheduler": {
            "type": "step",
            "step_size": 10,
            "gamma": 0.1
        },
        "data": {
            "batch_size": 64,
            "num_workers": 2
        },
        "training": {
            "num_epochs": 20,
            "early_stopping": 5
        }
    }
    
    # Save the configuration
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create trainer
    trainer = EightPuzzleTrainer('config.json')
    
    # Load data
    trainer.load_data(
        train_path='../Dataset/8puzzle_balanced_augmented.pt',
        val_path=None,  # Will be split from training data
        test_path='../Dataset/8puzzle_imbalanced_augmented.pt'
    )
    
    # Train model
    history = trainer.train()
    
    # Test model
    trainer.test()
    
    # Plot training history
    trainer.plot_training_history('plots/training_history.png')
    
    # Perform cross-testing
    results = trainer.cross_test(
        balanced_path='../Dataset/8puzzle_balanced_augmented.pt',
        imbalanced_path='../Dataset/8puzzle_imbalanced_augmented.pt'
    )
    
    print("Cross-testing results:")
    print(f"Balanced → Imbalanced: Accuracy = {results['balanced_to_imbalanced']['accuracy']:.4f}")
    print(f"Imbalanced → Balanced: Accuracy = {results['imbalanced_to_balanced']['accuracy']:.4f}")
    
    def validate(self, data_loader=None):
        """
        Validate the model
        
        Args:
            data_loader (DataLoader, optional): Data loader to use for validation
            
        Returns:
            tuple: (mean_loss, accuracy, precision, recall)
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        
        # Use validation loader if not specified
        if data_loader is None:
            data_loader = self.val_loader
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Validating")):
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss for each position
                loss = 0
                for i in range(9):
                    loss += self.criterion(outputs[i], target[:, i])
                
                # Update total loss
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                preds = self.model.predict(data)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        mean_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return mean_loss, accuracy, precision, recall