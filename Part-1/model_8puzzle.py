import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class EightPuzzleModel(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for classifying 8-puzzle states
    """
    def __init__(self, config_path):
        """
        Initialize the model with parameters from a configuration file
        
        Args:
            config_path (str): Path to the configuration file
        """
        super(EightPuzzleModel, self).__init__()
        
        # Load configuration from file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract model parameters
        input_size = config['model']['input_size']
        hidden_layers = config['model']['hidden_layers']
        output_size = config['model']['output_size']
        dropout_rate = config['model']['dropout_rate']
        
        # Create feature extractor for each position in the puzzle
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(28*28, hidden_layers[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(9)
        ])
        
        # Create hidden layers
        layers = []
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Create classifier for each position
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                *layers,
                nn.Linear(hidden_layers[-1], output_size)
            ) for _ in range(9)
        ])
        
        # Print model architecture
        print(f"Model initialized with:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Output size: {output_size}")
        print(f"  Dropout rate: {dropout_rate}")
        
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 9, 28, 28)
            
        Returns:
            list: List of 9 tensors, each of shape (batch_size, output_size)
        """
        batch_size = x.shape[0]
        
        # Flatten each position's image
        x = x.view(batch_size, 9, -1)
        
        # Extract features for each position
        features = [self.feature_extractors[i](x[:, i]) for i in range(9)]
        
        # Classify each position
        outputs = [self.classifiers[i](features[i]) for i in range(9)]
        
        return outputs
    
    def predict(self, x):
        """
        Predict labels for each position in the puzzle
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 9, 28, 28)
            
        Returns:
            torch.Tensor: Predicted labels of shape (batch_size, 9)
        """
        outputs = self.forward(x)
        preds = [torch.argmax(output, dim=1) for output in outputs]
        return torch.stack(preds, dim=1)
    
    def save_model(self, path):
        """
        Save the model to a file
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path, config_path):
        """
        Load a model from a file
        
        Args:
            path (str): Path to load the model from
            config_path (str): Path to the configuration file
            
        Returns:
            EightPuzzleModel: Loaded model
        """
        model = EightPuzzleModel(config_path)
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return model

# Example usage
if __name__ == "__main__":
    # Create a sample configuration file
    config = {
        "model": {
            "input_size": 28*28,
            "hidden_layers": [256, 128],
            "output_size": 9,  # 9 different digits (0-8)
            "dropout_rate": 0.2
        }
    }
    
    # Save the configuration
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create model
    model = EightPuzzleModel('config.json')
    
    # Test forward pass
    x = torch.randn(4, 9, 28, 28)  # Batch size 4, 9 positions, 28x28 images
    outputs = model(x)
    
    # Print output shapes
    print("\nModel outputs:")
    for i, output in enumerate(outputs):
        print(f"Position {i}: {output.shape}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0]}")