# 8-Puzzle Solver with Neural Networks and LLM

This repository contains a comprehensive solution for the 8-Puzzle problem using a combination of Neural Networks (for state recognition) and Large Language Models (for puzzle solving). The project also includes additional modules for phone number recognition to demonstrate transfer learning concepts.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [8-Puzzle Solver](#8-puzzle-solver)
- [Visualization Tools](#visualization-tools)
- [Phone Number Recognition](#phone-number-recognition)
- [Configuration](#configuration)
- [Examples](#examples)

## Overview

The 8-Puzzle is a classic sliding puzzle where the goal is to arrange 8 numbered tiles on a 3×3 grid. This project presents an end-to-end solution:

1. **Neural Network**: Recognizes the digits in the 8-puzzle state from images
2. **LLM Integration**: Uses Google's Gemini LLM to find the optimal solution path
3. **Visualization**: Tools to visualize the solution steps

Additionally, the project includes a phone number recognition model to demonstrate concepts of transfer learning and dataset imbalance handling.

## Project Structure

The codebase consists of several modules:

### 8-Puzzle Core Modules

- **Data Generation** (`data_8puzzle.py`): Creates MNIST-based datasets for 8-puzzle states
- **Model** (`model_8puzzle.py`): Neural network architecture for digit recognition
- **Trainer** (`trainer_8puzzle.py`): Training and evaluation pipeline
- **Utils** (`eight_puzzle_utils.py`): Utility functions for image processing and LLM integration
- **Solver** (`part2a_solver.py`): End-to-end solution combining NN and LLM

### Visualization Tools

- **State Visualizer** (`visualize_states.py`): Visualizes the sequence of states in a solution
- **Puzzle Visualizer** (`puzzle_visualizer.py`): Advanced visualization tools with animations
- **Dataset Loader** (`load_dataset.py`): Extracts and saves puzzle images from datasets

### Phone Number Recognition (Transfer Learning Demo)

- **Data** (`PhoneNum_data.py`): Generates phone number datasets
- **Model** (`PhoneNum_model.py`): CNN-based model for phone number recognition
- **Trainer** (`PhoneNum_trainer.py`): Training and evaluation pipeline

### Configuration and Main

- **Config Files** (`config.json`, `phone_config.json`): Configuration parameters
- **Main Script** (`main.py`): Entry point for dataset generation and model training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/8-puzzle-solver.git
cd 8-puzzle-solver

# Install dependencies
pip install -r requirements.txt

# Set environment variable for Gemini API
export GEMINI_API_KEY=your_api_key_here
```

Required dependencies:
- torch
- torchvision
- Pillow
- google-generativeai
- numpy
- matplotlib
- scikit-learn
- tqdm

## Dataset Generation

The project uses MNIST digits to create 8-puzzle datasets:

```bash
# Generate datasets with balanced and imbalanced variants
python main.py
```

This creates several dataset variants:
- Full dataset
- Balanced subset
- Imbalanced subset (with deliberate class imbalance)
- Augmented versions of each

## Model Architecture

### 8-Puzzle Model

The model architecture for digit recognition consists of:
- Feature extractors for each position in the puzzle
- Hidden layers with ReLU activations and dropout
- Classifiers for each position

```python
# Example model initialization
from model_8puzzle import EightPuzzleModel
model = EightPuzzleModel('config.json')
```

### Phone Number Model

A CNN-based model for phone number recognition:
- Convolutional layers for feature extraction
- MLP layers for classification
- Output classifiers for each digit position

## Training and Evaluation

### Training the 8-Puzzle Model

```bash
# Train and evaluate the model
python main.py
```

The training process includes:
- Model training on balanced/imbalanced datasets
- Cross-testing between datasets
- Performance metrics (accuracy, precision, recall)
- Visualization of training metrics

### Cross-Dataset Evaluation

The project includes tools to evaluate model performance across different dataset variants:

```python
# Example from trainer_8puzzle.py
cross_test_results = trainer.cross_test(
    balanced_train_path='./src/Dataset/8puzzle_balanced_train.pt',
    balanced_test_path='./src/Dataset/8puzzle_balanced_test.pt',
    imbalanced_train_path='./src/Dataset/8puzzle_imbalanced_train.pt',
    imbalanced_test_path='./src/Dataset/8puzzle_imbalanced_test.pt'
)
```

## 8-Puzzle Solver

The end-to-end solver combines neural network inference with Gemini LLM:

```bash
python part2a_solver.py \
    --config src/config.json \
    --checkpoint src/Results/models/8puzzle_model_on_full_dataset.pt \
    --source valid_puzzle_state_1.png \
    --goal valid_puzzle_state_2.png \
    --output states.json
```

### Solver Pipeline:

1. Neural network recognizes digits in source and goal images
2. Gemini LLM generates a sequence of states to solve the puzzle
3. Results are saved to a JSON file for visualization

## Visualization Tools

### Visualizing Solution Steps

```bash
python visualize_states.py states.json --output visualization.png
```

This creates a grid visualization of all steps in the solution.

### Advanced Visualization

The `puzzle_visualizer.py` module provides advanced visualization options:
- Static visualizations of solution paths
- Animated GIFs of solutions
- Comparative visualizations of different solution methods

## Phone Number Recognition

The project includes a separate module for phone number recognition as an example of transfer learning:

```bash
# Train the phone number recognition model
python PhoneNum_trainer.py --config src/config/phone_config.json --dataset_type balanced
```

### Phone Number Dataset Generation

```python
# From PhoneNum_data.py
# Generate balanced and imbalanced datasets
balanced_dataset = PhoneNumberDataset(num_samples=20000, is_balanced=True)
imbalanced_dataset = PhoneNumberDataset(num_samples=20000, is_balanced=False)
```

## Configuration

Configuration files control model architecture and training parameters:

### 8-Puzzle Configuration (`config.json`)

```json
{
    "model": {
        "input_size": 784,
        "hidden_layers": [256, 128],
        "output_size": 9,
        "dropout_rate": 0.2
    },
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-05
    },
    "scheduler": {
        "type": "step",
        "step_size": 10,
        "gamma": 0.1
    },
    "data": {
        "batch_size": 64,
        "num_workers": 2
    }
}
```

### Phone Number Configuration (`phone_config.json`)

```json
{
    "model": {
        "input_size": 7840,
        "hidden_sizes": [512, 256],
        "num_digits": 10,
        "num_classes": 10,
        "dropout_rate": 0.3
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs": 15
    }
}
```

## Examples

### Example 1: Generate Datasets and Train the Model

```bash
# Generate datasets and train the model
python main.py
```

### Example 2: Extract Images from Dataset for Testing

```bash
# Extract puzzle state images from test dataset
python load_dataset.py
```

### Example 3: Solving a Puzzle with the End-to-End Pipeline

```bash
# Solve a puzzle using neural network and LLM
python part2a_solver.py \
    --checkpoint src/Results/models/8puzzle_model_on_full_dataset.pt \
    --source valid_puzzle_state_1.png \
    --goal valid_puzzle_state_2.png
    
# Visualize the solution
python visualize_states.py states.json
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 