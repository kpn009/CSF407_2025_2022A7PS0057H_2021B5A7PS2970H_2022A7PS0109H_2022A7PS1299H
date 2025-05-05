import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from preprocess import preprocess_puzzle_image

def remove_white_edges(img):
    """
    Remove white edges from a screenshot by finding the content bounding box
    
    Args:
        img: PIL Image
        
    Returns:
        PIL Image: Cropped image with white edges removed
    """
    # Convert to numpy array for processing
    img_array = np.array(img)
    
    # If it's RGB, convert to grayscale for edge detection
    if len(img_array.shape) == 3:
        # If it's RGB, convert to grayscale
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Invert the image if it has white background
    if np.mean(gray) > 128:
        gray = 255 - gray
    
    # Find non-zero regions (content)
    rows = np.any(gray > 20, axis=1)
    cols = np.any(gray > 20, axis=0)
    
    # Get the content bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add some padding
    padding = 5
    rmin = max(0, rmin - padding)
    rmax = min(gray.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(gray.shape[1] - 1, cmax + padding)
    
    # Crop the image
    return img.crop((cmin, rmin, cmax, rmax))

def load_and_preprocess_real_image(image_path, visualize=False):
    """
    Load a real-world puzzle image and preprocess it for the model.
    Handles screenshots with white edges.
    
    Args:
        image_path (str): Path to the image file
        visualize (bool): If True, visualize the preprocessing steps
        
    Returns:
        torch.Tensor: Preprocessed image as a tensor of shape (9, 28, 28)
    """
    # Load the image
    img = Image.open(image_path)
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
    
    # First, remove any white edges (e.g., from screenshots)
    img = remove_white_edges(img)
    
    if visualize:
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        plt.title("After Edge Removal")
    
    # Ensure the image is square by taking the largest dimension
    width, height = img.size
    new_size = max(width, height)
    
    # Create a square image with black background
    square_img = Image.new('L', (new_size, new_size), 0)
    
    # Paste the original image in the center
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    square_img.paste(img, (paste_x, paste_y))
    
    # Resize to 84x84 using LANCZOS resampling for high quality
    img_84x84 = square_img.resize((84, 84), Image.Resampling.LANCZOS)
    
    # Apply contrast enhancement to make digits more visible
    enhancer = transforms.functional.adjust_contrast
    img_84x84 = enhancer(img_84x84, 1.5)  # Increase contrast by 50%
    
    if visualize:
        plt.subplot(1, 3, 3)
        plt.imshow(img_84x84, cmap='gray')
        plt.title("Final 84x84 Image")
        plt.tight_layout()
        plt.show()
    
    # Convert to tensor (values between 0 and 1)
    img_tensor = transforms.ToTensor()(img_84x84).squeeze(0)  # Shape (84, 84)
    
    # Use the preprocess_puzzle_image function to split into 9 tiles and normalize
    processed_tiles = preprocess_puzzle_image(img_tensor, normalize=True)
    
    # Optionally visualize the 9 tiles
    if visualize:
        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            # Denormalize for visualization
            mean, std = 0.1307, 0.3081
            tile_vis = processed_tiles[i] * std + mean
            plt.imshow(tile_vis, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return processed_tiles  # Shape (9, 28, 28)

def predict_from_real_image(model, image_path, device=None, visualize=True):
    """
    Process a real-world puzzle image and predict the digit in each tile.
    
    Args:
        model: The trained model for prediction
        image_path (str): Path to the image file
        device: Device to run the model on (cpu or cuda)
        visualize (bool): If True, visualize the preprocessing and prediction
        
    Returns:
        torch.Tensor: Predicted digits for each position in the puzzle
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess the image
    processed_tiles = load_and_preprocess_real_image(image_path, visualize=visualize)
    
    # Add batch dimension and move to device
    processed_tiles = processed_tiles.unsqueeze(0).to(device)  # Shape (1, 9, 28, 28)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(processed_tiles)
        predictions = model.predict(processed_tiles)
    
    predictions = predictions.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    
    # Print the predicted puzzle state in 3x3 grid format
    print("\nPredicted 8-Puzzle State:")
    print("-" * 13)
    for i in range(3):
        print("| ", end="")
        for j in range(3):
            print(f"{int(predictions[i * 3 + j])} | ", end="")
        print("\n" + "-" * 13)
    
    # Visualize the prediction result
    if visualize:
        plt.figure(figsize=(8, 8))
        for i in range(3):
            for j in range(3):
                pos = i * 3 + j
                plt.subplot(3, 3, pos + 1)
                plt.text(0.5, 0.5, str(int(predictions[pos])), 
                         fontsize=24, ha='center', va='center')
                plt.axis('off')
                plt.gca().add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                                fill=False, edgecolor='black'))
        plt.tight_layout()
        plt.suptitle("Predicted 8-Puzzle State", fontsize=16, y=0.98)
        plt.savefig('puzzle_prediction_result.png')
        plt.show()
    
    return predictions