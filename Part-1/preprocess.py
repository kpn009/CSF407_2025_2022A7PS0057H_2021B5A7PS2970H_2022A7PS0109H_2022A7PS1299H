import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def preprocess_puzzle_image(img, normalize=True):
    """
    Preprocesses an 84x84 puzzle image into 9 individual 28x28 images.
    
    Args:
        img: Either a PIL Image, numpy array, or PyTorch tensor with shape:
             - PIL Image: 84x84 grayscale image
             - numpy array: (84, 84) or (84, 84, 1) or (84, 84, 3)
             - torch tensor: (1, 84, 84) or (3, 84, 84) or (84, 84)
        normalize: If True, applies MNIST normalization to each tile
        
    Returns:
        torch.Tensor: Tensor of shape (9, 28, 28) containing the 9 individual tiles
    """
    # Convert to tensor if not already
    if isinstance(img, Image.Image):
        # Convert PIL image to grayscale and then to tensor
        if img.mode != 'L':
            img = img.convert('L')
        img = transforms.ToTensor()(img).squeeze(0)  # Shape becomes (H, W)
    elif isinstance(img, np.ndarray):
        # Convert numpy array to tensor
        if img.ndim == 3 and img.shape[2] in [1, 3]:
            # Convert RGB or grayscale with channel dimension to grayscale
            if img.shape[2] == 3:
                # Convert RGB to grayscale using standard coefficients
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                img = img.squeeze(2)  # Remove channel dimension
        
        # Ensure values are in [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
            
        img = torch.from_numpy(img).float()
    elif isinstance(img, torch.Tensor):
        # Handle tensor input
        if img.dim() == 3:
            if img.shape[0] in [1, 3]:
                # Convert RGB or grayscale with channel dimension to grayscale
                if img.shape[0] == 3:
                    # Convert RGB to grayscale using standard coefficients
                    img = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
                else:
                    img = img.squeeze(0)  # Remove channel dimension
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")
    
    # Ensure the image is of the correct size
    if img.shape != (84, 84):
        raise ValueError(f"Expected input of shape (84, 84), got {img.shape}")
    
    # Initialize tensor to store the 9 tiles
    tiles = torch.zeros((9, 28, 28))
    
    # Split the image into 9 tiles
    for i in range(3):
        for j in range(3):
            # Extract tile
            row_start = i * 28
            col_start = j * 28
            tile = img[row_start:row_start + 28, col_start:col_start + 28]
            
            # Store in output tensor (row-major order)
            tiles[i * 3 + j] = tile
    
    # Apply normalization if requested (MNIST normalization)
    if normalize:
        # MNIST mean and std values
        mean = 0.1307
        std = 0.3081
        
        # Apply normalization to each tile
        for i in range(9):
            tiles[i] = (tiles[i] - mean) / std
    
    return tiles

def detect_empty_tile(tiles):
    """
    Detects which of the 9 tiles is the empty space in the 8-puzzle.
    This is based on the intensity values - the empty tile should have higher
    average pixel value (closer to white).
    
    Args:
        tiles: Tensor of shape (9, 28, 28) containing the 9 individual tiles
        
    Returns:
        int: Index of the empty tile (0-8)
    """
    # Calculate mean intensity for each tile
    mean_intensities = [torch.mean(tile).item() for tile in tiles]
    
    # The empty tile should have the highest intensity (closest to white)
    empty_tile_idx = np.argmax(mean_intensities)
    
    return empty_tile_idx

def preprocess_batch(batch_images, normalize=True):
    """
    Preprocesses a batch of 84x84 puzzle images into batches of 9 individual 28x28 images.
    
    Args:
        batch_images: Tensor of shape (batch_size, 84, 84) or (batch_size, 1, 84, 84)
        normalize: If True, applies MNIST normalization to each tile
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size, 9, 28, 28)
    """
    batch_size = batch_images.shape[0]
    
    # Remove channel dimension if present
    if batch_images.dim() == 4 and batch_images.shape[1] == 1:
        batch_images = batch_images.squeeze(1)
    
    # Initialize tensor to store the processed batch
    processed_batch = torch.zeros((batch_size, 9, 28, 28))
    
    # Process each image in the batch
    for i in range(batch_size):
        processed_batch[i] = preprocess_puzzle_image(batch_images[i], normalize=normalize)
    
    return processed_batch

def preprocess_real_world_puzzle(image_path, output_size=(84, 84), debug=False):
    """
    Advanced preprocessing for real-world 8-puzzle images.
    Handles varying lighting conditions, rotations, and detects the grid structure.
    
    Args:
        image_path (str): Path to the input image
        output_size (tuple): Size of output image (height, width)
        debug (bool): If True, shows intermediate processing steps
        
    Returns:
        torch.Tensor: Processed image tensor of shape output_size
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the puzzle grid)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop to the puzzle area
        puzzle_area = gray[y:y+h, x:x+w]
        
        # Resize to desired output size
        puzzle_resized = cv2.resize(puzzle_area, output_size[::-1])  # cv2 uses (width, height)
        
        # Apply Otsu's thresholding to binarize the image
        _, puzzle_binary = cv2.threshold(puzzle_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Make sure digits are black (0) and background is white (255)
        # Calculate the mean value to determine if inversion is needed
        if np.mean(puzzle_binary) < 127:
            puzzle_binary = 255 - puzzle_binary
    else:
        # If no contours found, just resize and threshold the original
        puzzle_resized = cv2.resize(gray, output_size[::-1])
        _, puzzle_binary = cv2.threshold(puzzle_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure proper orientation (digits as black, background as white)
        if np.mean(puzzle_binary) < 127:
            puzzle_binary = 255 - puzzle_binary
    
    # Show debug images if requested
    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[2].imshow(thresh, cmap='gray')
        axes[2].set_title('Threshold')
        axes[3].imshow(puzzle_binary, cmap='gray')
        axes[3].set_title('Final Binary')
        plt.tight_layout()
        plt.show()
    
    # Convert to tensor (0 to 1 range)
    puzzle_tensor = torch.from_numpy(puzzle_binary).float() / 255.0
    
    return puzzle_tensor