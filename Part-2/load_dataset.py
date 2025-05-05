import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random

def load_and_save_puzzle_images(dataset_pt_path, num_images=2, output_prefix="puzzle_state"):
    """
    Loads an 8-puzzle dataset from a .pt file, selects random samples,
    reconstructs the 3x3 grid images, and saves them as PNG files.

    Args:
        dataset_pt_path (str): Path to the .pt dataset file
                               (e.g., 'src/Dataset/8puzzle_full_test.pt').
        num_images (int): Number of sample images to save.
        output_prefix (str): Prefix for the output PNG filenames.
    """
    # --- Input Validation ---
    if not os.path.exists(dataset_pt_path):
        print(f"Error: Dataset file not found at '{dataset_pt_path}'")
        print("Please ensure main.py has run successfully and created the dataset files.")
        return

    try:
        # Load the dataset
        data = torch.load(dataset_pt_path, map_location=torch.device('cpu')) # Load to CPU
        puzzle_states_tensor = data['puzzle_states'] # Expecting a tensor [N, 9, H, W] or list
        puzzle_labels_tensor = data['puzzle_labels'] # Expecting a tensor [N, 9] or list
        num_samples_in_file = len(puzzle_states_tensor)
        print(f"Loaded dataset with {num_samples_in_file} samples from '{dataset_pt_path}'.")

        if num_samples_in_file == 0:
             print("Error: The loaded dataset is empty.")
             return
        if num_images > num_samples_in_file:
             print(f"Warning: Requested {num_images} images, but dataset only has {num_samples_in_file}. Saving all.")
             num_images = num_samples_in_file

    except FileNotFoundError:
        print(f"Error: File not found at {dataset_pt_path}")
        return
    except KeyError as e:
         print(f"Error: Missing key '{e}' in the loaded .pt file. Expected 'puzzle_states' and 'puzzle_labels'.")
         return
    except Exception as e:
        print(f"Error loading dataset from {dataset_pt_path}: {e}")
        return

    # --- Select Random Indices ---
    selected_indices = random.sample(range(num_samples_in_file), num_images)
    print(f"Selected indices: {selected_indices}")

    # --- Reconstruct and Save Images ---
    saved_files = []
    for i, idx in enumerate(selected_indices):
        try:
            # Get the state tensor for the selected sample
            # Shape should be [9, H, W] e.g. [9, 28, 28]
            state_tensor = puzzle_states_tensor[idx]
            label_tensor = puzzle_labels_tensor[idx] # Shape [9]

            # Ensure tensor has the expected dimensions
            if state_tensor.dim() != 3 or state_tensor.shape[0] != 9:
                 print(f"Warning: Skipping sample {idx} due to unexpected state tensor shape: {state_tensor.shape}. Expected [9, H, W].")
                 continue

            tile_h, tile_w = state_tensor.shape[1], state_tensor.shape[2]

            # Create an empty canvas for the 3x3 grid
            # Multiply dimensions by 3
            grid_image = Image.new('L', (tile_w * 3, tile_h * 3)) # 'L' for grayscale

            # Paste each tile onto the grid
            for row in range(3):
                for col in range(3):
                    pos = row * 3 + col
                    # Convert the individual tile tensor to PIL Image
                    # Need to handle potential normalization if applied before saving
                    # Assuming ToTensor() was applied, values are 0-1. If Normalize was also applied, need inverse.
                    # For simplicity here, we'll just scale 0-1 tensor to 0-255.
                    # If normalization was applied, visualization might look weird without inverse transform.
                    tile_data = state_tensor[pos].numpy()
                    # Clamp values just in case and scale to 0-255
                    tile_data_scaled = np.clip(tile_data * 255, 0, 255).astype(np.uint8)
                    tile_image = Image.fromarray(tile_data_scaled, 'L')

                    # Calculate paste position
                    paste_x = col * tile_w
                    paste_y = row * tile_h
                    grid_image.paste(tile_image, (paste_x, paste_y))

            # Save the reconstructed grid image
            output_filename = f"{output_prefix}_{i+1}.png" # e.g., puzzle_state_1.png
            grid_image.save(output_filename)
            print(f"Saved sample {idx} (Label: {label_tensor.tolist()}) as '{output_filename}'")
            saved_files.append(output_filename)

        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")

    print("\nFinished saving images.")
    if len(saved_files) == 2:
        print(f"\nYou can now use '{saved_files[0]}' as your source and '{saved_files[1]}' as your goal image.")
    elif len(saved_files) > 0:
         print(f"\nSaved images: {saved_files}")


# --- Configuration ---
# Choose which dataset file to load samples from (e.g., the test set)
# Make sure this path is correct for your Colab environment after running main.py
DATASET_FILE_PATH = 'src/Dataset/8puzzle_full_test.pt'

# --- Run the extraction ---
load_and_save_puzzle_images(DATASET_FILE_PATH, num_images=2, output_prefix="valid_puzzle_state")
