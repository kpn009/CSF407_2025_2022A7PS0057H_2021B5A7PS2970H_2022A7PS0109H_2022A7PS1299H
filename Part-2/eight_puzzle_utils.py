# File: src/eight_puzzle_utils.py
"""
Utility class for the 8-Puzzle Solver Application (Task 2a).
Handles model loading, image processing, NN inference, and Gemini API interaction.
"""

import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import re # For parsing Gemini output
import sys # For sys.exit

# Try importing the model class
try:
    from model_8puzzle import EightPuzzleModel
except ImportError:
    print("Fatal Error: Could not import EightPuzzleModel from model_8puzzle.py.")
    print("Ensure model_8puzzle.py is in the same directory or accessible in the Python path.")
    sys.exit(1)

class EightPuzzleSolver:
    """
    Handles the logic for solving the 8-puzzle using a trained NN and Gemini LLM.
    """
    def __init__(self, config_path: str, checkpoint_path: str, gemini_model: str = "gemini-1.5-flash", api_key: str = None):
        """
        Initializes the solver.

        Args:
            config_path (str): Path to the NN model configuration JSON file.
            checkpoint_path (str): Path to the trained NN model checkpoint (.pt or .pth).
            gemini_model (str): Name of the Gemini model to use.
            api_key (str, optional): Google AI API key. Reads from GEMINI_API_KEY env var if None.

        Raises:
            FileNotFoundError: If config or checkpoint files are not found.
            ValueError: If API key is not provided or found.
            RuntimeError: If model loading fails.
            ImportError: If google.generativeai is not installed.
        """
        print("--- Initializing EightPuzzleSolver ---")

        # --- Validate Paths ---
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # --- Load NN Model ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        try:
            # Load model using the static method from EightPuzzleModel
            # Pass map_location for flexibility (CPU/GPU)
            self.model = EightPuzzleModel.load_model(checkpoint_path, config_path)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            print(f"NN Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the NN model: {e}")

        # --- Define Image Transformations ---
        # These should match the transformations used during training
        # Assuming input images are grayscale and need normalization
        # The model expects 9 individual 28x28 digit images as input
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
        ])
        print("Image transformer initialized.")

        # --- Configure Gemini API ---
        self.gemini_model_name = gemini_model
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided via argument or GEMINI_API_KEY environment variable.")

        try:
            genai.configure(api_key=resolved_api_key)
            self.llm = genai.GenerativeModel(self.gemini_model_name)
            print(f"Gemini API configured with model: {self.gemini_model_name}")
        except ImportError:
             raise ImportError("google.generativeai package not found. Please install it: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Failed to configure Gemini API: {e}")

        print("--- EightPuzzleSolver Initialized Successfully ---")

    def _preprocess_image(self, image_path: str) -> torch.Tensor | None:
        """
        Loads an image, splits it into 9 tiles, preprocesses each tile,
        and stacks them into a tensor suitable for the NN model.

        Args:
            image_path (str): Path to the input image file. Expected to be a grid of 3x3 digits.

        Returns:
            torch.Tensor | None: A tensor of shape (1, 9, 28, 28) ready for the model,
                                 or None if processing fails.
        """
        try:
            img = Image.open(image_path).convert('L') # Ensure grayscale
            img_width, img_height = img.size

            # Try to handle non-perfectly divisible dimensions slightly better
            tile_width = img_width / 3.0
            tile_height = img_height / 3.0
            if img_width % 3 != 0 or img_height % 3 != 0:
                 print(f"Warning: Image dimensions ({img_width}x{img_height}) are not perfectly divisible by 3. Tiles might not be exact.")


            tiles = []
            for i in range(3): # row
                for j in range(3): # col
                    # Define bounding box for each tile using potentially float coords
                    left = j * tile_width
                    top = i * tile_height
                    right = (j + 1) * tile_width
                    bottom = (i + 1) * tile_height
                    # Ensure integer coordinates for cropping
                    tile = img.crop((int(left), int(top), int(right), int(bottom)))

                    # Resize tile to 28x28 (expected by MNIST-based models)
                    tile = tile.resize((28, 28), Image.Resampling.LANCZOS)

                    # Apply transformations
                    tile_tensor = self.transform(tile) # Should output [1, 28, 28]
                    tiles.append(tile_tensor)

            # Stack tiles: list of 9 [1, 28, 28] -> [9, 1, 28, 28]
            stacked_tiles = torch.stack(tiles)

            # Squeeze the channel dimension: [9, 1, 28, 28] -> [9, 28, 28]
            stacked_tiles = stacked_tiles.squeeze(1)

            # Add batch dimension: [9, 28, 28] -> [1, 9, 28, 28]
            batch_tensor = stacked_tiles.unsqueeze(0)
            return batch_tensor

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def image_to_tiles_prediction(self, image_path: str) -> list[int] | None:
        """
        Processes an image file and predicts the 9 digit tiles using the loaded NN model.

        Args:
            image_path (str): Path to the input image file (3x3 grid of digits).

        Returns:
            list[int] | None: A list of 9 predicted integers (0-8), or None if prediction fails.
                              The order is row-major (top-left to bottom-right).
        """
        print(f"Processing image: {image_path}")
        # Preprocess image to get the input tensor
        input_tensor = self._preprocess_image(image_path)

        if input_tensor is None:
            return None

        # Move tensor to the correct device
        input_tensor = input_tensor.to(self.device)

        # Perform inference
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                predictions_tensor = self.model.predict(input_tensor) # Shape: [batch_size, 9]

            # Get predictions for the first (only) batch item and convert to list
            predicted_digits = predictions_tensor[0].cpu().tolist() # List of 9 integers
            print(f"NN Prediction for {os.path.basename(image_path)}: {predicted_digits}")
            return predicted_digits
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return None

    def _parse_gemini_response(self, response_text: str) -> list[list[int]] | None:
        """
        Parses the text response from Gemini to extract the list of states.
        Assumes the response contains a Python-style list of lists of integers.

        Args:
            response_text (str): The text generated by the Gemini model.

        Returns:
            list[list[int]] | None: A list where each inner list represents a puzzle state (9 ints),
                                    or None if parsing fails.
        """
        print("--- Attempting to parse Gemini response ---")
        # print(f"Raw response text:\n{response_text}") # Uncomment for deep debugging
        try:
            # Use regex to find the list structure robustly
            # Looks for [[...], [...], ...]
            # Allow for potential markdown backticks around the JSON
            response_text_cleaned = response_text.strip().strip('`')
            if response_text_cleaned.startswith("json"):
                 response_text_cleaned = response_text_cleaned[4:].strip()

            match = re.search(r'^\s*\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]\s*$', response_text_cleaned, re.DOTALL | re.MULTILINE)

            if match:
                list_str = match.group(0)
                # Use json.loads for safe evaluation of the list string
                states = json.loads(list_str)

                # Validate structure (list of lists, each inner list has 9 ints)
                if not isinstance(states, list):
                    raise ValueError("Parsed structure is not a list.")
                # Allow empty list as a valid response if LLM couldn't find path
                if not states:
                     print("Gemini returned an empty list, indicating no path found or possible.")
                     return [] # Return empty list, not None

                if not all(isinstance(s, list) and len(s) == 9 and all(isinstance(i, int) for i in s) for s in states):
                     raise ValueError("Parsed list does not contain lists of 9 integers.")

                print(f"Successfully parsed {len(states)} states from Gemini response.")
                return states
            else:
                print("Error: Could not find a valid JSON list of states in the Gemini response.")
                print("--- Gemini Response ---")
                print(response_text)
                print("-----------------------")
                return None # Parsing failed
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error parsing Gemini response: {e}")
            print("--- Gemini Response ---")
            print(response_text)
            print("-----------------------")
            return None # Parsing failed


    def get_intermediate_states_from_gemini(self, source_state: list[int], goal_state: list[int]) -> list[list[int]] | None:
        """
        Prompts the Gemini LLM to find the intermediate states between source and goal,
        attempting to do so even if the inputs aren't strictly valid 8-puzzle states.

        Args:
            source_state (list[int]): The starting state (9 digits, row-major).
            goal_state (list[int]): The target state (9 digits, row-major).

        Returns:
            list[list[int]] | None: A list of intermediate states (including source and goal),
                                    an empty list if no path is found,
                                    or None if the API call or parsing fails completely.
        """
        if not source_state or not goal_state or len(source_state) != 9 or len(goal_state) != 9:
             print("Error: Invalid source or goal state list provided to Gemini prompt function.")
             return None

        print("\n--- Querying Gemini for Intermediate States (Relaxed Rules) ---")
        print(f"Source State: {source_state}")
        print(f"Goal State:   {goal_state}")

        # Construct the modified prompt
        # Acknowledge potential invalidity, ask for best effort using adjacent swaps
        prompt = f"""
You are an expert puzzle solver. Your task is to find a sequence of states to transform a source configuration into a goal configuration on a 3x3 grid. The configurations are represented as flat lists (row-major).

Source Configuration:
{source_state}

Goal Configuration:
{goal_state}

IMPORTANT NOTE: The source and goal configurations provided might NOT be standard 8-puzzle states (they might contain duplicate numbers or miss some numbers from 0-8).

TASK: Despite potential invalidity, attempt to find a sequence of intermediate configurations to transform the Source into the Goal. Each step should involve swapping the position of ONE number with an ADJACENT number (up, down, left, or right).
* If the number '0' is present in the configuration, prioritize swapping the '0' with an adjacent number. Treat '0' like the "blank tile" if it exists.
* If '0' is not present, or if multiple '0's exist, try to find a sequence of adjacent swaps involving any number to reach the goal.
* The sequence should start with the Source Configuration and should end definitely with the Goal Configuration and the goal state {goal_state}.

OUTPUT FORMAT: Provide the output ONLY as a single JSON list of lists. Each inner list must contain 9 integers representing a configuration in the sequence. Do not include any other text, explanations, or formatting before or after the JSON list. If no path can be found, return an empty JSON list `[]`.

Example format: [[1, 2, 3, 4, 5, 6, 7, 8, 0], [1, 2, 3, 4, 0, 6, 7, 8, 5], ...]
"""

        try:
            # Make the API call
            # Consider adding safety settings if needed
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
            response = self.llm.generate_content(
                prompt,
                # safety_settings=safety_settings # Optional
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default is 1
                    # stop_sequences=['\n'], # Could help ensure cleaner output
                    # max_output_tokens=1024, # Adjust if needed
                    temperature=0.5 # Lower temp might give more deterministic results
                )
            )


            # Check for safety ratings or blocks if necessary (optional but good practice)
            if response.prompt_feedback.block_reason:
                print(f"Warning: Prompt blocked due to {response.prompt_feedback.block_reason}")
                # Consider how to handle blocked prompts, maybe return None or specific error
                return None
            # Check candidates and finish reason
            if not response.candidates:
                 print("Warning: Gemini generation returned no candidates.")
                 # Try to parse anyway, might be in parts or text attribute
                 # return self._parse_gemini_response(response.text) # Risky
                 return None # Safer to return None if no candidates
            # Log finish reason for debugging
            finish_reason = response.candidates[0].finish_reason
            print(f"Gemini finish reason: {finish_reason}")
            # if finish_reason not in ('STOP', 'MAX_TOKENS'): # Allow MAX_TOKENS as potentially partial
            #      print(f"Warning: Gemini generation finished unexpectedly: {finish_reason}")
                 # Decide whether to proceed or return None


            # Parse the response text
            # The parsing function now handles the empty list case correctly
            return self._parse_gemini_response(response.text)

        except Exception as e:
            print(f"Error during Gemini API call or processing: {e}")
            # You might want to print response details here for debugging if available
            # try:
            #     print(f"Gemini Raw Response Parts: {response.parts}")
            # except Exception:
            #     pass
            return None # Indicate failure

# Example usage (optional, for testing the class directly)
if __name__ == "__main__":
    print("Testing EightPuzzleSolver...")
    # Create a dummy config file
    dummy_config = {
        "model": {
            "input_size": 28*28,
            "hidden_layers": [128], # Simpler model for dummy test
            "output_size": 9,
            "dropout_rate": 0.1
        },
        "optimizer": {"type": "adam", "learning_rate": 0.001}, # Need optimizer section for trainer load
        "data": {"batch_size": 64}, # Need data section for trainer load
        "training": {"num_epochs": 1} # Need training section for trainer load
    }
    dummy_config_path = "dummy_config.json"
    with open(dummy_config_path, 'w') as f:
        json.dump(dummy_config, f, indent=4)

    # Create a dummy model and save checkpoint
    try:
        # Need to instantiate via trainer or ensure model class handles config directly
        # Assuming EightPuzzleModel init is sufficient as per original code
        dummy_model_instance = EightPuzzleModel(dummy_config_path)
        dummy_checkpoint_path = "dummy_checkpoint.pt"
        torch.save(dummy_model_instance.state_dict(), dummy_checkpoint_path)
        print(f"Dummy model and checkpoint created at {dummy_checkpoint_path}")

        # Create a dummy 3x3 image (e.g., 84x84 pixels)
        dummy_image = Image.new('L', (84, 84), color=200) # Grayscale
        from PIL import ImageDraw
        draw = ImageDraw.Draw(dummy_image)
        draw.text((10, 10), "1", fill=0); draw.text((40, 10), "1", fill=0); draw.text((70, 10), "3", fill=0) # Duplicate 1
        draw.text((10, 40), "4", fill=0); draw.text((40, 40), "5", fill=0); draw.text((70, 40), "5", fill=0) # Duplicate 5
        draw.text((10, 70), "6", fill=0); draw.text((40, 70), "7", fill=0); draw.text((70, 70), "8", fill=0) # Missing 0, 2
        dummy_image_path_invalid = "dummy_puzzle_image_invalid.png"
        dummy_image.save(dummy_image_path_invalid)
        print(f"Dummy INVALID puzzle image saved to {dummy_image_path_invalid}")


        # --- Test Initialization ---
        # Set API key via environment variable: export GEMINI_API_KEY="YOUR_API_KEY"
        solver = EightPuzzleSolver(
            config_path=dummy_config_path,
            checkpoint_path=dummy_checkpoint_path,
            # api_key="YOUR_API_KEY" # Or pass directly
        )

        # --- Test Image Processing and Prediction ---
        predicted_state = solver.image_to_tiles_prediction(dummy_image_path_invalid)
        if predicted_state:
            print(f"Predicted state from dummy image: {predicted_state}")
            # Note: Prediction will be random as the model is untrained

            # --- Test Gemini Interaction (if prediction worked) ---
            # Use another potentially invalid state for testing
            goal_state_example = [1, 3, 5, 4, 1, 5, 6, 7, 8] # Also invalid
            intermediate_states = solver.get_intermediate_states_from_gemini(predicted_state, goal_state_example)

            # Check the result type
            if intermediate_states is None:
                 print("\nFailed to get intermediate states from Gemini (API/Parsing error).")
            elif not intermediate_states: # Empty list
                 print("\nGemini returned an empty list (no path found or possible).")
            else: # Got a list of states
                print("\n--- Intermediate States from Gemini ---")
                for i, state in enumerate(intermediate_states):
                    print(f"Step {i}: {state}")
                print("-------------------------------------")

    except FileNotFoundError as e:
        print(f"Setup Error: {e}")
    except ValueError as e:
         print(f"Configuration Error: {e}. Ensure GEMINI_API_KEY is set.")
    except ImportError as e:
         print(f"Dependency Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_config_path): os.remove(dummy_config_path)
        if os.path.exists(dummy_checkpoint_path): os.remove(dummy_checkpoint_path)
        if os.path.exists(dummy_image_path_invalid): os.remove(dummy_image_path_invalid)
        print("Cleaned up dummy files.")

