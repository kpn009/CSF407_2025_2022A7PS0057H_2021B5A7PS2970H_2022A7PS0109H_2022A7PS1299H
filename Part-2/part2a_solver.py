# File: src/part2a_solver.py
"""
Task 2a: End-to-end 8-Puzzle Solver Application.
Accepts command-line arguments for the NN checkpoint, source/goal images,
and Gemini configuration. Uses the NN to infer puzzle states from images,
then prompts the Gemini LLM to generate the sequence of intermediate states
required to solve the puzzle. Saves the resulting sequence to a JSON file.
"""

import argparse
import json
import os
import sys # For sys.exit

# Import the solver utility class
# Ensure eight_puzzle_utils.py is in the same directory or Python path
try:
    from eight_puzzle_utils import EightPuzzleSolver
except ImportError as e:
    print(f"Error: Could not import EightPuzzleSolver: {e}")
    print("Ensure eight_puzzle_utils.py is in the 'src' directory or accessible in the Python path.")
    sys.exit(1) # Exit if the essential utility class is missing

# --- Crucial Dependency Check ---
# Ensure the model class is available before running main, as utils might need it
try:
    from model_8puzzle import EightPuzzleModel
except ImportError:
    print("Fatal Error: Could not import EightPuzzleModel from model_8puzzle.py.")
    print("Ensure model_8puzzle.py is in the 'src' directory or accessible in the Python path.")
    sys.exit(1)
# --- Check End ---


def main():
    """
    Main function to parse arguments, run the solver, and save results.
    """
    parser = argparse.ArgumentParser(
        description="8-Puzzle Solver using NN (Task 1 checkpoint) + Gemini LLM (Task 2a)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # --- Command Line Arguments ---
    parser.add_argument(
        "--config",
        type=str,
        default="src/config.json", # Default path assuming script is run from project root
        help="Path to the NN model configuration JSON file used for training."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained 8-Puzzle NN model checkpoint (.pt or .pth) from Task 1."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source state image (e.g., from 8-puzzle test set, 3x3 grid)."
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="Path to the goal state image (e.g., from 8-puzzle test set, 3x3 grid)."
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-1.5-flash", # A capable and often free/cost-effective model
        help="Name of the Gemini model to use (e.g., 'gemini-1.5-flash', 'gemini-1.0-pro')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="states.json",
        help="Output JSON file path to save the sequence of intermediate states."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Optional: Your Google AI (Gemini) API key. If not provided, uses GEMINI_API_KEY environment variable."
    )
    # --- Arguments Defined ---

    args = parser.parse_args()

    # --- Input Validation ---
    # Basic path existence checks (Solver class does more thorough checks)
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'")
        sys.exit(1)
    # Checkpoint existence check moved to Solver init
    if not os.path.exists(args.source):
        print(f"Error: Source image file not found at '{args.source}'")
        sys.exit(1)
    if not os.path.exists(args.goal):
        print(f"Error: Goal image file not found at '{args.goal}'")
        sys.exit(1)
    # --- Validation End ---

    try:
        # Initialize the solver with paths and API key
        solver = EightPuzzleSolver(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            gemini_model=args.gemini_model,
            api_key=args.api_key # Pass the key argument (can be None)
        )
    except (FileNotFoundError, ValueError, RuntimeError, TypeError, ImportError) as e:
        # Catch errors during initialization (API key missing, model loading issues etc.)
        print(f"Error initializing EightPuzzleSolver: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during solver initialization: {e}")
         sys.exit(1)

    print("\n--- Processing Images with NN ---")
    # Use the NN to get the initial and goal states from images
    source_state_digits = solver.image_to_tiles_prediction(args.source)
    goal_state_digits = solver.image_to_tiles_prediction(args.goal)

    if source_state_digits is None:
        print(f"Error: Failed to process source image '{args.source}' using the NN.")
        sys.exit(1)
    if goal_state_digits is None:
        print(f"Error: Failed to process goal image '{args.goal}' using the NN.")
        sys.exit(1)

    # --- Get Intermediate States from Gemini ---
    # Call the method that handles the Gemini API interaction
    intermediate_states = solver.get_intermediate_states_from_gemini(
        source_state=source_state_digits,
        goal_state=goal_state_digits
    )

    if not intermediate_states:
         print("Error: Failed to get intermediate states from Gemini. Check logs, API key/quota, and prompt.")
         sys.exit(1) # Exit if solving failed or returned empty

    # --- Save Results ---
    output_data = {"states": intermediate_states} # Structure as required by visualize_states.py
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
             print(f"Creating output directory: {output_dir}")
             os.makedirs(output_dir, exist_ok=True)

        # Write the JSON file
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=4) # Use indent for readability
        print(f"\nSuccessfully saved sequence of {len(intermediate_states)} states to '{args.output}'")
    except IOError as e:
        print(f"Error saving states to '{args.output}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving the output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
