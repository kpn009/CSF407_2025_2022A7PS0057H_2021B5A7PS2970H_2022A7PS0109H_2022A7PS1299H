import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML
import time
from PIL import Image
import argparse
import glob

from part2a_solver import EightPuzzleSolver
class PuzzleVisualizer:
    """
    Visualizer for the 8-puzzle solver simulation results
    """
    def __init__(self, results_dir="./src/Results"):
        """
        Initialize the visualizer
        
        Args:
            results_dir (str): Directory containing the results
        """
        self.results_dir = results_dir
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    
    def load_results(self, json_path):
        """
        Load results from a JSON file
        
        Args:
            json_path (str): Path to JSON file with results
            
        Returns:
            dict: Loaded results
        """
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        return results
    
    def create_state_grid(self, state_str):
        """
        Convert state string to a 3x3 grid for visualization
        
        Args:
            state_str (str): State string in format "123/456/780"
            
        Returns:
            numpy.ndarray: 3x3 grid of digits
        """
        grid = np.zeros((3, 3), dtype=int)
        rows = state_str.split('/')
        
        for i, row in enumerate(rows):
            for j, char in enumerate(row):
                if char == '_':
                    grid[i, j] = 0
                else:
                    grid[i, j] = int(char)
        
        return grid
    
    def visualize_single_state(self, state_str, ax=None, title=None):
        """
        Visualize a single puzzle state
        
        Args:
            state_str (str): State string in format "123/456/780"
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            title (str, optional): Title for the plot
            
        Returns:
            matplotlib.axes.Axes: The axes with the plot
        """
        grid = self.create_state_grid(state_str)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        
        # Create a grid
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks(np.arange(0, 4, 1))
        ax.set_yticks(np.arange(0, 4, 1))
        ax.grid(True)
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Plot digits
        for i in range(3):
            for j in range(3):
                digit = grid[i, j]
                if digit == 0:
                    ax.text(j + 0.5, 2.5 - i, " ", ha='center', va='center', fontsize=20)
                else:
                    ax.text(j + 0.5, 2.5 - i, str(digit), ha='center', va='center', fontsize=20)
        
        # Set title
        if title:
            ax.set_title(title)
        
        return ax
    
    def visualize_solution_path(self, states, explanations=None, save_path=None):
        """
        Visualize the solution path as a sequence of states
        
        Args:
            states (list): List of state strings
            explanations (list, optional): List of explanations for each step
            save_path (str, optional): Path to save the visualization
            
        Returns:
            None
        """
        # Determine grid size
        n_states = len(states)
        n_cols = min(5, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(n_cols * 3, n_rows * 3 + 1))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot each state
        for i, state in enumerate(states):
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Create title based on step number and explanation
            title = f"Step {i}"
            
            self.visualize_single_state(state, ax, title)
            
            # Add explanation as text
            if explanations and i < len(explanations):
                ax.text(1.5, -0.3, f"↓ {explanations[i]}", 
                        ha='center', va='center', fontsize=8,
                        transform=ax.transData, wrap=True)
        
        # Add overall title
        plt.suptitle(f"8-Puzzle Solution Path ({n_states} steps)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if needed
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Solution path visualization saved to {save_path}")
        
        plt.show()
    
    def create_animation(self, states, explanations=None, save_path=None, interval=1000):
        """
        Create an animation of the solution path
        
        Args:
            states (list): List of state strings
            explanations (list, optional): List of explanations for each step
            save_path (str, optional): Path to save the animation
            interval (int): Interval between frames in milliseconds
            
        Returns:
            matplotlib.animation.Animation: Animation object
        """
        # Create figure with two subplots - one for state, one for explanation
        fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(2, 1, height_ratios=[4, 1])
        ax_state = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        
        # Hide axes for text
        ax_text.axis('off')
        
        # Function to update the animation
        def update(frame):
            # Clear axes
            ax_state.clear()
            ax_text.clear()
            
            # Plot current state
            self.visualize_single_state(states[frame], ax_state, f"Step {frame}")
            
            # Add explanation
            if explanations and frame < len(explanations):
                explanation = explanations[frame] if frame > 0 else "Initial state"
                ax_text.text(0.5, 0.5, explanation, ha='center', va='center', wrap=True)
            
            return ax_state, ax_text
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(states),
                                       interval=interval, blit=False)
        
        # Save if needed
        if save_path:
            anim.save(save_path, dpi=100, writer='pillow')
            print(f"Animation saved to {save_path}")
        
        plt.close()
        return anim
    
    def compare_model_results(self, comparison_json_path, save_dir=None):
        """
        Compare results from multiple models
        
        Args:
            comparison_json_path (str): Path to JSON file with comparison results
            save_dir (str, optional): Directory to save visualizations
            
        Returns:
            None
        """
        # Load comparison results
        results = self.load_results(comparison_json_path)
        
        # Create directory for saving visualizations
        if save_dir is None:
            save_dir = os.path.join(self.results_dir, "visualizations")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create summary plot
        n_models = len(results)
        
        # Metrics for comparison
        model_names = []
        steps_taken = []
        
        for model_key, model_result in results.items():
            model_names.append(model_key)
            steps_taken.append(model_result["steps_taken"])
            
            # Create individual solution path visualization
            solution_path = model_result["path"]
            explanations = model_result.get("explanations", None)
            save_path = os.path.join(save_dir, f"{model_key}_solution_path.png")
            
            self.visualize_solution_path(solution_path, explanations, save_path)
            
            # Create animation
            anim_path = os.path.join(save_dir, f"{model_key}_animation.gif")
            self.create_animation(solution_path, explanations, anim_path)
        
        # Create performance comparison bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, steps_taken)
        
        # Add value labels to bars
        for bar, steps in zip(bars, steps_taken):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(steps),
                ha='center', va='bottom'
            )
        
        plt.xlabel('Model')
        plt.ylabel('Steps Taken')
        plt.title('Performance Comparison: Steps to Solution')
        plt.tight_layout()
        
        # Save comparison chart
        comparison_path = os.path.join(save_dir, "model_comparison_chart.png")
        plt.savefig(comparison_path)
        print(f"Model comparison chart saved to {comparison_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, results_json_path, save_path=None):
        """
        Create a comprehensive HTML report of the puzzle solver results
        
        Args:
            results_json_path (str): Path to JSON file with results
            save_path (str, optional): Path to save the HTML report
            
        Returns:
            str: HTML report
        """
        # Load results
        results = self.load_results(results_json_path)
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>8-Puzzle Solver Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1000px; margin: 0 auto; }
                .state-grid { display: grid; grid-template-columns: repeat(3, 1fr); width: 150px; height: 150px; }
                .cell { border: 1px solid #000; display: flex; justify-content: center; align-items: center; font-size: 24px; }
                .step { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; }
                .explanation { margin-top: 10px; color: #555; }
                .stats { background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }
                h1, h2 { color: #333; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>8-Puzzle Solver Report</h1>
        """
        
        # Add initial and goal states
        html += f"""
                <div class="stats">
                    <h2>Overview</h2>
                    <p>Initial State: {results["initial_state"]}</p>
                    <p>Goal State: {results["goal_state"]}</p>
                    <p>Total Steps: {len(results["path"]) - 1}</p>
                </div>
                
                <h2>Solution Path</h2>
        """
        
        # Add each step
        for i, state in enumerate(results["path"]):
            explanation = results.get("explanations", [])[i-1] if i > 0 and "explanations" in results else ""
            
            # Create state grid
            grid_html = '<div class="state-grid">'
            rows = state.split('/')
            for row in rows:
                for cell in row:
                    value = "&nbsp;" if cell == "_" else cell
                    grid_html += f'<div class="cell">{value}</div>'
            grid_html += '</div>'
            
            # Add step
            html += f"""
                <div class="step">
                    <h3>Step {i}</h3>
                    {grid_html}
                    <div class="explanation">{explanation}</div>
                </div>
            """
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save if needed
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            print(f"HTML report saved to {save_path}")
        
        return html

def main():
    """Main function for the puzzle visualizer"""
    parser = argparse.ArgumentParser(description='8-Puzzle Visualizer')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='./src/Results/visualizations', help='Directory to save visualizations')
    parser.add_argument('--compare', action='store_true', help='Compare multiple model results')
    parser.add_argument('--animate', action='store_true', help='Create animations')
    parser.add_argument('--report', action='store_true', help='Create HTML report')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PuzzleVisualizer()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare:
        # Compare model results
        visualizer.compare_model_results(args.results, args.output_dir)
    else:
        # Visualize single result
        results = visualizer.load_results(args.results)
        
        # Visualize solution path
        visualizer.visualize_solution_path(
            results["path"],
            results.get("explanations", None),
            os.path.join(args.output_dir, "solution_path.png")
        )
        
        if args.animate:
            # Create animation
            visualizer.create_animation(
                results["path"],
                results.get("explanations", None),
                os.path.join(args.output_dir, "solution_animation.gif")
            )
        
        if args.report:
            # Create HTML report
            visualizer.create_comprehensive_report(
                args.results,
                os.path.join(args.output_dir, "puzzle_report.html")
            )

if __name__ == "__main__":
    main()