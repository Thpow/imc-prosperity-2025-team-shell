import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import pandas as pd

def load_results(results_dir="tuning_results"):
    """Load all result files from the directory"""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
    
    return results

def analyze_parameter_impact(results):
    """Analyze the impact of each parameter on profit"""
    # Extract all results history entries
    all_entries = []
    for result in results:
        all_entries.extend(result.get("results_history", []))
    
    if not all_entries:
        print("No results history found.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {**entry["params"], "total_profit": entry["total_profit"]}
        for entry in all_entries
    ])
    
    # Analyze each parameter's correlation with profit
    correlations = {}
    for param in df.columns:
        if param != "total_profit":
            correlations[param] = df[param].corr(df["total_profit"])
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Parameter impact on profit (correlation):")
    for param, corr in sorted_correlations:
        print(f"{param}: {corr:.4f}")
    
    # Plot top parameters by correlation
    top_params = [param for param, _ in sorted_correlations[:5]]
    
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(top_params):
        plt.subplot(2, 3, i+1)
        plt.scatter(df[param], df["total_profit"], alpha=0.5)
        plt.title(f"{param} vs Profit (corr={correlations[param]:.4f})")
        plt.xlabel(param)
        plt.ylabel("Total Profit")
        
        # Add trend line
        z = np.polyfit(df[param], df["total_profit"], 1)
        p = np.poly1d(z)
        plt.plot(sorted(df[param].unique()), p(sorted(df[param].unique())), "r--")
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    
    plt.savefig("visualizations/parameter_impact.png")
    plt.close()
    
    return correlations

def plot_profit_progression(results):
    """Plot the progression of profit over iterations"""
    # Extract all results history entries with iteration numbers
    all_entries = []
    for result in results:
        all_entries.extend(result.get("results_history", []))
    
    if not all_entries:
        print("No results history found.")
        return
    
    # Sort by iteration
    all_entries.sort(key=lambda x: x.get("iteration", 0))
    
    # Extract profits
    iterations = [entry.get("iteration", i) for i, entry in enumerate(all_entries)]
    profits = [entry["total_profit"] for entry in all_entries]
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(profits)
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, profits, 'b-', alpha=0.5, label="Profit")
    plt.plot(iterations, running_max, 'r-', label="Best profit so far")
    plt.xlabel("Iteration")
    plt.ylabel("Total Profit")
    plt.title("Profit Progression Over Iterations")
    plt.legend()
    plt.grid(True)
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    
    plt.savefig("visualizations/profit_progression.png")
    plt.close()

def find_best_parameters(results):
    """Find the best parameters across all result files"""
    best_profit = -float('inf')
    best_params = None
    
    for result in results:
        if result.get("best_profit", -float('inf')) > best_profit:
            best_profit = result["best_profit"]
            best_params = result["best_params"]
    
    if best_params:
        print(f"Best parameters found (profit: {best_profit}):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print("No best parameters found.")
    
    return best_profit, best_params

def analyze_parameter_distributions(results):
    """Analyze the distribution of parameter values in top-performing runs"""
    # Extract all results history entries
    all_entries = []
    for result in results:
        all_entries.extend(result.get("results_history", []))
    
    if not all_entries:
        print("No results history found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {**entry["params"], "total_profit": entry["total_profit"]}
        for entry in all_entries
    ])
    
    # Get top 10% performing runs
    threshold = np.percentile(df["total_profit"], 90)
    top_df = df[df["total_profit"] >= threshold]
    
    print(f"Analyzing parameter distributions in top 10% of runs (profit >= {threshold})")
    
    # Plot distributions for each parameter
    params = [col for col in df.columns if col != "total_profit"]
    
    # Calculate grid dimensions based on number of parameters
    num_params = len(params)
    grid_cols = 4
    grid_rows = (num_params + grid_cols - 1) // grid_cols  # Ceiling division
    
    plt.figure(figsize=(15, 3 * grid_rows))
    for i, param in enumerate(params):
        if i < grid_rows * grid_cols:  # Ensure we don't exceed subplot grid
            plt.subplot(grid_rows, grid_cols, i+1)
            
            # Get unique values for this parameter
            unique_values = sorted(df[param].unique())
            
            # Count occurrences in top runs
            counts = [len(top_df[top_df[param] == val]) for val in unique_values]
            
            plt.bar(range(len(unique_values)), counts)
            plt.xticks(range(len(unique_values)), unique_values, rotation=90)
            plt.title(param)
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    
    plt.savefig("visualizations/parameter_distributions.png")
    plt.close()

def main():
    # Create directories if they don't exist
    for directory in ["tuning_results", "test_results", "backtests", "visualizations"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Check if we have results to analyze
    if not os.path.exists("tuning_results") or not os.listdir("tuning_results"):
        print("No tuning results found. Run parameter_tuner.py first.")
        return
    
    # Load results
    results = load_results()
    print(f"Loaded {len(results)} result files.")
    
    # Find best parameters
    best_profit, best_params = find_best_parameters(results)
    
    # Analyze parameter impact
    correlations = analyze_parameter_impact(results)
    
    # Plot profit progression
    plot_profit_progression(results)
    
    # Analyze parameter distributions in top runs
    analyze_parameter_distributions(results)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
