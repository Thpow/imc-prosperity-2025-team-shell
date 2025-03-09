"""
RSI Parameter Optimizer for KELP Trading Strategy
This script uses hill climbing with random jumps to find optimal RSI parameters.
Runs indefinitely and automatically applies improvements.
"""

import subprocess
import os
import re
import time
import random
from datetime import datetime
import csv
import signal
import sys
import numpy as np

# Parameter ranges and steps for hill climbing
PARAM_RANGES = {
    "rsi_period": {"min": 0, "max": 500, "step": 5},
    "rsi_overbought": {"min": 10, "max": 90, "step": 2},
    "rsi_oversold": {"min": 10, "max": 90, "step": 2},
    "rsi_neutral_high": {"min": 10, "max": 90, "step": 2},
    "rsi_neutral_low": {"min": 10, "max": 90, "step": 2}
}

# Global variables for tracking optimization state
best_params = None
best_kelp_profit = 0
stagnation_counter = 0
results_history = []

# Create results directory
results_dir = "optimizer_results"
os.makedirs(results_dir, exist_ok=True)

# Setup CSV logging
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_file = os.path.join(results_dir, f"rsi_optimization_{timestamp}.csv")
log_file = os.path.join(results_dir, f"optimization_log_{timestamp}.txt")

def log_message(message):
    """Log a message to both console and log file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def setup_csv():
    """Setup the CSV file with headers"""
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = list(PARAM_RANGES.keys()) + ["KELP_Profit", "RESIN_Profit", "Total_Profit", "Iteration", "Method"]
        writer.writerow(header)

def modify_main_py(params, backup_first=True):
    """
    Modify main.py with the specified RSI parameters
    
    Args:
        params (dict): Dictionary of RSI parameters to update
        backup_first (bool): Whether to create a backup of the original file
    """
    # Backup main.py if this is the first run
    if backup_first:
        with open('main.py', 'r') as file:
            original_content = file.read()
        
        with open('main.py.bak', 'w') as file:
            file.write(original_content)
        log_message("Backed up main.py to main.py.bak")
    
    # Read the current content
    with open('main.py', 'r') as file:
        content = file.readlines()
    
    # Find and modify the RSI parameters in the Trader.__init__ method
    in_params_block = False
    kelp_block_start = None
    
    for i, line in enumerate(content):
        if '"KELP": {' in line:
            in_params_block = True
            kelp_block_start = i
        elif in_params_block and '},' in line and kelp_block_start is not None:
            # End of KELP parameters block
            for j in range(kelp_block_start, i):
                for param_name, param_value in params.items():
                    if f'"{param_name}"' in content[j]:
                        # Extract indentation
                        indent = len(content[j]) - len(content[j].lstrip())
                        content[j] = " " * indent + f'"{param_name}": {param_value},      # Modified by optimizer\n'
            break
    
    # Write the modified content back
    with open('main.py', 'w') as file:
        file.writelines(content)

def run_backtest():
    """
    Run the prosperity backtest and extract the profit results
    
    Returns:
        tuple: (kelp_profit, resin_profit, total_profit)
    """
    process = subprocess.run(
        ["prosperity3bt", "main.py", "0"],
        capture_output=True,
        text=True
    )
    
    output = process.stdout
    
    # Extract profit values using regex
    kelp_match = re.search(r'KELP: ([0-9,]+)', output)
    resin_match = re.search(r'RAINFOREST_RESIN: ([0-9,]+)', output)
    total_match = re.search(r'Total profit: ([0-9,]+)', output)
    
    kelp_profit = int(kelp_match.group(1).replace(',', '')) if kelp_match else 0
    resin_profit = int(resin_match.group(1).replace(',', '')) if resin_match else 0
    total_profit = int(total_match.group(1).replace(',', '')) if total_match else 0
    
    return kelp_profit, resin_profit, total_profit

def restore_backup():
    """Restore the original main.py from backup"""
    try:
        with open('main.py.bak', 'r') as file:
            original_content = file.read()
        
        with open('main.py', 'w') as file:
            file.write(original_content)
        log_message("Restored main.py from backup")
    except FileNotFoundError:
        log_message("No backup file found")

def get_current_params():
    """Extract current parameters from main.py"""
    with open('main.py', 'r') as file:
        content = file.read()
    
    current_params = {}
    for param_name in PARAM_RANGES.keys():
        pattern = f'"{param_name}":\\s*(\\d+)'
        match = re.search(pattern, content)
        if match:
            current_params[param_name] = int(match.group(1))
    
    return current_params

def log_result(params, kelp_profit, resin_profit, total_profit, iteration, method):
    """Log a result to the CSV file"""
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        row = [params[name] for name in PARAM_RANGES.keys()] + [kelp_profit, resin_profit, total_profit, iteration, method]
        writer.writerow(row)

def get_neighbors(params):
    """Get all neighbors by adjusting each parameter by one step in both directions"""
    neighbors = []
    
    for param_name, param_value in params.items():
        param_range = PARAM_RANGES[param_name]
        step = param_range["step"]
        
        # Try increasing the parameter
        if param_value + step <= param_range["max"]:
            neighbor = params.copy()
            neighbor[param_name] = param_value + step
            neighbors.append(neighbor)
        
        # Try decreasing the parameter
        if param_value - step >= param_range["min"]:
            neighbor = params.copy()
            neighbor[param_name] = param_value - step
            neighbors.append(neighbor)
    
    return neighbors

def get_random_neighbor(params, jump_size_factor=2):
    """Get a random neighbor with a larger jump to escape local maxima"""
    neighbor = params.copy()
    
    # Randomly select 1-3 parameters to modify
    num_params_to_modify = random.randint(1, min(3, len(PARAM_RANGES)))
    params_to_modify = random.sample(list(PARAM_RANGES.keys()), num_params_to_modify)
    
    for param_name in params_to_modify:
        param_range = PARAM_RANGES[param_name]
        step = param_range["step"] * jump_size_factor  # Larger step for random jumps
        
        # Randomly decide direction (increase or decrease)
        direction = random.choice([-1, 1])
        new_value = params[param_name] + (direction * step)
        
        # Ensure the new value is within bounds
        new_value = max(param_range["min"], min(param_range["max"], new_value))
        neighbor[param_name] = new_value
    
    return neighbor

def evaluate_params(params, iteration, method="hill_climbing"):
    """Evaluate a parameter set and log results"""
    global best_params, best_kelp_profit, stagnation_counter, results_history
    
    modify_main_py(params, backup_first=False)
    kelp_profit, resin_profit, total_profit = run_backtest()
    
    # Log results
    log_result(params, kelp_profit, resin_profit, total_profit, iteration, method)
    
    # Track result history (keep most recent 100)
    results_history.append((params, kelp_profit))
    if len(results_history) > 100:
        results_history.pop(0)
    
    # Check if this is a new best result
    if kelp_profit > best_kelp_profit:
        improvement = kelp_profit - best_kelp_profit
        best_kelp_profit = kelp_profit
        best_params = params.copy()
        stagnation_counter = 0
        
        log_message(f"IMPROVEMENT FOUND! KELP Profit: {kelp_profit} (+{improvement})")
        log_message(f"New best parameters: {params}")
        
        # Save best parameters immediately
        save_best_params()
        
        return True
    else:
        stagnation_counter += 1
        return False

def save_best_params():
    """Save the best parameters to main.py"""
    if best_params:
        log_message(f"Applying best parameters to main.py: {best_params}")
        modify_main_py(best_params, backup_first=False)

def random_restart():
    """Generate completely random parameters within ranges"""
    random_params = {}
    for param_name, param_range in PARAM_RANGES.items():
        min_val, max_val, step = param_range["min"], param_range["max"], param_range["step"]
        # Generate random value as a multiple of step
        steps = (max_val - min_val) // step
        random_steps = random.randint(0, steps)
        random_params[param_name] = min_val + (random_steps * step)
    
    return random_params

def hill_climbing_optimizer():
    """Hill climbing optimization with random jumps to escape local maxima"""
    global stagnation_counter, best_kelp_profit, best_params
    
    # Initialize parameters
    if best_params is None:
        # Start with current parameters from main.py
        current_params = get_current_params()
        if not current_params or len(current_params) < len(PARAM_RANGES):
            log_message("Could not extract all parameters from main.py, using defaults")
            current_params = {
                "rsi_period": 30,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "rsi_neutral_high": 60,
                "rsi_neutral_low": 40
            }
        
        # Evaluate starting point
        log_message(f"Starting optimization with parameters: {current_params}")
        evaluate_params(current_params, 0, "initial")
        best_params = current_params
    else:
        current_params = best_params.copy()
    
    iteration = 1
    max_stagnation = 10  # Number of iterations before trying random jumps
    
    # Register signal handler for graceful exit
    def signal_handler(sig, frame):
        log_message("\nOptimization interrupted! Saving best parameters...")
        save_best_params()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:  # Run indefinitely
            # ALWAYS use best_params as reference to avoid drifting away from optimal solution
            reference_params = best_params.copy()
            
            # Choose optimization method based on stagnation
            if stagnation_counter >= max_stagnation:
                method = random.choices(
                    ["random_jump", "random_restart", "hill_climbing"], 
                    weights=[0.6, 0.3, 0.1], 
                    k=1
                )[0]
                
                if method == "random_jump":
                    # Random jump to escape local maximum, but using best_params as reference
                    jump_size = min(2 + (stagnation_counter // 10), 5)  # Increase jump size with stagnation
                    next_params = get_random_neighbor(reference_params, jump_size_factor=jump_size)
                    log_message(f"RANDOM JUMP (size {jump_size}) after {stagnation_counter} stagnant iterations")
                
                elif method == "random_restart":
                    # Complete random restart
                    next_params = random_restart()
                    log_message(f"RANDOM RESTART after {stagnation_counter} stagnant iterations")
                    
                else:
                    # Fall back to hill climbing with probability 0.1
                    neighbors = get_neighbors(reference_params)
                    next_params = random.choice(neighbors)
                    method = "hill_climbing"
            else:
                # Standard hill climbing - always use best_params as base reference
                method = "hill_climbing"
                neighbors = get_neighbors(reference_params)
                next_params = random.choice(neighbors)
            
            # Evaluate the selected parameters
            improvement = evaluate_params(next_params, iteration, method)
            
            # Log information
            param_desc = ", ".join([f"{k}={v}" for k, v in next_params.items()])
            log_message(f"Iteration {iteration}, Method: {method}")
            log_message(f"Parameters: {param_desc}")
            log_message(f"KELP Profit: {results_history[-1][1]}, Best: {best_kelp_profit}")
            log_message(f"Stagnation counter: {stagnation_counter}")
            log_message("-" * 80)
            
            # Only update current parameters if there was an improvement
            # This prevents drifting from best solution
            if improvement:
                current_params = next_params.copy()
                
            # Reset stagnation counter after a big jump
            if method in ["random_jump", "random_restart"]:
                stagnation_counter = max(stagnation_counter // 2, 0)  # Reduce stagnation counter but don't reset completely
            
            iteration += 1
            
            # Periodically remind about best parameters
            if iteration % 10 == 0:
                log_message(f"REMINDER - Best parameters so far: {best_params}")
                log_message(f"Best KELP profit: {best_kelp_profit}")
            
            # Optional: sleep to prevent overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        log_message("\nOptimization interrupted! Saving best parameters...")
        save_best_params()

if __name__ == "__main__":
    setup_csv()
    log_message("Starting RSI Parameter Optimizer with Hill Climbing")
    log_message("Press Ctrl+C to stop optimization and save best parameters")
    log_message("=" * 80)
    
    hill_climbing_optimizer()
