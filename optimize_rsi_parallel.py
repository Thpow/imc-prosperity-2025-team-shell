"""
Parallel CPU-based RSI Parameter Optimizer
Uses multiprocessing to evaluate multiple parameter combinations simultaneously
"""

import numpy as np
import subprocess
import os
import re
import time
from datetime import datetime
import csv
import signal
import sys
from typing import Dict, List, Tuple
import json
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import random

# Parameter ranges adjusted to standard technical analysis practices
PARAM_RANGES = {
    "rsi_period": {"min": 10, "max": 30, "step": 2},        # Centered around standard 14-period RSI
    "rsi_overbought": {"min": 65, "max": 80, "step": 1},    # Standard range for overbought
    "rsi_oversold": {"min": 20, "max": 35, "step": 1},      # Standard range for oversold
    "rsi_neutral_high": {"min": 55, "max": 65, "step": 1},  # Upper neutral zone
    "rsi_neutral_low": {"min": 35, "max": 45, "step": 1}    # Lower neutral zone
}

# Weight parameters for fitness calculation
PROFIT_WEIGHT = 0.7  # Weight for pure profit
STABILITY_WEIGHT = 0.3  # Weight for trading stability

# Number of parallel processes
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Leave one CPU free
BATCH_SIZE = NUM_PROCESSES * 2  # 2 evaluations per process

def log_message(message: str, log_file: Path):
    """Log message to both console and file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def generate_random_params() -> Dict[str, int]:
    """Generate random parameters within defined ranges"""
    params = {}
    for param_name, param_range in PARAM_RANGES.items():
        if param_name == "rsi_period":
            # Bias towards 14-period RSI
            center = 14
            std = 4
            value = int(round(np.random.normal(center, std)))
            value = max(param_range["min"], min(param_range["max"], value))
            value = round(value / param_range["step"]) * param_range["step"]
        else:
            min_val = param_range["min"]
            max_val = param_range["max"]
            step = param_range["step"]
            num_steps = (max_val - min_val) // step
            value = min_val + (random.randint(0, num_steps) * step)
        params[param_name] = value
    return params

def get_neighbor_params(params: Dict[str, int], jump_size: float = 1.0) -> Dict[str, int]:
    """Generate neighboring parameters with variable jump size"""
    neighbor = {}
    for param_name, current_value in params.items():
        param_range = PARAM_RANGES[param_name]
        step = param_range["step"]
        
        # Scale the step by jump size
        effective_step = int(step * jump_size)
        if effective_step < step:
            effective_step = step
            
        # Random step in either direction
        change = random.choice([-effective_step, effective_step])
        new_value = current_value + change
        
        # Ensure value is within bounds and aligned to step size
        new_value = max(param_range["min"], min(param_range["max"], new_value))
        new_value = round((new_value - param_range["min"]) / step) * step + param_range["min"]
        
        neighbor[param_name] = new_value
    return neighbor

def calculate_fitness(profit: float, params: Dict[str, int]) -> float:
    """Calculate fitness score considering both profit and parameter stability"""
    # Base fitness from profit
    profit_score = profit / 1000  # Normalize profit
    
    # Parameter stability score
    stability_score = 0
    if params["rsi_period"] == 14:  # Bonus for standard RSI period
        stability_score += 0.2
    if 70 <= params["rsi_overbought"] <= 75:  # Standard overbought range
        stability_score += 0.1
    if 25 <= params["rsi_oversold"] <= 30:    # Standard oversold range
        stability_score += 0.1
    
    # Ensure neutral zones make sense
    if params["rsi_neutral_low"] < params["rsi_oversold"]:
        stability_score -= 0.2
    if params["rsi_neutral_high"] > params["rsi_overbought"]:
        stability_score -= 0.2
        
    return PROFIT_WEIGHT * profit_score + STABILITY_WEIGHT * stability_score

def modify_main_py(params: Dict[str, int]):
    """Modify main.py with given parameters"""
    with open('main.py', 'r') as file:
        content = file.readlines()
    
    in_params_block = False
    kelp_block_start = None
    
    for i, line in enumerate(content):
        if '"KELP": {' in line:
            in_params_block = True
            kelp_block_start = i
        elif in_params_block and '},' in line and kelp_block_start is not None:
            for j in range(kelp_block_start, i):
                for param_name, param_value in params.items():
                    if f'"{param_name}"' in content[j]:
                        indent = len(content[j]) - len(content[j].lstrip())
                        content[j] = " " * indent + f'"{param_name}": {param_value},      # Modified by parallel optimizer\n'
            break
    
    with open('main.py', 'w') as file:
        file.writelines(content)

def evaluate_params(params: Dict[str, int]) -> Tuple[float, float]:
    """Evaluate a parameter set and return profit and fitness"""
    # Create a copy of main.py for this process
    process_id = os.getpid()
    temp_main = f'main_{process_id}.py'
    with open('main.py', 'r') as src, open(temp_main, 'w') as dst:
        dst.write(src.read())
    
    try:
        # Modify the temporary main.py
        with open(temp_main, 'r') as file:
            content = file.readlines()
        
        in_params_block = False
        kelp_block_start = None
        
        for i, line in enumerate(content):
            if '"KELP": {' in line:
                in_params_block = True
                kelp_block_start = i
            elif in_params_block and '},' in line and kelp_block_start is not None:
                for j in range(kelp_block_start, i):
                    for param_name, param_value in params.items():
                        if f'"{param_name}"' in content[j]:
                            indent = len(content[j]) - len(content[j].lstrip())
                            content[j] = " " * indent + f'"{param_name}": {param_value},      # Modified by parallel optimizer\n'
                break
        
        with open(temp_main, 'w') as file:
            file.writelines(content)
        
        # Run backtest with temporary file
        process = subprocess.run(
            ["prosperity3bt", temp_main, "0"],
            capture_output=True,
            text=True
        )
        
        # Extract KELP profit
        kelp_match = re.search(r'KELP: ([0-9,]+)', process.stdout)
        if kelp_match:
            profit = float(kelp_match.group(1).replace(',', ''))
        else:
            profit = float('-inf')
            
        # Calculate fitness
        fitness = calculate_fitness(profit, params)
        
        return profit, fitness
    
    finally:
        # Clean up temporary file
        try:
            os.remove(temp_main)
        except:
            pass

def optimize(num_generations: int = 100):
    """Run parallel optimization"""
    results_dir = Path("optimizer_results")
    results_dir.mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = results_dir / f"parallel_rsi_optimization_{timestamp}.csv"
    log_file = results_dir / f"parallel_optimization_log_{timestamp}.txt"
    
    # Backup main.py
    with open('main.py', 'r') as f:
        original_content = f.read()
    with open('main.py.bak', 'w') as f:
        f.write(original_content)
    
    try:
        # Setup CSV logging
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = list(PARAM_RANGES.keys()) + ["Profit", "Fitness", "Generation"]
            writer.writerow(header)
        
        log_message(f"Starting parallel optimization with {NUM_PROCESSES} processes", log_file)
        log_message(f"Batch size: {BATCH_SIZE}", log_file)
        log_message("Using standard RSI practices with emphasis on 14-period RSI", log_file)
        
        best_params = None
        best_profit = float('-inf')
        best_fitness = float('-inf')
        start_time = time.time()
        total_evaluations = 0
        stagnation_counter = 0
        
        # Create process pool
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            for generation in range(num_generations):
                # Generate parameter sets for this generation
                if generation == 0 or stagnation_counter >= 10:
                    # Random exploration or restart
                    param_sets = [generate_random_params() for _ in range(BATCH_SIZE)]
                else:
                    # Generate neighbors of best parameters with increasing jump size
                    jump_size = 1.0 + (stagnation_counter * 0.1)
                    param_sets = [get_neighbor_params(best_params, jump_size) for _ in range(BATCH_SIZE)]
                
                # Evaluate parameter sets in parallel
                future_results = [executor.submit(evaluate_params, params) for params in param_sets]
                results = [future.result() for future in future_results]
                
                # Process results
                for params, (profit, fitness) in zip(param_sets, results):
                    total_evaluations += 1
                    
                    # Log to CSV
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [params[k] for k in PARAM_RANGES.keys()]
                        row.extend([profit, fitness, generation])
                        writer.writerow(row)
                    
                    # Update best if improved
                    if fitness > best_fitness:
                        improvement = profit - best_profit
                        best_profit = profit
                        best_fitness = fitness
                        best_params = params.copy()
                        stagnation_counter = 0
                        
                        log_message(f"\nNew best found in generation {generation}!", log_file)
                        log_message(f"Profit improvement: +{improvement}", log_file)
                        log_message(f"Fitness improvement: +{fitness - best_fitness:.2f}", log_file)
                        log_message(f"Parameters: {best_params}", log_file)
                        
                        # Save best parameters
                        modify_main_py(best_params)
                    
                # Update stagnation counter
                if stagnation_counter >= 10:
                    log_message(f"Stagnation detected ({stagnation_counter} generations). Performing random restart.", log_file)
                stagnation_counter += 1
                
                # Log progress
                if generation % 5 == 0:
                    elapsed = time.time() - start_time
                    iterations_per_sec = total_evaluations / elapsed
                    
                    log_message(f"\nGeneration {generation}", log_file)
                    log_message(f"Best profit: {best_profit}", log_file)
                    log_message(f"Best fitness: {best_fitness:.2f}", log_file)
                    log_message(f"Iterations/sec: {iterations_per_sec:.2f}", log_file)
                    log_message(f"Stagnation counter: {stagnation_counter}", log_file)
                    
                    # Save intermediate results
                    modify_main_py(best_params)
        
    except KeyboardInterrupt:
        log_message("\nOptimization interrupted!", log_file)
    finally:
        # Save final best parameters
        if best_params:
            log_message("\nFinal best parameters:", log_file)
            for param, value in best_params.items():
                log_message(f"  {param}: {value}", log_file)
            modify_main_py(best_params)
        
        # Ask about restoring original main.py
        response = input("\nRestore original main.py? (y/n): ")
        if response.lower() == 'y':
            with open('main.py', 'w') as f:
                f.write(original_content)
            log_message("Restored original main.py", log_file)
        else:
            log_message("Kept optimized parameters in main.py", log_file)

if __name__ == "__main__":
    # Get number of generations from user
    num_gens = input("Enter number of generations (default 100): ")
    num_gens = int(num_gens) if num_gens.strip() else 100
    
    optimize(num_generations=num_gens)
