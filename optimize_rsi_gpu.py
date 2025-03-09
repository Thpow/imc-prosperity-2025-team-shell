"""
GPU-Accelerated RSI Parameter Optimizer
Uses PyTorch CUDA to parallelize parameter testing and maximize iterations/second
Optimizes RSI parameters while maintaining standard technical analysis practices
"""

import torch
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

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print("WARNING: CUDA not available, falling back to CPU")

# Parameter ranges adjusted to standard technical analysis practices
PARAM_RANGES = {
    "rsi_period": {"min": 10, "max": 30, "step": 2},        # Centered around standard 14-period RSI
    "rsi_overbought": {"min": 65, "max": 80, "step": 1},    # Standard range for overbought
    "rsi_oversold": {"min": 20, "max": 35, "step": 1},      # Standard range for oversold
    "rsi_neutral_high": {"min": 55, "max": 65, "step": 1},  # Upper neutral zone
    "rsi_neutral_low": {"min": 35, "max": 45, "step": 1}    # Lower neutral zone
}

# Batch size for parallel processing
BATCH_SIZE = 128 if DEVICE.type == 'cuda' else 8

# Weight parameters for fitness calculation
PROFIT_WEIGHT = 0.7  # Weight for pure profit
STABILITY_WEIGHT = 0.3  # Weight for trading stability

class GPUOptimizer:
    def __init__(self):
        self.best_params = None
        self.best_profit = float('-inf')
        self.best_fitness = float('-inf')
        self.results_dir = Path("optimizer_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_file = self.results_dir / f"gpu_rsi_optimization_{timestamp}.csv"
        self.log_file = self.results_dir / f"gpu_optimization_log_{timestamp}.txt"
        
        # Initialize parameter tensors
        self.setup_parameter_tensors()
        
        # Cache for evaluated parameters
        self.param_cache = {}
        
        # Trading stability tracking
        self.trade_history = []
        
    def log_message(self, message: str):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    def setup_parameter_tensors(self):
        """Create parameter combinations as tensors on GPU"""
        param_values = {}
        for param_name, param_range in PARAM_RANGES.items():
            values = torch.arange(
                param_range["min"], 
                param_range["max"] + param_range["step"], 
                param_range["step"],
                device=DEVICE
            )
            param_values[param_name] = values
        
        self.param_values = param_values
        
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of parameter combinations with bias towards standard values"""
        batch = {}
        for param_name, values in self.param_values.items():
            if param_name == "rsi_period":
                # Bias towards 14-period RSI with proper tensor format
                center = (14 - PARAM_RANGES[param_name]["min"]) / PARAM_RANGES[param_name]["step"]
                indices = torch.normal(
                    mean=float(center),
                    std=4.0,
                    size=(batch_size,),
                    device=DEVICE
                ).long().clamp(0, len(values)-1)
            else:
                indices = torch.randint(0, len(values), (batch_size,), device=DEVICE)
            batch[param_name] = values[indices]
        
        return batch
    
    def mutate_parameters(self, params: Dict[str, torch.Tensor], mutation_rate: float = 0.1) -> Dict[str, torch.Tensor]:
        """Apply conservative mutations to parameters"""
        mutated = {}
        for param_name, values in params.items():
            # Smaller mutations for RSI period
            if param_name == "rsi_period":
                mutation_rate = 0.05
                step_size = 2
            else:
                mutation_rate = 0.1
                step_size = PARAM_RANGES[param_name]["step"]
            
            mask = torch.rand(values.shape, device=DEVICE) < mutation_rate
            steps = torch.randint(-1, 2, values.shape, device=DEVICE)
            
            mutated_values = values + (steps * step_size * mask)
            mutated_values = torch.clamp(
                mutated_values,
                PARAM_RANGES[param_name]["min"],
                PARAM_RANGES[param_name]["max"]
            )
            
            mutated[param_name] = mutated_values
            
        return mutated

    def calculate_fitness(self, profit: float, params: Dict[str, int]) -> float:
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

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of parameters in parallel"""
        batch_size = len(next(iter(batch.values())))
        profits = torch.zeros(batch_size, device=DEVICE)
        fitness_scores = torch.zeros(batch_size, device=DEVICE)
        
        cpu_batch = {k: v.cpu().numpy() for k, v in batch.items()}
        
        for i in range(batch_size):
            params = {k: int(v[i]) for k, v in cpu_batch.items()}
            param_key = json.dumps(params, sort_keys=True)
            
            if param_key in self.param_cache:
                profit = self.param_cache[param_key]
            else:
                profit = self.evaluate_single(params)
                self.param_cache[param_key] = profit
            
            profits[i] = profit
            fitness_scores[i] = self.calculate_fitness(profit, params)
            
        return profits, fitness_scores
    
    def evaluate_single(self, params: Dict[str, int]) -> float:
        """Evaluate a single parameter set"""
        self.modify_main_py(params)
        
        process = subprocess.run(
            ["prosperity3bt", "main.py", "0"],
            capture_output=True,
            text=True
        )
        
        kelp_match = re.search(r'KELP: ([0-9,]+)', process.stdout)
        if kelp_match:
            return float(kelp_match.group(1).replace(',', ''))
        return float('-inf')
    
    def modify_main_py(self, params: Dict[str, int]):
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
                            content[j] = " " * indent + f'"{param_name}": {param_value},      # Modified by GPU optimizer\n'
                break
        
        with open('main.py', 'w') as file:
            file.writelines(content)
    
    def save_best_params(self):
        """Save the best parameters found"""
        if self.best_params:
            self.log_message(f"Saving best parameters (profit: {self.best_profit}, fitness: {self.best_fitness:.2f}):")
            for param, value in self.best_params.items():
                self.log_message(f"  {param}: {value}")
            
            self.modify_main_py(self.best_params)
    
    def optimize(self, num_generations: int = 100):
        """Run GPU-accelerated optimization"""
        try:
            # Backup main.py
            with open('main.py', 'r') as f:
                original_content = f.read()
            with open('main.py.bak', 'w') as f:
                f.write(original_content)
            
            # Setup CSV logging
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = list(PARAM_RANGES.keys()) + ["Profit", "Fitness", "Generation"]
                writer.writerow(header)
            
            self.log_message(f"Starting GPU-accelerated optimization on {DEVICE}")
            self.log_message(f"Batch size: {BATCH_SIZE}")
            self.log_message("Using standard RSI practices with emphasis on 14-period RSI")
            
            start_time = time.time()
            total_evaluations = 0
            stagnation_counter = 0
            
            for generation in range(num_generations):
                # Generate parameter batch
                batch = self.generate_batch(BATCH_SIZE)
                
                # Evaluate batch
                profits, fitness_scores = self.evaluate_batch(batch)
                total_evaluations += BATCH_SIZE
                
                # Find best in batch
                best_idx = torch.argmax(fitness_scores)
                batch_best_profit = profits[best_idx].item()
                batch_best_fitness = fitness_scores[best_idx].item()
                batch_best_params = {
                    k: int(v[best_idx].item()) 
                    for k, v in batch.items()
                }
                
                # Update global best if needed
                if batch_best_fitness > self.best_fitness:
                    improvement = batch_best_profit - self.best_profit
                    self.best_profit = batch_best_profit
                    self.best_fitness = batch_best_fitness
                    self.best_params = batch_best_params
                    stagnation_counter = 0
                    
                    self.log_message(f"\nNew best found in generation {generation}!")
                    self.log_message(f"Profit improvement: +{improvement}")
                    self.log_message(f"Fitness improvement: +{batch_best_fitness - self.best_fitness:.2f}")
                    self.log_message(f"Parameters: {batch_best_params}")
                    
                    # Save immediately
                    self.save_best_params()
                else:
                    stagnation_counter += 1
                
                # Log progress
                if generation % 10 == 0:
                    elapsed = time.time() - start_time
                    iterations_per_sec = total_evaluations / elapsed
                    
                    self.log_message(f"\nGeneration {generation}")
                    self.log_message(f"Best profit: {self.best_profit}")
                    self.log_message(f"Best fitness: {self.best_fitness:.2f}")
                    self.log_message(f"Iterations/sec: {iterations_per_sec:.2f}")
                    self.log_message(f"Stagnation counter: {stagnation_counter}")
                
                # Log to CSV
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for i in range(BATCH_SIZE):
                        row = [int(batch[k][i].item()) for k in PARAM_RANGES.keys()]
                        row.extend([profits[i].item(), fitness_scores[i].item(), generation])
                        writer.writerow(row)
                
                # Adjust mutation rate based on stagnation
                mutation_rate = min(0.1 + (stagnation_counter * 0.01), 0.3)
                batch = self.mutate_parameters(batch, mutation_rate)
            
        except KeyboardInterrupt:
            self.log_message("\nOptimization interrupted!")
        finally:
            # Save best parameters
            self.save_best_params()
            
            # Restore original main.py if requested
            response = input("\nRestore original main.py? (y/n): ")
            if response.lower() == 'y':
                with open('main.py.bak', 'r') as f:
                    original_content = f.read()
                with open('main.py', 'w') as f:
                    f.write(original_content)
                self.log_message("Restored original main.py")

if __name__ == "__main__":
    optimizer = GPUOptimizer()
    
    # Get number of generations from user
    num_gens = input("Enter number of generations (default 100): ")
    num_gens = int(num_gens) if num_gens.strip() else 100
    
    optimizer.optimize(num_generations=num_gens)
