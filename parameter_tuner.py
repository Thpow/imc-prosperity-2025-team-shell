import subprocess
import json
import os
import time
import datetime
import re
import random
import numpy as np
from typing import Dict, List, Tuple, Any

class ParameterTuner:
    def __init__(self, main_file="main.py", round_number=0, results_dir="tuning_results"):
        self.main_file = main_file
        self.round_number = round_number
        self.results_dir = results_dir
        self.best_profit = -float('inf')
        self.best_params = None
        self.results_history = []
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Parameters to tune and their ranges
        self.param_ranges = {
            "position_limit": [30, 50, 70, 100],
            "max_history_length": [30, 50, 70],
            "spread_multiplier": [0.5, 0.8, 1.0, 1.5],
            "min_spread": [1, 2],
            "max_position_spread_adjustment": [0.3, 0.5, 0.7],
            "base_volume": [5, 10, 15],
            "volume_scaling": [0.8, 1.0, 1.2],
            "volatility_window": [5, 8, 10],
            "momentum_threshold": [0.0005, 0.001, 0.002],
            "momentum_boost": [1.0, 1.5, 2.0],
            "arbitrage_aggressiveness": [0.6, 0.8, 0.9],
            "price_improve_threshold": [0.4, 0.5, 0.6],
            "rsi_period": [14],  # Standard RSI period, not tuning
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "bb_period": [20],  # Standard BB period, not tuning
            "bb_std": [1.5, 2.0, 2.5]
        }
        
        # Initialize the best parameters with current values
        self.best_params = self.get_current_params()
        
    def get_current_params(self) -> Dict[str, Any]:
        """Extract current parameters from main.py"""
        params = {}
        with open(self.main_file, 'r') as f:
            content = f.read()
            
        # Extract parameters using regex
        for param_name in self.param_ranges.keys():
            pattern = rf'self\.{param_name}\s*=\s*([0-9.]+)'
            match = re.search(pattern, content)
            if match:
                value_str = match.group(1)
                # Convert to appropriate type
                if '.' in value_str:
                    params[param_name] = float(value_str)
                else:
                    params[param_name] = int(value_str)
        
        return params
    
    def update_params_in_file(self, params: Dict[str, Any]) -> None:
        """Update parameters in main.py"""
        with open(self.main_file, 'r') as f:
            content = f.read()
        
        # Update each parameter
        for param_name, value in params.items():
            pattern = rf'self\.{param_name}\s*=\s*[0-9.]+'
            replacement = f'self.{param_name} = {value}'
            content = re.sub(pattern, replacement, content)
        
        # Write updated content back to file
        with open(self.main_file, 'w') as f:
            f.write(content)
    
    def run_backtest(self) -> Tuple[float, Dict[str, float]]:
        """Run backtest and return total profit and per-product profit"""
        cmd = f"prosperity3bt {self.main_file} {self.round_number}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse output to get profit
        output = result.stdout
        
        # Extract profits using regex
        total_profit = 0
        product_profits = {}
        
        # Extract product profits
        product_pattern = r'([A-Z_]+): ([0-9,]+)'
        for match in re.finditer(product_pattern, output):
            product = match.group(1)
            profit_str = match.group(2).replace(',', '')
            profit = int(profit_str)
            product_profits[product] = profit
        
        # Extract total profit
        total_pattern = r'Total profit: ([0-9,]+)'
        total_match = re.search(total_pattern, output)
        if total_match:
            total_profit_str = total_match.group(1).replace(',', '')
            total_profit = int(total_profit_str)
        
        return total_profit, product_profits
    
    def random_search(self, iterations=20) -> None:
        """Perform random search for parameter optimization"""
        print(f"Starting random search with {iterations} iterations...")
        
        for i in range(iterations):
            # Generate random parameters
            params = {}
            for param_name, values in self.param_ranges.items():
                params[param_name] = random.choice(values)
            
            # Update parameters in file
            self.update_params_in_file(params)
            
            # Run backtest
            total_profit, product_profits = self.run_backtest()
            
            # Record results
            result = {
                "iteration": i + 1,
                "params": params.copy(),
                "total_profit": total_profit,
                "product_profits": product_profits
            }
            self.results_history.append(result)
            
            # Update best parameters if profit improved
            if total_profit > self.best_profit:
                self.best_profit = total_profit
                self.best_params = params.copy()
                print(f"New best profit: {total_profit} with parameters: {params}")
            
            # Save results periodically
            if (i + 1) % 5 == 0 or i == iterations - 1:
                self.save_results()
            
            print(f"Iteration {i+1}/{iterations}: Profit = {total_profit}")
    
    def grid_search(self, param_names=None) -> None:
        """Perform grid search on specified parameters"""
        if param_names is None:
            # Choose a subset of parameters to tune (to avoid combinatorial explosion)
            param_names = ["position_limit", "spread_multiplier", "base_volume", "momentum_boost"]
        
        print(f"Starting grid search on parameters: {param_names}...")
        
        # Get current best parameters as starting point
        current_params = self.best_params.copy() if self.best_params else self.get_current_params()
        
        # Generate all combinations of specified parameters
        param_values = [self.param_ranges[name] for name in param_names]
        combinations = self._generate_combinations(param_values)
        
        total_combinations = len(combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(combinations):
            # Update parameters with current combination
            params = current_params.copy()
            for j, name in enumerate(param_names):
                params[name] = combination[j]
            
            # Update parameters in file
            self.update_params_in_file(params)
            
            # Run backtest
            total_profit, product_profits = self.run_backtest()
            
            # Record results
            result = {
                "iteration": i + 1,
                "params": params.copy(),
                "total_profit": total_profit,
                "product_profits": product_profits
            }
            self.results_history.append(result)
            
            # Update best parameters if profit improved
            if total_profit > self.best_profit:
                self.best_profit = total_profit
                self.best_params = params.copy()
                print(f"New best profit: {total_profit} with parameters: {params}")
            
            # Save results periodically
            if (i + 1) % 10 == 0 or i == total_combinations - 1:
                self.save_results()
            
            print(f"Combination {i+1}/{total_combinations}: Profit = {total_profit}")
    
    def bayesian_optimization(self, iterations=20) -> None:
        """Simple Bayesian-inspired optimization (exploration vs. exploitation)"""
        print(f"Starting Bayesian-inspired optimization with {iterations} iterations...")
        
        # Start with the current best parameters
        current_params = self.best_params.copy() if self.best_params else self.get_current_params()
        
        for i in range(iterations):
            # Decide whether to explore or exploit
            if i < iterations * 0.3 or random.random() < 0.3:  # 30% exploration
                # Exploration: randomly change multiple parameters
                params = current_params.copy()
                num_params_to_change = random.randint(1, min(5, len(self.param_ranges)))
                params_to_change = random.sample(list(self.param_ranges.keys()), num_params_to_change)
                
                for param_name in params_to_change:
                    params[param_name] = random.choice(self.param_ranges[param_name])
            else:
                # Exploitation: change one parameter slightly from best known
                params = self.best_params.copy()
                param_name = random.choice(list(self.param_ranges.keys()))
                current_value_index = 0
                
                # Find the index of the current value in the parameter range
                if params[param_name] in self.param_ranges[param_name]:
                    current_value_index = self.param_ranges[param_name].index(params[param_name])
                
                # Move one step left or right in the parameter range
                possible_indices = []
                if current_value_index > 0:
                    possible_indices.append(current_value_index - 1)
                if current_value_index < len(self.param_ranges[param_name]) - 1:
                    possible_indices.append(current_value_index + 1)
                
                if possible_indices:
                    new_index = random.choice(possible_indices)
                    params[param_name] = self.param_ranges[param_name][new_index]
            
            # Update parameters in file
            self.update_params_in_file(params)
            
            # Run backtest
            total_profit, product_profits = self.run_backtest()
            
            # Record results
            result = {
                "iteration": i + 1,
                "params": params.copy(),
                "total_profit": total_profit,
                "product_profits": product_profits
            }
            self.results_history.append(result)
            
            # Update best parameters if profit improved
            if total_profit > self.best_profit:
                self.best_profit = total_profit
                self.best_params = params.copy()
                print(f"New best profit: {total_profit} with parameters: {params}")
            
            # Save results periodically
            if (i + 1) % 5 == 0 or i == iterations - 1:
                self.save_results()
            
            print(f"Iteration {i+1}/{iterations}: Profit = {total_profit}")
    
    def hill_climbing(self, iterations=20) -> None:
        """Perform hill climbing optimization"""
        print(f"Starting hill climbing with {iterations} iterations...")
        
        # Start with the current best parameters
        current_params = self.best_params.copy() if self.best_params else self.get_current_params()
        current_profit = self.best_profit
        
        for i in range(iterations):
            # Generate a neighbor by changing one parameter
            neighbor_params = current_params.copy()
            param_name = random.choice(list(self.param_ranges.keys()))
            
            # Get current value and possible alternatives
            current_value = neighbor_params[param_name]
            alternatives = [v for v in self.param_ranges[param_name] if v != current_value]
            
            if alternatives:
                neighbor_params[param_name] = random.choice(alternatives)
                
                # Update parameters in file
                self.update_params_in_file(neighbor_params)
                
                # Run backtest
                neighbor_profit, product_profits = self.run_backtest()
                
                # Record results
                result = {
                    "iteration": i + 1,
                    "params": neighbor_params.copy(),
                    "total_profit": neighbor_profit,
                    "product_profits": product_profits
                }
                self.results_history.append(result)
                
                # Move to neighbor if it's better
                if neighbor_profit > current_profit:
                    current_params = neighbor_params.copy()
                    current_profit = neighbor_profit
                    
                    # Update global best if needed
                    if current_profit > self.best_profit:
                        self.best_profit = current_profit
                        self.best_params = current_params.copy()
                        print(f"New best profit: {current_profit} with parameters: {current_params}")
                
                # Save results periodically
                if (i + 1) % 5 == 0 or i == iterations - 1:
                    self.save_results()
                
                print(f"Iteration {i+1}/{iterations}: Profit = {neighbor_profit}")
    
    def _generate_combinations(self, param_values):
        """Generate all combinations of parameter values"""
        if not param_values:
            return [[]]
        
        result = []
        for value in param_values[0]:
            for combination in self._generate_combinations(param_values[1:]):
                result.append([value] + combination)
        
        return result
    
    def save_results(self) -> None:
        """Save results to a JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.results_dir, f"tuning_results_{timestamp}.json")
        
        results = {
            "best_profit": self.best_profit,
            "best_params": self.best_params,
            "results_history": self.results_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def apply_best_params(self) -> None:
        """Apply the best parameters found to main.py"""
        if self.best_params:
            self.update_params_in_file(self.best_params)
            print(f"Applied best parameters with profit {self.best_profit}:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        else:
            print("No best parameters found.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter tuner for trading strategy")
    parser.add_argument("--method", type=str, default="bayesian", 
                        choices=["random", "grid", "bayesian", "hill"],
                        help="Optimization method to use")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of iterations for optimization")
    parser.add_argument("--round", type=int, default=0,
                        help="Round number for backtesting")
    parser.add_argument("--apply-best", action="store_true",
                        help="Apply best parameters after optimization")
    
    args = parser.parse_args()
    
    tuner = ParameterTuner(round_number=args.round)
    
    # Run the selected optimization method
    if args.method == "random":
        tuner.random_search(iterations=args.iterations)
    elif args.method == "grid":
        tuner.grid_search()
    elif args.method == "bayesian":
        tuner.bayesian_optimization(iterations=args.iterations)
    elif args.method == "hill":
        tuner.hill_climbing(iterations=args.iterations)
    
    # Apply best parameters if requested
    if args.apply_best:
        tuner.apply_best_params()
