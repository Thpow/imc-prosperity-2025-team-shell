import subprocess
import re
import json
import os
import datetime
from typing import Dict, Any, Tuple

def update_params_in_file(main_file: str, params: Dict[str, Any]) -> None:
    """Update parameters in main.py"""
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Update each parameter
    for param_name, value in params.items():
        pattern = rf'self\.{param_name}\s*=\s*[0-9.]+'
        replacement = f'self.{param_name} = {value}'
        content = re.sub(pattern, replacement, content)
    
    # Write updated content back to file
    with open(main_file, 'w') as f:
        f.write(content)

def run_backtest(main_file: str, round_number: int) -> Tuple[float, Dict[str, float]]:
    """Run backtest and return total profit and per-product profit"""
    cmd = f"prosperity3bt {main_file} {round_number}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse output to get profit
    output = result.stdout
    print(output)  # Display the output
    
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

def save_results(params: Dict[str, Any], total_profit: float, product_profits: Dict[str, float]) -> None:
    """Save test results to a JSON file"""
    results_dir = "test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(results_dir, f"test_results_{timestamp}.json")
    
    results = {
        "params": params,
        "total_profit": total_profit,
        "product_profits": product_profits
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    main_file = "main.py"
    round_number = 0
    
    # Define parameter combinations to test
    param_sets = [
        # Test 1: Best parameters from grid search
        {
            "position_limit": 50,
            "max_history_length": 50,
            "spread_multiplier": 0.5,
            "min_spread": 1,
            "max_position_spread_adjustment": 0.5,
            "base_volume": 15,
            "volume_scaling": 1.0,
            "volatility_window": 8,
            "momentum_threshold": 0.001,
            "momentum_boost": 1.0,
            "arbitrage_aggressiveness": 0.7,
            "price_improve_threshold": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std": 2.5
        },
        # Test 2: Slightly more aggressive position sizing
        {
            "position_limit": 60,
            "max_history_length": 50,
            "spread_multiplier": 0.5,
            "min_spread": 1,
            "max_position_spread_adjustment": 0.5,
            "base_volume": 18,
            "volume_scaling": 1.0,
            "volatility_window": 8,
            "momentum_threshold": 0.001,
            "momentum_boost": 1.2,
            "arbitrage_aggressiveness": 0.7,
            "price_improve_threshold": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std": 2.5
        },
        # Test 3: More conservative position sizing
        {
            "position_limit": 40,
            "max_history_length": 50,
            "spread_multiplier": 0.6,
            "min_spread": 1,
            "max_position_spread_adjustment": 0.5,
            "base_volume": 12,
            "volume_scaling": 1.0,
            "volatility_window": 8,
            "momentum_threshold": 0.001,
            "momentum_boost": 0.8,
            "arbitrage_aggressiveness": 0.7,
            "price_improve_threshold": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "bb_period": 20,
            "bb_std": 2.5
        },
        # Test 4: Adjusted technical indicators
        {
            "position_limit": 50,
            "max_history_length": 50,
            "spread_multiplier": 0.5,
            "min_spread": 1,
            "max_position_spread_adjustment": 0.5,
            "base_volume": 15,
            "volume_scaling": 1.0,
            "volatility_window": 8,
            "momentum_threshold": 0.001,
            "momentum_boost": 1.0,
            "arbitrage_aggressiveness": 0.7,
            "price_improve_threshold": 0.5,
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "bb_period": 20,
            "bb_std": 2.0
        },
    ]
    
    # Run tests for each parameter set
    for i, params in enumerate(param_sets):
        print(f"\n=== Test {i+1}: Running with parameters ===")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        # Update parameters in file
        update_params_in_file(main_file, params)
        
        # Run backtest
        total_profit, product_profits = run_backtest(main_file, round_number)
        
        # Save results
        save_results(params, total_profit, product_profits)
        
        print(f"Test {i+1} completed with total profit: {total_profit}")
        print("Product profits:")
        for product, profit in product_profits.items():
            print(f"  {product}: {profit}")
        print("=" * 50)

if __name__ == "__main__":
    main()
