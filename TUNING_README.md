# Parameter Tuning for Trading Strategy

This directory contains scripts for systematically tuning the parameters of the trading strategy in `main.py` to maximize profit.

## Available Scripts

### 1. Parameter Tuner (`parameter_tuner.py`)

This script performs systematic parameter optimization using various methods:

- **Random Search**: Randomly samples parameter combinations
- **Grid Search**: Tests all combinations of selected parameters
- **Bayesian Optimization**: Uses a simple Bayesian-inspired approach to balance exploration and exploitation
- **Hill Climbing**: Iteratively improves parameters by testing neighboring values
- **Coordinate Ascent**: Optimizes one parameter at a time, finding the local optimum for each parameter before moving to the next

#### Usage:

```bash
# Run Bayesian optimization with 20 iterations
python parameter_tuner.py --method bayesian --iterations 20 --round 0

# Run grid search on selected parameters
python parameter_tuner.py --method grid --round 0

# Run random search with 30 iterations and apply the best parameters found
python parameter_tuner.py --method random --iterations 30 --round 0 --apply-best

# Run hill climbing with 15 iterations
python parameter_tuner.py --method hill --iterations 15 --round 0

# Run coordinate ascent with 5 iterations
python parameter_tuner.py --method coordinate --iterations 5 --round 0
```

The `--iterations` parameter specifies the maximum number of iterations per parameter for the coordinate ascent method.

### 2. Quick Test (`quick_test.py`)

This script allows you to quickly test a few specific parameter combinations and compare their performance.

#### Usage:

```bash
# Run the predefined parameter sets
python quick_test.py
```

You can edit the `param_sets` list in the script to test different parameter combinations.

### 3. Results Analysis (`analyze_results.py`)

This script analyzes the results of parameter tuning runs, providing insights into:

- The impact of each parameter on profit
- The progression of profit over iterations
- The best parameter combination found
- The distribution of parameter values in top-performing runs

#### Usage:

```bash
# Analyze the results of parameter tuning
python analyze_results.py
```

## Workflow for Parameter Tuning

1. **Initial Exploration**: Run `quick_test.py` to test a few parameter combinations and get a sense of their impact.

2. **Systematic Tuning**: Use `parameter_tuner.py` with the Bayesian or random search method to explore a wider range of parameter combinations.

3. **Focused Tuning**: Based on the results of the initial exploration, use grid search to fine-tune the most important parameters.

4. **Analysis**: Run `analyze_results.py` to analyze the results and gain insights into parameter impact.

5. **Final Optimization**: Use hill climbing to fine-tune the best parameter combination found.

6. **Fine-tuning**: Use the coordinate ascent method to find local optima for each parameter.

## Example Tuning Session

```bash
# Step 1: Run quick tests to get a baseline
python quick_test.py

# Step 2: Run Bayesian optimization with 20 iterations
python parameter_tuner.py --method bayesian --iterations 20 --round 0

# Step 3: Analyze the results
python analyze_results.py

# Step 4: Run grid search on the most important parameters
python parameter_tuner.py --method grid --round 0

# Step 5: Run hill climbing to fine-tune the best parameters
python parameter_tuner.py --method hill --iterations 15 --round 0 --apply-best

# Step 6: Fine-tune using coordinate ascent
python parameter_tuner.py --method coordinate --iterations 5 --round 0 --apply

# Step 7: Verify the final performance
python quick_test.py
```

## Tips for Effective Parameter Tuning

1. **Start with a small number of iterations** to get a sense of parameter impact.

2. **Focus on the most important parameters** identified by the analysis.

3. **Use a combination of methods** for more thorough exploration of the parameter space.

4. **Save intermediate results** to avoid losing progress.

5. **Analyze results frequently** to guide the tuning process.

6. **Be patient** - parameter tuning is an iterative process that takes time.

7. **Consider the trade-off between exploration and exploitation** - too much exploration may not converge to the best parameters, while too much exploitation may get stuck in local optima.

8. **Use the `--apply` flag** to automatically apply the best parameters found.
