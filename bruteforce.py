# Exchange rate matrix
rates = [
    [1.00, 1.45, 0.52, 0.72],  # Snowballs
    [0.70, 1.00, 0.31, 0.48],  # Pizzas
    [1.95, 3.10, 1.00, 1.49],  # Silicon Nuggets
    [1.34, 1.98, 0.64, 1.00]   # Seashells
]

# Currency names for readable output
names = ["Snowballs", "Pizzas", "Silicon Nuggets", "Seashells"]

# Global variables to track the best result
max_amount = 0
best_path = []

def dfs(current_currency, current_amount, path, trades):
    """
    Recursively explore all conversion paths with exactly 5 trades.
    
    Args:
        current_currency (int): Current currency index (0-3)
        current_amount (float): Current amount after conversions
        path (list): Sequence of currencies visited
        trades (int): Number of trades made so far
    """
    global max_amount, best_path
    
    # Base case: exactly 5 trades and back to Seashells
    if trades == 5 and current_currency == 3:
        if current_amount > max_amount:
            max_amount = current_amount
            best_path = path.copy()
        return
    
    # Stop if we've exceeded 5 trades or can't reach 5 trades with remaining steps
    if trades >= 5:
        return
    
    # Try converting to each currency
    for next_currency in range(4):
        new_amount = current_amount * rates[current_currency][next_currency]
        dfs(next_currency, new_amount, path + [next_currency], trades + 1)

# Start with 500 Seashells
initial_currency = 3  # Seashells
initial_amount = 500
dfs(initial_currency, initial_amount, [initial_currency], 0)

# Output the result
if max_amount > 500:
    print("Most profitable path with exactly 5 trades:", " -> ".join(names[c] for c in best_path))
    print(f"Final amount: {max_amount:.2f} Seashells")
else:
    print("No profitable path with exactly 5 trades increases the initial 500 Seashells.")