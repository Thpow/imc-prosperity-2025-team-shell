from main import black_scholes

# Example usage
S = 100  # Stock price
K = 100  # Strike price
T = 1    # Time to maturity (in years)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)

# Calculate call option price
call_price = black_scholes(S, K, T, r, sigma, 'call')
print(f"Call option price: {call_price:.2f}")

# Calculate put option price
put_price = black_scholes(S, K, T, r, sigma, 'put')
print(f"Put option price: {put_price:.2f}")
