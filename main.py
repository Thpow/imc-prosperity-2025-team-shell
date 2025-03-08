from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math
import numpy as np
from scipy.stats import norm

from funcs import *

product = "KELP"

class Trader:
    def __init__(self):
        # Initialize position tracking
        self.position = 0
        self.positions = {}
        self.position_limit = 50

        #=======history tacking=======#
        self.price_history = {"KELP":[]}
        self.volatility_history = {"KELP":[]}
        self.low = {}
        self.high = {}

        #=======params=======#
        #periods
        self.max_history_length = 200
        self.max_volatility_length = 200   
        self.short_ma_window = 50
        self.long_ma_window = 200
        self.momentum_window = 10
        self.rsi_period = 14
        self.bollinger_band_period = 20

        #thresholds
        self.momentum_threshold = 0.1
        self.rsi_long_indicator = 30
        self.rsi_short_indicator = 50

        
    def calculate_mid_price(self, order_depth):
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def calculate_spread(self, order_depth):
        """Calculate spread"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_ask - best_bid)

    def calculate_momentum(self, order_depth, price_history, momentum_window):
        """Calculate momentum only if we have enough price history"""
        todays_price = self.calculate_mid_price(order_depth)
        
        # Check if we have enough data points for momentum calculation
        if len(price_history[product]) > momentum_window:
            return todays_price - price_history[product][-momentum_window]
        else:
            return 0  # Return neutral momentum when not enough data
    
    def calculate_rsi(self, prices, period):
        """Calculate Relative Strength Index"""
        # Check if we have enough data
        if len(prices[product]) <= period:
            return 50  # Return neutral RSI when not enough data
            
        # Calculate price changes
        price_data = prices[product][-period-1:] # Get enough data for calculating changes
        deltas = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100  # Prevent division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_bollinger_bands(self, prices, period, num_std=2):
        """Calculate Bollinger Bands"""
        # Check if we have enough data
        if len(prices[product]) < period:
            mid_price = prices[product][-1] if prices[product] else 0
            return mid_price * 0.95, mid_price, mid_price * 1.05  # Return placeholder bands when not enough data
        
        # Calculate SMA
        price_data = prices[product][-period:]
        sma = sum(price_data) / period
        
        # Calculate standard deviation
        std = statistics.stdev(price_data)
        
        # Calculate upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return lower_band, sma, upper_band
        
    def run(self, state: TradingState):
        # Init
        result = {}
        orders: List[Order] = []
        
        # Get order depth for the product
        order_depth: OrderDepth = state.order_depths[product]
        
        # Update position
        if product in state.position:
            self.position = state.position[product]
        else:
            self.position = 0
            
        # Get current mid price and update history
        current_price = self.calculate_mid_price(order_depth)
        if current_price is not None:
            self.price_history[product].append(current_price)
            
        # Limit history length
        if len(self.price_history[product]) > self.max_history_length:
            self.price_history[product] = self.price_history[product][-self.max_history_length:]
        
        # We need at least rsi_period+1 data points to calculate indicators properly
        min_data_required = max(self.rsi_period + 1, self.bollinger_band_period, self.momentum_window + 1)
        
        # Check if we have enough data to calculate indicators
        if len(self.price_history[product]) >= min_data_required:
            # Calculate indicators
            rsi = self.calculate_rsi(self.price_history, self.rsi_period)
            lower_band, sma, upper_band = self.calculate_bollinger_bands(self.price_history, self.bollinger_band_period)
            momentum = self.calculate_momentum(order_depth, self.price_history, self.momentum_window)
            
            # Trading logic - RSI based
            if rsi < self.rsi_long_indicator and self.position < self.position_limit:
                # Buy signal - RSI oversold
                orders.append(Order(product, current_price, self.position_limit - self.position))
                
            elif rsi > self.rsi_short_indicator and self.position > 0:
                # Sell signal - RSI overbought
                orders.append(Order(product, current_price, -self.position))
                
            # Add bollinger band logic here if needed
                
        # Add all the orders to the result
        result[product] = orders
        
        return result, 0, ""