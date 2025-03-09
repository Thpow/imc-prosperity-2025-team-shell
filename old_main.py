from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math
import numpy as np
from scipy.stats import norm

from funcs import *

class Trader:
    def __init__(self):
        # Initialize position tracking
        self.position = 0
        self.positions = {}
        
        # Position limits for each product
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50
        }

        #=======history tracking=======#
        self.price_history = {"RAINFOREST_RESIN":[], "KELP":[]}
        self.volatility_history = {"RAINFOREST_RESIN":[], "KELP":[]}
        self.low = {}
        self.high = {}

        #=======global params=======#
        self.max_history_length = 200
        self.max_volatility_length = 200
        
        #=======RAINFOREST_RESIN params=======#
        self.resin_params = {
            # Periods
            "short_ma_window": 12,
            "long_ma_window": 300,
            "momentum_window": 11,
            "rsi_period": 13,
            "bollinger_band_period": 15,
            
            # Thresholds
            "momentum_threshold": 0.09,
            "rsi_long_indicator": 45,  # Oversold threshold
            "rsi_short_indicator": 67,  # Overbought threshold
            
            # Counter-trading parameters
            "counter_trade_size": 0,  # Amount to trade in opposite direction for KELP
        }
        
        #=======KELP params=======#
        self.kelp_params = {
            # Periods
            "short_ma_window": 15,
            "long_ma_window": 106,
            "momentum_window": 8,
            "rsi_period": 17,
            "bollinger_band_period": 17,
            
            # Thresholds
            "momentum_threshold": 0.17,
            "rsi_long_indicator": 34,  # Oversold threshold  
            "rsi_short_indicator": 75,  # Overbought threshold
            
            # Counter-trading parameters
            "counter_trade_size": 0,  # Amount to trade in opposite direction for RESIN
        }

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

    def calculate_momentum(self, order_depth, price_history, product, momentum_window):
        """Calculate momentum only if we have enough price history"""
        todays_price = self.calculate_mid_price(order_depth)
        
        # Check if we have enough data points for momentum calculation
        if len(price_history[product]) > momentum_window:
            return todays_price - price_history[product][-momentum_window]
        else:
            return 0  # Return neutral momentum when not enough data
    
    def calculate_rsi(self, prices, product, period):
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

    def calculate_bollinger_bands(self, prices, product, period, num_std=2):
        """Calculate Bollinger Bands"""
        # Ensure the product exists in prices and we have enough data
        if product not in prices or len(prices[product]) < period:
            mid_price = prices[product][-1] if product in prices and prices[product] else 0
            return mid_price * 0.95, mid_price, mid_price * 1.05  # Return placeholder bands when not enough data

        # Calculate SMA using the most recent 'period' prices
        price_data = prices[product][-period:]
        sma = sum(price_data) / period

        # Calculate standard deviation and the bands
        std = statistics.stdev(price_data)
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return lower_band, sma, upper_band

    def process_product(self, product, state, params):
        """Process trading logic for a specific product"""
        orders = []
        counter_orders = []
        counter_product = "KELP" if product == "RAINFOREST_RESIN" else "RAINFOREST_RESIN"
        
        # Get order depth for the product
        order_depth = state.order_depths.get(product, None)
        counter_order_depth = state.order_depths.get(counter_product, None)
        
        if not order_depth or not counter_order_depth:
            return orders, counter_orders
            
        # Update position
        position = state.position.get(product, 0)
        position_limit = self.position_limits[product]
        
        # Get current mid price and update history
        current_price = self.calculate_mid_price(order_depth)
        counter_price = self.calculate_mid_price(counter_order_depth)
        
        if current_price is None or counter_price is None:
            return orders, counter_orders
            
        self.price_history[product].append(current_price)
        
        # Limit history length
        if len(self.price_history[product]) > self.max_history_length:
            self.price_history[product] = self.price_history[product][-self.max_history_length:]
            
        # We need at least enough data for our indicators
        min_data_required = max(
            params["rsi_period"] + 1, 
            params["bollinger_band_period"], 
            params["momentum_window"] + 1
        )
        
        # Check if we have enough data to calculate indicators
        if len(self.price_history[product]) >= min_data_required:
            # Calculate indicators
            rsi = self.calculate_rsi(self.price_history, product, params["rsi_period"])
            lower_band, sma, upper_band = self.calculate_bollinger_bands(
                self.price_history, product, params["bollinger_band_period"]
            )
            momentum = self.calculate_momentum(
                order_depth, self.price_history, product, params["momentum_window"]
            )
            
            trade_size = 0
            counter_trade_size = 0
            
            # Bollinger Bands based trading logic
            if current_price < lower_band and position < position_limit:
                # Buy signal: price is below the lower Bollinger Band (oversold condition)
                buy_size = min(2, position_limit - position)
                if buy_size > 0:
                    orders.append(Order(product, current_price, buy_size))
                    # Counter-trade in opposite direction with smaller size
                    counter_trade_size -= 1
            elif current_price > upper_band and position > -position_limit:
                # Sell signal: price is above the upper Bollinger Band (overbought condition)
                sell_size = min(2, position_limit + position)
                if sell_size > 0:
                    orders.append(Order(product, current_price, -sell_size))
                    # Counter-trade in opposite direction with smaller size
                    counter_trade_size += 1
                    
            # RSI based signals
            if rsi < params["rsi_long_indicator"] and position < position_limit:
                # Buy signal based on RSI oversold
                buy_size = min(5, position_limit - position)
                if buy_size > 0:
                    orders.append(Order(product, current_price, buy_size))
                    # Counter-trade in opposite direction
                    counter_trade_size -= params["counter_trade_size"]
            elif rsi > params["rsi_short_indicator"] and position > -position_limit:
                # Sell signal based on RSI overbought
                sell_size = min(5, position_limit + position)
                if sell_size > 0:
                    orders.append(Order(product, current_price, -sell_size))
                    # Counter-trade in opposite direction
                    counter_trade_size += params["counter_trade_size"]
            
            # Process counter orders only if we have a valid counter price and trade size
            if counter_trade_size != 0 and counter_price is not None:
                counter_position = state.position.get(counter_product, 0)
                counter_limit = self.position_limits[counter_product]
                
                # Ensure we don't exceed position limits for counter trades
                if counter_trade_size > 0:
                    counter_trade_size = min(counter_trade_size, counter_limit - counter_position)
                else:
                    counter_trade_size = max(counter_trade_size, -counter_limit - counter_position)
                
                if counter_trade_size != 0:
                    counter_orders.append(Order(counter_product, counter_price, counter_trade_size))
        
        return orders, counter_orders

    def run(self, state: TradingState):
        # Init
        result = {}
        
        # Initialize empty orders for each product
        result["RAINFOREST_RESIN"] = []
        result["KELP"] = []
        
        # Process RAINFOREST_RESIN
        resin_orders, resin_counter_orders = self.process_product("RAINFOREST_RESIN", state, self.resin_params)
        result["RAINFOREST_RESIN"].extend(resin_orders)
        result["KELP"].extend(resin_counter_orders)
        
        # Process KELP
        kelp_orders, kelp_counter_orders = self.process_product("KELP", state, self.kelp_params)
        result["KELP"].extend(kelp_orders)
        result["RAINFOREST_RESIN"].extend(kelp_counter_orders)
        
        return result, 0, ""