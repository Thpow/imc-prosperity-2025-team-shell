from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math

class Trader:
    def __init__(self):
        self.positions = {"KELP": 0, "RAINFOREST_RESIN": 0}
        self.position_limits = {"KELP": 50, "RAINFOREST_RESIN": 50}
        # Price history for each product
        self.price_history = {"KELP": [], "RAINFOREST_RESIN": []}
        # Maximum history length
        self.max_history_length = 20
        # RSI periods
        self.rsi_period = 14
        
    def calculate_mid_price(self, order_depth):
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Default to neutral when not enough data
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100  # Prevent division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            # Not enough data, return None
            return None, None, None
        
        # Calculate SMA
        sma = sum(prices[-period:]) / period
        
        # Calculate standard deviation
        std = statistics.stdev(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return lower_band, sma, upper_band

    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Skip if no orders on either side
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue
                
            # Get current position
            current_position = self.positions.get(product, 0)
            
            # Calculate mid price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price is None:
                result[product] = orders
                continue
                
            # Update price history
            self.price_history[product].append(mid_price)
            if len(self.price_history[product]) > self.max_history_length:
                self.price_history[product].pop(0)
            
            # Get best bid and ask
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            if product == "RAINFOREST_RESIN":
                # Mean reversion for stable asset using Bollinger Bands
                if len(self.price_history[product]) >= 5:  # Need at least some price history
                    # Calculate Bollinger Bands with 1.5 standard deviations (tighter than usual)
                    bb_result = self.calculate_bollinger_bands(self.price_history[product], period=5, num_std=1.5)
                    
                    if bb_result:
                        lower_band, sma, upper_band = bb_result
                        
                        # Buy when price is near lower band (oversold)
                        if best_ask < lower_band * 1.01 and current_position < self.position_limits[product]:
                            # Calculate volume - more aggressive when deeper below band
                            buy_pct = min(1.0, (lower_band - best_ask) / lower_band * 10)
                            buy_vol = math.ceil(buy_pct * min(5, self.position_limits[product] - current_position))
                            buy_vol = min(buy_vol, abs(order_depth.sell_orders[best_ask]))
                            
                            if buy_vol > 0:
                                orders.append(Order(product, best_ask, buy_vol))
                                current_position += buy_vol
                        
                        # Sell when price is near upper band (overbought)
                        if best_bid > upper_band * 0.99 and current_position > -self.position_limits[product]:
                            # Calculate volume - more aggressive when higher above band
                            sell_pct = min(1.0, (best_bid - upper_band) / upper_band * 10)
                            sell_vol = math.ceil(sell_pct * min(5, self.position_limits[product] + current_position))
                            sell_vol = min(sell_vol, abs(order_depth.buy_orders[best_bid]))
                            
                            if sell_vol > 0:
                                orders.append(Order(product, best_bid, -sell_vol))
                                current_position -= sell_vol
            
            else:  # KELP
                # Trend following for volatile asset using RSI
                if len(self.price_history[product]) >= 5:  # Need at least some price history
                    # Calculate RSI
                    rsi = self.calculate_rsi(self.price_history[product], period=min(5, len(self.price_history[product])-1))
                    
                    # Buy when RSI is low (oversold)
                    if rsi < 30 and current_position < self.position_limits[product]:
                        # More oversold = larger position
                        buy_strength = (30 - rsi) / 30
                        buy_vol = math.ceil(buy_strength * min(10, self.position_limits[product] - current_position))
                        buy_vol = min(buy_vol, abs(order_depth.sell_orders[best_ask]))
                        
                        if buy_vol > 0:
                            orders.append(Order(product, best_ask, buy_vol))
                            current_position += buy_vol
                    
                    # Sell when RSI is high (overbought)
                    elif rsi > 70 and current_position > -self.position_limits[product]:
                        # More overbought = larger position
                        sell_strength = (rsi - 70) / 30
                        sell_vol = math.ceil(sell_strength * min(10, self.position_limits[product] + current_position))
                        sell_vol = min(sell_vol, abs(order_depth.buy_orders[best_bid]))
                        
                        if sell_vol > 0:
                            orders.append(Order(product, best_bid, -sell_vol))
                            current_position -= sell_vol
                    
                    # Close positions if RSI is returning to middle
                    elif 40 <= rsi <= 60:
                        if current_position > 0:  # Long position
                            sell_vol = min(current_position, abs(order_depth.buy_orders[best_bid]))
                            if sell_vol > 0:
                                orders.append(Order(product, best_bid, -sell_vol))
                                current_position -= sell_vol
                        elif current_position < 0:  # Short position
                            buy_vol = min(-current_position, abs(order_depth.sell_orders[best_ask]))
                            if buy_vol > 0:
                                orders.append(Order(product, best_ask, buy_vol))
                                current_position += buy_vol
            
            # Update position
            self.positions[product] = current_position
            result[product] = orders
        
        return result, 0, ""
