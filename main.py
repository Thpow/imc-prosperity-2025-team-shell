from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math

class Trader:
    def __init__(self):
        # Initialize position tracking
        self.position = 0
        self.position_limit = 50  # Position limit for KELP
        
        # Price history for KELP
        self.price_history = []
        self.max_history_length = 30
        
        # Market making parameters
        self.spread_multiplier = 2.0  # Multiplier for spread calculation
        self.min_spread = 2  # Minimum spread in ticks
        self.max_position_spread_adjustment = 1.5  # Increase spread when position grows
        
        # Volume parameters
        self.base_volume = 5  # Base order volume
        self.volume_scaling = 0.8  # Scaling factor for volume based on position
        
        # Volatility tracking
        self.volatility_window = 10
        self.volatility_history = []
        
        # Trading state tracking
        self.last_mid_price = None
        self.trader_data = ""
    
    def calculate_mid_price(self, order_depth):
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def calculate_volatility(self, prices):
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0
        
        # Calculate returns
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        
        # Return standard deviation of returns as volatility measure
        if not returns:
            return 0
        return statistics.stdev(returns) if len(returns) > 1 else 0
    
    def calculate_spread(self, volatility, position):
        """Calculate dynamic spread based on volatility and position"""
        # Base spread from volatility
        base_spread = max(self.min_spread, int(volatility * 100 * self.spread_multiplier))
        
        # Position-based spread adjustment
        position_factor = 1 + (abs(position) / self.position_limit) * self.max_position_spread_adjustment
        
        return max(self.min_spread, int(base_spread * position_factor))
    
    def calculate_order_volume(self, side, position):
        """Calculate order volume based on position and side"""
        # Reduce volume as position grows toward limit
        remaining_capacity = self.position_limit - abs(position)
        position_factor = remaining_capacity / self.position_limit
        
        # Asymmetric volume: larger orders when reducing position, smaller when increasing
        if (side == "buy" and position < 0) or (side == "sell" and position > 0):
            # Reducing position - more aggressive
            volume_adjustment = 1.2
        else:
            # Increasing position - more conservative
            volume_adjustment = 0.8
        
        volume = max(1, int(self.base_volume * position_factor * volume_adjustment))
        return volume

    def run(self, state: TradingState):
        """
        Market making strategy for KELP product
        """
        product = "KELP"
        result = {}
        orders: List[Order] = []
        
        # Get order depth for KELP
        if product not in state.order_depths:
            return result, self.trader_data, ""
        
        order_depth = state.order_depths[product]
        
        # Skip if no orders on either side
        if not order_depth.buy_orders or not order_depth.sell_orders:
            result[product] = orders
            return result, self.trader_data, ""
        
        # Get current position
        current_position = state.position.get(product, 0)
        self.position = current_position
        
        # Calculate mid price
        mid_price = self.calculate_mid_price(order_depth)
        if mid_price is None:
            result[product] = orders
            return result, self.trader_data, ""
        
        # Update price history
        self.price_history.append(mid_price)
        if len(self.price_history) > self.max_history_length:
            self.price_history.pop(0)
        
        # Calculate volatility
        volatility = self.calculate_volatility(self.price_history[-min(len(self.price_history), self.volatility_window):])
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.max_history_length:
            self.volatility_history.pop(0)
        
        # Get best bid and ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate spread based on volatility and position
        spread = self.calculate_spread(volatility, current_position)
        
        # Market making logic
        # Calculate our bid and ask prices
        our_bid = best_bid - 1  # Improve best bid by 1
        our_ask = best_ask + 1  # Improve best ask by 1
        
        # Widen spread if we're getting close to position limits
        position_ratio = abs(current_position) / self.position_limit
        if position_ratio > 0.7:
            # Widen the spread on the side that would increase position
            if current_position > 0:
                our_bid = best_bid - max(2, int(spread * position_ratio))
            else:
                our_ask = best_ask + max(2, int(spread * position_ratio))
        
        # Calculate volumes
        buy_volume = self.calculate_order_volume("buy", current_position)
        sell_volume = self.calculate_order_volume("sell", current_position)
        
        # Reduce volume when approaching position limits
        remaining_buy_capacity = self.position_limit - current_position
        remaining_sell_capacity = self.position_limit + current_position
        
        buy_volume = min(buy_volume, remaining_buy_capacity)
        sell_volume = min(sell_volume, remaining_sell_capacity)
        
        # Place orders if volumes are positive
        if buy_volume > 0:
            orders.append(Order(product, our_bid, buy_volume))
        
        if sell_volume > 0:
            orders.append(Order(product, our_ask, -sell_volume))
        
        # Take advantage of arbitrage opportunities
        for bid_price, bid_volume in order_depth.buy_orders.items():
            for ask_price, ask_volume in order_depth.sell_orders.items():
                if bid_price > ask_price:  # Arbitrage opportunity
                    # Calculate how much we can trade
                    max_trade = min(abs(bid_volume), abs(ask_volume))
                    
                    # Consider position limits
                    if current_position + max_trade > self.position_limit:
                        max_trade = self.position_limit - current_position
                    elif current_position - max_trade < -self.position_limit:
                        max_trade = current_position + self.position_limit
                    
                    if max_trade > 0:
                        # Buy at ask price and sell at bid price
                        orders.append(Order(product, ask_price, max_trade))
                        orders.append(Order(product, bid_price, -max_trade))
                        
                        # Update position for subsequent calculations
                        current_position = current_position  # Net zero effect on position
        
        # Save market state for next iteration
        self.last_mid_price = mid_price
        
        # Store result
        result[product] = orders
        
        return result, self.trader_data, ""