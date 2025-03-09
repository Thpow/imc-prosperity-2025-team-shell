from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np

class Trader:
    def __init__(self):
        # Position limits per product
        self.POSITION_LIMITS = {"KELP": 45, "RAINFOREST_RESIN": 50}
        # Historical price tracking
        self.historical_prices = {"KELP": [], "RAINFOREST_RESIN": []}
        # Strategy parameters
        self.params = {
            "KELP": {
                "window": 51, 
                "std_multiplier": .91,
                "rsi_period": 50,      # Standard RSI period
                "rsi_overbought": 70,  # RSI threshold for overbought
                "rsi_oversold": 30,    # RSI threshold for oversold
                "rsi_neutral_high": 60, # Upper neutral zone - avoid buying
                "rsi_neutral_low": 40   # Lower neutral zone - avoid selling
            },  
            "RAINFOREST_RESIN": {"acceptable_price": 10000}  # Existing threshold
        }
        # Track market conditions for KELP
        self.market_conditions = {"KELP": "normal"}  # Can be "normal", "overbought", "oversold"

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid-price from best bid and ask."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return 0
        
    def calculate_rsi(self, product: str) -> float:
        """Calculate Relative Strength Index (RSI) with standard period of 14."""
        prices = self.historical_prices[product]
        period = self.params[product]["rsi_period"]
        
        # Need at least period+1 prices to calculate RSI
        if len(prices) < period + 1:
            return 50  # Default to neutral when not enough data
            
        # Calculate price changes
        deltas = np.diff(prices[-(period+1):])
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        # Calculate RSI
        if avg_loss == 0:
            return 100  # No losses, market is completely bullish
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def optimize_kelp_params(self, product: str):
        """Calculate dynamic thresholds for KELP based on historical prices."""
        prices = self.historical_prices[product]
        if len(prices) < self.params[product]["window"]:
            return None  # Not enough data yet
        window = self.params[product]["window"]
        recent_prices = prices[-window:]
        ma = np.mean(recent_prices)  # Moving average
        std = np.std(recent_prices)  # Standard deviation
        std_multiplier = self.params[product]["std_multiplier"]
        return ma - std_multiplier * std, ma + std_multiplier * std  # Buy/sell thresholds

    def assess_market_condition(self, product: str, rsi: float) -> str:
        """Assess market condition based on RSI value."""
        if rsi >= self.params[product]["rsi_overbought"]:
            return "overbought"
        elif rsi <= self.params[product]["rsi_oversold"]:
            return "oversold"
        else:
            return "normal"

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS.get(product, 20)

            # Track mid-price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price > 0:
                self.historical_prices[product].append(mid_price)
                if len(self.historical_prices[product]) > 100:
                    self.historical_prices[product].pop(0)

            if product == "KELP":
                # Calculate RSI for KELP
                rsi = self.calculate_rsi(product)
                
                # Update market condition assessment
                self.market_conditions[product] = self.assess_market_condition(product, rsi)
                
                # Mean-reversion strategy enhanced with RSI
                thresholds = self.optimize_kelp_params(product)
                if thresholds:
                    buy_threshold, sell_threshold = thresholds
                    
                    # RSI RULES FOR KELP:
                    # 1. In overbought conditions (RSI > 70), avoid buying and prefer shorting
                    # 2. In oversold conditions (RSI < 30), avoid shorting and prefer buying
                    # 3. In neutral zone, use standard mean-reversion but with caution
                    
                    # BUY LOGIC - Only buy when RSI is below neutral high or in oversold condition
                    if mid_price < buy_threshold and position < position_limit and (
                        rsi < self.params[product]["rsi_neutral_high"] or 
                        self.market_conditions[product] == "oversold"
                    ):
                        # Buy more aggressively if oversold
                        buy_quantity = min(8 if rsi < self.params[product]["rsi_oversold"] else 5, 
                                          position_limit - position)
                        
                        best_ask = min(order_depth.sell_orders.keys())
                        if best_ask <= buy_threshold:
                            orders.append(Order(product, best_ask, buy_quantity))
                    
                    # SELL/SHORT LOGIC - Only sell when RSI is above neutral low or in overbought condition  
                    elif mid_price > sell_threshold and position > -position_limit and (
                        rsi > self.params[product]["rsi_neutral_low"] or 
                        self.market_conditions[product] == "overbought"
                    ):
                        # Sell more aggressively if overbought
                        sell_quantity = min(8 if rsi > self.params[product]["rsi_overbought"] else 5, 
                                           position_limit + position)
                        
                        best_bid = max(order_depth.buy_orders.keys())
                        if best_bid >= sell_threshold:
                            orders.append(Order(product, best_bid, -sell_quantity))
                    
                    # Special case: Strong RSI signals can override price thresholds
                    # Very high RSI -> short even if price isn't at threshold
                    elif rsi > 75 and position > -position_limit:
                        sell_quantity = min(10, position_limit + position)
                        best_bid = max(order_depth.buy_orders.keys())
                        orders.append(Order(product, best_bid, -sell_quantity))
                    
                    # Very low RSI -> buy even if price isn't at threshold  
                    elif rsi < 25 and position < position_limit:
                        buy_quantity = min(10, position_limit - position)
                        best_ask = min(order_depth.sell_orders.keys())
                        orders.append(Order(product, best_ask, buy_quantity))

            elif product == "RAINFOREST_RESIN":
                # Existing strategy (optimized slightly)
                acceptable_price = self.params[product]["acceptable_price"]
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    if best_ask < acceptable_price:
                        buy_quantity = min(-order_depth.sell_orders[best_ask], position_limit - position)
                        if buy_quantity > 0:
                            orders.append(Order(product, best_ask, buy_quantity))
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    if best_bid > acceptable_price+1:
                        sell_quantity = min(order_depth.buy_orders[best_bid], position_limit + position)
                        if sell_quantity > 0:
                            orders.append(Order(product, best_bid, -sell_quantity))

            result[product] = orders

        conversions = 0
        traderData = ""
        return result, conversions, traderData