from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np

class Trader:
    def __init__(self):
        # Position limits per product
        self.POSITION_LIMITS = {"KELP": 45, "RAINFOREST_RESIN": 50, "SQUID_INK": 50}
        # Historical price tracking
        self.historical_prices = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        # Strategy parameters
        self.params = {
            "KELP": {
                "acceptable_price": 2000  # Simplified threshold strategy
            },  
            "RAINFOREST_RESIN": {"acceptable_price": 10000},  # Existing threshold
            "SQUID_INK": {"acceptable_price": 2000}  # Placeholder - not actually used for trading
        }
        # Track market conditions for KELP
        self.market_conditions = {"KELP": "normal", "SQUID_INK": "normal"}  # Can be "normal", "overbought", "oversold"

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

            # SQUID_INK placeholder strategy - no trading
            if product == "SQUID_INK":
                # Just track prices but don't trade
                result[product] = []  # Empty orders list means no trades
                
            # KELP threshold strategy
            elif product == "KELP":
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

            # RAINFOREST_RESIN threshold strategy
            elif product == "RAINFOREST_RESIN":
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