from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np

class Trader:
    def __init__(self):
        # Position limits per product
        self.POSITION_LIMITS = {"KELP": 20, "RAINFOREST_RESIN": 20}
        # Historical price tracking
        self.historical_prices = {"KELP": [], "RAINFOREST_RESIN": []}
        # Strategy parameters
        self.params = {
            "KELP": {"window": 50, "std_multiplier": 1.0},  # For mean-reversion
            "RAINFOREST_RESIN": {"acceptable_price": 10000}  # Existing threshold
        }

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid-price from best bid and ask."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return 0

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
                # Mean-reversion strategy
                thresholds = self.optimize_kelp_params(product)
                if thresholds:
                    buy_threshold, sell_threshold = thresholds
                    if mid_price < buy_threshold and position < position_limit:
                        buy_quantity = min(5, position_limit - position)
                        best_ask = min(order_depth.sell_orders.keys())
                        if best_ask <= buy_threshold:
                            orders.append(Order(product, best_ask, buy_quantity))
                    elif mid_price > sell_threshold and position > -position_limit:
                        sell_quantity = min(5, position_limit + position)
                        best_bid = max(order_depth.buy_orders.keys())
                        if best_bid >= sell_threshold:
                            orders.append(Order(product, best_bid, -sell_quantity))

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
                    if best_bid > acceptable_price:
                        sell_quantity = min(order_depth.buy_orders[best_bid], position_limit + position)
                        if sell_quantity > 0:
                            orders.append(Order(product, best_bid, -sell_quantity))

            result[product] = orders

        conversions = 0
        traderData = ""
        return result, conversions, traderData