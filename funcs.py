from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string



def calculate_mid_price(order_depth):
    """Calculate mid price from order book"""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2