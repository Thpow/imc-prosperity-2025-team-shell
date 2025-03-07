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
     
        self.position = 0
        self.positions = {}
        self.position_limit = 50

        #=======history tacking=======#
        self.price_history = {"KELP":[]}
        self.volatility_history = {"KELP":[]}
        self.position_limit = 50
        self.low = {}
        self.high = {}

        #=======params=======#
        self.max_history_length = 200
        self.max_volatility_length = 200   
        self.short_ma_window = 50
        self.long_ma_window = 200
        
    def calculate_ma(self, prices, window):
        return sum(prices)/window

    def run(self, state: TradingState):
        #init
        result = {}
        orders: List[Order] = []
        
        #init dephth with product
        order_depth: OrderDepth = state.order_depths[product]

        #update historic data
        self.price_history[product].append(calculate_mid_price(order_depth))
        if len(self.price_history[product]) > self.max_history_length:
            self.price_history[product].pop(0)

        #indicators
        # Track current position
        current_position = 0
        if product in state.position:
            current_position = state.position[product]

        #check if its been 200 days
        if len(self.price_history[product]) >= self.long_ma_window:
            short_MA = self.calculate_ma(self.price_history[product][-self.short_ma_window:], self.short_ma_window)
            long_MA = self.calculate_ma(self.price_history[product][-self.long_ma_window:], self.long_ma_window)
            
            # Get best bid and ask prices
            if order_depth.buy_orders:
                best_bid, best_bid_amount = max(order_depth.buy_orders.items(), key=lambda x: x[0])
            else:
                best_bid, best_bid_amount = 0, 0
                
            if order_depth.sell_orders:
                best_ask, best_ask_amount = min(order_depth.sell_orders.items(), key=lambda x: x[0])
            else:
                best_ask, best_ask_amount = float('inf'), 0
            
            # Golden Cross (Entry Signal) - Short MA crosses above Long MA
            if short_MA > long_MA and current_position < self.position_limit:
                # Buy signal - go long
                if self.position == -50:
                    orders.append(Order(product, best_ask, 100))
                    self.position += 100
                elif self.position == 0:
                    orders.append(Order(product, best_ask, 50))
                    self.position += 50
                else:
                    pass
            
            # Death Cross (Exit Signal) - Short MA crosses below Long MA
            elif short_MA < long_MA:
                # Sell signal - exit long position
                if self.position == 50:
                    orders.append(Order(product, best_bid, -100))
                elif self.position == 0:
                    orders.append(Order(product, best_ask, -50))
                else:
                    pass
        
        # Add all the orders to the result
        result[product] = orders
        
        return result, 0, ""