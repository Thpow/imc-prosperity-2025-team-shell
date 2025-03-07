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
        if len(self.price_history) > 200:
            self.price_history.pop(0)

        #indicators

        #check if its been 200 days
        if state.timestamp >=200:
            short_MA = self.calculate_ma(self.price_history[product][-self.short_ma_window:], self.short_ma_window)
            long_MA = self.calculate_ma(self.price_history[product], self.long_ma_window)

        # get best price and how many orders
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        if state.timestamp >=200:
            if short_MA > long_MA:
                                            #price #quant
                result[product] = [Order(product, best_bid, 50)]
    
        
    
        return result, 0, ""
    