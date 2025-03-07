from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math
import numpy as np
from scipy.stats import norm

from funcs import *

product = "RAINFOREST_RESIN"

class Trader:
    def __init__(self):
        # Initialize position tracking
        self.position = 0
        self.positions = {}
     
        self.position = 0
        self.positions = {}
        self.position_limit = 50

        #=======history tacking=======#
        self.price_history = {}
        self.volatility_history = {}
        self.position_limit = 50
        self.price_history = {}
        self.volatility_history = {}
        self.low = {}
        self.high = {}

        #=======params=======#
        self.max_history_length = 200
        self.max_volatility_length = 200   
        short_ma_window = 50
        long_ma_window = 200
        
    def calculate_ma(self, prices, window):
        pass

    def run(self, state: TradingState):
        #init
        result = {}
        orders: List[Order] = []
        
        #init dephth with product
        order_depth: OrderDepth = state.order_depths[product]

        #update historic data
        self.price_history[product].append(calculate_mid_price(order_depth))

        #indicators

        #check if its been 200 days
        short_MA = self.calculate_ma(self.price_history[-1*self.short_ma_window:], self.short_ma_window)
        long_MA = self.calculate_ma(self.price_history, self.long_ma_window)

        # get best price and how many orders
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        
                                        #price #quant
        result[product] = [Order(product, best_bid, -50)]
        
    
        return result, 0, ""
    