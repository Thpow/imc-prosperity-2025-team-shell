from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    def __init__(self):
        self.current_position = 0

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent.
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        
        # Iterate through all products
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Calculate the majority order side (buy/sell)
            buy_order_count = len(order_depth.buy_orders)
            sell_order_count = len(order_depth.sell_orders)
            
            print(f"Buy Order depth: {buy_order_count}, Sell Order depth: {sell_order_count}")
            
            if buy_order_count > sell_order_count:
                # If the buy side has more orders, we try to sell (because the majority is buying)
                if self.current_position == 1:
                    pass
                elif self.current_position == 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    orders.append(Order(product, best_ask, 1))
                    self.current_position = self.current_position + 1
                elif self.current_position == -1:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    orders.append(Order(product, best_ask, 2))
                    self.current_position = self.current_position + 2
                else:
                    print('something went pretty wrong')
                    exit(1)
            else:
                if self.current_position == -1:
                    pass
                elif self.current_position == 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    orders.append(Order(product, best_bid, -1))
                    self.current_position = self.current_position - 1
                elif self.current_position == 1:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    orders.append(Order(product, best_bid, -2))
                    self.current_position = self.current_position - 2
                else:
                    print('something went pretty wrong')
                    exit(1)
            result[product] = orders
        
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
