from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
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
                if len(order_depth.sell_orders) > 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    print(f"SELL {best_ask_amount}x at {best_ask}")
                    orders.append(Order(product, best_ask, -best_ask_amount))
            else:
                # If the sell side has more orders, we try to buy (because the majority is selling)
                if len(order_depth.buy_orders) > 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    print(f"BUY {best_bid_amount}x at {best_bid}")
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
        
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
