from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import statistics
import math

class Trader:
    def __init__(self):
        # Initialize position tracking
        self.position = 0
        self.position_limit = 50  # Standard position limit for KELP
        
        # Price history for products
        self.price_history = {}
        self.max_history_length = 50  # Increased to accommodate longer indicator periods
        
        # Market making parameters
        self.spread_multiplier = 0.5
        self.min_spread = 1
        self.max_position_spread_adjustment = 0.5
        
        # Volume parameters
        self.base_volume = 15
        self.volume_scaling = 1.0
        
        # Volatility tracking
        self.volatility_window = 8
        self.volatility_history = {}
        
        # Trading state tracking
        self.last_mid_price = {}
        self.trader_data = ""
        
        # Directional trading parameters
        self.momentum_threshold = 0.001
        self.momentum_boost = 1.0  # Reduced from 2.0 for more conservative trading
        
        # Arbitrage parameters
        self.arbitrage_aggressiveness = 0.7  # Reduced from 0.9 for more conservative trading
        
        # Price improvement parameters
        self.price_improve_threshold = 0.5  # Reduced from 0.6 for more conservative trading
        
        # RSI parameters
        self.rsi_period = 14  # Standard RSI period
        self.rsi_oversold = 35  # Adjusted from 30 for better performance
        self.rsi_overbought = 65  # Adjusted from 70 for better performance
        
        # Bollinger Bands parameters
        self.bb_period = 20  # Standard Bollinger Bands period
        self.bb_std = 2.0  # Adjusted from 2.5 for better performance
    
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
        base_spread = max(self.min_spread, int(volatility * 50 * self.spread_multiplier))
        
        # Position-based spread adjustment
        position_factor = 1 + (abs(position) / self.position_limit) * self.max_position_spread_adjustment
        
        return max(self.min_spread, int(base_spread * position_factor))
    
    def calculate_order_volume(self, side, position):
        """Calculate order volume based on position and side"""
        # More conservative volume calculation
        remaining_capacity = self.position_limit - abs(position)
        position_factor = max(0.3, remaining_capacity / self.position_limit)
        
        # Asymmetric volume: larger orders when reducing position
        if (side == "buy" and position < 0) or (side == "sell" and position > 0):
            # Reducing position
            volume_adjustment = 1.5
        else:
            # Increasing position - more conservative
            volume_adjustment = 0.8
        
        volume = max(1, int(self.base_volume * position_factor * volume_adjustment))
        return volume
    
    def detect_momentum(self, prices, window=5):
        """Detect price momentum for directional bias"""
        if len(prices) < window:
            return 0
        
        # Calculate short-term trend
        recent_prices = prices[-window:]
        if recent_prices[0] == 0:  # Avoid division by zero
            return 0
            
        total_return = recent_prices[-1] / recent_prices[0] - 1
        return total_return
        
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        if len(prices) <= period:
            return 50  # Default to neutral when not enough data
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100  # Prevent division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            # Not enough data, return None
            return None, None, None
        
        # Calculate SMA
        sma = sum(prices[-period:]) / period
        
        # Calculate standard deviation
        std = statistics.stdev(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return lower_band, sma, upper_band

    def run(self, state: TradingState):
        """
        Trading strategy for KELP product with improved technical indicators
        """
        result = {}
        
        # Process each available product
        for product in state.order_depths.keys():
            orders: List[Order] = []
            
            # Get order depth for the product
            order_depth = state.order_depths[product]
            
            # Skip if no orders on either side
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue
            
            # Get current position
            current_position = state.position.get(product, 0)
            
            # Calculate mid price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price is None:
                result[product] = orders
                continue
            
            # Initialize price history for this product if it doesn't exist
            if product not in self.price_history:
                self.price_history[product] = []
                self.volatility_history[product] = []
                self.last_mid_price[product] = None
            
            # Update price history
            self.price_history[product].append(mid_price)
            if len(self.price_history[product]) > self.max_history_length:
                self.price_history[product].pop(0)
            
            # Calculate volatility
            volatility = self.calculate_volatility(self.price_history[product][-min(len(self.price_history[product]), self.volatility_window):])
            
            # Update volatility history
            if product not in self.volatility_history:
                self.volatility_history[product] = []
            self.volatility_history[product].append(volatility)
            if len(self.volatility_history[product]) > self.max_history_length:
                self.volatility_history[product].pop(0)
            
            # Detect momentum for directional bias
            momentum = self.detect_momentum(self.price_history[product])
            
            # Calculate RSI - only when we have enough data
            rsi = 50  # Default to neutral
            if len(self.price_history[product]) >= self.rsi_period:
                rsi = self.calculate_rsi(self.price_history[product], period=self.rsi_period)
            
            # Calculate Bollinger Bands - only when we have enough data
            bb_result = None
            if len(self.price_history[product]) >= self.bb_period:
                bb_result = self.calculate_bollinger_bands(self.price_history[product], period=self.bb_period, num_std=self.bb_std)
            
            # Get best bid and ask
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            
            # Calculate spread based on volatility and position
            spread = self.calculate_spread(volatility, current_position)
            
            # Market making logic
            # Calculate our bid and ask prices with momentum bias
            if momentum > self.momentum_threshold:
                # Upward momentum - be more aggressive on buys
                our_bid = best_bid
                our_ask = best_ask
            elif momentum < -self.momentum_threshold:
                # Downward momentum - be more aggressive on sells
                our_bid = best_bid
                our_ask = best_ask
            else:
                # No strong momentum - be neutral
                our_bid = best_bid
                our_ask = best_ask
            
            # Improve price (go inside the spread) when position is not extreme
            position_ratio = abs(current_position) / self.position_limit
            if position_ratio < self.price_improve_threshold:
                # Calculate mid-price
                mid_price = (best_bid + best_ask) / 2
                
                # Price improvement - go inside the spread
                if (momentum > 0 and current_position < 0) or (momentum < 0 and current_position > 0):
                    # More aggressive when momentum helps reduce position
                    price_improve_factor = 0.6
                else:
                    price_improve_factor = 0.3
                    
                # Apply price improvement
                bid_improve = int((mid_price - best_bid) * price_improve_factor)
                ask_improve = int((best_ask - mid_price) * price_improve_factor)
                
                if bid_improve > 0 and current_position < self.position_limit * 0.7:
                    our_bid = best_bid + bid_improve
                
                if ask_improve > 0 and current_position > -self.position_limit * 0.7:
                    our_ask = best_ask - ask_improve
            
            # Calculate base volumes - more conservative sizing
            base_buy_volume = self.calculate_order_volume("buy", current_position)
            base_sell_volume = self.calculate_order_volume("sell", current_position)
            
            # Apply momentum boost to volumes - more conservative
            if momentum > self.momentum_threshold:
                buy_volume = int(base_buy_volume * (1 + abs(momentum) * self.momentum_boost))
                sell_volume = base_sell_volume
            elif momentum < -self.momentum_threshold:
                buy_volume = base_buy_volume
                sell_volume = int(base_sell_volume * (1 + abs(momentum) * self.momentum_boost))
            else:
                buy_volume = base_buy_volume
                sell_volume = base_sell_volume
            
            # Apply RSI adjustments to volume - only when we have enough data
            if len(self.price_history[product]) >= self.rsi_period:
                if rsi < self.rsi_oversold and current_position < self.position_limit * 0.7:
                    # Oversold condition - increase buy volume
                    rsi_factor = (self.rsi_oversold - rsi) / self.rsi_oversold
                    buy_volume = int(buy_volume * (1 + rsi_factor * 0.8))  # More conservative adjustment
                elif rsi > self.rsi_overbought and current_position > -self.position_limit * 0.7:
                    # Overbought condition - increase sell volume
                    rsi_factor = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                    sell_volume = int(sell_volume * (1 + rsi_factor * 0.8))  # More conservative adjustment
            
            # Apply Bollinger Bands adjustments - only when we have enough data
            if bb_result:
                lower_band, sma, upper_band = bb_result
                
                # Buy when price is near lower band (oversold)
                if best_ask < lower_band * 1.01 and current_position < self.position_limit * 0.7:
                    # Calculate volume - conservative multiplier of 2 (was 10)
                    buy_pct = min(1.0, (lower_band - best_ask) / lower_band * 2)
                    bb_buy_vol = math.ceil(buy_pct * min(8, self.position_limit - current_position))
                    buy_volume = max(buy_volume, bb_buy_vol)
                
                # Sell when price is near upper band (overbought)
                if best_bid > upper_band * 0.99 and current_position > -self.position_limit * 0.7:
                    # Calculate volume - conservative multiplier of 2 (was 10)
                    sell_pct = min(1.0, (best_bid - upper_band) / upper_band * 2)
                    bb_sell_vol = math.ceil(sell_pct * min(8, self.position_limit + current_position))
                    sell_volume = max(sell_volume, bb_sell_vol)
            
            # More conservative volume reduction when approaching limits
            remaining_buy_capacity = self.position_limit - current_position
            remaining_sell_capacity = self.position_limit + current_position
            
            # More conservative position sizing
            buy_volume = min(buy_volume, int(remaining_buy_capacity * 0.8))
            sell_volume = min(sell_volume, int(remaining_sell_capacity * 0.8))
            
            # Place orders if volumes are positive
            if buy_volume > 0:
                orders.append(Order(product, our_bid, buy_volume))
            
            if sell_volume > 0:
                orders.append(Order(product, our_ask, -sell_volume))
            
            # Take advantage of arbitrage opportunities - more conservative
            arb_opportunities = []
            for bid_price, bid_volume in order_depth.buy_orders.items():
                for ask_price, ask_volume in order_depth.sell_orders.items():
                    if bid_price > ask_price:  # Arbitrage opportunity
                        profit_per_unit = bid_price - ask_price
                        max_trade = min(abs(bid_volume), abs(ask_volume))
                        arb_opportunities.append((profit_per_unit, ask_price, bid_price, max_trade))
            
            # Sort arbitrage opportunities by profit
            arb_opportunities.sort(reverse=True)
            
            # Execute arbitrage opportunities more conservatively
            for profit, ask_price, bid_price, max_trade in arb_opportunities:
                # Take a more conservative percentage of the arbitrage opportunity
                trade_size = int(max_trade * self.arbitrage_aggressiveness)
                
                # Limit trade size based on position limits
                trade_size = min(trade_size, int(min(remaining_buy_capacity, remaining_sell_capacity) * 0.8))
                
                if trade_size > 0:
                    # Buy at ask price and sell at bid price
                    orders.append(Order(product, ask_price, trade_size))
                    orders.append(Order(product, bid_price, -trade_size))
                    
                    # Update remaining capacity after this trade
                    remaining_buy_capacity -= trade_size
                    remaining_sell_capacity -= trade_size
            
            # Add position-building logic based on order book imbalance - more conservative
            if len(self.price_history[product]) > 5:
                # Calculate order book imbalance
                total_bids = sum(abs(vol) for vol in order_depth.buy_orders.values())
                total_asks = sum(abs(vol) for vol in order_depth.sell_orders.values())
                
                if total_bids > 0 and total_asks > 0:
                    imbalance = (total_bids - total_asks) / (total_bids + total_asks)
                    
                    # Take directional position on strong imbalance - more conservative
                    if imbalance > 0.4 and current_position < self.position_limit * 0.7:  # Strong buying pressure
                        aggressive_buy_size = int(min(remaining_buy_capacity * 0.4, total_asks * 0.2))
                        if aggressive_buy_size > 0:
                            orders.append(Order(product, best_ask, aggressive_buy_size))
                    
                    elif imbalance < -0.4 and current_position > -self.position_limit * 0.7:  # Strong selling pressure
                        aggressive_sell_size = int(min(remaining_sell_capacity * 0.4, total_bids * 0.2))
                        if aggressive_sell_size > 0:
                            orders.append(Order(product, best_bid, -aggressive_sell_size))
            
            # Save market state for next iteration
            self.last_mid_price[product] = mid_price
            
            # Store result
            result[product] = orders
        
        return result, self.trader_data, ""