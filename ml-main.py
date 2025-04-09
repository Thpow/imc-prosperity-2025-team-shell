from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np
import joblib
import os

class Trader:
    def __init__(self):
        # Position limits per product
        self.POSITION_LIMITS = {"KELP": 45, "RAINFOREST_RESIN": 50, "SQUID_INK": 50}
        
        # Historical price and volume data (short-term, just for indicators)
        self.price_history = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        self.volume_history = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        
        # Tracking profit and volatility
        self.profits = {"KELP": 0, "RAINFOREST_RESIN": 0, "SQUID_INK": 0}
        self.volatility = {"KELP": 0, "RAINFOREST_RESIN": 0, "SQUID_INK": 0}
        
        # Market state tracking
        self.market_state = {"KELP": "normal", "RAINFOREST_RESIN": "normal", "SQUID_INK": "normal"}
        
        # Risk management parameters
        self.max_drawdown_allowed = 0.15  # 15% max drawdown
        self.drawdown = {"KELP": 0, "RAINFOREST_RESIN": 0, "SQUID_INK": 0}
        
        # Position sizing parameters
        self.base_risk = {"KELP": 0.20, "RAINFOREST_RESIN": 0.30, "SQUID_INK": 0.20}
        
        # Default parameters from memories
        self.params = {
            "KELP": {
                "rsi_period": 24,
                "rsi_overbought": 72,
                "rsi_oversold": 28,
                "rsi_neutral_high": 64,
                "rsi_neutral_low": 36,
                "std_multiplier": 0.89,
                "short_std_multiplier": 0.89,
                "short_rsi_threshold": 70,
                "acceptable_price": 2000
            },
            "RAINFOREST_RESIN": {
                "acceptable_price": 10000,
                "rsi_period": 24
            },
            "SQUID_INK": {
                "acceptable_price": 2000,
                "rsi_period": 24
            }
        }
        
        # Try to load pre-trained models
        self.models = {}
        self.scalers = {}
        self.load_models()

    def load_models(self):
        """Try to load pre-trained models if they exist."""
        for product in self.POSITION_LIMITS.keys():
            model_path = f"models/{product}_model.joblib"
            scaler_path = f"models/{product}_scaler.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[product] = joblib.load(model_path)
                    self.scalers[product] = joblib.load(scaler_path)
                except:
                    print(f"Could not load model for {product}")

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid-price from best bid and ask."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return 0

    def calculate_rsi(self, product: str) -> float:
        """Calculate RSI with period from params."""
        prices = self.price_history[product]
        period = self.params[product]["rsi_period"]
        
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

    def predict_price_movement(self, product: str) -> float:
        """Use ML model to predict future price movement."""
        if product not in self.models or len(self.price_history[product]) < 30:
            return 0
        
        prices = self.price_history[product]
        volumes = self.volume_history[product]
        
        # Create features for current state
        window_sizes = [5, 10, 20]
        feature_row = []
        
        i = len(prices) - 1  # Latest data point
        
        # Price momentum features
        for window in window_sizes:
            if i >= window:
                # Price change over window
                price_change = (prices[i] - prices[i-window]) / prices[i-window]
                feature_row.append(price_change)
                
                # Moving average
                ma = np.mean(prices[i-window:i])
                feature_row.append(prices[i] / ma - 1)  # Distance from MA
                
                # Volatility
                volatility = np.std(prices[i-window:i])
                feature_row.append(volatility / prices[i])
            else:
                # Insufficient history, use zeros
                feature_row.extend([0, 0, 0])
        
        # Volume features
        if volumes and len(volumes) > 0:
            recent_volumes = volumes[-min(max(window_sizes), len(volumes)):]
            avg_volume = np.mean(recent_volumes) if recent_volumes else 0
            feature_row.append(avg_volume)
            
            # Volume momentum
            for window in window_sizes:
                if i >= window and len(volumes) > i:
                    vol_momentum = np.mean(volumes[i-window:i]) / avg_volume if avg_volume > 0 else 1
                    feature_row.append(vol_momentum)
                else:
                    feature_row.append(1)
        else:
            feature_row.extend([0, 1, 1, 1])  # Default volume features
        
        # RSI feature
        rsi = self.calculate_rsi(product)
        feature_row.append(rsi)
        
        try:
            # Scale features
            scaled_features = self.scalers[product].transform([feature_row])
            
            # Make prediction
            predicted_return = self.models[product].predict(scaled_features)[0]
            
            return predicted_return
        except:
            return 0

    def calculate_position_size(self, product: str, signal_strength: float, is_buy: bool) -> int:
        """Calculate position size based on signal strength and risk parameters."""
        position_limit = self.POSITION_LIMITS[product]
        
        # Base risk as percentage of position limit
        max_risk_per_trade = self.base_risk[product] * position_limit
        
        # Adjust for volatility (reduce size during high volatility)
        volatility_factor = 1.0
        if self.volatility[product] > 0.01:  # If volatility is high
            volatility_factor = 0.01 / self.volatility[product]
            volatility_factor = max(0.5, min(1.0, volatility_factor))
        
        # Adjust for drawdown (reduce position size during drawdowns)
        drawdown_factor = 1.0
        if self.drawdown[product] > 0:
            drawdown_factor = 1.0 - (self.drawdown[product] / self.max_drawdown_allowed)
            drawdown_factor = max(0.25, min(1.0, drawdown_factor))
        
        # Calculate position size
        position_size = int(position_limit * max_risk_per_trade * signal_strength * volatility_factor * drawdown_factor)
        
        # Minimum position size of 1 if there's any signal
        if signal_strength > 0.1 and position_size == 0:
            position_size = 1
        
        return max(1, position_size)

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic."""
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            
            # Track mid-price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price > 0:
                self.price_history[product].append(mid_price)
                if len(self.price_history[product]) > 100:  # Keep history manageable
                    self.price_history[product].pop(0)
            
            # Calculate average volume
            if order_depth.buy_orders and order_depth.sell_orders:
                avg_vol = (
                    sum(abs(v) for v in order_depth.buy_orders.values()) + 
                    sum(abs(v) for v in order_depth.sell_orders.values())
                ) / (len(order_depth.buy_orders) + len(order_depth.sell_orders))
                
                self.volume_history[product].append(avg_vol)
                if len(self.volume_history[product]) > 100:
                    self.volume_history[product].pop(0)
            
            # Skip if no price data
            if not mid_price:
                result[product] = []
                continue
            
            # Default to threshold-based trading if no ML model available
            predicted_return = self.predict_price_movement(product)
            
            # Trading logic based on predictions and thresholds
            if order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                
                # Buy signal based on price being below threshold or positive prediction
                if best_ask < self.params[product]["acceptable_price"] or predicted_return > 0.001:
                    signal_strength = 0.7 if best_ask < self.params[product]["acceptable_price"] else abs(predicted_return)
                    buy_size = self.calculate_position_size(product, signal_strength, True)
                    buy_volume = min(buy_size, -order_depth.sell_orders[best_ask], 
                                   self.POSITION_LIMITS[product] - position)
                    
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
            
            if order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                
                # Sell signal based on price being above threshold or negative prediction
                if best_bid > self.params[product]["acceptable_price"] + 1 or predicted_return < -0.001:
                    signal_strength = 0.7 if best_bid > self.params[product]["acceptable_price"] + 1 else abs(predicted_return)
                    sell_size = self.calculate_position_size(product, signal_strength, False)
                    sell_volume = min(sell_size, order_depth.buy_orders[best_bid],
                                    self.POSITION_LIMITS[product] + position)
                    
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
            
            result[product] = orders
            
            # Update P&L if available for drawdown calculation
            if hasattr(state, 'profit_and_loss') and product in state.profit_and_loss:
                current_pnl = state.profit_and_loss.get(product, 0)
                if self.profits[product] > 0 and current_pnl < self.profits[product]:
                    self.drawdown[product] = (self.profits[product] - current_pnl) / self.profits[product]
                self.profits[product] = current_pnl
        
        return result, 0, ""  # No conversions in this strategy
