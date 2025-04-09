from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import glob
import joblib

class Trader:
    def __init__(self):
        # Position limits per product
        self.POSITION_LIMITS = {"KELP": 45, "RAINFOREST_RESIN": 50, "SQUID_INK": 50}
        
        # Track the last day/timestamp to know when to retrain
        self.last_day = None
        self.last_timestamp = None
        
        # Historical price and volume data
        self.price_history = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        self.volume_history = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        self.mid_prices = {"KELP": [], "RAINFOREST_RESIN": [], "SQUID_INK": []}
        
        # Tracking profit and volatility
        self.profits = {"KELP": 0, "RAINFOREST_RESIN": 0, "SQUID_INK": 0}
        self.volatility = {"KELP": 0, "RAINFOREST_RESIN": 0, "SQUID_INK": 0}
        
        # Market state tracking
        self.market_state = {"KELP": "normal", "RAINFOREST_RESIN": "normal", "SQUID_INK": "normal"}
        
        # Machine learning models
        self.models = {}
        self.scalers = {}
        
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
        
        # Try to load pre-trained models if they exist
        self.load_models()
        
        # Load and preprocess historical data on initialization
        self.load_historical_data()
        
        # Train initial models if no pre-trained models exist
        if not self.models:
            self.train_models()

    def load_models(self):
        """Try to load pre-trained models if they exist."""
        for product in self.POSITION_LIMITS.keys():
            model_path = f"models/{product}_model.joblib"
            scaler_path = f"models/{product}_scaler.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[product] = joblib.load(model_path)
                self.scalers[product] = joblib.load(scaler_path)

    def save_models(self):
        """Save trained models for future use."""
        os.makedirs("models", exist_ok=True)
        for product in self.models:
            joblib.dump(self.models[product], f"models/{product}_model.joblib")
            joblib.dump(self.scalers[product], f"models/{product}_scaler.joblib")

    def load_historical_data(self):
        """Load historical price and trade data from day1 directory."""
        # Find all price files
        price_files = glob.glob("day1/prices_*.csv")
        trade_files = glob.glob("day1/trades_*.csv")
        
        # Load and preprocess price data
        all_price_data = []
        for file in price_files:
            df = pd.read_csv(file, sep=';')
            all_price_data.append(df)
        
        if all_price_data:
            price_data = pd.concat(all_price_data, ignore_index=True)
            
            # Process price data into features for each product
            for product in self.POSITION_LIMITS.keys():
                product_data = price_data[price_data['product'] == product]
                if not product_data.empty:
                    # Sort by day and timestamp
                    product_data = product_data.sort_values(['day', 'timestamp'])
                    
                    # Store mid prices
                    self.mid_prices[product] = product_data['mid_price'].tolist()
                    
                    # Calculate average volume
                    product_data['avg_volume'] = (
                        product_data['bid_volume_1'].fillna(0) + 
                        product_data['bid_volume_2'].fillna(0) + 
                        product_data['bid_volume_3'].fillna(0) + 
                        product_data['ask_volume_1'].fillna(0) + 
                        product_data['ask_volume_2'].fillna(0) + 
                        product_data['ask_volume_3'].fillna(0)
                    ) / 6
                    
                    self.volume_history[product] = product_data['avg_volume'].tolist()
                    
                    # Calculate volatility
                    if len(self.mid_prices[product]) > 1:
                        returns = np.diff(self.mid_prices[product]) / self.mid_prices[product][:-1]
                        self.volatility[product] = np.std(returns) if returns.size > 0 else 0

    def prepare_features(self, product, prices, volumes):
        """Prepare features for the ML model."""
        if len(prices) < 30:  # Need sufficient history
            return None
        
        # Basic features
        features = []
        labels = []
        
        window_sizes = [5, 10, 20]
        prediction_horizon = 5  # Steps ahead to predict price
        
        for i in range(max(window_sizes), len(prices) - prediction_horizon):
            feature_row = []
            
            # Price momentum features
            for window in window_sizes:
                # Price change over window
                price_change = (prices[i] - prices[i-window]) / prices[i-window]
                feature_row.append(price_change)
                
                # Moving average
                ma = np.mean(prices[i-window:i])
                feature_row.append(prices[i] / ma - 1)  # Distance from MA
                
                # Volatility
                volatility = np.std(prices[i-window:i])
                feature_row.append(volatility / prices[i])
            
            # Volume features
            if volumes:
                recent_volumes = volumes[i-max(window_sizes):i]
                avg_volume = np.mean(recent_volumes) if recent_volumes else 0
                feature_row.append(avg_volume)
                
                # Volume momentum
                for window in window_sizes:
                    if i >= window and len(volumes) > i:
                        vol_momentum = np.mean(volumes[i-window:i]) / avg_volume if avg_volume > 0 else 1
                        feature_row.append(vol_momentum)
            
            # RSI feature
            if i >= 24:  # RSI period from memories
                price_changes = np.diff(prices[i-24:i])
                gains = np.sum(np.where(price_changes > 0, price_changes, 0))
                losses = -np.sum(np.where(price_changes < 0, price_changes, 0))
                
                if losses == 0:
                    rsi = 100
                else:
                    rs = gains / losses if losses > 0 else 1
                    rsi = 100 - (100 / (1 + rs))
                
                feature_row.append(rsi)
            else:
                feature_row.append(50)  # Neutral RSI when not enough data
            
            # Target: future price movement
            future_return = (prices[i+prediction_horizon] - prices[i]) / prices[i]
            
            features.append(feature_row)
            labels.append(future_return)
        
        return np.array(features), np.array(labels)

    def train_models(self):
        """Train ML models for each product."""
        for product in self.POSITION_LIMITS.keys():
            prices = self.mid_prices[product]
            volumes = self.volume_history[product]
            
            if len(prices) < 50:  # Need sufficient data
                continue
                
            # Prepare features and labels
            data = self.prepare_features(product, prices, volumes)
            if data is None:
                continue
                
            features, labels = data
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Train a Random Forest model
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(scaled_features, labels)
            
            # Save model and scaler
            self.models[product] = model
            self.scalers[product] = scaler
        
        # Save trained models
        self.save_models()

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid-price from best bid and ask."""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return 0

    def calculate_rsi(self, product: str) -> float:
        """Calculate RSI with period from params."""
        prices = self.mid_prices[product]
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
        if product not in self.models or len(self.mid_prices[product]) < 30:
            return 0
        
        prices = self.mid_prices[product]
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
        
        # Scale features
        scaled_features = self.scalers[product].transform([feature_row])
        
        # Make prediction
        predicted_return = self.models[product].predict(scaled_features)[0]
        
        return predicted_return

    def calculate_signal_strength(self, product: str, predicted_return: float, current_price: float) -> float:
        """Calculate trade signal strength (0 to 1) based on predictions and technical indicators."""
        # Get RSI
        rsi = self.calculate_rsi(product)
        
        if product == "KELP":
            # Combine RSI signal with price prediction
            rsi_signal = 0
            
            # RSI component - distance from thresholds
            if rsi <= self.params[product]["rsi_oversold"]:
                # Buy signal gets stronger as RSI drops
                rsi_dist = (self.params[product]["rsi_oversold"] - rsi) / self.params[product]["rsi_oversold"]
                rsi_signal = rsi_dist
            elif rsi >= self.params[product]["rsi_overbought"]:
                # Sell signal gets stronger as RSI rises
                rsi_dist = (rsi - self.params[product]["rsi_overbought"]) / (100 - self.params[product]["rsi_overbought"])
                rsi_signal = -rsi_dist
            
            # Price prediction component (convert to -1 to 1 scale)
            price_signal = np.clip(predicted_return * 10, -1, 1)  # Scale and clip prediction
            
            # Combined signal (-1 to 1)
            combined_signal = 0.6 * price_signal + 0.4 * rsi_signal
            
            # Convert to 0-1 scale for position sizing
            signal_strength = (combined_signal + 1) / 2
            
        elif product == "RAINFOREST_RESIN":
            # For RAINFOREST_RESIN, use price discount/premium relative to threshold
            acceptable_price = self.params[product]["acceptable_price"]
            
            if current_price < acceptable_price:
                # Price below threshold - buy signal
                discount = (acceptable_price - current_price) / acceptable_price
                signal_strength = min(0.8, discount * 10)  # Cap at 0.8
            elif current_price > acceptable_price:
                # Price above threshold - sell signal
                premium = (current_price - acceptable_price) / acceptable_price
                signal_strength = 0  # No buy signal
            else:
                signal_strength = 0.5  # Neutral
                
            # Adjust by prediction
            signal_strength += predicted_return * 0.2
            signal_strength = np.clip(signal_strength, 0, 1)
            
        elif product == "SQUID_INK":
            # Similar approach as KELP
            acceptable_price = self.params[product]["acceptable_price"]
            
            if current_price < acceptable_price:
                # Price below threshold - buy signal
                discount = (acceptable_price - current_price) / acceptable_price
                signal_strength = min(0.8, discount * 10)  # Cap at 0.8
            else:
                signal_strength = 0.2  # Weak buy signal
                
            # Adjust by prediction
            signal_strength = signal_strength * 0.7 + (predicted_return + 1) / 2 * 0.3
            signal_strength = np.clip(signal_strength, 0, 1)
        
        else:
            signal_strength = 0.5  # Default neutral
            
        return signal_strength

    def calculate_position_size(self, product: str, signal_strength: float, is_buy: bool) -> int:
        """Calculate position size based on signal strength, volatility, and drawdown."""
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
        
        # Adjust for signal strength (0-1)
        position_size = int(position_limit * max_risk_per_trade * signal_strength * volatility_factor * drawdown_factor)
        
        # Minimum position size of 1 if there's any signal
        if signal_strength > 0.1 and position_size == 0:
            position_size = 1
            
        # Adjust for short positions if needed
        if not is_buy and product == "KELP":
            rsi = self.calculate_rsi(product)
            # More aggressive shorts when RSI is high
            if rsi > self.params[product]["short_rsi_threshold"]:
                position_size = int(position_size * 1.2)  # 20% larger short positions when overbought
        
        return max(1, position_size)  # Ensure at least 1 unit

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic."""
        result = {}
        
        # Check if we need to retrain (day changed)
        current_day = state.timestamp // 1000000  # Assuming timestamp format
        if self.last_day is not None and current_day != self.last_day:
            # Day changed, retrain models
            self.train_models()
        
        self.last_day = current_day
        self.last_timestamp = state.timestamp
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            position_limit = self.POSITION_LIMITS.get(product, 20)
            
            # Track mid-price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price > 0:
                self.mid_prices[product].append(mid_price)
                if len(self.mid_prices[product]) > 200:  # Keep history manageable
                    self.mid_prices[product].pop(0)
            
            # Calculate average volume
            if order_depth.buy_orders and order_depth.sell_orders:
                avg_vol = (
                    sum(abs(v) for v in order_depth.buy_orders.values()) + 
                    sum(abs(v) for v in order_depth.sell_orders.values())
                ) / (len(order_depth.buy_orders) + len(order_depth.sell_orders))
                
                self.volume_history[product].append(avg_vol)
                if len(self.volume_history[product]) > 200:
                    self.volume_history[product].pop(0)
            
            # Skip if no price data
            if not mid_price or not self.mid_prices[product]:
                result[product] = []
                continue
            
            # Get ML prediction
            predicted_return = self.predict_price_movement(product)
            
            # Trading logic
            # Buy logic - different for each product
            if product == "KELP" or product == "SQUID_INK":
                # For KELP and SQUID_INK, we combine ML with RSI and threshold
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    
                    # Calculate signal strength for buying
                    buy_signal = self.calculate_signal_strength(product, predicted_return, best_ask)
                    
                    # Strong buy signal or below threshold price
                    if buy_signal > 0.6 or best_ask < self.params[product]["acceptable_price"]:
                        # Calculate position size
                        order_size = self.calculate_position_size(product, buy_signal, True)
                        buy_quantity = min(order_size, -order_depth.sell_orders[best_ask], position_limit - position)
                        
                        if buy_quantity > 0:
                            orders.append(Order(product, best_ask, buy_quantity))
            
                # Sell logic
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    
                    # Calculate signal strength for selling
                    sell_signal = 1 - self.calculate_signal_strength(product, predicted_return, best_bid)
                    
                    # Strong sell signal or above threshold price
                    if sell_signal > 0.6 or best_bid > self.params[product]["acceptable_price"] + 1:
                        # Calculate position size
                        order_size = self.calculate_position_size(product, sell_signal, False)
                        sell_quantity = min(order_size, order_depth.buy_orders[best_bid], position_limit + position)
                        
                        if sell_quantity > 0:
                            orders.append(Order(product, best_bid, -sell_quantity))
            
            elif product == "RAINFOREST_RESIN":
                # Similar to KELP but with different parameters
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    
                    # Calculate signal strength for buying
                    buy_signal = self.calculate_signal_strength(product, predicted_return, best_ask)
                    
                    # Strong buy signal or below threshold price
                    if buy_signal > 0.5 or best_ask < self.params[product]["acceptable_price"]:
                        # Calculate position size
                        order_size = self.calculate_position_size(product, buy_signal, True)
                        buy_quantity = min(order_size, -order_depth.sell_orders[best_ask], position_limit - position)
                        
                        if buy_quantity > 0:
                            orders.append(Order(product, best_ask, buy_quantity))
            
                # Sell logic
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    
                    # Calculate signal strength for selling
                    sell_signal = 1 - self.calculate_signal_strength(product, predicted_return, best_bid)
                    
                    # Strong sell signal or above threshold price
                    if sell_signal > 0.6 or best_bid > self.params[product]["acceptable_price"] + 1:
                        # Calculate position size
                        order_size = self.calculate_position_size(product, sell_signal, False)
                        sell_quantity = min(order_size, order_depth.buy_orders[best_bid], position_limit + position)
                        
                        if sell_quantity > 0:
                            orders.append(Order(product, best_bid, -sell_quantity))
            
            result[product] = orders
            
            # Update P&L if available for drawdown calculation
            if hasattr(state, 'profit_and_loss') and product in state.profit_and_loss:
                current_pnl = state.profit_and_loss.get(product, 0)
                if self.profits[product] > 0 and current_pnl < self.profits[product]:
                    # Calculate drawdown as percentage of previous profit
                    self.drawdown[product] = (self.profits[product] - current_pnl) / self.profits[product]
                self.profits[product] = current_pnl
        
        # No conversions in this strategy
        conversions = 0
        trader_data = ""
        
        return result, conversions, trader_data


if __name__ == "__main__":
    trader = Trader()
    
    # This loads and trains the model on historical data
    # In a real trading environment, you would use the run method instead
    
    print("ML model trained and ready for trading")
