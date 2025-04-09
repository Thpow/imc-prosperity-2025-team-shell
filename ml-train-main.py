import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import glob
import joblib

def prepare_features(prices, volumes):
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

def train_models():
    """Train ML models for each product using historical data."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Find all price files
    price_files = glob.glob("day1/prices_*.csv")
    
    # Load and preprocess price data
    all_price_data = []
    for file in price_files:
        df = pd.read_csv(file, sep=';')
        all_price_data.append(df)
    
    if not all_price_data:
        print("No historical data found in day1 directory!")
        return
    
    price_data = pd.concat(all_price_data, ignore_index=True)
    products = ["KELP", "RAINFOREST_RESIN", "SQUID_INK"]
    
    for product in products:
        print(f"Training model for {product}...")
        product_data = price_data[price_data['product'] == product]
        
        if product_data.empty:
            print(f"No data found for {product}")
            continue
        
        # Sort by day and timestamp
        product_data = product_data.sort_values(['day', 'timestamp'])
        
        # Get mid prices
        prices = product_data['mid_price'].tolist()
        
        # Calculate average volume
        product_data['avg_volume'] = (
            product_data['bid_volume_1'].fillna(0) + 
            product_data['bid_volume_2'].fillna(0) + 
            product_data['bid_volume_3'].fillna(0) + 
            product_data['ask_volume_1'].fillna(0) + 
            product_data['ask_volume_2'].fillna(0) + 
            product_data['ask_volume_3'].fillna(0)
        ) / 6
        
        volumes = product_data['avg_volume'].tolist()
        
        # Prepare features and labels
        data = prepare_features(prices, volumes)
        if data is None:
            print(f"Not enough data for {product}")
            continue
        
        features, labels = data
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(scaled_features, labels)
        
        # Save model and scaler
        joblib.dump(model, f"models/{product}_model.joblib")
        joblib.dump(scaler, f"models/{product}_scaler.joblib")
        print(f"Saved model and scaler for {product}")

if __name__ == "__main__":
    train_models()
    print("Training complete! Models saved in models/ directory")
