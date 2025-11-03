import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class PricePredictor:
    """Machine Learning model for price prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        df = df.copy()
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Technical indicators as features
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target variable: future price change (next 5 periods)
        df['target'] = df['close'].shift(-5) / df['close'] - 1
        
        # Drop NaN values
        df = df.dropna()
        
        self.feature_columns = [col for col in df.columns if col not in [
            'open_time', 'close_time', 'target', 'symbol', 'interval'
        ]]
        
        return df
    
    def train_models(self, df: pd.DataFrame, symbol: str, interval: str):
        """Train multiple ML models for prediction."""
        try:
            # Prepare features
            df_processed = self.prepare_features(df)
            
            if len(df_processed) < 100:
                logger.warning(f"Not enough data for training {symbol} {interval}")
                return
            
            X = df_processed[self.feature_columns]
            y = df_processed['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[symbol] = scaler
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
            }
            
            best_score = -np.inf
            best_model_name = None
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                
                logger.info(f"Model {name} for {symbol} {interval}: R² = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    self.models[symbol] = model
            
            self.is_trained = True
            logger.info(f"🎯 Best model for {symbol} {interval}: {best_model_name} (R² = {best_score:.4f})")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol} {interval}: {e}")
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Make prediction using trained model."""
        if symbol not in self.models:
            return {'error': f'No model trained for {symbol}'}
        
        try:
            # Prepare features for prediction
            df_processed = self.prepare_features(df)
            
            if len(df_processed) == 0:
                return {'error': 'No data for prediction'}
            
            # Use latest data point
            latest_features = df_processed[self.feature_columns].iloc[-1:].values
            
            # Scale features
            scaled_features = self.scalers[symbol].transform(latest_features)
            
            # Make prediction
            prediction = self.models[symbol].predict(scaled_features)[0]
            
            # Calculate confidence based on recent prediction accuracy
            confidence = self._calculate_confidence(df_processed, symbol)
            
            return {
                'predicted_change': prediction,
                'confidence': confidence,
                'direction': 'up' if prediction > 0 else 'down',
                'magnitude': abs(prediction)
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence(self, df: pd.DataFrame, symbol: str) -> float:
        """Calculate prediction confidence based on recent performance."""
        # Simplified confidence calculation
        if len(df) < 50:
            return 0.5
        
        recent_data = df.tail(50)
        price_volatility = recent_data['close'].pct_change().std()
        
        # Higher volatility -> lower confidence
        confidence = max(0.1, 1.0 - price_volatility * 10)
        return min(confidence, 0.95)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, sma, lower_band
    
    def save_models(self, directory: str = "ml_models"):
        """Save trained models to disk."""
        if not self.is_trained:
            logger.warning("No models to save")
            return
        
        os.makedirs(directory, exist_ok=True)
        
        for symbol, model in self.models.items():
            model_path = os.path.join(directory, f"{symbol}_model.joblib")
            scaler_path = os.path.join(directory, f"{symbol}_scaler.joblib")
            
            joblib.dump(model, model_path)
            joblib.dump(self.scalers[symbol], scaler_path)
        
        logger.info(f"💾 Models saved to {directory}")
    
    def load_models(self, directory: str = "ml_models"):
        """Load trained models from disk."""
        if not os.path.exists(directory):
            logger.warning(f"Model directory {directory} not found")
            return
        
        model_files = [f for f in os.listdir(directory) if f.endswith('_model.joblib')]
        
        for model_file in model_files:
            symbol = model_file.replace('_model.joblib', '')
            
            model_path = os.path.join(directory, model_file)
            scaler_path = os.path.join(directory, f"{symbol}_scaler.joblib")
            
            if os.path.exists(scaler_path):
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                logger.info(f"📂 Loaded model for {symbol}")
        
        self.is_trained = len(self.models) > 0

# ML-based trading strategy
class MLStrategy:
    """Trading strategy using ML predictions."""
    
    def __init__(self, predictor: PricePredictor):
        self.predictor = predictor
        self.name = "MLStrategy"
        self.parameters = {
            'confidence_threshold': 0.6,
            'min_prediction_strength': 0.02,
            'max_position_size': 0.1
        }
    
    def calculate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate trading signals based on ML predictions."""
        result_df = df.copy()
        
        # Get ML prediction
        prediction = self.predictor.predict(df, symbol)
        
        # Initialize signal column
        result_df['ml_signal'] = 0
        result_df['ml_confidence'] = 0.0
        result_df['ml_prediction'] = 0.0
        
        if 'error' not in prediction:
            # Add prediction info to the latest row
            result_df.loc[result_df.index[-1], 'ml_confidence'] = prediction['confidence']
            result_df.loc[result_df.index[-1], 'ml_prediction'] = prediction['predicted_change']
            
            # Generate signal based on prediction
            if (prediction['confidence'] > self.parameters['confidence_threshold'] and
                abs(prediction['predicted_change']) > self.parameters['min_prediction_strength']):
                
                if prediction['direction'] == 'up':
                    result_df.loc[result_df.index[-1], 'ml_signal'] = 1
                else:
                    result_df.loc[result_df.index[-1], 'ml_signal'] = -1
        
        return result_df
    
    def should_buy(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check if we should buy based on ML prediction."""
        if len(df) < 50:
            return False
        
        prediction = self.predictor.predict(df, symbol)
        
        return ('error' not in prediction and
                prediction['direction'] == 'up' and
                prediction['confidence'] > self.parameters['confidence_threshold'] and
                prediction['predicted_change'] > self.parameters['min_prediction_strength'])
    
    def should_sell(self, df: pd.DataFrame, symbol: str) -> bool:
        """Check if we should sell based on ML prediction."""
        if len(df) < 50:
            return False
        
        prediction = self.predictor.predict(df, symbol)
        
        return ('error' not in prediction and
                prediction['direction'] == 'down' and
                prediction['confidence'] > self.parameters['confidence_threshold'] and
                abs(prediction['predicted_change']) > self.parameters['min_prediction_strength'])

# Test function
def test_ml_predictor():
    """Test the ML predictor with sample data."""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    
    # Create realistic price data with some trend
    prices = [100]
    for i in range(1, 500):
        # Add some trend and noise
        trend = np.sin(i * 0.05) * 2  # Cyclical trend
        noise = np.random.normal(0, 1)
        new_price = prices[-1] + trend + noise
        prices.append(max(new_price, 1))
    
    sample_data = {
        'open_time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }
    df = pd.DataFrame(sample_data)
    
    # Test ML predictor
    predictor = PricePredictor()
    predictor.train_models(df, "TEST", "1h")
    
    # Make prediction
    prediction = predictor.predict(df, "TEST")
    print("ML Prediction Results:")
    print(f"Predicted Change: {prediction.get('predicted_change', 0):.4f}")
    print(f"Direction: {prediction.get('direction', 'N/A')}")
    print(f"Confidence: {prediction.get('confidence', 0):.2f}")
    
    # Test ML strategy
    strategy = MLStrategy(predictor)
    signals = strategy.calculate_signals(df, "TEST")
    
    print(f"\nML Signals generated: {len(signals[signals['ml_signal'] != 0])}")
    
    return predictor, strategy

if __name__ == "__main__":
    test_ml_predictor()