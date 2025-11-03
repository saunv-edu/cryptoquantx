import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class IndicatorCalculator:
    """Calculate technical indicators from OHLCV data."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame."""
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original data
        result_df = df.copy()
        
        # Price-based indicators
        result_df = self._calculate_trend_indicators(result_df)
        result_df = self._calculate_momentum_indicators(result_df)
        result_df = self._calculate_volatility_indicators(result_df)
        result_df = self._calculate_volume_indicators(result_df)
        
        logger.info(f"Calculated {len(result_df.columns) - len(df.columns)} indicators")
        return result_df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators."""
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # MACD (Manual calculation)
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        # RSI (Manual calculation)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Stochastic (Manual calculation)
        df['stoch_k'] = self._calculate_stochastic(df, 14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Bollinger Bands (Manual calculation)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Manual calculation)
        df['atr'] = self._calculate_atr(df, 14)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # OBV (Manual calculation)
        df['obv'] = self._calculate_obv(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K manually."""
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        stoch_k = 100 * (df['close'] - lowest_low) / denominator.replace(0, 1)
        
        return stoch_k
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR manually."""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate OBV manually."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def calculate_for_symbol(self, symbol: str, interval: str, limit: int = 1000):
        """Calculate indicators for a specific symbol and interval."""
        try:
            # Get klines data
            df = self.db.get_klines_df(symbol, interval, limit)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {interval}")
                return
            
            # Calculate indicators
            df_with_indicators = self.calculate_all_indicators(df)
            
            # Save indicators to database
            self._save_indicators_to_db(symbol, interval, df_with_indicators)
            
            logger.info(f"Calculated indicators for {symbol} {interval} - {len(df_with_indicators)} rows")
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {interval}: {e}")
    
    def _save_indicators_to_db(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save calculated indicators to database."""
        indicators_data = []
        
        for _, row in df.iterrows():
            # Only save rows where we have valid indicator data
            if pd.notna(row.get('rsi')) and pd.notna(row.get('close')):
                indicator_data = {
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': row['open_time'],
                    
                    # Trend Indicators
                    'rsi': float(row.get('rsi', 0)),
                    'macd': float(row.get('macd', 0)),
                    'macd_signal': float(row.get('macd_signal', 0)),
                    'macd_histogram': float(row.get('macd_histogram', 0)),
                    
                    # Moving Averages
                    'ema_20': float(row.get('ema_20', 0)),
                    'ema_50': float(row.get('ema_50', 0)),
                    'sma_20': float(row.get('sma_20', 0)),
                    'sma_50': float(row.get('sma_50', 0)),
                    
                    # Bollinger Bands
                    'bb_upper': float(row.get('bb_upper', 0)),
                    'bb_middle': float(row.get('bb_middle', 0)),
                    'bb_lower': float(row.get('bb_lower', 0)),
                    'bb_width': float(row.get('bb_width', 0)),
                    'bb_position': float(row.get('bb_position', 0)),
                    
                    # Volatility
                    'atr': float(row.get('atr', 0)),
                    
                    # Momentum
                    'stoch_k': float(row.get('stoch_k', 0)),
                    'stoch_d': float(row.get('stoch_d', 0)),
                    
                    # Volume
                    'volume_sma_20': float(row.get('volume_sma_20', 0)),
                    'obv': float(row.get('obv', 0))
                }
                indicators_data.append(indicator_data)
        
        if indicators_data:
            self.db.save_indicators(indicators_data)
            logger.info(f"Saved {len(indicators_data)} indicators for {symbol} {interval}")

# Test function
def test_indicator_calculation():
    """Test indicator calculation with sample data."""
    calculator = IndicatorCalculator()
    
    # Create sample data
    np.random.seed(42)  # For reproducible results
    sample_data = {
        'open_time': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(150, 250, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }
    df = pd.DataFrame(sample_data)
    
    # Calculate indicators
    result_df = calculator.calculate_all_indicators(df)
    
    print("✅ Indicator calculation test successful!")
    print(f"Original columns: {len(sample_data)}")
    print(f"With indicators: {len(result_df.columns)}")
    print("\nSample indicators:")
    print(result_df[['close', 'rsi', 'macd', 'ema_20', 'bb_upper', 'stoch_k']].tail())
    
    # Check for NaN values
    print(f"\nNaN values in indicators:")
    for col in result_df.columns:
        nan_count = result_df[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    return result_df

if __name__ == "__main__":
    test_indicator_calculation()