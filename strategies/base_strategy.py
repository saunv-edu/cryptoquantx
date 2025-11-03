from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.signals = []
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on indicators."""
        pass
    
    @abstractmethod
    def should_buy(self, df: pd.DataFrame) -> bool:
        """Determine if we should enter a long position."""
        pass
    
    @abstractmethod
    def should_sell(self, df: pd.DataFrame) -> bool:
        """Determine if we should exit a long position."""
        pass
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)
        logger.info(f"Set parameters for {self.name}: {kwargs}")
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """Get signal strength from 0 to 1."""
        # Default implementation - can be overridden
        return 0.5

class MovingAverageCrossover(BaseStrategy):
    """Simple Moving Average Crossover strategy."""
    
    def __init__(self):
        super().__init__("MovingAverageCrossover")
        self.default_parameters = {
            'fast_ma': 20,
            'slow_ma': 50,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        self.set_parameters(**self.default_parameters)
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MA crossover signals."""
        result_df = df.copy()
        
        fast_ma = self.parameters['fast_ma']
        slow_ma = self.parameters['slow_ma']
        
        # Calculate moving averages if not present
        if f'ema_{fast_ma}' not in df.columns:
            result_df[f'ema_{fast_ma}'] = result_df['close'].ewm(span=fast_ma).mean()
        if f'ema_{slow_ma}' not in df.columns:
            result_df[f'ema_{slow_ma}'] = result_df['close'].ewm(span=slow_ma).mean()
        
        # Generate signals - FIXED: Use proper boolean logic instead of ~
        result_df['ma_signal'] = 0
        
        # Buy signal: fast MA crosses above slow MA
        result_df['ma_crossover'] = (result_df[f'ema_{fast_ma}'] > result_df[f'ema_{slow_ma}']) & \
                                  (result_df[f'ema_{fast_ma}'].shift(1) <= result_df[f'ema_{slow_ma}'].shift(1))
        
        # Sell signal: fast MA crosses below slow MA  
        result_df['ma_crossunder'] = (result_df[f'ema_{fast_ma}'] < result_df[f'ema_{slow_ma}']) & \
                                   (result_df[f'ema_{fast_ma}'].shift(1) >= result_df[f'ema_{slow_ma}'].shift(1))
        
        result_df.loc[result_df['ma_crossover'], 'ma_signal'] = 1
        result_df.loc[result_df['ma_crossunder'], 'ma_signal'] = -1
        
        return result_df
    
    def should_buy(self, df: pd.DataFrame) -> bool:
        """Check if we should buy based on MA crossover and RSI."""
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # MA crossover condition
        fast_ma = self.parameters['fast_ma']
        slow_ma = self.parameters['slow_ma']
        
        # Check if columns exist, if not calculate them
        if f'ema_{fast_ma}' not in df.columns:
            df_copy = df.copy()
            df_copy[f'ema_{fast_ma}'] = df_copy['close'].ewm(span=fast_ma).mean()
            df_copy[f'ema_{slow_ma}'] = df_copy['close'].ewm(span=slow_ma).mean()
            current = df_copy.iloc[-1]
            previous = df_copy.iloc[-2]
        
        ma_crossover = (current[f'ema_{fast_ma}'] > current[f'ema_{slow_ma}'] and 
                       previous[f'ema_{fast_ma}'] <= previous[f'ema_{slow_ma}'])
        
        # RSI condition (optional)
        rsi_condition = True
        if 'rsi' in df.columns:
            rsi_condition = current['rsi'] < self.parameters['rsi_overbought']
        
        return ma_crossover and rsi_condition
    
    def should_sell(self, df: pd.DataFrame) -> bool:
        """Check if we should sell based on MA crossunder and RSI."""
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # MA crossunder condition
        fast_ma = self.parameters['fast_ma']
        slow_ma = self.parameters['slow_ma']
        
        # Check if columns exist, if not calculate them
        if f'ema_{fast_ma}' not in df.columns:
            df_copy = df.copy()
            df_copy[f'ema_{fast_ma}'] = df_copy['close'].ewm(span=fast_ma).mean()
            df_copy[f'ema_{slow_ma}'] = df_copy['close'].ewm(span=slow_ma).mean()
            current = df_copy.iloc[-1]
            previous = df_copy.iloc[-2]
        
        ma_crossunder = (current[f'ema_{fast_ma}'] < current[f'ema_{slow_ma}'] and 
                        previous[f'ema_{fast_ma}'] >= previous[f'ema_{slow_ma}'])
        
        # RSI condition (optional)
        rsi_condition = True
        if 'rsi' in df.columns:
            rsi_condition = current['rsi'] > self.parameters['rsi_oversold']
        
        return ma_crossunder and rsi_condition

class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self):
        super().__init__("RSIStrategy")
        self.default_parameters = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        self.set_parameters(**self.default_parameters)
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI-based signals."""
        result_df = df.copy()
        
        # Calculate RSI if not present
        if 'rsi' not in df.columns:
            result_df['rsi'] = self._calculate_rsi(result_df['close'], self.parameters['rsi_period'])
        
        # Generate signals
        result_df['rsi_signal'] = 0
        
        # Buy when RSI crosses above oversold level
        result_df['rsi_buy'] = (result_df['rsi'] > self.parameters['rsi_oversold']) & \
                              (result_df['rsi'].shift(1) <= self.parameters['rsi_oversold'])
        
        # Sell when RSI crosses below overbought level
        result_df['rsi_sell'] = (result_df['rsi'] < self.parameters['rsi_overbought']) & \
                               (result_df['rsi'].shift(1) >= self.parameters['rsi_overbought'])
        
        result_df.loc[result_df['rsi_buy'], 'rsi_signal'] = 1
        result_df.loc[result_df['rsi_sell'], 'rsi_signal'] = -1
        
        return result_df
    
    def should_buy(self, df: pd.DataFrame) -> bool:
        """Check if RSI indicates oversold condition."""
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if 'rsi' not in df.columns:
            # Calculate RSI if not present
            rsi_current = self._calculate_rsi_single(df['close'].tail(15), self.parameters['rsi_period'])
            rsi_previous = self._calculate_rsi_single(df['close'].tail(15).shift(1).dropna(), self.parameters['rsi_period'])
            return rsi_current > self.parameters['rsi_oversold'] and rsi_previous <= self.parameters['rsi_oversold']
        
        # Buy when RSI crosses above oversold level
        return (current['rsi'] > self.parameters['rsi_oversold'] and 
               previous['rsi'] <= self.parameters['rsi_oversold'])
    
    def should_sell(self, df: pd.DataFrame) -> bool:
        """Check if RSI indicates overbought condition."""
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if 'rsi' not in df.columns:
            # Calculate RSI if not present
            rsi_current = self._calculate_rsi_single(df['close'].tail(15), self.parameters['rsi_period'])
            rsi_previous = self._calculate_rsi_single(df['close'].tail(15).shift(1).dropna(), self.parameters['rsi_period'])
            return rsi_current < self.parameters['rsi_overbought'] and rsi_previous >= self.parameters['rsi_overbought']
        
        # Sell when RSI crosses below overbought level
        return (current['rsi'] < self.parameters['rsi_overbought'] and 
               previous['rsi'] >= self.parameters['rsi_overbought'])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_rsi_single(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate single RSI value for the last point."""
        if len(prices) < period + 1:
            return 50  # Default neutral value
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).tail(period).mean()
        loss = (-delta.where(delta < 0, 0)).tail(period).mean()
        
        if loss == 0:
            return 100 if gain > 0 else 50
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# Strategy Factory
class StrategyFactory:
    """Factory class to create strategy instances."""
    
    @staticmethod
    def create_strategy(strategy_name: str, **parameters) -> BaseStrategy:
        """Create a strategy instance by name."""
        strategies = {
            'MovingAverageCrossover': MovingAverageCrossover,
            'RSIStrategy': RSIStrategy
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = strategies[strategy_name]()
        if parameters:
            strategy.set_parameters(**parameters)
        
        return strategy
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy names."""
        return ['MovingAverageCrossover', 'RSIStrategy']

# Test function
def test_strategies():
    """Test strategy implementations."""
    # Create sample data
    sample_data = {
        'close': [100, 102, 101, 105, 108, 106, 104, 107, 110, 112] * 10
    }
    df = pd.DataFrame(sample_data)
    
    # Test MA Crossover
    ma_strategy = MovingAverageCrossover()
    ma_strategy.set_parameters(fast_ma=5, slow_ma=10)
    ma_signals = ma_strategy.calculate_signals(df)
    
    print("✅ MA Crossover strategy test successful!")
    print(f"Signals generated: {len(ma_signals[ma_signals['ma_signal'] != 0])}")
    
    # Test RSI Strategy
    rsi_strategy = RSIStrategy()
    rsi_signals = rsi_strategy.calculate_signals(df)
    
    print("✅ RSI strategy test successful!")
    print(f"Signals generated: {len(rsi_signals[rsi_signals['rsi_signal'] != 0])}")
    
    return ma_signals, rsi_signals

if __name__ == "__main__":
    test_strategies()