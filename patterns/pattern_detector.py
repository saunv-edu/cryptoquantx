import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

@dataclass
class PatternSignal:
    """Container for pattern detection signals."""
    pattern_type: str
    symbol: str
    interval: str
    timestamp: pd.Timestamp
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    price_level: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class PatternDetector:
    """Detect technical chart patterns."""
    
    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
    
    def detect_all_patterns(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect all supported patterns."""
        signals = []
        
        # Ensure we have enough data
        if len(df) < 20:
            return signals
        
        # Detect different pattern types
        patterns_to_check = [
            self.detect_support_resistance,
            self.detect_double_top_bottom,
            self.detect_head_shoulders,
            self.detect_triangle,
            self.detect_breakout
        ]
        
        for pattern_func in patterns_to_check:
            try:
                pattern_signals = pattern_func(df, symbol, interval)
                signals.extend(pattern_signals)
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern_func.__name__}: {e}")
        
        # Filter by confidence
        signals = [s for s in signals if s.confidence >= self.min_confidence]
        
        logger.info(f"Detected {len(signals)} patterns for {symbol} {interval}")
        return signals
    
    def detect_support_resistance(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect support and resistance levels."""
        signals = []
        
        # Use recent price action to identify levels
        recent_data = df.tail(50)
        prices = recent_data['close'].values
        
        # Find local minima and maxima
        from scipy.signal import argrelextrema
        minima_indices = argrelextrema(prices, np.less, order=3)[0]
        maxima_indices = argrelextrema(prices, np.greater, order=3)[0]
        
        # Identify significant support levels (minima)
        support_levels = []
        for idx in minima_indices:
            price_level = prices[idx]
            # Check if this is a significant level (multiple touches)
            touches = np.sum(np.abs(prices - price_level) < price_level * 0.01)
            if touches >= 2:
                support_levels.append(price_level)
        
        # Identify significant resistance levels (maxima)
        resistance_levels = []
        for idx in maxima_indices:
            price_level = prices[idx]
            touches = np.sum(np.abs(prices - price_level) < price_level * 0.01)
            if touches >= 2:
                resistance_levels.append(price_level)
        
        current_price = prices[-1]
        
        # Generate signals for nearby levels
        for level in support_levels:
            if abs(current_price - level) / level < 0.02:  # Within 2%
                confidence = 0.8 if len(support_levels) > 1 else 0.6
                signals.append(PatternSignal(
                    pattern_type="SUPPORT",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bullish",
                    price_level=level
                ))
        
        for level in resistance_levels:
            if abs(current_price - level) / level < 0.02:  # Within 2%
                confidence = 0.8 if len(resistance_levels) > 1 else 0.6
                signals.append(PatternSignal(
                    pattern_type="RESISTANCE",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bearish",
                    price_level=level
                ))
        
        return signals
    
    def detect_double_top_bottom(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect double top and double bottom patterns."""
        signals = []
        
        # Use the last 30 candles for pattern detection
        recent_data = df.tail(30)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Find potential double top (M pattern)
        max_idx = np.argmax(highs)
        if max_idx > 5 and max_idx < len(highs) - 5:
            left_peak = highs[max_idx - 5:max_idx].max()
            right_peak = highs[max_idx + 1:max_idx + 6].max()
            
            # Check if peaks are similar (within 1%)
            if abs(left_peak - right_peak) / left_peak < 0.01:
                neckline = lows[max_idx - 2:max_idx + 3].min()
                confidence = 0.75
                
                signals.append(PatternSignal(
                    pattern_type="DOUBLE_TOP",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bearish",
                    price_level=right_peak,
                    target_price=neckline - (right_peak - neckline),
                    stop_loss=right_peak * 1.01
                ))
        
        # Find potential double bottom (W pattern)
        min_idx = np.argmin(lows)
        if min_idx > 5 and min_idx < len(lows) - 5:
            left_trough = lows[min_idx - 5:min_idx].min()
            right_trough = lows[min_idx + 1:min_idx + 6].min()
            
            # Check if troughs are similar (within 1%)
            if abs(left_trough - right_trough) / left_trough < 0.01:
                neckline = highs[min_idx - 2:min_idx + 3].max()
                confidence = 0.75
                
                signals.append(PatternSignal(
                    pattern_type="DOUBLE_BOTTOM",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bullish",
                    price_level=right_trough,
                    target_price=neckline + (neckline - right_trough),
                    stop_loss=right_trough * 0.99
                ))
        
        return signals
    
    def detect_head_shoulders(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect head and shoulders patterns."""
        signals = []
        
        recent_data = df.tail(40)
        highs = recent_data['high'].values
        
        # Simplified head and shoulders detection
        if len(highs) >= 10:
            # Look for pattern: lower high - highest high - lower high
            middle_idx = len(highs) // 2
            left_high = highs[:middle_idx].max()
            head_high = highs[middle_idx-2:middle_idx+2].max()
            right_high = highs[middle_idx:].max()
            
            # Check head and shoulders pattern conditions
            if (head_high > left_high and head_high > right_high and
                abs(left_high - right_high) / head_high < 0.02):
                
                neckline = recent_data['low'].min()
                confidence = 0.7
                
                signals.append(PatternSignal(
                    pattern_type="HEAD_SHOULDERS",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bearish",
                    price_level=head_high,
                    target_price=neckline - (head_high - neckline),
                    stop_loss=head_high * 1.02
                ))
        
        return signals
    
    def detect_triangle(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect triangle patterns (symmetrical, ascending, descending)."""
        signals = []
        
        recent_data = df.tail(20)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Calculate trend lines
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)
        
        # Symmetrical triangle (converging highs and lows)
        if (high_slope < -0.001 and low_slope > 0.001 and 
            abs(high_slope) > 0.0005 and abs(low_slope) > 0.0005):
            
            current_price = recent_data['close'].iloc[-1]
            avg_price = (highs[-1] + lows[-1]) / 2
            confidence = 0.65
            
            # Direction is neutral for symmetrical triangle
            direction = "neutral"
            
            signals.append(PatternSignal(
                pattern_type="TRIANGLE_SYMMETRICAL",
                symbol=symbol,
                interval=interval,
                timestamp=df.iloc[-1]['open_time'],
                confidence=confidence,
                direction=direction,
                price_level=avg_price
            ))
        
        # Ascending triangle (flat highs, rising lows)
        elif (abs(high_slope) < 0.0005 and low_slope > 0.001):
            resistance_level = np.mean(highs)
            current_price = recent_data['close'].iloc[-1]
            confidence = 0.7
            
            if current_price > resistance_level * 0.98:
                signals.append(PatternSignal(
                    pattern_type="TRIANGLE_ASCENDING",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bullish",
                    price_level=resistance_level,
                    target_price=resistance_level + (resistance_level - lows[-1])
                ))
        
        # Descending triangle (falling highs, flat lows)
        elif (high_slope < -0.001 and abs(low_slope) < 0.0005):
            support_level = np.mean(lows)
            current_price = recent_data['close'].iloc[-1]
            confidence = 0.7
            
            if current_price < support_level * 1.02:
                signals.append(PatternSignal(
                    pattern_type="TRIANGLE_DESCENDING",
                    symbol=symbol,
                    interval=interval,
                    timestamp=df.iloc[-1]['open_time'],
                    confidence=confidence,
                    direction="bearish",
                    price_level=support_level,
                    target_price=support_level - (highs[-1] - support_level)
                ))
        
        return signals
    
    def detect_breakout(self, df: pd.DataFrame, symbol: str, interval: str) -> List[PatternSignal]:
        """Detect breakout patterns."""
        signals = []
        
        recent_data = df.tail(15)
        current_price = recent_data['close'].iloc[-1]
        
        # Calculate recent volatility
        volatility = recent_data['close'].pct_change().std()
        
        # Check for resistance breakout
        resistance = recent_data['high'].iloc[:-1].max()
        if current_price > resistance and (current_price - resistance) / resistance > volatility * 2:
            signals.append(PatternSignal(
                pattern_type="BREAKOUT_RESISTANCE",
                symbol=symbol,
                interval=interval,
                timestamp=df.iloc[-1]['open_time'],
                confidence=0.8,
                direction="bullish",
                price_level=resistance,
                target_price=current_price + (current_price - resistance),
                stop_loss=resistance * 0.99
            ))
        
        # Check for support breakdown
        support = recent_data['low'].iloc[:-1].min()
        if current_price < support and (support - current_price) / support > volatility * 2:
            signals.append(PatternSignal(
                pattern_type="BREAKOUT_SUPPORT",
                symbol=symbol,
                interval=interval,
                timestamp=df.iloc[-1]['open_time'],
                confidence=0.8,
                direction="bearish",
                price_level=support,
                target_price=current_price - (support - current_price),
                stop_loss=support * 1.01
            ))
        
        return signals
    
    def _calculate_slope(self, values: np.ndarray) -> float:
        """Calculate slope of values using linear regression."""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

# Pattern monitoring system
class PatternMonitor:
    """Monitor patterns in real-time."""
    
    def __init__(self):
        self.detector = PatternDetector()
        self.active_patterns = []
    
    async def analyze_new_data(self, kline_data: dict):
        """Analyze new kline data for patterns."""
        from data.databases.db_manager import DatabaseManager
        
        db = DatabaseManager()
        symbol = kline_data['symbol']
        interval = kline_data['interval']
        
        # Get recent data for analysis
        df = db.get_klines_df(symbol, interval, limit=100)
        
        if len(df) < 20:
            return
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df, symbol, interval)
        
        # Log significant patterns
        for pattern in patterns:
            if pattern.confidence > 0.75:
                logger.info(f"🔍 Pattern detected: {pattern.pattern_type} "
                           f"({pattern.direction}) for {symbol} {interval} "
                           f"- Confidence: {pattern.confidence:.2f}")
                
                # Store pattern for tracking
                self.active_patterns.append({
                    'pattern': pattern,
                    'detected_time': datetime.now(),
                    'status': 'active'
                })

# Test function
def test_pattern_detection():
    """Test pattern detection with sample data."""
    # Create sample price data with patterns
    np.random.seed(42)
    
    # Create a double bottom pattern
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')
    prices = []
    
    # First decline
    for i in range(15):
        prices.append(100 - i * 2 + np.random.normal(0, 1))
    
    # First trough
    for i in range(5):
        prices.append(70 + np.random.normal(0, 0.5))
    
    # Rally
    for i in range(10):
        prices.append(70 + i * 1.5 + np.random.normal(0, 1))
    
    # Second decline to similar level
    for i in range(10):
        prices.append(85 - i * 1.5 + np.random.normal(0, 1))
    
    # Second trough and breakout
    for i in range(10):
        prices.append(71 + i * 2 + np.random.normal(0, 1))
    
    sample_data = {
        'open_time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 50)
    }
    df = pd.DataFrame(sample_data)
    
    # Test pattern detection
    detector = PatternDetector()
    patterns = detector.detect_all_patterns(df, "TEST", "1h")
    
    print("Pattern Detection Test Results:")
    for pattern in patterns:
        print(f"- {pattern.pattern_type}: {pattern.direction} "
              f"(confidence: {pattern.confidence:.2f})")
    
    return patterns

if __name__ == "__main__":
    test_pattern_detection()