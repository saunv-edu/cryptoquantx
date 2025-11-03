import asyncio
from datetime import datetime
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patterns.pattern_detector import PatternDetector, PatternSignal
from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class PatternMonitor:
    """Monitor patterns in real-time."""
    
    def __init__(self, min_confidence: float = 0.7):
        self.detector = PatternDetector(min_confidence=min_confidence)
        self.active_patterns: List[Dict] = []
        self.pattern_history: List[Dict] = []
    
    async def analyze_new_data(self, kline_data: dict) -> List[PatternSignal]:
        """Analyze new kline data for patterns."""
        db = DatabaseManager()
        symbol = kline_data['symbol']
        interval = kline_data['interval']
        
        # Get recent data for analysis
        df = db.get_klines_df(symbol, interval, limit=100)
        
        if len(df) < 20:
            return []
        
        # Detect patterns
        patterns = self.detector.detect_all_patterns(df, symbol, interval)
        
        # Process new patterns
        new_signals = []
        for pattern in patterns:
            if pattern.confidence > 0.75:
                new_signals.append(pattern)
                
                # Check if this is a new pattern
                is_new = self._is_new_pattern(pattern)
                
                if is_new:
                    logger.info(f"🔍 NEW Pattern: {pattern.pattern_type} "
                               f"({pattern.direction}) for {symbol} {interval} "
                               f"- Confidence: {pattern.confidence:.2f}")
                    
                    # Store pattern for tracking
                    pattern_record = {
                        'pattern': pattern,
                        'detected_time': datetime.now(),
                        'status': 'active',
                        'symbol': symbol,
                        'interval': interval
                    }
                    self.active_patterns.append(pattern_record)
                    self.pattern_history.append(pattern_record)
        
        return new_signals
    
    def _is_new_pattern(self, pattern: PatternSignal) -> bool:
        """Check if this pattern is new (not recently detected)."""
        recent_time = datetime.now().timestamp() - 3600  # 1 hour ago
        
        for active_pattern in self.active_patterns:
            existing = active_pattern['pattern']
            if (existing.pattern_type == pattern.pattern_type and
                existing.symbol == pattern.symbol and
                existing.interval == pattern.interval and
                abs(existing.price_level - pattern.price_level) / pattern.price_level < 0.05):
                return False
        
        return True
    
    def get_active_patterns(self, symbol: str = None) -> List[Dict]:
        """Get currently active patterns."""
        if symbol:
            return [p for p in self.active_patterns if p['symbol'] == symbol]
        return self.active_patterns
    
    def cleanup_old_patterns(self, max_age_hours: int = 24):
        """Remove old patterns from active tracking."""
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        self.active_patterns = [
            p for p in self.active_patterns 
            if (current_time - p['detected_time'].timestamp()) < max_age_seconds
        ]

# Test function
async def test_pattern_monitor():
    """Test the pattern monitor."""
    # Create sample kline data
    sample_kline = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'open_time': datetime.now(),
        'open': 50000,
        'high': 51000,
        'low': 49000,
        'close': 50500,
        'volume': 1000,
        'close_time': datetime.now(),
        'quote_asset_volume': 50000000,
        'number_of_trades': 1000,
        'taker_buy_base_volume': 500,
        'taker_buy_quote_volume': 25000000,
        'is_final': True
    }
    
    monitor = PatternMonitor()
    signals = await monitor.analyze_new_data(sample_kline)
    
    print(f"Detected {len(signals)} patterns")
    for signal in signals:
        print(f"- {signal.pattern_type}: {signal.direction}")

if __name__ == "__main__":
    asyncio.run(test_pattern_monitor())