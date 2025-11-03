import asyncio
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.collectors.binance_collector import BinanceDataCollector
from utils.logger import logger

class FallbackDataCollector:
    """Fallback data collector using REST API when WebSocket fails."""
    
    def __init__(self, poll_interval: int = 60):  # Poll every 60 seconds
        self.poll_interval = poll_interval
        self.is_running = False
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add callback function."""
        self.callbacks.append(callback)
    
    async def start_polling(self, symbols: List[str], intervals: List[str]):
        """Start polling data via REST API."""
        self.is_running = True
        collector = BinanceDataCollector()
        
        logger.info(f"Starting fallback polling for {len(symbols)} symbols")
        
        while self.is_running:
            try:
                # Collect latest data
                for symbol in symbols:
                    for interval in intervals:
                        # Get only the latest kline
                        klines = await collector.fetch_klines(symbol, interval, limit=1)
                        if klines:
                            kline_data = klines[0]
                            
                            # Notify callbacks
                            for callback in self.callbacks:
                                try:
                                    await callback(kline_data)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                
                logger.debug(f"Fallback polling completed for {len(symbols)} symbols")
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Fallback polling error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def stop_polling(self):
        """Stop polling."""
        self.is_running = False