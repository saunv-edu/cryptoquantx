import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class BinanceDataCollector:
    def __init__(self):
        self.base_url = "https://testnet.binance.vision/api/v3" if settings.BINANCE_TESTNET else "https://api.binance.com/api/v3"
        self.db = DatabaseManager()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_klines(self, symbol: str, interval: str, limit: int = 1000) -> List[Dict]:
        """Fetch klines data from Binance API."""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    klines = []
                    for kline in data:
                        klines.append({
                            'symbol': symbol,
                            'interval': interval,
                            'open_time': datetime.fromtimestamp(kline[0] / 1000),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'close_time': datetime.fromtimestamp(kline[6] / 1000),
                            'quote_asset_volume': float(kline[7]),
                            'number_of_trades': kline[8],
                            'taker_buy_base_volume': float(kline[9]),
                            'taker_buy_quote_volume': float(kline[10]),
                            'is_final': True
                        })
                    logger.info(f"Fetched {len(klines)} klines for {symbol} {interval}")
                    return klines
                else:
                    logger.error(f"Error fetching klines: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Exception fetching klines for {symbol}: {e}")
            return []
    
    async def collect_historical_data(self, symbols: List[str], intervals: List[str], days: int = 30):
        """Collect historical data for multiple symbols and intervals."""
        tasks = []
        for symbol in symbols:
            for interval in intervals:
                # Calculate appropriate limit based on interval and days
                limit = self._calculate_limit(interval, days)
                task = self.fetch_klines(symbol, interval, limit)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_klines = 0
        for klines in results:
            if klines:
                self.db.save_klines(klines)
                total_klines += len(klines)
        
        logger.info(f"Total klines saved: {total_klines}")
    
    def _calculate_limit(self, interval: str, days: int) -> int:
        """Calculate appropriate limit based on interval and days."""
        intervals_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        minutes_per_day = 24 * 60
        total_minutes = days * minutes_per_day
        
        if interval in intervals_minutes:
            interval_minutes = intervals_minutes[interval]
            limit = total_minutes // interval_minutes
            return min(limit, 1000)  # Binance max limit is 1000
        
        return 500  # Default limit

    async def collect_realtime_data(self, symbols: List[str], intervals: List[str]):
        """Start real-time data collection (to be implemented with WebSocket)."""
        logger.info(f"Starting real-time data collection for {symbols}")
        # WebSocket implementation will be added in next phase
        pass

# Simple test function
async def main():
    """Test the data collector."""
    async with BinanceDataCollector() as collector:
        symbols = settings.DEFAULT_SYMBOLS
        intervals = ["1m", "5m", "15m"]
        
        logger.info("Starting historical data collection...")
        await collector.collect_historical_data(symbols, intervals, days=7)
        logger.info("Historical data collection completed!")

if __name__ == "__main__":
    asyncio.run(main())