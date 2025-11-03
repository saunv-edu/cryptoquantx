import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Callable, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class BinanceWebSocketCollector:
    """Real-time data collection via Binance WebSocket - Mainnet Version."""
    
    def __init__(self, use_testnet: bool = False):  # Mặc định dùng Mainnet
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.testnet_url = "wss://testnet.binance.vision/ws"
        self.use_testnet = use_testnet
        self.db = DatabaseManager()
        self.callbacks = []
        self.is_running = False
        self.connections = {}
        
    def add_callback(self, callback: Callable):
        """Add callback function for real-time data."""
        self.callbacks.append(callback)
    
    def get_websocket_url(self, stream_name: str) -> str:
        """Get WebSocket URL - LUÔN dùng Mainnet vì Testnet không hỗ trợ WebSocket."""
        # LUÔN sử dụng Mainnet vì Testnet không hỗ trợ WebSocket
        return f"{self.base_url}/{stream_name}"
    
    async def handle_single_stream(self, stream_name: str, symbol: str, interval: str):
        """Handle a single WebSocket stream."""
        url = self.get_websocket_url(stream_name)
        
        logger.info(f"🔌 Connecting to: {stream_name}")
        
        retry_count = 0
        max_retries = 3
        retry_delay = 5
        
        while self.is_running and retry_count < max_retries:
            try:
                async with websockets.connect(
                    url, 
                    ping_interval=30, 
                    ping_timeout=20,
                    close_timeout=10
                ) as websocket:
                    
                    logger.info(f"✅ Connected to: {stream_name}")
                    retry_count = 0  # Reset retry count on success
                    
                    while self.is_running:
                        try:
                            # Receive message with timeout
                            message = await asyncio.wait_for(websocket.recv(), timeout=60)
                            data = json.loads(message)
                            
                            # Process kline data
                            if self._is_valid_kline_message(data):
                                kline_data = self._parse_kline_message(data, symbol, interval)
                                
                                if kline_data:
                                    # Save to database
                                    self.db.save_klines([kline_data])
                                    
                                    # Notify callbacks
                                    await self._notify_callbacks(kline_data)
                                    
                                    logger.info(f"📊 {symbol} {interval}: ${kline_data['close']:.2f}")
                            
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            try:
                                await websocket.ping()
                                logger.debug(f"🏓 Ping sent to {stream_name}")
                                continue
                            except Exception as e:
                                logger.warning(f"Ping failed for {stream_name}: {e}")
                                break
                                
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"🔌 Connection closed for {stream_name}")
                            break
                        except Exception as e:
                            logger.error(f"❌ Error in {stream_name}: {e}")
                            break
                            
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Connection failed for {stream_name} (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    await asyncio.sleep(retry_delay)
        
        if retry_count >= max_retries:
            logger.error(f"🚨 Max retries reached for {stream_name}, giving up")
    
    def _is_valid_kline_message(self, data: Dict) -> bool:
        """Check if message is a valid kline message."""
        return (isinstance(data, dict) and 
                data.get('e') == 'kline' and 
                'k' in data and 
                data['k'].get('x') is True)  # Only closed klines
    
    def _parse_kline_message(self, data: Dict, symbol: str, interval: str) -> Optional[Dict]:
        """Parse kline WebSocket message."""
        try:
            kline = data['k']
            
            return {
                'symbol': symbol.upper(),
                'interval': interval,
                'open_time': datetime.fromtimestamp(kline['t'] / 1000),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'close_time': datetime.fromtimestamp(kline['T'] / 1000),
                'quote_asset_volume': float(kline['q']),
                'number_of_trades': kline['n'],
                'taker_buy_base_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q']),
                'is_final': kline['x']
            }
        except Exception as e:
            logger.error(f"Error parsing kline message: {e}")
            return None
    
    async def _notify_callbacks(self, kline_data: Dict):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                await callback(kline_data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def start_monitoring(self, symbols: List[str], intervals: List[str]):
        """Start real-time monitoring for multiple symbols and intervals."""
        self.is_running = True
        
        tasks = []
        for symbol in symbols:
            # Convert to lowercase for WebSocket
            ws_symbol = symbol.lower()
            for interval in intervals:
                stream_name = f"{ws_symbol}@kline_{interval}"
                task = asyncio.create_task(
                    self.handle_single_stream(stream_name, symbol, interval)
                )
                tasks.append(task)
        
        logger.info(f"🚀 Started monitoring {len(tasks)} streams on MAINNET")
        
        try:
            # Use asyncio.gather with return_exceptions to prevent one failed task from stopping others
            await asyncio.gather(*tasks, return_exceptions=True)
                    
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.is_running = False
            logger.info("🛑 Monitoring stopped")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_running = False
        logger.info("🔚 WebSocket monitoring stopped")

# AlertManager giữ nguyên
class AlertManager:
    """Manage trading alerts and notifications."""
    
    def __init__(self):
        self.alerts = []
        self.triggered_alerts = []
    
    def add_price_alert(self, symbol: str, condition: str, price: float, message: str):
        """Add price-based alert."""
        alert = {
            'type': 'price',
            'symbol': symbol,
            'condition': condition,
            'price': price,
            'message': message,
            'triggered': False
        }
        self.alerts.append(alert)
        logger.info(f"Added price alert: {symbol} {condition} {price}")
    
    def add_indicator_alert(self, symbol: str, indicator: str, condition: str, 
                          value: float, message: str):
        """Add indicator-based alert."""
        alert = {
            'type': 'indicator',
            'symbol': symbol,
            'indicator': indicator,
            'condition': condition,
            'value': value,
            'message': message,
            'triggered': False
        }
        self.alerts.append(alert)
        logger.info(f"Added indicator alert: {symbol} {indicator} {condition} {value}")
    
    async def check_alerts(self, symbol: str, current_price: float, indicators: Dict = None):
        """Check all alerts for a symbol."""
        for alert in self.alerts:
            if alert['symbol'] == symbol and not alert['triggered']:
                triggered = False
                
                if alert['type'] == 'price':
                    triggered = self._check_price_alert(alert, current_price)
                elif alert['type'] == 'indicator' and indicators:
                    triggered = self._check_indicator_alert(alert, indicators)
                
                if triggered:
                    alert['triggered'] = True
                    alert['triggered_time'] = datetime.now()
                    self.triggered_alerts.append(alert)
                    
                    await self._send_alert_notification(alert, current_price)
    
    def _check_price_alert(self, alert: Dict, current_price: float) -> bool:
        """Check if price alert condition is met."""
        condition = alert['condition']
        target_price = alert['price']
        
        if condition == 'above' and current_price > target_price:
            return True
        elif condition == 'below' and current_price < target_price:
            return True
        elif condition == 'equals' and abs(current_price - target_price) < 0.01:
            return True
        
        return False
    
    def _check_indicator_alert(self, alert: Dict, indicators: Dict) -> bool:
        """Check if indicator alert condition is met."""
        indicator_value = indicators.get(alert['indicator'])
        if indicator_value is None:
            return False
        
        condition = alert['condition']
        target_value = alert['value']
        
        if condition == 'above' and indicator_value > target_value:
            return True
        elif condition == 'below' and indicator_value < target_value:
            return True
        elif condition == 'equals' and abs(indicator_value - target_value) < 0.01:
            return True
        
        return False
    
    async def _send_alert_notification(self, alert: Dict, current_price: float):
        """Send alert notification."""
        message = f"🚨 ALERT: {alert['message']}\n"
        message += f"Symbol: {alert['symbol']}\n"
        message += f"Current Price: ${current_price:.2f}\n"
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        logger.info(f"ALERT TRIGGERED: {message}")
        
        print(f"\n{'='*50}")
        print(message)
        print(f"{'='*50}\n")

# Test function với Mainnet
async def test_mainnet_websocket():
    """Test the Mainnet WebSocket collector."""
    collector = BinanceWebSocketCollector(use_testnet=False)  # Sử dụng Mainnet
    
    async def print_callback(kline_data):
        print(f"✅ {kline_data['symbol']} {kline_data['interval']}: "
              f"${kline_data['close']:.2f} (Vol: {kline_data['volume']:.0f})")
    
    collector.add_callback(print_callback)
    
    symbols = ['BTCUSDT', 'ETHUSDT']  # Uppercase symbols
    intervals = ['1m']  # Chỉ test 1 interval để đơn giản
    
    print("🚀 Starting Mainnet WebSocket test (30 seconds)...")
    
    async def stop_after_delay():
        await asyncio.sleep(30)
        collector.stop_monitoring()
        print("🛑 Test completed")
    
    await asyncio.gather(
        collector.start_monitoring(symbols, intervals),
        stop_after_delay(),
        return_exceptions=True
    )

if __name__ == "__main__":
    asyncio.run(test_mainnet_websocket())