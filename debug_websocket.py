import asyncio
import websockets
import json
from datetime import datetime

async def debug_websocket_connection():
    """Debug WebSocket connection với các URL khác nhau."""
    
    test_cases = [
        # Testnet connections
        ("wss://testnet.binance.vision/ws/btcusdt@kline_1m", "Testnet BTC 1m"),
        ("wss://testnet.binance.vision/ws/ethusdt@kline_1m", "Testnet ETH 1m"),
        ("wss://testnet.binance.vision/ws/btcusdt@kline_5m", "Testnet BTC 5m"),
        ("wss://testnet.binance.vision/ws/ethusdt@kline_5m", "Testnet ETH 5m"),
        
        # Mainnet connections  
        ("wss://stream.binance.com:9443/ws/btcusdt@kline_1m", "Mainnet BTC 1m"),
        ("wss://stream.binance.com:9443/ws/ethusdt@kline_1m", "Mainnet ETH 1m"),
    ]
    
    for url, description in test_cases:
        print(f"\n🔌 Testing: {description}")
        print(f"URL: {url}")
        
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as websocket:
                print(f"✅ Connected successfully!")
                
                # Try to receive a message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    
                    if 'e' in data and data['e'] == 'kline':
                        kline = data['k']
                        print(f"📊 Received kline: {kline['s']} {kline['i']} - Close: {kline['c']}")
                    else:
                        print(f"📨 Received: {data}")
                        
                except asyncio.TimeoutError:
                    print("⏰ Timeout - no data received")
                
                await websocket.close()
                print("✅ Connection closed properly")
                
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"❌ HTTP Error: {e.status_code} - {e}")
        except websockets.exceptions.InvalidURI as e:
            print(f"❌ Invalid URI: {e}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

async def test_stream_combinations():
    """Test các combination stream khác nhau."""
    
    base_urls = [
        "wss://testnet.binance.vision/ws",
        "wss://stream.binance.com:9443/ws"
    ]
    
    streams = [
        "btcusdt@kline_1m",
        "ethusdt@kline_1m", 
        "btcusdt@kline_5m",
        "ethusdt@kline_5m",
        "!ticker@arr",  # Test all tickers stream
        "btcusdt@ticker"  # Test ticker stream
    ]
    
    for base_url in base_urls:
        print(f"\n🎯 Testing base URL: {base_url}")
        
        for stream in streams[:4]:  # Chỉ test 4 streams đầu
            url = f"{base_url}/{stream}"
            print(f"  Testing: {stream}")
            
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"    ✅ Connected to {stream}")
                    
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5)
                        print(f"    📨 Received data from {stream}")
                        await ws.close()
                    except asyncio.TimeoutError:
                        print(f"    ⏰ No data from {stream}")
                        await ws.close()
                        
            except Exception as e:
                print(f"    ❌ Failed: {e}")

if __name__ == "__main__":
    print("🔧 WebSocket Connection Debug Tool")
    print("=" * 60)
    
    asyncio.run(debug_websocket_connection())
    
    print("\n" + "=" * 60)
    print("🔄 Testing Stream Combinations")
    print("=" * 60)
    
    asyncio.run(test_stream_combinations())