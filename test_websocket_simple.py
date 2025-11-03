import asyncio
import websockets
import json
from datetime import datetime

async def test_binance_websocket():
    """Test Binance WebSocket connection directly."""
    symbols = ['btcusdt', 'ethusdt']
    intervals = ['1m', '5m']
    
    for symbol in symbols:
        for interval in intervals:
            stream_name = f"{symbol}@kline_{interval}"
            url = f"wss://stream.binance.com:9443/ws/{stream_name}"
            
            print(f"Testing: {stream_name}")
            
            try:
                async with websockets.connect(url) as websocket:
                    print(f"✅ Connected to {stream_name}")
                    
                    # Receive one message to verify it works
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    
                    if 'k' in data:
                        kline = data['k']
                        print(f"📊 First kline: {kline['s']} - Close: {kline['c']}")
                    
                    await websocket.close()
                    print(f"✅ Test passed for {stream_name}\n")
                    
            except asyncio.TimeoutError:
                print(f"❌ Timeout for {stream_name}\n")
            except Exception as e:
                print(f"❌ Error for {stream_name}: {e}\n")

async def test_binance_testnet():
    """Test Binance Testnet WebSocket."""
    symbols = ['btcusdt', 'ethusdt']
    intervals = ['1m']
    
    for symbol in symbols:
        for interval in intervals:
            stream_name = f"{symbol}@kline_{interval}"
            url = f"wss://testnet.binance.vision/ws/{stream_name}"
            
            print(f"Testing Testnet: {stream_name}")
            
            try:
                async with websockets.connect(url) as websocket:
                    print(f"✅ Connected to Testnet: {stream_name}")
                    
                    # Receive one message
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    
                    if 'k' in data:
                        kline = data['k']
                        print(f"📊 Testnet kline: {kline['s']} - Close: {kline['c']}")
                    
                    await websocket.close()
                    print(f"✅ Testnet test passed for {stream_name}\n")
                    
            except asyncio.TimeoutError:
                print(f"❌ Testnet timeout for {stream_name}\n")
            except Exception as e:
                print(f"❌ Testnet error for {stream_name}: {e}\n")

if __name__ == "__main__":
    print("🔌 Testing Binance WebSocket Connections...")
    print("=" * 50)
    
    asyncio.run(test_binance_websocket())
    
    print("\n🔌 Testing Binance Testnet WebSocket Connections...")
    print("=" * 50)
    
    asyncio.run(test_binance_testnet())