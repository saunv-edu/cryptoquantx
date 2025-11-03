import asyncio
import websockets
import json
from datetime import datetime

async def test_mainnet_simple():
    """Simple test for Mainnet WebSocket."""
    symbols = ['btcusdt', 'ethusdt']
    intervals = ['1m']
    
    print("🚀 Testing Mainnet WebSocket Connections")
    print("=" * 50)
    
    for symbol in symbols:
        for interval in intervals:
            stream_name = f"{symbol}@kline_{interval}"
            url = f"wss://stream.binance.com:9443/ws/{stream_name}"
            
            print(f"\n🔌 Testing: {stream_name}")
            
            try:
                async with websockets.connect(url) as websocket:
                    print(f"✅ Connected to Mainnet: {stream_name}")
                    
                    # Nhận 3 messages để xác nhận hoạt động
                    for i in range(3):
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        data = json.loads(message)
                        
                        if 'k' in data:
                            kline = data['k']
                            if kline['x']:  # Chỉ hiển thị kline đã đóng
                                print(f"📊 {i+1}. {kline['s']} {kline['i']}: ${float(kline['c']):.2f}")
                    
                    await websocket.close()
                    print(f"✅ Test passed for {stream_name}")
                    
            except asyncio.TimeoutError:
                print(f"❌ Timeout for {stream_name}")
            except Exception as e:
                print(f"❌ Error for {stream_name}: {e}")

if __name__ == "__main__":
    asyncio.run(test_mainnet_simple())