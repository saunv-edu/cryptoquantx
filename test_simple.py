import asyncio
import aiohttp
import os
from datetime import datetime

async def test_binance_connection():
    """Test simple Binance API connection."""
    async with aiohttp.ClientSession() as session:
        url = "https://testnet.binance.vision/api/v3/ping"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    print("✅ Binance Testnet connection successful!")
                    return True
                else:
                    print(f"❌ Binance connection failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

async def test_database():
    """Test database connection."""
    try:
        import sqlite3
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        
        # Create simple table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                timestamp TEXT
            )
        ''')
        
        # Insert test data
        cursor.execute('INSERT INTO test_table (timestamp) VALUES (?)', 
                      (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
        print("✅ Database test successful!")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def main():
    print("🧪 Running system tests...")
    
    # Test 1: Binance connection
    binance_ok = await test_binance_connection()
    
    # Test 2: Database
    db_ok = await test_database()
    
    if binance_ok and db_ok:
        print("🎉 All tests passed! System is ready.")
    else:
        print("💥 Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())