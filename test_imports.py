# test_imports.py
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.settings import settings
    print("✅ Settings import successful!")
    print(f"Binance Testnet: {settings.BINANCE_TESTNET}")
    print(f"Default symbols: {settings.DEFAULT_SYMBOLS}")
    
    from data.databases.db_manager import DatabaseManager
    print("✅ DatabaseManager import successful!")
    
    from data.collectors.binance_collector import BinanceDataCollector
    print("✅ BinanceDataCollector import successful!")
    
    from utils.logger import logger
    print("✅ Logger import successful!")
    
    print("🎉 All imports successful! System is ready.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")