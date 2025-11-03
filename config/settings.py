import os
from typing import List

class Settings:
    """Application settings - simplified version."""
    
    def __init__(self):
        # Binance API
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "qNlX9Kg43E3JmbcGiw1pHdX8rvgBhPuoehghEn7rRHTixCHfpgncUmSgOCxrNLvt")
        self.BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "FvhRBvFnajwMTEigOtsATqEffv5yb1FroeME0CzQVC6zkgYKmggEJLy1Bo64PjoD")
        self.BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        
        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")
        
        # Trading Parameters
        self.RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))
        self.MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))
        
        # Default symbols
        symbols_str = os.getenv("DEFAULT_SYMBOLS", "BTCUSDT,ETHUSDT,ADAUSDT")
        self.DEFAULT_SYMBOLS = [s.strip() for s in symbols_str.split(",")]
        
        # Data Collection
        intervals_str = os.getenv("COLLECTION_INTERVALS", "1m, 2m, 3m, 5m, 10m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")
        self.COLLECTION_INTERVALS = [s.strip() for s in intervals_str.split(",")]

settings = Settings()