from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import pandas as pd

Base = declarative_base()

class Kline(Base):
    __tablename__ = "klines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(5), nullable=False)
    open_time = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    close_time = Column(DateTime, nullable=False)
    quote_asset_volume = Column(Float)
    number_of_trades = Column(Integer)
    taker_buy_base_volume = Column(Float)
    taker_buy_quote_volume = Column(Float)
    is_final = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Kline(symbol={self.symbol}, interval={self.interval}, time={self.open_time})>"

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now(), index=True)
    strategy_name = Column(String(50))
    profit_loss = Column(Float)
    is_test = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Trade(symbol={self.symbol}, side={self.side}, price={self.price})>"

class Indicator(Base):
    __tablename__ = "indicators"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(5), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Trend Indicators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    
    # Moving Averages
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    
    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    bb_position = Column(Float)
    
    # Volatility
    atr = Column(Float)
    
    # Momentum
    stoch_k = Column(Float)  # ADDED: Stochastic %K
    stoch_d = Column(Float)  # ADDED: Stochastic %D
    
    # Volume
    volume_sma_20 = Column(Float)
    obv = Column(Float)
    
    def __repr__(self):
        return f"<Indicator(symbol={self.symbol}, timestamp={self.timestamp})>"