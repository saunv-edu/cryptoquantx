from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from data.models import Base, Kline, Trade, Indicator
from utils.logger import logger

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Define model classes for easy access
        self.Kline = Kline
        self.Trade = Trade
        self.Indicator = Indicator
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_klines(self, klines_data: list):
        """Save klines data to database."""
        with self.get_session() as session:
            for kline_data in klines_data:
                # Check if kline already exists
                existing = session.query(self.Kline).filter(
                    self.Kline.symbol == kline_data['symbol'],
                    self.Kline.interval == kline_data['interval'],
                    self.Kline.open_time == kline_data['open_time']
                ).first()
                
                if not existing:
                    kline = self.Kline(**kline_data)
                    session.add(kline)
            logger.info(f"Processed {len(klines_data)} klines")
    
    def log_trade(self, trade_data: dict):
        """Log a trade to database."""
        with self.get_session() as session:
            trade = self.Trade(**trade_data)
            session.add(trade)
            logger.info(f"Trade logged: {trade_data}")
    
    def get_latest_kline(self, symbol: str, interval: str):
        """Get the latest kline for a symbol and interval."""
        with self.get_session() as session:
            kline = session.query(self.Kline).filter(
                self.Kline.symbol == symbol,
                self.Kline.interval == interval
            ).order_by(self.Kline.open_time.desc()).first()
            return kline
    
    def get_klines_df(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Get klines as pandas DataFrame."""
        with self.get_session() as session:
            query = session.query(self.Kline).filter(
                self.Kline.symbol == symbol,
                self.Kline.interval == interval
            ).order_by(self.Kline.open_time.desc()).limit(limit)
            
            df = pd.read_sql(query.statement, session.bind)
            return df.sort_values('open_time')
    
    def save_indicators(self, indicators_data: list):
        """Save indicators data to database."""
        with self.get_session() as session:
            for indicator_data in indicators_data:
                # Check if indicator already exists
                existing = session.query(self.Indicator).filter(
                    self.Indicator.symbol == indicator_data['symbol'],
                    self.Indicator.interval == indicator_data['interval'],
                    self.Indicator.timestamp == indicator_data['timestamp']
                ).first()
                
                if not existing:
                    indicator = self.Indicator(**indicator_data)
                    session.add(indicator)
            logger.info(f"Saved {len(indicators_data)} indicators")