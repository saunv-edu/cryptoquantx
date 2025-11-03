import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class AlternativeIndicatorCalculator:
    """Calculate technical indicators using pandas_ta."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators using pandas_ta."""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Moving Averages
        result_df.ta.sma(length=20, append=True)
        result_df.ta.sma(length=50, append=True)
        result_df.ta.ema(length=20, append=True)
        result_df.ta.ema(length=50, append=True)
        
        # MACD
        result_df.ta.macd(append=True)
        
        # RSI
        result_df.ta.rsi(append=True)
        
        # Bollinger Bands
        result_df.ta.bbands(append=True)
        
        # Stochastic
        result_df.ta.stoch(append=True)
        
        # ATR
        result_df.ta.atr(append=True)
        
        # ADX
        result_df.ta.adx(append=True)
        
        # OBV
        result_df.ta.obv(append=True)
        
        logger.info(f"Calculated indicators using pandas_ta")
        return result_df
    
    def calculate_for_symbol(self, symbol: str, interval: str, limit: int = 1000):
        """Calculate indicators for a specific symbol and interval."""
        try:
            df = self.db.get_klines_df(symbol, interval, limit)
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {interval}")
                return
            
            df_with_indicators = self.calculate_all_indicators(df)
            self._save_indicators_to_db(symbol, interval, df_with_indicators)
            
            logger.info(f"Calculated indicators for {symbol} {interval}")
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} {interval}: {e}")
    
    def _save_indicators_to_db(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save calculated indicators to database."""
        with self.db.get_session() as session:
            for _, row in df.iterrows():
                if 'RSI_14' in df.columns and pd.notna(row.get('RSI_14')):
                    indicator_data = {
                        'symbol': symbol,
                        'interval': interval,
                        'timestamp': row['open_time'],
                        'rsi': float(row.get('RSI_14', 0)),
                        'macd': float(row.get('MACD_12_26_9', 0)),
                        'macd_signal': float(row.get('MACDs_12_26_9', 0)),
                        'macd_histogram': float(row.get('MACDh_12_26_9', 0)),
                        'bb_upper': float(row.get('BBU_20_2.0', 0)),
                        'bb_middle': float(row.get('BBM_20_2.0', 0)),
                        'bb_lower': float(row.get('BBL_20_2.0', 0)),
                        'ema_20': float(row.get('EMA_20', 0)),
                        'ema_50': float(row.get('EMA_50', 0)),
                        'sma_20': float(row.get('SMA_20', 0)),
                        'atr': float(row.get('ATRr_14', 0)),
                        'adx': float(row.get('ADX_14', 0)),
                        'stoch_k': float(row.get('STOCHk_14_3_3', 0)),
                        'stoch_d': float(row.get('STOCHd_14_3_3', 0))
                    }
                    
                    existing = session.query(self.db.Indicator).filter(
                        self.db.Indicator.symbol == symbol,
                        self.db.Indicator.interval == interval,
                        self.db.Indicator.timestamp == row['open_time']
                    ).first()
                    
                    if not existing:
                        indicator = self.db.Indicator(**indicator_data)
                        session.add(indicator)