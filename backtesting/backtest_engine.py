import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import BaseStrategy
from utils.logger import logger

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        self.initial_capital = 0
        self.final_capital = 0
        self.total_return = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.profit_factor = 0
        self.trades = []
        self.equity_curve = []

class BacktestEngine:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission
        
    def run_backtest(self, strategy: BaseStrategy, df: pd.DataFrame, 
                    symbol: str, position_size: float = 0.1) -> BacktestResult:
        """Run backtest for a strategy on historical data."""
        
        result = BacktestResult()
        result.initial_capital = self.initial_capital
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Ensure we have the required indicators
        df_with_signals = strategy.calculate_signals(df)
        
        for i in range(1, len(df_with_signals)):
            current_data = df_with_signals.iloc[:i+1]
            current_row = df_with_signals.iloc[i]
            previous_row = df_with_signals.iloc[i-1]
            
            # Calculate current portfolio value
            if position > 0:
                current_value = capital + (position * current_row['close'])
            else:
                current_value = capital
                
            equity_curve.append({
                'timestamp': current_row['open_time'],
                'equity': current_value,
                'price': current_row['close']
            })
            
            # Check for buy signal
            if position == 0 and strategy.should_buy(current_data):
                # Calculate position size
                position_value = capital * position_size
                position = position_value / current_row['close']
                entry_price = current_row['close']
                capital -= position_value
                
                trade = {
                    'entry_time': current_row['open_time'],
                    'entry_price': entry_price,
                    'position': position,
                    'type': 'LONG',
                    'commission': position_value * self.commission
                }
                capital -= trade['commission']
                trades.append(trade)
                
                logger.debug(f"BUY: {position:.4f} {symbol} at {entry_price:.2f}")
            
            # Check for sell signal
            elif position > 0 and strategy.should_sell(current_data):
                exit_price = current_row['close']
                exit_value = position * exit_price
                capital += exit_value
                
                # Calculate PnL
                pnl = exit_value - (position * entry_price)
                pnl_percent = (pnl / (position * entry_price)) * 100
                
                # Update last trade
                last_trade = trades[-1]
                last_trade['exit_time'] = current_row['open_time']
                last_trade['exit_price'] = exit_price
                last_trade['pnl'] = pnl
                last_trade['pnl_percent'] = pnl_percent
                last_trade['commission'] += exit_value * self.commission
                
                capital -= last_trade['commission']
                
                logger.debug(f"SELL: {position:.4f} {symbol} at {exit_price:.2f}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
                
                position = 0
                entry_price = 0
        
        # Close any open position at the end
        if position > 0:
            exit_price = df_with_signals.iloc[-1]['close']
            exit_value = position * exit_price
            capital += exit_value
            
            pnl = exit_value - (position * entry_price)
            pnl_percent = (pnl / (position * entry_price)) * 100
            
            last_trade = trades[-1]
            last_trade['exit_time'] = df_with_signals.iloc[-1]['open_time']
            last_trade['exit_price'] = exit_price
            last_trade['pnl'] = pnl
            last_trade['pnl_percent'] = pnl_percent
            last_trade['commission'] += exit_value * self.commission
            
            capital -= last_trade['commission']
        
        # Calculate performance metrics
        result.final_capital = capital
        result.total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
        result.trades = trades
        result.total_trades = len(trades)
        
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
            
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            result.win_rate = (result.winning_trades / result.total_trades) * 100
            
            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in losing_trades))
            
            result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate max drawdown
            equity_series = pd.Series([e['equity'] for e in equity_curve])
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max * 100
            result.max_drawdown = drawdowns.min()
            
            # Calculate Sharpe ratio (simplified)
            returns = equity_series.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365)
        
        result.equity_curve = equity_curve
        
        return result
    
    def generate_report(self, result: BacktestResult, strategy_name: str, symbol: str) -> str:
        """Generate a comprehensive backtest report."""
        
        report = f"""
📊 BACKTEST REPORT - {strategy_name} - {symbol}
{'='*50}

💼 CAPITAL & RETURNS
  Initial Capital: ${result.initial_capital:,.2f}
  Final Capital:   ${result.final_capital:,.2f}
  Total Return:    {result.total_return:+.2f}%

📈 PERFORMANCE METRICS
  Total Trades:    {result.total_trades}
  Winning Trades:  {result.winning_trades}
  Losing Trades:   {result.losing_trades}
  Win Rate:        {result.win_rate:.1f}%
  Profit Factor:   {result.profit_factor:.2f}
  Max Drawdown:    {result.max_drawdown:.2f}%
  Sharpe Ratio:    {result.sharpe_ratio:.2f}

💸 TRADE ANALYSIS
"""
        if result.trades:
            avg_win = np.mean([t['pnl'] for t in result.trades if t['pnl'] > 0])
            avg_loss = np.mean([t['pnl'] for t in result.trades if t['pnl'] <= 0])
            largest_win = max([t['pnl'] for t in result.trades])
            largest_loss = min([t['pnl'] for t in result.trades])
            
            report += f"""
  Average Win:     ${avg_win:.2f}
  Average Loss:    ${avg_loss:.2f}
  Largest Win:     ${largest_win:.2f}
  Largest Loss:    ${largest_loss:.2f}
"""
        
        return report

# Test function
def test_backtest_engine():
    """Test the backtest engine with sample data."""
    from strategies.base_strategy import MovingAverageCrossover
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    prices = [100]
    for i in range(1, 500):
        change = np.random.normal(0, 2)
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))  # Ensure price doesn't go below 1
    
    sample_data = {
        'open_time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 1)) for p in prices],
        'low': [p - abs(np.random.normal(0, 1)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 500)
    }
    df = pd.DataFrame(sample_data)
    
    # Create strategy and run backtest
    strategy = MovingAverageCrossover()
    strategy.set_parameters(fast_ma=10, slow_ma=20)
    
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run_backtest(strategy, df, "TEST", position_size=0.1)
    
    report = engine.generate_report(result, "MA Crossover", "TEST")
    print(report)
    
    return result

if __name__ == "__main__":
    test_backtest_engine()