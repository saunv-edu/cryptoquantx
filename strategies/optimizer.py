import itertools
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import StrategyFactory
from backtesting.backtest_engine import BacktestEngine
from utils.logger import logger

class StrategyOptimizer:
    """Optimize strategy parameters using grid search."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.backtest_engine = BacktestEngine(initial_capital)
    
    def grid_search(self, strategy_name: str, df: pd.DataFrame, symbol: str,
                   param_grid: Dict[str, List[Any]], position_size: float = 0.1) -> pd.DataFrame:
        """Perform grid search optimization."""
        
        results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Running grid search for {strategy_name} with {len(param_combinations)} combinations")
        
        for params in tqdm(param_combinations, desc="Optimizing"):
            param_dict = dict(zip(param_names, params))
            
            try:
                # Create strategy with current parameters
                strategy = StrategyFactory.create_strategy(strategy_name, **param_dict)
                
                # Run backtest
                result = self.backtest_engine.run_backtest(strategy, df, symbol, position_size)
                
                # Store results
                result_data = {
                    **param_dict,
                    'total_return': result.total_return,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'final_capital': result.final_capital
                }
                
                results.append(result_data)
                
            except Exception as e:
                logger.warning(f"Failed backtest with params {param_dict}: {e}")
                continue
        
        # Convert to DataFrame and sort by total return
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('total_return', ascending=False)
        
        return results_df
    
    def find_best_parameters(self, results_df: pd.DataFrame, 
                           min_trades: int = 5,
                           max_drawdown: float = -20.0) -> Dict[str, Any]:
        """Find the best parameters from optimization results."""
        
        if results_df.empty:
            return {}
        
        # Filter results by constraints
        filtered_df = results_df[
            (results_df['total_trades'] >= min_trades) &
            (results_df['max_drawdown'] >= max_drawdown)
        ]
        
        if filtered_df.empty:
            logger.warning("No results meet the constraints, using best overall")
            filtered_df = results_df
        
        # Get the best result
        best_result = filtered_df.iloc[0]
        
        # Extract parameters (exclude performance metrics)
        param_columns = [col for col in filtered_df.columns 
                        if col not in ['total_return', 'win_rate', 'total_trades', 
                                      'profit_factor', 'max_drawdown', 'sharpe_ratio', 
                                      'final_capital']]
        
        best_params = best_result[param_columns].to_dict()
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Performance: {best_result['total_return']:.2f}% return, "
                   f"{best_result['win_rate']:.1f}% win rate, "
                   f"{best_result['max_drawdown']:.2f}% max drawdown")
        
        return best_params

# Parameter grids for different strategies
PARAM_GRIDS = {
    'MovingAverageCrossover': {
        'fast_ma': [5, 10, 15, 20],
        'slow_ma': [20, 30, 50],
        'rsi_period': [14, 21],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75]
    },
    'RSIStrategy': {
        'rsi_period': [10, 14, 21],
        'rsi_oversold': [20, 25, 30, 35],
        'rsi_overbought': [65, 70, 75, 80]
    }
}

# Test function
def test_optimizer():
    """Test the strategy optimizer."""
    from strategies.base_strategy import MovingAverageCrossover
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='1H')
    prices = [100]
    for i in range(1, 300):
        change = np.random.normal(0, 1.5)
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))
    
    sample_data = {
        'open_time': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 300)
    }
    df = pd.DataFrame(sample_data)
    
    # Test optimization
    optimizer = StrategyOptimizer(initial_capital=10000)
    
    # Use smaller grid for testing
    test_grid = {
        'fast_ma': [10, 15],
        'slow_ma': [20, 30],
        'rsi_period': [14]
    }
    
    results = optimizer.grid_search('MovingAverageCrossover', df, "TEST", test_grid)
    
    print("Optimization Results:")
    print(results.head(10))
    
    best_params = optimizer.find_best_parameters(results)
    print(f"\nBest Parameters: {best_params}")
    
    return results

if __name__ == "__main__":
    test_optimizer()