# portfolio/portfolio_manager.py
class PortfolioManager:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.positions = {}
        self.performance_metrics = {}
    
    def calculate_position_sizes(self, signals: dict, risk_per_trade=0.02) -> dict:
        """Calculate position sizes based on portfolio optimization"""
        total_equity = self.get_total_equity()
        position_sizes = {}
        
        for symbol, signal in signals.items():
            if signal['confidence'] > 0.7:  # Only high confidence signals
                risk_amount = total_equity * risk_per_trade
                position_size = risk_amount / signal['atr']  # ATR-based position sizing
                position_sizes[symbol] = position_size
        
        return position_sizes
    
    def rebalance_portfolio(self, target_allocations: dict):
        current_allocations = self.get_current_allocations()
        rebalance_orders = []
        
        for symbol, target_alloc in target_allocations.items():
            current_alloc = current_allocations.get(symbol, 0)
            if abs(current_alloc - target_alloc) > 0.05:  # 5% threshold
                rebalance_orders.append({
                    'symbol': symbol,
                    'side': 'BUY' if target_alloc > current_alloc else 'SELL',
                    'quantity': abs(target_alloc - current_alloc) * self.get_total_equity()
                })
        
        return rebalance_orders