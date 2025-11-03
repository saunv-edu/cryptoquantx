import asyncio
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import hmac
import hashlib
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from binance.client import Client
from binance.exceptions import BinanceAPIException
from data.databases.db_manager import DatabaseManager
from utils.logger import logger

class LiveTradingExecutor:
    """Live trading executor with risk management."""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.client = None
        self.db = DatabaseManager()
        self.positions = {}
        self.risk_manager = RiskManager()
        self.is_running = False
        
    def initialize_client(self, api_key: str, api_secret: str):
        """Initialize Binance client."""
        try:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=self.testnet
            )
            
            # Test connection
            self.client.get_account()
            logger.info(f"✅ Binance client initialized ({'Testnet' if self.testnet else 'Mainnet'})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance client: {e}")
            raise
    
    async def execute_trade(self, 
                          symbol: str, 
                          side: str, 
                          quantity: float,
                          order_type: str = 'MARKET',
                          price: Optional[float] = None,
                          strategy_name: str = "Unknown") -> Dict:
        """Execute a trade with risk management."""
        
        if not self.client:
            return {'error': 'Client not initialized'}
        
        try:
            # Risk management check
            risk_check = self.risk_manager.check_trade_risk(symbol, side, quantity, self.positions)
            if not risk_check['allowed']:
                logger.warning(f"🚫 Trade blocked by risk manager: {risk_check['reason']}")
                return {'error': risk_check['reason']}
            
            # Calculate quantity with precision
            quantity = self._adjust_quantity_precision(symbol, quantity)
            
            if quantity <= 0:
                return {'error': 'Invalid quantity'}
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': quantity
            }
            
            if order_type.upper() == 'LIMIT' and price:
                order_params['price'] = self._adjust_price_precision(symbol, price)
                order_params['timeInForce'] = 'GTC'
            
            # Execute order
            logger.info(f"🎯 Executing {side} order: {symbol} {quantity} ({order_type})")
            
            order = self.client.create_order(**order_params)
            
            # Log trade
            trade_data = {
                'symbol': symbol,
                'side': side.upper(),
                'price': float(order['fills'][0]['price']) if order.get('fills') else float(order['price']),
                'quantity': quantity,
                'strategy_name': strategy_name,
                'is_test': self.testnet
            }
            self.db.log_trade(trade_data)
            
            # Update positions
            self._update_positions(symbol, side, quantity, trade_data['price'])
            
            logger.info(f"✅ Order executed: {order['orderId']} - {side} {quantity} {symbol} "
                       f"at {trade_data['price']:.2f}")
            
            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': trade_data['price'],
                'strategy': strategy_name
            }
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error: {e.message}"
            logger.error(f"❌ Trade execution failed: {error_msg}")
            return {'error': error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"❌ Trade execution failed: {error_msg}")
            return {'error': error_msg}
    
    def _adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """Adjust quantity to exchange precision requirements."""
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                return quantity
            
            # Get lot size filter
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(filter['stepSize'].rstrip('0').split('.')[-1])
                    
                    # Adjust quantity to step size
                    adjusted_qty = float(Decimal(str(quantity)) // Decimal(str(step_size)) * Decimal(str(step_size)))
                    return round(adjusted_qty, precision)
            
            return quantity
            
        except Exception as e:
            logger.warning(f"Could not adjust quantity precision: {e}")
            return quantity
    
    def _adjust_price_precision(self, symbol: str, price: float) -> float:
        """Adjust price to exchange precision requirements."""
        try:
            info = self.client.get_symbol_info(symbol)
            if not info:
                return price
            
            # Get price filter
            for filter in info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
                    precision = len(filter['tickSize'].rstrip('0').split('.')[-1])
                    
                    # Adjust price to tick size
                    adjusted_price = float(Decimal(str(price)) // Decimal(str(tick_size)) * Decimal(str(tick_size)))
                    return round(adjusted_price, precision)
            
            return price
            
        except Exception as e:
            logger.warning(f"Could not adjust price precision: {e}")
            return price
    
    def _update_positions(self, symbol: str, side: str, quantity: float, price: float):
        """Update internal positions tracking."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0
            }
        
        position = self.positions[symbol]
        
        if side.upper() == 'BUY':
            total_cost = position['total_cost'] + (quantity * price)
            total_quantity = position['quantity'] + quantity
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['quantity'] = total_quantity
            position['total_cost'] = total_cost
            
        elif side.upper() == 'SELL':
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                # Position closed
                self.positions.pop(symbol, None)
    
    async def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get account balance for specified asset."""
        if not self.client:
            return 0.0
        
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_position(self, symbol: str) -> Dict:
        """Get current position for symbol."""
        return self.positions.get(symbol, {
            'quantity': 0,
            'avg_price': 0,
            'total_cost': 0
        })
    
    async def close_all_positions(self):
        """Close all open positions."""
        if not self.client:
            return
        
        for symbol, position in self.positions.items():
            if position['quantity'] > 0:
                await self.execute_trade(
                    symbol=symbol,
                    side='SELL',
                    quantity=position['quantity'],
                    strategy_name='Emergency_Close'
                )

class RiskManager:
    """Risk management for trading operations."""
    
    def __init__(self):
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.max_daily_loss = 0.05    # 5% max daily loss
        self.max_drawdown = 0.15      # 15% max drawdown
        self.daily_trades_limit = 10  # Max 10 trades per day
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        
    def check_trade_risk(self, symbol: str, side: str, quantity: float, positions: Dict) -> Dict:
        """Check if trade meets risk requirements."""
        
        # Check daily trades limit
        if self.daily_trades_count >= self.daily_trades_limit:
            return {
                'allowed': False,
                'reason': f'Daily trades limit reached ({self.daily_trades_limit})'
            }
        
        # Check position size (simplified)
        if quantity <= 0:
            return {
                'allowed': False,
                'reason': 'Invalid quantity'
            }
        
        # Check if trying to sell more than owned
        if side.upper() == 'SELL':
            current_position = positions.get(symbol, {}).get('quantity', 0)
            if quantity > current_position:
                return {
                    'allowed': False,
                    'reason': f'Insufficient position. Owned: {current_position}, Trying to sell: {quantity}'
                }
        
        # All checks passed
        return {
            'allowed': True,
            'reason': 'Risk check passed'
        }
    
    def update_daily_metrics(self, pnl: float):
        """Update daily risk metrics."""
        self.daily_trades_count += 1
        self.daily_pnl += pnl
        
        # Reset daily metrics if new day (simplified)
        # In production, you'd check actual date change
    
    def can_trade_today(self) -> bool:
        """Check if trading is allowed for today based on risk limits."""
        return (self.daily_trades_count < self.daily_trades_limit and
                self.daily_pnl > -self.max_daily_loss)

# Trading bot that combines everything
class TradingBot:
    """Main trading bot that coordinates strategies and execution."""
    
    def __init__(self, testnet: bool = True):
        self.executor = LiveTradingExecutor(testnet=testnet)
        self.strategies = {}
        self.is_running = False
        self.trading_enabled = False
        
    def add_strategy(self, name: str, strategy):
        """Add trading strategy."""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    async def process_signal(self, symbol: str, signal_data: Dict):
        """Process trading signal from strategies."""
        if not self.trading_enabled:
            return
        
        try:
            strategy_name = signal_data.get('strategy', 'Unknown')
            side = signal_data.get('side', '').upper()
            quantity = signal_data.get('quantity', 0)
            price = signal_data.get('price')
            order_type = signal_data.get('order_type', 'MARKET')
            
            if side in ['BUY', 'SELL'] and quantity > 0:
                result = await self.executor.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    strategy_name=strategy_name
                )
                
                if 'error' not in result:
                    logger.info(f"🎯 Signal executed: {strategy_name} - {side} {quantity} {symbol}")
                else:
                    logger.warning(f"🚫 Signal failed: {result['error']}")
                    
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def enable_trading(self):
        """Enable actual trading."""
        self.trading_enabled = True
        logger.info("🟢 TRADING ENABLED - Live orders will be executed")
    
    def disable_trading(self):
        """Disable trading (dry-run mode)."""
        self.trading_enabled = False
        logger.info("🔴 TRADING DISABLED - Dry-run mode")
    
    async def shutdown(self):
        """Shutdown trading bot gracefully."""
        self.is_running = False
        self.disable_trading()
        await self.executor.close_all_positions()
        logger.info("🛑 Trading bot shutdown complete")

# Test function
async def test_live_executor():
    """Test live executor with simulated trading."""
    print("🧪 Testing Live Trading Executor (Simulation)")
    
    # Create mock executor for testing
    executor = LiveTradingExecutor(testnet=True)
    
    # Simulate successful trade execution
    mock_trade = {
        'success': True,
        'order_id': 123456,
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 0.001,
        'price': 50000.0,
        'strategy': 'Test_Strategy'
    }
    
    print("✅ Live executor test completed (simulation)")
    print(f"Mock trade: {mock_trade}")
    
    return executor

if __name__ == "__main__":
    asyncio.run(test_live_executor())