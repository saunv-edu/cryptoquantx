from abc import ABC, abstractmethod
from binance.client import Client
from ...config.settings import settings
from ...data.databases.db_manager import DatabaseManager
from ...utils.logger import get_logger
from ...utils.risk_manager import RiskManager

logger = get_logger(__name__)

class BaseExecutor(ABC):
    def __init__(self):
        self.db = DatabaseManager()
        self.risk_manager = RiskManager()
        
        # Cấu hình Binance client
        self.client = Client(
            api_key=settings.BINANCE_API_KEY,
            api_secret=settings.BINANCE_SECRET_KEY,
            testnet=settings.BINANCE_TESTNET
        )

    @abstractmethod
    def execute_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET'):
        """Thực thi lệnh - abstract method để triển khai cụ thể."""
        pass

    def calculate_quantity(self, symbol: str, risk_per_trade: float = 0.02) -> float:
        """Tính toán khối lượng lệnh dựa trên quản lý rủi ro."""
        # Ví dụ đơn giản: sử dụng 2% vốn cho mỗi lệnh
        account_info = self.client.get_account()
        free_balance = float([asset for asset in account_info['balances'] if asset['asset'] == 'USDT'][0]['free'])
        return (free_balance * risk_per_trade)  # Trong thực tế, cần quy đổi sang base asset của symbol

class LiveExecutor(BaseExecutor):
    def execute_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET'):
        """Thực thi lệnh LIVE trên Binance."""
        try:
            # Kiểm tra rủi ro trước khi đặt lệnh
            if not self.risk_manager.check_risk(symbol, side, quantity):
                logger.warning(f"Risk check failed for {side} {quantity} {symbol}. Order cancelled.")
                return

            # Gửi lệnh
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            
            # Ghi log vào database
            self.db.log_trade(
                symbol=symbol,
                side=side,
                price=float(order['fills'][0]['price']),
                quantity=quantity,
                strategy_name="Live_Strategy"
            )
            
            logger.info(f"LIVE ORDER EXECUTED: {side} {quantity} {symbol} at {order['fills'][0]['price']}")
            return order

        except Exception as e:
            logger.error(f"Failed to execute live order: {e}")
            raise

class TestnetExecutor(BaseExecutor):
    def execute_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET'):
        """Thực thi lệnh trên Binance Testnet."""
        try:
            # Testnet không cần check risk khắt khe, nhưng vẫn log để phân tích
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            
            self.db.log_trade(
                symbol=symbol,
                side=side,
                price=float(order['fills'][0]['price']),
                quantity=quantity,
                strategy_name="Testnet_Strategy"
            )
            
            logger.info(f"TESTNET ORDER EXECUTED: {side} {quantity} {symbol} at {order['fills'][0]['price']}")
            return order

        except Exception as e:
            logger.error(f"Failed to execute testnet order: {e}")
            raise