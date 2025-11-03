# exchanges/multi_exchange_manager.py
class MultiExchangeManager:
    def __init__(self):
        self.exchanges = {
            'binance': BinanceExchange(),
            'bybit': BybitExchange(),
            'okx': OKXExchange()
        }
        self.unified_interface = UnifiedTradingInterface()
    
    async def get_best_price(self, symbol: str, side: str) -> dict:
        prices = {}
        for exchange_name, exchange in self.exchanges.items():
            try:
                price = await exchange.get_ticker_price(symbol)
                prices[exchange_name] = price
            except Exception as e:
                print(f"Error getting price from {exchange_name}: {e}")
        
        return self._find_best_price(prices, side)