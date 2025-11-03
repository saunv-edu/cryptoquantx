import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.collectors.binance_collector import BinanceDataCollector
from data.collectors.websocket_collector import BinanceWebSocketCollector, AlertManager
from patterns.pattern_monitor import PatternMonitor
from data.calculators.indicator_calculator import IndicatorCalculator
from strategies.base_strategy import StrategyFactory
from backtesting.backtest_engine import BacktestEngine
from strategies.optimizer import StrategyOptimizer
from ml.price_predictor import PricePredictor, MLStrategy
from execution.live_executor import LiveTradingExecutor, TradingBot
from utils.logger import logger

async def setup_database():
    """Setup database."""
    logger.info("Setting up database...")
    from data.databases.db_manager import DatabaseManager
    db_manager = DatabaseManager()
    logger.info("Database setup completed")

async def collect_historical_data():
    """Collect historical data."""
    async with BinanceDataCollector() as collector:
        from config.settings import settings
        symbols = settings.DEFAULT_SYMBOLS
        intervals = ["1m", "5m", "15m", "1h", "4h"]
        
        logger.info("Starting historical data collection...")
        await collector.collect_historical_data(symbols, intervals, days=60)  # More data for ML
        logger.info("Historical data collection completed!")

async def calculate_indicators():
    """Calculate technical indicators."""
    calculator = IndicatorCalculator()
    from config.settings import settings
    
    logger.info("Starting indicator calculation...")
    
    for symbol in settings.DEFAULT_SYMBOLS:
        for interval in ["15m", "1h", "4h"]:
            logger.info(f"Calculating indicators for {symbol} {interval}...")
            calculator.calculate_for_symbol(symbol, interval, limit=1000)
    
    logger.info("Indicator calculation completed!")

async def train_ml_models():
    """Train machine learning models for prediction."""
    from config.settings import settings
    from data.databases.db_manager import DatabaseManager
    
    logger.info("🤖 Training ML models...")
    
    db = DatabaseManager()
    predictor = PricePredictor()
    
    # Train models for major symbols
    for symbol in settings.DEFAULT_SYMBOLS[:2]:  # BTC and ETH
        for interval in ["1h", "4h"]:
            df = db.get_klines_df(symbol, interval, limit=1000)
            
            if len(df) > 200:
                logger.info(f"Training ML model for {symbol} {interval}...")
                predictor.train_models(df, symbol, interval)
    
    # Save trained models
    predictor.save_models()
    
    logger.info("✅ ML model training completed!")
    return predictor

async def run_backtests():
    """Run backtests for all strategies including ML."""
    from config.settings import settings
    from data.databases.db_manager import DatabaseManager
    
    logger.info("🧪 Running comprehensive backtests...")
    
    db = DatabaseManager()
    backtest_engine = BacktestEngine(initial_capital=10000)
    
    # Test traditional strategies
    strategies_to_test = ['MovingAverageCrossover', 'RSIStrategy']
    
    for symbol in settings.DEFAULT_SYMBOLS[:2]:
        for interval in ["1h", "4h"]:
            df = db.get_klines_df(symbol, interval, limit=500)
            
            if len(df) < 100:
                continue
                
            for strategy_name in strategies_to_test:
                try:
                    strategy = StrategyFactory.create_strategy(strategy_name)
                    result = backtest_engine.run_backtest(strategy, df, symbol, position_size=0.1)
                    
                    logger.info(f"Backtest {strategy_name} {symbol} {interval}: "
                               f"Return: {result.total_return:.2f}%, "
                               f"Trades: {result.total_trades}, "
                               f"Win Rate: {result.win_rate:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Backtest failed: {e}")
    
    # Test ML strategy if models are trained
    try:
        predictor = PricePredictor()
        predictor.load_models()
        
        if predictor.is_trained:
            ml_strategy = MLStrategy(predictor)
            
            for symbol in settings.DEFAULT_SYMBOLS[:2]:
                df = db.get_klines_df(symbol, "1h", limit=500)
                if len(df) > 100:
                    result = backtest_engine.run_backtest(ml_strategy, df, symbol, position_size=0.1)
                    logger.info(f"Backtest MLStrategy {symbol} 1h: "
                               f"Return: {result.total_return:.2f}%, "
                               f"Trades: {result.total_trades}, "
                               f"Win Rate: {result.win_rate:.1f}%")
                    
    except Exception as e:
        logger.warning(f"ML strategy backtest skipped: {e}")

async def setup_live_trading():
    """Setup live trading bot (disabled by default)."""
    from config.settings import settings
    
    logger.info("🤖 Setting up live trading bot...")
    
    # Initialize trading bot (DISABLED by default for safety)
    trading_bot = TradingBot(testnet=True)
    
    # Load ML models for strategies
    try:
        predictor = PricePredictor()
        predictor.load_models()
        
        if predictor.is_trained:
            ml_strategy = MLStrategy(predictor)
            trading_bot.add_strategy("ML_Strategy", ml_strategy)
            logger.info("✅ ML strategy loaded for live trading")
            
    except Exception as e:
        logger.warning(f"Could not load ML strategy: {e}")
    
    # Add traditional strategies
    for strategy_name in ['MovingAverageCrossover', 'RSIStrategy']:
        strategy = StrategyFactory.create_strategy(strategy_name)
        trading_bot.add_strategy(strategy_name, strategy)
    
    logger.info("🟡 Live trading bot READY (trading DISABLED for safety)")
    logger.info("💡 Call trading_bot.enable_trading() to activate live trading")
    
    return trading_bot

async def setup_real_time_monitoring():
    """Setup real-time monitoring and alerts."""
    from config.settings import settings
    
    logger.info("📡 Setting up real-time monitoring...")
    
    # Sử dụng Mainnet WebSocket
    ws_collector = BinanceWebSocketCollector(use_testnet=False)
    alert_manager = AlertManager()
    pattern_monitor = PatternMonitor()
    
    # Add pattern monitoring callback
    async def pattern_analysis_callback(kline_data):
        try:
            await pattern_monitor.analyze_new_data(kline_data)
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
    
    ws_collector.add_callback(pattern_analysis_callback)
    
    # Add ML prediction callback
    async def ml_prediction_callback(kline_data):
        try:
            from data.databases.db_manager import DatabaseManager
            from ml.price_predictor import PricePredictor
            
            db = DatabaseManager()
            predictor = PricePredictor()
            predictor.load_models()
            
            symbol = kline_data['symbol']
            interval = kline_data['interval']
            
            if predictor.is_trained and symbol in predictor.models:
                df = db.get_klines_df(symbol, interval, limit=100)
                if len(df) > 50:
                    prediction = predictor.predict(df, symbol)
                    
                    if 'error' not in prediction and prediction['confidence'] > 0.7:
                        logger.info(f"🤖 ML Prediction {symbol}: {prediction['direction']} "
                                   f"(confidence: {prediction['confidence']:.2f})")
                        
        except Exception as e:
            logger.debug(f"ML prediction callback skipped: {e}")
    
    ws_collector.add_callback(ml_prediction_callback)
    
    # Add some example alerts với giá thực tế từ Mainnet
    from data.databases.db_manager import DatabaseManager
    db = DatabaseManager()
    
    for symbol in settings.DEFAULT_SYMBOLS[:2]:  # Chỉ 2 symbols đầu
        # Lấy giá hiện tại để đặt alert hợp lý
        try:
            latest_kline = db.get_latest_kline(symbol, "1h")
            if latest_kline:
                current_price = latest_kline.close
                # Đặt alert ở mức ±5% so với giá hiện tại
                upper_alert = current_price * 1.05
                lower_alert = current_price * 0.95
                
                alert_manager.add_price_alert(
                    symbol=symbol,
                    condition='above',
                    price=upper_alert,
                    message=f"{symbol} broke above ${upper_alert:.0f}"
                )
                
                alert_manager.add_price_alert(
                    symbol=symbol,
                    condition='below', 
                    price=lower_alert,
                    message=f"{symbol} fell below ${lower_alert:.0f}"
                )
                
                logger.info(f"Set alerts for {symbol}: ${lower_alert:.0f} - ${upper_alert:.0f}")
        except Exception as e:
            logger.warning(f"Could not set alerts for {symbol}: {e}")
            # Đặt alert mặc định nếu không lấy được giá
            alert_manager.add_price_alert(
                symbol=symbol,
                condition='above',
                price=100000,
                message=f"{symbol} broke above $100,000"
            )
    
    # Add alert checking callback
    async def alert_checking_callback(kline_data):
        try:
            from data.databases.db_manager import DatabaseManager
            from data.calculators.indicator_calculator import IndicatorCalculator
            
            db = DatabaseManager()
            calculator = IndicatorCalculator()
            
            symbol = kline_data['symbol']
            interval = kline_data['interval']
            
            # Get latest indicators
            df = db.get_klines_df(symbol, interval, limit=50)
            if len(df) > 20:
                df_with_indicators = calculator.calculate_all_indicators(df)
                latest_indicators = df_with_indicators.iloc[-1].to_dict()
                
                # Check alerts
                await alert_manager.check_alerts(
                    symbol, kline_data['close'], latest_indicators
                )
        except Exception as e:
            logger.error(f"Alert checking error: {e}")
    
    ws_collector.add_callback(alert_checking_callback)
    
    # Add simple logging callback
    async def logging_callback(kline_data):
        logger.info(f"📊 {kline_data['symbol']} {kline_data['interval']}: ${kline_data['close']:.2f}")
    
    ws_collector.add_callback(logging_callback)

    # Start monitoring
    monitoring_task = asyncio.create_task(
        ws_collector.start_monitoring(
            symbols=settings.DEFAULT_SYMBOLS[:2],
            intervals=['1m', '5m']
        )
    )
    
    return monitoring_task, ws_collector

async def run_system():
    """Main function to run the complete trading system."""
    monitoring_task = None
    ws_collector = None
    
    try:
        logger.info("🚀 Starting Complete Crypto Trading System...")
        
        # Step 0: Setup database
        logger.info("🗄️ Step 0: Setting up database...")
        await setup_database()
        
        # Step 1: Collect historical data
        logger.info("📊 Step 1: Collecting historical data...")
        await collect_historical_data()
        
        # Step 2: Calculate indicators
        logger.info("📈 Step 2: Calculating indicators...")
        await calculate_indicators()
        
        # Step 3: Train ML models
        logger.info("🤖 Step 3: Training ML models...")
        await train_ml_models()
        
        # Step 4: Run backtests
        logger.info("🧪 Step 4: Running backtests...")
        await run_backtests()
        
        # Step 5: Setup live trading (disabled)
        logger.info("💼 Step 5: Setting up live trading...")
        trading_bot = await setup_live_trading()
        
        # Step 6: Start real-time monitoring
        logger.info("📡 Step 6: Starting real-time monitoring...")
        monitoring_task, ws_collector = await setup_real_time_monitoring()
        
        logger.info("✅ SYSTEM READY!")
        logger.info("🎯 Real-time monitoring ACTIVE")
        logger.info("🤖 ML models TRAINED") 
        logger.info("💼 Live trading READY (disabled for safety)")
        logger.info("=" * 60)
        logger.info("💡 To enable live trading, call: trading_bot.enable_trading()")
        logger.info("=" * 60)
        
        # Keep system running
        await asyncio.sleep(3600)  # Run for 1 hour
        
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running system: {e}")
    finally:
        # Cleanup
        if ws_collector:
            ws_collector.stop_monitoring()
        if monitoring_task:
            monitoring_task.cancel()
        logger.info("🔚 System shutdown completed")

if __name__ == "__main__":
    # Install required packages for new features
    try:
        import sklearn
        import xgboost
        from binance.client import Client
    except ImportError:
        logger.warning("Some required packages missing. Install with: "
                      "pip install scikit-learn xgboost python-binance")
    
    asyncio.run(run_system())