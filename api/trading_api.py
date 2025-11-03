# api/trading_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Trading System API")

class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    strategy: str

@app.post("/api/trade")
async def execute_trade(trade_request: TradeRequest):
    try:
        result = await trading_bot.execute_trade(
            symbol=trade_request.symbol,
            side=trade_request.side,
            quantity=trade_request.quantity,
            strategy_name=trade_request.strategy
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/strategies")
async def get_strategies():
    return {
        "strategies": StrategyFactory.get_available_strategies(),
        "active_strategies": trading_bot.get_active_strategies()
    }

@app.get("/api/performance/{strategy_name}")
async def get_strategy_performance(strategy_name: str):
    metrics = backtest_engine.get_strategy_metrics(strategy_name)
    return {"strategy": strategy_name, "metrics": metrics}