"""Configuration defaults for strategy backtests."""

# Simulation start date (inclusive) in YYYY-MM-DD format.
# Feel free to change this to the date you want your backtests to begin.
SIMULATION_START_DATE = "2020-01-01"

INITIAL_CAPITAL_KRW = 10_000_000

BACKTEST_SLIPPAGE = {
    "buy_pct": 0.25,
    "sell_pct": 0.25,
}
