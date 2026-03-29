import ccxt
import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import EXCHANGE, SYMBOL, TIMEFRAME, LIMIT, TOTAL_CANDLES


def fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, total=TOTAL_CANDLES) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance in paginated batches.
    Returns clean DataFrame with no lookahead contamination.
    """
    exchange = getattr(ccxt, EXCHANGE)({
        "enableRateLimit": True,
    })

    all_candles = []
    since = None

    print(f"[DataPipeline] Fetching {total} candles for {symbol} {timeframe}...")

    while len(all_candles) < total:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=LIMIT)
        if not candles:
            break
        all_candles = candles + all_candles if since else all_candles + candles

        # paginate backwards from first candle
        since = candles[0][0] - (LIMIT * _tf_to_ms(timeframe))
        fetched = len(all_candles)
        print(f"  → {fetched} candles fetched", end="\r")

        if len(candles) < LIMIT:
            break
        if len(all_candles) >= total:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.tail(total)

    print(f"\n[DataPipeline] Done. Shape: {df.shape} | From: {df.index[0]} → {df.index[-1]}")
    return df


def _tf_to_ms(timeframe: str) -> int:
    """Convert timeframe string to milliseconds."""
    mapping = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(timeframe, 3_600_000)


def add_target(df: pd.DataFrame, target_pct: float, target_candles: int) -> pd.DataFrame:
    """
    Binary target: 1 if price rises by target_pct% within next target_candles candles.
    Uses future max close — NO lookahead in model (label only used in training).
    """
    future_max = df["close"].shift(-1).rolling(target_candles).max().shift(-(target_candles - 1))
    df["target"] = ((future_max - df["close"]) / df["close"] >= target_pct).astype(int)

    # drop last N rows where target is NaN (no future data)
    df = df.iloc[:-target_candles]
    print(f"[Target] Label distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    return df


if __name__ == "__main__":
    from configs.config import TARGET_PCT, TARGET_CANDLES
    df = fetch_ohlcv()
    df = add_target(df, TARGET_PCT, TARGET_CANDLES)
    df.to_parquet("/home/claude/crypto_bot/data/raw_btc.parquet")
    print("[DataPipeline] Saved to data/raw_btc.parquet")
