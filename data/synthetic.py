"""
Synthetic BTC/USDT data generator.
Uses geometric Brownian motion with:
- Realistic volatility clustering (GARCH-like)
- Momentum bursts
- Volume correlation with price moves
Used for pipeline validation; swap with real Binance data in production.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_ohlcv(
    n_candles: int = 5000,
    start_price: float = 40000,
    timeframe_hours: int = 1,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)

    dt    = timeframe_hours / (24 * 365)  # fraction of a year
    mu    = 0.0003                         # slight upward drift
    sigma = 0.008                          # ~0.8% hourly vol (realistic BTC)

    closes = [start_price]
    vols   = []
    vol    = sigma

    # GARCH-like volatility clustering
    for i in range(n_candles - 1):
        shock   = np.random.randn()
        vol     = 0.94 * vol + 0.06 * abs(shock) * sigma
        vol     = np.clip(vol, sigma * 0.3, sigma * 4)
        ret     = mu + vol * shock
        closes.append(closes[-1] * (1 + ret))
        vols.append(vol)

    closes = np.array(closes)
    vols   = np.array([sigma] + vols)

    # Construct OHLCV from closes
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]

    highs  = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(n_candles)) * vols * 0.5)
    lows   = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(n_candles)) * vols * 0.5)

    # Volume: higher on big moves
    price_move  = np.abs(closes / opens - 1)
    base_volume = np.random.lognormal(mean=10, sigma=0.5, size=n_candles)
    volumes     = base_volume * (1 + price_move * 20)

    # Timestamps
    start_dt = datetime(2023, 1, 1)
    index    = [start_dt + timedelta(hours=i * timeframe_hours) for i in range(n_candles)]

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=pd.DatetimeIndex(index, name="timestamp"))

    return df


if __name__ == "__main__":
    df = generate_synthetic_ohlcv(5000)
    print(df.tail())
    print(f"Shape: {df.shape}")
