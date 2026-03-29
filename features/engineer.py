import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  LAYER 1 — CLASSIC INDICATORS (context)
# ─────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_9"]  = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)
    df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(int)

    # crossover signals
    df["ema_9_21_cross"]  = np.sign(df["ema_9"] - df["ema_21"])
    df["ema_21_50_cross"] = np.sign(df["ema_21"] - df["ema_50"])
    df["price_vs_ema50"]  = (df["close"] - df["ema_50"]) / df["ema_50"]

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]        = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"]   = macd["MACDh_12_26_9"]
    df["macd_cross"]  = np.sign(df["macd"] - df["macd_signal"])

    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"]   = adx["ADX_14"]
    df["di_pos"] = adx["DMP_14"]
    df["di_neg"] = adx["DMN_14"]
    df["di_diff"] = df["di_pos"] - df["di_neg"]
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"]  = ta.rsi(df["close"], length=7)
    df["rsi_slope"] = df["rsi_14"].diff(3)

    stochrsi = ta.stochrsi(df["close"], length=14)
    df["stochrsi_k"] = stochrsi["STOCHRSIk_14_14_3_3"]
    df["stochrsi_d"] = stochrsi["STOCHRSId_14_14_3_3"]

    df["roc_5"]  = ta.roc(df["close"], length=5)
    df["roc_10"] = ta.roc(df["close"], length=10)
    df["roc_20"] = ta.roc(df["close"], length=20)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_pct"] = df["atr_14"] / df["close"]  # normalized

    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bb["BBU_20_2.0_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0_2.0"]
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["close"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["candle_range"] = (df["high"] - df["low"]) / df["close"]
    df["candle_body"]  = abs(df["close"] - df["open"]) / df["close"]
    df["upper_wick"]   = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"]
    df["lower_wick"]   = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["obv_slope"] = df["obv"].diff(5) / (df["obv"].abs().rolling(5).mean() + 1e-9)

    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma_20"] + 1e-9)
    df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)

    # VWAP deviation (rolling 24-period)
    df["vwap"] = (df["close"] * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]
    return df


# ─────────────────────────────────────────────
#  LAYER 2 — EDGE INDICATORS (conviction)
# ─────────────────────────────────────────────

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximated microstructure from OHLCV.
    Real CVD needs tick data — this is the best proxy from candle data.
    """
    # Buying pressure proxy: close relative to high-low range
    df["buy_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
    df["sell_pressure"] = 1 - df["buy_pressure"]

    # Cumulative Volume Delta approximation
    df["cvd_proxy"] = ((df["buy_pressure"] - 0.5) * 2 * df["volume"])
    df["cvd_cum"]   = df["cvd_proxy"].rolling(20).sum()
    df["cvd_slope"] = df["cvd_cum"].diff(5)

    # Trade flow imbalance proxy
    df["tfi"] = df["cvd_proxy"].rolling(10).mean() / (df["vol_ma_20"] + 1e-9)

    # Price vs open (candle direction strength)
    df["candle_dir"] = np.sign(df["close"] - df["open"])
    df["consecutive_dir"] = df["candle_dir"].rolling(5).sum()  # -5 to +5

    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """ML-native features the model can use directly."""
    returns = df["close"].pct_change()

    # Rolling statistics
    df["ret_mean_10"]  = returns.rolling(10).mean()
    df["ret_std_10"]   = returns.rolling(10).std()
    df["ret_skew_20"]  = returns.rolling(20).skew()
    df["ret_kurt_20"]  = returns.rolling(20).kurt()

    # Autocorrelation lag 1 (momentum persistence)
    df["autocorr_1"] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)

    # Hurst exponent (simplified via RS analysis)
    df["hurst"] = returns.rolling(50).apply(_hurst_simple, raw=True)

    # Return entropy (market randomness)
    df["entropy"] = returns.rolling(20).apply(_entropy, raw=True)

    # Z-score of price vs 50-period mean
    df["price_zscore"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).std() + 1e-9)

    return df


def _hurst_simple(series: np.ndarray) -> float:
    """Simplified Hurst exponent. <0.5 = mean-reverting, >0.5 = trending."""
    try:
        if len(series) < 20:
            return 0.5
        lags = range(2, min(20, len(series)//2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return 0.5


def _entropy(series: np.ndarray) -> float:
    """Approximate entropy — higher = more random/choppy market."""
    try:
        bins = np.histogram(series, bins=10)[0]
        probs = bins / bins.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-9))
    except:
        return 0.0


# ─────────────────────────────────────────────
#  LAYER 3 — REGIME DETECTION (meta)
# ─────────────────────────────────────────────

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple regime classification without HMM dependency.
    HMM layer added separately in models/.
    """
    # Trend regime: ADX > 25 = trending
    df["regime_trending"] = (df["adx"] > 25).astype(int)

    # Volatility regime: ATR % vs its own MA
    df["atr_ma"] = df["atr_pct"].rolling(50).mean()
    df["regime_high_vol"] = (df["atr_pct"] > df["atr_ma"] * 1.3).astype(int)

    # Squeeze regime: BB width at 20-period low
    df["bb_width_min"] = df["bb_width"].rolling(20).min()
    df["regime_squeeze"] = (df["bb_width"] <= df["bb_width_min"] * 1.05).astype(int)

    # Combined regime score: 0 = bad, 3 = ideal conditions
    df["regime_score"] = df["regime_trending"] + (1 - df["regime_high_vol"]) + (1 - df["regime_squeeze"])

    return df


# ─────────────────────────────────────────────
#  MASTER PIPELINE
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[Features] Building feature matrix...")

    df = add_trend_features(df)
    print("  ✓ Trend")
    df = add_momentum_features(df)
    print("  ✓ Momentum")
    df = add_volatility_features(df)
    print("  ✓ Volatility")
    df = add_volume_features(df)
    print("  ✓ Volume")
    df = add_microstructure_features(df)
    print("  ✓ Microstructure (CVD proxy)")
    df = add_statistical_features(df)
    print("  ✓ Statistical / ML-native")
    df = add_regime_features(df)
    print("  ✓ Regime")

    # Drop rows with NaN from rolling windows
    before = len(df)
    df = df.dropna()
    print(f"  ✓ Dropped {before - len(df)} NaN rows | Final shape: {df.shape}")

    return df


FEATURE_COLS = [
    # Trend
    "ema_9_21_cross", "ema_21_50_cross", "price_vs_ema50",
    "macd", "macd_hist", "macd_cross", "adx", "di_diff",
    # Momentum
    "rsi_14", "rsi_7", "rsi_slope", "stochrsi_k", "stochrsi_d",
    "roc_5", "roc_10", "roc_20",
    # Volatility
    "atr_pct", "bb_width", "bb_position", "candle_range",
    "candle_body", "upper_wick", "lower_wick",
    # Volume
    "obv_slope", "vol_ratio", "vol_spike", "vwap_dev",
    # Microstructure
    "buy_pressure", "cvd_cum", "cvd_slope", "tfi", "consecutive_dir",
    # Statistical
    "ret_mean_10", "ret_std_10", "ret_skew_20", "ret_kurt_20",
    "autocorr_1", "hurst", "entropy", "price_zscore",
    # Regime
    "regime_trending", "regime_high_vol", "regime_squeeze", "regime_score","price_above_ema200"
]


if __name__ == "__main__":
    df = pd.read_parquet("/home/claude/crypto_bot/data/raw_btc.parquet")
    df = build_features(df)
    df.to_parquet("/home/claude/crypto_bot/data/features_btc.parquet")
    print(f"\n[Features] Saved. Total features: {len(FEATURE_COLS)}")
