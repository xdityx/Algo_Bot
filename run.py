"""
CRYPTO BOT — MASTER RUNNER
Run this file to execute the full pipeline:
  1. Fetch data
  2. Build features
  3. Train model (walk-forward)
  4. Backtest
  5. Print results + save equity curve
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config     import TARGET_PCT, TARGET_CANDLES, SYMBOL, TIMEFRAME
from data.pipeline      import fetch_ohlcv, add_target
from features.engineer  import build_features
from models.trainer     import walk_forward_train, predict, save_model
from backtest.engine    import BacktestEngine


def run_pipeline():
    print("=" * 55)
    print("  CRYPTO BOT — FULL PIPELINE")
    print(f"  {SYMBOL} | {TIMEFRAME} | Target: {TARGET_PCT*100:.1f}% in {TARGET_CANDLES} candles")
    print("=" * 55)

    # ── 1. Data ──────────────────────────────────────────
    print("\n[1/4] Fetching data...")
    try:
        df = fetch_ohlcv()
        print("[Data] Loaded from Binance")
    except Exception as e:
        print(f"[Data] Binance unavailable ({type(e).__name__}), using synthetic data")
        from data.synthetic import generate_synthetic_ohlcv
        df = generate_synthetic_ohlcv(5000)
    df = add_target(df, TARGET_PCT, TARGET_CANDLES)
    df.to_parquet("data/raw_btc.parquet")

    # ── 2. Features ───────────────────────────────────────
    print("\n[2/4] Engineering features...")
    df = build_features(df)
    df.to_parquet("data/features_btc.parquet")

    # ── 3. Train ──────────────────────────────────────────
    print("\n[3/4] Training model...")
    final_model, _, oos_preds, feat_imp = walk_forward_train(df)
    save_model(final_model, "models/lgbm_final.txt")

# ── 4. Backtest ───────────────────────────────────────
    print("\n[4/4] Running backtest...")
    predictions = predict(final_model, df)

    auto_threshold = float(predictions.quantile(0.97))
    print(f"[Backtest] Auto-calibrated threshold: {auto_threshold:.4f}")
    print(f"[Backtest] Signals at threshold: {(predictions >= auto_threshold).sum()}")

    engine  = BacktestEngine(initial_capital=10_000, threshold=auto_threshold)
    results = engine.run(df, predictions)

    if not results:
        print("No trades generated.")
        return

    # ── Plot equity curve ─────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Crypto Bot — {SYMBOL} {TIMEFRAME}", fontsize=14, fontweight="bold")

    # Equity curve
    eq = results["equity_curve"]
    axes[0].plot(eq, color="#00d4aa", linewidth=1.5)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Capital ($)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(10_000, color="gray", linestyle="--", alpha=0.5)

    # Drawdown
    rolling_max = eq.cummax()
    drawdown    = (eq - rolling_max) / rolling_max * 100
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#ff4444", alpha=0.5)
    axes[1].set_title("Drawdown %")
    axes[1].set_ylabel("%")
    axes[1].grid(True, alpha=0.3)

    # Feature importance (top 15)
    feat_imp.head(15).sort_values().plot(
        kind="barh", ax=axes[2], color="#4488ff"
    )
    axes[2].set_title("Top 15 Features (Gain)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    print("\n[Done] Plot saved to backtest_results.png")
    print("=" * 55)

    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()
