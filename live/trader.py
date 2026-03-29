"""
LIVE TRADING BOT
Runs every 4h, checks signal, places real orders on Binance Futures.

SETUP:
1. Create Binance Futures API key (enable Futures trading, NO withdrawal)
2. Set your keys in .env file:
   BINANCE_API_KEY=your_key
   BINANCE_SECRET=your_secret
3. Set TESTNET=True first to paper trade
4. Run: python live/trader.py
"""

import os
import time
import ccxt
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.config     import SYMBOL, TIMEFRAME, ATR_SL_MULT, ATR_TP_MULT, MAX_POSITION
from data.pipeline      import fetch_ohlcv
from features.engineer  import build_features, FEATURE_COLS

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TESTNET          = True        # ← SET TO FALSE FOR REAL MONEY
CAPITAL          = 10000         # USDT to allocate per trade
CONFIDENCE_QUANTILE = 0.97     # top 3% signals only
MODEL_PATH       = "models/lgbm_final.txt"
LOG_PATH         = "live/trade_log.csv"

# Load API keys from environment
API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_SECRET", "")


# ─────────────────────────────────────────────
#  EXCHANGE SETUP
# ─────────────────────────────────────────────

def get_exchange():
    exchange = ccxt.binance({
        "apiKey":  API_KEY,
        "secret":  API_SECRET,
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    if TESTNET:
        exchange.set_sandbox_mode(True)
        print("[Exchange] ⚠️  TESTNET MODE — no real money")
    else:
        print("[Exchange] 🔴 LIVE MODE — real money")
    return exchange


# ─────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────

def get_signal(model, threshold: float) -> dict:
    """Fetch latest candles, engineer features, generate signal."""
    print(f"\n[Signal] Fetching latest {TIMEFRAME} candles...")
    df = fetch_ohlcv(total=500)  # enough for all indicators

    from configs.config import TARGET_PCT, TARGET_CANDLES
    # Don't add target for live — just features
    df = build_features(df)

    if len(df) < 10:
        return {"action": "HOLD", "reason": "insufficient data"}

    latest     = df.iloc[-1]
    confidence = model.predict(df[FEATURE_COLS].tail(1))[0]

    atr        = latest["atr_14"]
    price      = latest["close"]
    regime     = latest.get("regime_score", 0)

    print(f"[Signal] Price: ${price:,.2f} | Confidence: {confidence:.4f} | "
          f"Threshold: {threshold:.4f} | Regime: {regime}/3")

    if confidence >= threshold and regime >= 2:
        sl = price - ATR_SL_MULT * atr
        tp = price + ATR_TP_MULT * atr
        return {
            "action":     "BUY",
            "price":      price,
            "confidence": confidence,
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
            "atr":        atr,
            "regime":     regime,
        }

    return {
        "action":     "HOLD",
        "confidence": confidence,
        "price":      price,
        "reason":     f"confidence {confidence:.4f} < threshold {threshold:.4f}"
                      if confidence < threshold else "regime too low",
    }


# ─────────────────────────────────────────────
#  ORDER EXECUTION
# ─────────────────────────────────────────────

def place_order(exchange, signal: dict) -> dict:
    """Place market entry + SL/TP bracket orders."""
    symbol   = SYMBOL.replace("/", "")  # BTCUSDT for futures
    price    = signal["price"]
    sl       = signal["sl"]
    tp       = signal["tp"]

    # Position size in contracts
    quantity = round(CAPITAL / price, 3)
    quantity = max(quantity, 0.001)  # Binance minimum

    print(f"\n[Order] Placing BUY | Qty: {quantity} BTC | "
          f"Entry: ${price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}")

    try:
        # Market entry
        entry_order = exchange.create_market_buy_order(symbol, quantity)
        print(f"[Order] ✅ Entry filled: {entry_order['id']}")

        time.sleep(1)

        # Stop loss
        sl_order = exchange.create_order(
            symbol, "STOP_MARKET", "sell", quantity,
            params={"stopPrice": sl, "reduceOnly": True}
        )
        print(f"[Order] ✅ SL set at ${sl:,.2f}: {sl_order['id']}")

        # Take profit
        tp_order = exchange.create_order(
            symbol, "TAKE_PROFIT_MARKET", "sell", quantity,
            params={"stopPrice": tp, "reduceOnly": True}
        )
        print(f"[Order] ✅ TP set at ${tp:,.2f}: {tp_order['id']}")

        return {
            "status":       "filled",
            "entry_id":     entry_order["id"],
            "sl_id":        sl_order["id"],
            "tp_id":        tp_order["id"],
            "quantity":     quantity,
            "entry_price":  price,
            "sl":           sl,
            "tp":           tp,
        }

    except Exception as e:
        print(f"[Order] ❌ Failed: {e}")
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────
#  POSITION MONITOR
# ─────────────────────────────────────────────

def has_open_position(exchange) -> bool:
    """Check if bot already has an open position."""
    try:
        symbol    = SYMBOL.replace("/", "")
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if float(pos.get("positionAmt", 0)) != 0:
                print(f"[Position] Open position detected: {pos['positionAmt']} BTC")
                return True
        return False
    except Exception as e:
        print(f"[Position] Check failed: {e}")
        return True  # Assume open if check fails — safety first


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def log_signal(signal: dict, order: dict = None):
    """Append signal and order result to CSV log."""
    os.makedirs("live", exist_ok=True)
    row = {
        "timestamp":  datetime.utcnow().isoformat(),
        "action":     signal.get("action"),
        "price":      signal.get("price"),
        "confidence": signal.get("confidence"),
        "sl":         signal.get("sl"),
        "tp":         signal.get("tp"),
        "regime":     signal.get("regime"),
        "order_status": order.get("status") if order else "no_order",
        "entry_id":   order.get("entry_id", "") if order else "",
    }
    log_df = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, index=False)

    print(f"[Log] Saved to {LOG_PATH}")


# ─────────────────────────────────────────────
#  THRESHOLD CALIBRATION
# ─────────────────────────────────────────────

def calibrate_threshold(model) -> float:
    """Compute live threshold from recent data distribution."""
    print("[Calibrate] Computing threshold from recent data...")
    df    = fetch_ohlcv(total=500)
    df    = build_features(df)
    preds = model.predict(df[FEATURE_COLS])
    threshold = float(pd.Series(preds).quantile(CONFIDENCE_QUANTILE))
    print(f"[Calibrate] Threshold set to {threshold:.4f} (top {(1-CONFIDENCE_QUANTILE)*100:.0f}%)")
    return threshold


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run_live():
    print("=" * 55)
    print("  CRYPTO BOT — LIVE TRADER")
    print(f"  {SYMBOL} | {TIMEFRAME} | {'TESTNET' if TESTNET else '🔴 LIVE'}")
    print("=" * 55)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found at {MODEL_PATH}")
        print("[Error] Run python run.py first to train the model")
        return

    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"[Model] Loaded from {MODEL_PATH}")

    # Setup exchange
    exchange  = get_exchange()
    threshold = calibrate_threshold(model)

    tf_seconds = {
        "1h": 3600, "4h": 14400, "1d": 86400
    }.get(TIMEFRAME, 14400)

    print(f"\n[Bot] Running every {TIMEFRAME}. Press Ctrl+C to stop.\n")

    while True:
        try:
            now = datetime.utcnow()
            print(f"\n{'='*55}")
            print(f"[Bot] Cycle at {now.strftime('%Y-%m-%d %H:%M UTC')}")

            # Skip if already in position
            if has_open_position(exchange):
                print("[Bot] Position open — skipping signal check")
            else:
                signal = get_signal(model, threshold)
                print(f"[Bot] Signal: {signal['action']}")

                if signal["action"] == "BUY":
                    order = place_order(exchange, signal)
                    log_signal(signal, order)
                else:
                    log_signal(signal)
                    print(f"[Bot] Holding — {signal.get('reason', '')}")

            # Wait for next candle close
            print(f"[Bot] Next check in {TIMEFRAME}...")
            time.sleep(tf_seconds)

        except KeyboardInterrupt:
            print("\n[Bot] Stopped by user.")
            break
        except Exception as e:
            print(f"[Bot] Error: {e}")
            print("[Bot] Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_live()
