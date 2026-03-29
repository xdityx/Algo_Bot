# ─────────────────────────────────────────────
#  CRYPTO BOT — MASTER CONFIG
# ─────────────────────────────────────────────

EXCHANGE       = "binance"
SYMBOL         = "BTC/USDT"
TIMEFRAME      = "4h"
LIMIT          = 1000
TOTAL_CANDLES  = 5000

# ── Target ──────────────────────────────────
TARGET_PCT     = 0.010         # 1.0% move = positive label
TARGET_CANDLES = 12            # look 8 candles ahead

# ── Risk ────────────────────────────────────
ATR_SL_MULT    = 0.8
ATR_TP_MULT    = 1.2
MAX_POSITION   = 0.10
CONFIDENCE_THRESHOLD = 0.38   # recalibrated for real data

# ── Walk-forward ────────────────────────────
TRAIN_PCT      = 0.60
VAL_PCT        = 0.20
TEST_PCT       = 0.20
N_SPLITS       = 5

# ── LightGBM ────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "is_unbalance":     False,
    "learning_rate":    0.03,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "n_estimators":     1000,
    "early_stopping_rounds": 50,
    "verbose":          -1,
    "scale_pos_weight": 3.0
}