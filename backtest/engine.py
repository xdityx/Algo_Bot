import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import (
    ATR_SL_MULT, ATR_TP_MULT, MAX_POSITION,
    CONFIDENCE_THRESHOLD, TARGET_CANDLES
)


class BacktestEngine:
    """
    Event-driven backtester.
    - No lookahead: signals generated at candle close, executed at next open
    - ATR-based dynamic SL/TP
    - Confidence-weighted position sizing
    - Regime filter: only trades when regime_score >= 2
    """

    def __init__(self, initial_capital=10_000, fee=0.001, slippage=0.0005, threshold = None):
        self.initial_capital = initial_capital
        self.fee             = fee        # 0.1% per side
        self.slippage        = slippage   # 0.05% slippage
        self.threshold       = threshold or CONFIDENCE_THRESHOLD

    def run(self, df: pd.DataFrame, predictions: pd.Series) -> dict:
        df = df.copy()
        df["confidence"] = predictions

        capital    = self.initial_capital
        equity     = [capital]
        trades     = []
        in_trade   = False
        entry_price = sl = tp = 0
        trade_size  = 0

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]

            # ── Manage open trade ──────────────────────────
            if in_trade:
                exit_price  = None
                exit_reason = None
                hit_sl = (trade_direction == 1  and row["low"]  <= sl) or \
                         (trade_direction == -1 and row["high"] >= sl)
                hit_tp = (trade_direction == 1  and row["high"] >= tp) or \
                         (trade_direction == -1 and row["low"]  <= tp)

                if hit_sl:
                    exit_price  = sl
                    exit_reason = "SL"
                elif hit_tp:
                    exit_price  = tp
                    exit_reason = "TP"
                elif i == trade_entry_idx + TARGET_CANDLES:
                    exit_price  = row["open"]
                    exit_reason = "TIMEOUT"

                if exit_price:
                    pnl_pct  = trade_direction * (exit_price - entry_price) / entry_price
                    pnl_pct -= (self.fee * 2 + self.slippage * 2)
                    pnl      = trade_size * pnl_pct
                    capital += pnl

                    trades.append({
                        "entry_time":  df.index[trade_entry_idx],
                        "exit_time":   df.index[i],
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "exit_reason": exit_reason,
                        "pnl_pct":     pnl_pct,
                        "pnl":         pnl,
                        "confidence":  trade_confidence,
                        "regime":      trade_regime,
                    })
                    in_trade = False

            # ── Check for new signal ───────────────────────
            ema_200 = df["close"].ewm(span=200).mean()
            

            if (not in_trade
                and prev["confidence"] >= self.threshold
                and prev.get("regime_score", 3) >= 2):

                entry_price  = row["open"] * (1 + self.slippage)
                atr          = row["atr_14"]
                sl           = entry_price - ATR_SL_MULT * atr
                tp           = entry_price + ATR_TP_MULT * atr
                conf_scale   = (prev["confidence"] - self.threshold) / (1 - self.threshold)
                position_pct = MAX_POSITION * (0.5 + 0.5 * conf_scale)
                trade_size   = capital * position_pct
                trade_entry_idx  = i
                trade_confidence = prev["confidence"]
                trade_regime     = prev.get("regime_score", 3)
                trade_direction  = 1
                in_trade         = True

            equity.append(capital)

        return self._summarize(trades, equity, df)

    def _summarize(self, trades: list, equity: list, df: pd.DataFrame) -> dict:
        if not trades:
            print("[Backtest] No trades generated.")
            return {}

        trade_df = pd.DataFrame(trades)
        equity_s = pd.Series(equity, index=df.index[:len(equity)])

        wins     = trade_df[trade_df["pnl"] > 0]
        losses   = trade_df[trade_df["pnl"] <= 0]

        total_ret   = (equity[-1] - self.initial_capital) / self.initial_capital
        win_rate    = len(wins) / len(trade_df)
        avg_win     = wins["pnl_pct"].mean() if len(wins) else 0
        avg_loss    = losses["pnl_pct"].mean() if len(losses) else 0
        profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) else float("inf")

        # Sharpe (annualized, hourly)
        daily_ret   = equity_s.pct_change().dropna()
        sharpe      = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(24 * 365)

        # Max drawdown
        rolling_max = equity_s.cummax()
        drawdown    = (equity_s - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        # Exit breakdown
        exit_counts = trade_df["exit_reason"].value_counts()

        results = {
            "total_trades":   len(trade_df),
            "win_rate":       win_rate,
            "total_return":   total_ret,
            "final_capital":  equity[-1],
            "profit_factor":  profit_factor,
            "sharpe":         sharpe,
            "max_drawdown":   max_dd,
            "avg_win_pct":    avg_win,
            "avg_loss_pct":   avg_loss,
            "exit_breakdown": exit_counts.to_dict(),
            "trades":         trade_df,
            "equity_curve":   equity_s,
        }

        print("\n[Backtest] ─── Results ───────────────────────────")
        print(f"  Trades:         {results['total_trades']}")
        print(f"  Win Rate:       {win_rate*100:.1f}%")
        print(f"  Total Return:   {total_ret*100:.1f}%")
        print(f"  Profit Factor:  {profit_factor:.2f}")
        print(f"  Sharpe Ratio:   {sharpe:.2f}")
        print(f"  Max Drawdown:   {max_dd*100:.1f}%")
        print(f"  Avg Win:        {avg_win*100:.2f}%")
        print(f"  Avg Loss:       {avg_loss*100:.2f}%")
        print(f"  Exit Breakdown: {exit_counts.to_dict()}")
        print(f"  Final Capital:  ₹{equity[-1]:,.0f} (started ₹{self.initial_capital:,})")
        print("──────────────────────────────────────────────────\n")

        return results


if __name__ == "__main__":
    from models.trainer import load_model, predict
    from features.engineer import FEATURE_COLS

    df      = pd.read_parquet("/home/claude/crypto_bot/data/features_btc.parquet")
    model   = load_model()
    preds   = predict(model, df)
    engine  = BacktestEngine(initial_capital=10_000)
    results = engine.run(df, preds)
