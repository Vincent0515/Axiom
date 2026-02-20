from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Max Drawdown measures the worst peak-to-trough drop in equity curve.
    Returned as a negative number (e.g., -0.22 means -22%).
    """
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    return float(drawdown.min())


def compute_sharpe(strategy_ret: pd.Series, annual_trading_days: int = 252) -> float:
    """
    Sharpe ratio (risk-free ~0): mean(strategy_ret) / std(strategy_ret) * sqrt(252).
    Higher is better, but can be misleading if overfit.
    """
    r = strategy_ret.dropna()
    if len(r) < 2:
        return 0.0
    std = r.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((r.mean() / std) * np.sqrt(annual_trading_days))


def run_ma_backtest(feature_path: Path, ma_window: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    MA Trend Strategy:
      - signal = 1 when Close > MA(window), else 0
      - strategy_ret = yesterday_signal * today_ret_1d   (avoid look-ahead bias)
      - equity starts at 1.0 and compounds over time

    Returns:
      - df: dataframe with signal/strategy_ret/equity columns
      - metrics: dict with total_return, max_drawdown, sharpe
    """
    # ------------------------------------------------------------
    # Load input features (produced by build_features.py)
    # ------------------------------------------------------------
    df = pd.read_parquet(feature_path).sort_values("Date")

    # ------------------------------------------------------------
    # Ensure we have the required columns
    # ------------------------------------------------------------
    required = ["Date", "Close", "ret_1d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features file: {missing}")

    # ------------------------------------------------------------
    # Compute MA(window) if not present (keeps this backtest reusable)
    # ------------------------------------------------------------
    ma_col = f"ma_{ma_window}"
    if ma_col not in df.columns:
        df[ma_col] = df["Close"].rolling(ma_window).mean()

    # ------------------------------------------------------------
    # Strategy signal: long when Close > MA(window)
    # ------------------------------------------------------------
    df["signal"] = (df["Close"] > df[ma_col]).astype(int)

    # ------------------------------------------------------------
    # Strategy returns: use yesterday's signal to avoid look-ahead bias
    # ------------------------------------------------------------
    df["strategy_ret"] = df["signal"].shift(1) * df["ret_1d"]

    # ------------------------------------------------------------
    # Equity curve: start at 1.0 and compound daily
    # ------------------------------------------------------------
    df["equity"] = (1.0 + df["strategy_ret"].fillna(0.0)).cumprod()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    total_return = float(df["equity"].iloc[-1] - 1.0)
    max_dd = compute_max_drawdown(df["equity"])
    sharpe = compute_sharpe(df["strategy_ret"])

    metrics = {
        "feature_file": str(feature_path),
        "ma_window": int(ma_window),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }
    return df, metrics


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Demo run: pick the first feature file and backtest MA20
    # ------------------------------------------------------------
    feature_files = list((Path("data") / "features").glob("*_feat.parquet"))
    if not feature_files:
        raise FileNotFoundError("No feature files found in data/features")

    df_out, metrics = run_ma_backtest(feature_files[0], ma_window=20)

    print("Metrics:", metrics)
    print(df_out[["Date", "signal", "strategy_ret", "equity"]].head(35))