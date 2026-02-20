from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_features(raw_path: Path) -> Path:
    """
    Read raw OHLCV parquet, compute simple features, save to data/features.
    Features:
      - ret_1d: daily return based on Close
      - ma_20: 20-day moving average of Close
      - vol_20: 20-day rolling std of returns
    """
    df = pd.read_parquet(raw_path)

    # --- Normalize column names (handle yfinance multi-index saved as strings) ---
    cols = list(df.columns)

    def pick_col(prefix: str) -> str:
        # match exact "Date" if present
        if prefix in cols:
            return prefix
        # match stringified tuple like "('Close', 'AAPL')"
        candidates = [c for c in cols if isinstance(c, str) and c.startswith(f"('{prefix}'")]
        if not candidates:
            raise ValueError(f"Could not find a '{prefix}' column. Columns={cols}")
        return candidates[0]

    date_col = pick_col("Date")
    close_col = pick_col("Close")

    # rename to standard names
    df = df.rename(columns={date_col: "Date", close_col: "Close"})

    df = df.sort_values("Date")

    # 1) Daily return
    df["ret_1d"] = df["Close"].pct_change()

    # 2) Moving average of price
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # 3) Rolling volatility of returns
    df["vol_20"] = df["ret_1d"].rolling(window=20).std()

    out_dir = Path("data") / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / raw_path.name.replace(".parquet", "_feat.parquet")
    df.to_parquet(out_path, index=False)

    return out_path


if __name__ == "__main__":
    # Pick the first parquet file under data/raw and build features for it
    raw_files = list((Path("data") / "raw").glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError("No raw parquet files found in data/raw. Run download.py first.")

    out = build_features(raw_files[0])
    print("Saved features:", out)