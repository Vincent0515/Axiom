from __future__ import annotations

from pathlib import Path
import yfinance as yf


def download_ohlcv(ticker: str, start: str, end: str) -> Path:
    """
    Download adjusted OHLCV data and save to data/raw as parquet.
    Uses a simple cache: if file already exists, returns it.
    """
    out_dir = Path("data") / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{ticker}_{start}_{end}.parquet"

    # Cache: if already downloaded, skip
    if out_path.exists():
        print("Using cached file:", out_path)
        return out_path

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    df = df.reset_index()
    df.to_parquet(out_path, index=False)

    print("Downloaded and saved:", out_path)
    return out_path


if __name__ == "__main__":
    path = download_ohlcv("AAPL", "2020-01-01", "2025-01-01")
    print("Done:", path)