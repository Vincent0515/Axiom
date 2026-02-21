from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize batch runs into a single CSV.")
    p.add_argument("--out_root", default="data/reports/batch", help="Root folder that contains run subfolders")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)

    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    rows: list[dict] = []

    # Each run is a folder under out_root, containing best_config.json
    for run_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        best_path = run_dir / "best_config.json"
        if not best_path.exists():
            # Skip folders without the expected artifact
            continue

        with best_path.open("r", encoding="utf-8") as f:
            best = json.load(f)

        # Add run metadata
        best["run_name"] = run_dir.name
        rows.append(best)

    if not rows:
        raise RuntimeError(f"No best_config.json found under: {out_root}")

    df = pd.DataFrame(rows)

    # Put run_name first for readability
    cols = ["run_name"] + [c for c in df.columns if c != "run_name"]
    df = df[cols]

    # Sort by sharpe descending
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False)

    out_path = out_root / "summary.csv"
    df.to_csv(out_path, index=False)

    print("Saved summary:", out_path)
    print(df[["run_name", "ma_window", "total_return", "max_drawdown", "sharpe"]])


if __name__ == "__main__":
    main()