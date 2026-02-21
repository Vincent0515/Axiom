from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize batch runs into a single CSV.")
    p.add_argument("--out_root", default="data/reports/batch", help="Root folder that contains run subfolders")
    return p.parse_args()


def summarize(out_root: str) -> Path:
    """
    Read best_config.json from each run folder and write a summary.csv.
    Returns the summary.csv path.
    """
    out_root_path = Path(out_root)
    if not out_root_path.exists():
        raise FileNotFoundError(f"out_root not found: {out_root_path}")

    rows: list[dict] = []
    for run_dir in sorted([p for p in out_root_path.iterdir() if p.is_dir()]):
        best_path = run_dir / "best_config.json"
        if not best_path.exists():
            continue

        with best_path.open("r", encoding="utf-8") as f:
            best = json.load(f)

        best["run_name"] = run_dir.name
        rows.append(best)

    if not rows:
        raise RuntimeError(f"No best_config.json found under: {out_root_path}")

    df = pd.DataFrame(rows)
    cols = ["run_name"] + [c for c in df.columns if c != "run_name"]
    df = df[cols]

    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False)

    out_path = out_root_path / "summary.csv"
    df.to_csv(out_path, index=False)

    print("Saved summary:", out_path)
    print(df[["run_name", "ma_window", "total_return", "max_drawdown", "sharpe"]])
    return out_path


def main() -> None:
    args = parse_args()
    summarize(args.out_root)


if __name__ == "__main__":
    main()