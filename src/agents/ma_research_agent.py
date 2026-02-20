from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.backtest.run_ma_backtest import run_ma_backtest


def main() -> None:
    # ------------------------------------------------------------
    # Agent Goal:
    # Automatically search MA window parameter and pick the best one.
    # "Best" here = highest Sharpe ratio (risk-adjusted return).
    # ------------------------------------------------------------

    # 1) Select one feature file to work on (later we can support multi-ticker)
    feature_files = list((Path("data") / "features").glob("*_feat.parquet"))
    if not feature_files:
        raise FileNotFoundError("No feature files found in data/features")

    feature_path = feature_files[0]

    # 2) Candidate parameter space (this is the agent's search space)
    windows = [10, 20, 30, 50, 100, 150, 200]

    # 3) Run experiments by calling infra tool (backtest function)
    results: list[dict] = []
    for w in windows:
        _, metrics = run_ma_backtest(feature_path, ma_window=w)
        results.append(metrics)

    # 4) Rank experiments and pick the best config
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    best = df.iloc[0].to_dict()

    # 5) Save artifacts (industrial habit)
    out_dir = Path("data") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "ma_sweep.csv", index=False)

    with (out_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # 6) Print a human-readable summary
    print("=== MA Research Agent Summary ===")
    print("Feature file:", feature_path)
    print("Tried windows:", windows)
    print("\nTop 5 by Sharpe:")
    print(df[["ma_window", "total_return", "max_drawdown", "sharpe"]].head(5))
    print("\nBest config saved to:", out_dir / "best_config.json")


if __name__ == "__main__":
    main()