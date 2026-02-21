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

    # Select one feature file to work on (later we can support multi-ticker)
    feature_files = list((Path("data") / "features").glob("*_feat.parquet"))
    if not feature_files:
        raise FileNotFoundError("No feature files found in data/features")

    feature_path = feature_files[0]

    # Candidate parameter space (this is the agent's search space)
    # ------------------------------------------------------------
    # Adaptive search (2-stage):
    # Stage 1: coarse search
    # Stage 2: refine around the best window from Stage 1
    # ------------------------------------------------------------
    coarse_windows = [10, 20, 50, 100, 200]

    # Stage 1 experiments
    results: list[dict] = []
    for w in coarse_windows:
        _, metrics = run_ma_backtest(feature_path, ma_window=w)
        results.append(metrics)

    df_stage1 = pd.DataFrame(results)

    # Pick best from Stage 1 by Sharpe (we'll apply risk filter later in final selection)
    best_stage1 = df_stage1.sort_values("sharpe", ascending=False).iloc[0]
    w_star = int(best_stage1["ma_window"])

    # Stage 2: refine around w_star (clamp to sensible bounds)
    low = max(5, w_star - 20)
    high = min(250, w_star + 20)

    refine_windows = list(range(low, high + 1, 5))  # step=5 is a good start

    # Run Stage 2 experiments (avoid duplicates)
    seen = set(df_stage1["ma_window"].astype(int).tolist())
    for w in refine_windows:
        if w in seen:
            continue
        _, metrics = run_ma_backtest(feature_path, ma_window=w)
        results.append(metrics)

    windows = sorted({m["ma_window"] for m in results})


    # Rank experiments and pick the best config
    # ------------------------------------------------------------
    # Risk-aware selection:
    # 1) Filter out strategies with too large drawdown (risk constraint)
    # 2) Choose the best Sharpe among the survivors
    # ------------------------------------------------------------
    df = pd.DataFrame(results)

    max_dd_limit = -0.30  # e.g. reject strategies worse than -30% drawdown
    survivors = df[df["max_drawdown"] >= max_dd_limit].copy()

    if len(survivors) > 0:
        ranked = survivors.sort_values("sharpe", ascending=False)
    else:
        # If everything fails the risk constraint, fall back to pure Sharpe ranking
        ranked = df.sort_values("sharpe", ascending=False)

    best = ranked.iloc[0].to_dict()

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
    print(ranked[["ma_window", "total_return", "max_drawdown", "sharpe"]].head(5))
    print("\nBest config saved to:", out_dir / "best_config.json")


if __name__ == "__main__":
    main()