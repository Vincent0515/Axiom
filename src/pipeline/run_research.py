from __future__ import annotations

import argparse

from src.tools.run_batch import run_batch
from src.tools.summarize_runs import summarize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full Axiom research pipeline (batch + summarize).")
    p.add_argument("--configs_dir", default="configs", help="Folder containing *.yaml configs")
    p.add_argument("--features", default="data/features", help="Folder containing *_feat.parquet files")
    p.add_argument("--out_root", default="data/reports/batch", help="Output root folder for all runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Axiom Research Pipeline ===")
    print("Configs:", args.configs_dir)
    print("Features:", args.features)
    print("Out root:", args.out_root)

    out_root_path = run_batch(args.configs_dir, args.features, args.out_root)
    summary_path = summarize(str(out_root_path))

    print("\nPipeline complete.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()