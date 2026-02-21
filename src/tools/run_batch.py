from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    """
    Batch runner for running multiple experiment configs.
    This is infra glue: it orchestrates many agent runs reproducibly.
    """
    p = argparse.ArgumentParser(description="Run Axiom agents for all YAML configs in a folder.")
    p.add_argument("--configs_dir", default="configs", help="Folder containing *.yaml configs")
    p.add_argument("--features", default="data/features", help="Folder containing *_feat.parquet files")
    p.add_argument("--out_root", default="data/reports/batch", help="Root folder to store each run outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    configs_dir = Path(args.configs_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = sorted(configs_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No .yaml configs found in: {configs_dir}")

    # ------------------------------------------------------------
    # For each config:
    # - create a dedicated output folder
    # - call the agent via 'python -m ...' so imports work
    # ------------------------------------------------------------
    for cfg in configs:
        run_name = cfg.stem  # filename without extension
        run_out = out_root / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "src.agents.ma_research_agent",
            "--config", str(cfg),
            "--features", args.features,
            "--outdir", str(run_out),
        ]

        print("\n=== Running:", run_name, "===")
        print("Command:", " ".join(cmd))

        # Run and stream output. If any run fails, stop immediately.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Run failed for config={cfg} (exit={result.returncode})")

    print("\nAll runs completed. Output root:", out_root)


if __name__ == "__main__":
    main()