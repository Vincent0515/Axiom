from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Axiom agents for all YAML configs in a folder.")
    p.add_argument("--configs_dir", default="configs", help="Folder containing *.yaml configs")
    p.add_argument("--features", default="data/features", help="Folder containing *_feat.parquet files")
    p.add_argument("--out_root", default="data/reports/batch", help="Root folder to store each run outputs")
    return p.parse_args()


def run_batch(configs_dir: str, features: str, out_root: str) -> Path:
    """
    Run MA research agent for every YAML config in configs_dir.
    Returns the output root folder path.
    """
    configs_path = Path(configs_dir)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    configs = sorted(configs_path.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No .yaml configs found in: {configs_path}")

    for cfg in configs:
        run_name = cfg.stem
        run_out = out_root_path / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "src.agents.ma_research_agent",
            "--config", str(cfg),
            "--features", features,
            "--outdir", str(run_out),
        ]

        print("\n=== Running:", run_name, "===")
        print("Command:", " ".join(cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Run failed for config={cfg} (exit={result.returncode})")

    print("\nAll runs completed. Output root:", out_root_path)
    return out_root_path


def main() -> None:
    args = parse_args()
    run_batch(args.configs_dir, args.features, args.out_root)


if __name__ == "__main__":
    main()