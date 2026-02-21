from __future__ import annotations
from google import genai
from pathlib import Path

import re
import os
import pandas as pd

def call_gemini(prompt: str, model: str = "models/gemini-2.5-flash") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return resp.text or ""

def generate_gemini_summary(df: pd.DataFrame) -> str:
    table_csv = df.to_csv(index=False)

    prompt = f"""
You are a quant research assistant.

You are given backtest summary results as CSV. Your job:
1) Write a short Markdown memo (<= 250 words) comparing the configs.
2) Then output EXACTLY 3 new experiment configs as YAML code blocks.
   - Each YAML block must start with a comment line: # filename: <name>.yaml
   - Use forward slashes in paths (data/features/...)
   - Include keys: run_name, feature_file, ma_window, windows (list)
   - Make the 3 configs meaningfully different (aggressive / baseline / conservative)
3) Do not include any other code blocks besides those 3 YAML blocks.

CSV:
{df.to_csv(index=False)}
""".strip()

    return call_gemini(prompt, model="models/gemini-2.5-flash")


def generate_template_summary(df: pd.DataFrame) -> str:
    """
    Create a human-readable summary of the research results.
    This is a simple template version (no LLM).
    """
    lines = []
    lines.append("# Axiom Research Summary\n")
    lines.append("## Best Config Comparison\n")

    # Best row
    best = df.iloc[0]
    lines.append(f"* **Best strategy:** {best['run_name']}")
    lines.append(f"  * MA window: {best['ma_window']}")
    lines.append(f"  * Total return: {best['total_return']:.2f}")
    lines.append(f"  * Max drawdown: {best['max_drawdown']:.2f}")
    lines.append(f"  * Sharpe: {best['sharpe']:.2f}\n")

    lines.append("## Full Ranking")
    lines.append(df.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Insights (template)")
    lines.append("- Higher Sharpe indicates better risk-adjusted performance.")
    lines.append("- Compare conservative vs aggressive behavior.")
    lines.append("- Future experiments could explore narrower windows.\n")

    return "\n".join(lines)

def extract_yaml_blocks(md: str) -> list[tuple[str, str]]:
    """
    Extract YAML code blocks that start with:
    # filename: xxx.yaml
    Returns list of (filename, yaml_text)
    """
    blocks = []
    pattern = r"```yaml\s*(#\s*filename:\s*(.+?)\s*\n)(.*?)```"
    for m in re.finditer(pattern, md, flags=re.DOTALL | re.IGNORECASE):
        filename = m.group(2).strip()
        yaml_body = m.group(1) + m.group(3)
        blocks.append((filename, yaml_body.strip() + "\n"))
    return blocks


def write_next_configs(blocks: list[tuple[str, str]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, yaml_text in blocks:
        path = out_dir / filename
        path.write_text(yaml_text, encoding="utf-8")
        written.append(path)
    return written

def main() -> None:
    summary_path = Path("data/reports/batch/summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError("No summary.csv found—run research pipeline first.")

    df = pd.read_csv(summary_path)

    use_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if use_gemini:
        text = generate_gemini_summary(df)
    else:
        text = generate_template_summary(df)

    out_file = summary_path.parent / "report.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)

    print("Generated summary report:", out_file)

    # ============================
    # NEW: extract YAML configs
    # ============================

    blocks = extract_yaml_blocks(text)

    if blocks:
        next_dir = Path("configs/llm_next")
        written = write_next_configs(blocks, next_dir)

        print("Wrote next configs:")
        for p in written:
            print(" -", p)
    else:
        print("No YAML config blocks found in LLM output.")


if __name__ == "__main__":
    main()