import yaml
from pathlib import Path

def load_config(path: str | Path) -> dict:
    """
    Load YAML configuration file.
    Returns a dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)