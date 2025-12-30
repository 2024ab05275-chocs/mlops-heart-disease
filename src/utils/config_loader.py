import json
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:
    """
    Load JSON configuration file.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        return json.load(f)
