"""Config & paths helpers.

- Put environment-specific settings in configs/local.yaml and configs/colab.yaml
- Select config via:
    - env var: PROFILE=local|colab
    - or CLI:  --profile local|colab
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml

# Project root: <project>/
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

def load_config(profile: str | None = None) -> dict:
    profile = profile or os.getenv("PROFILE", "local")
    cfg_path = CONFIG_DIR / f"{profile}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path}. "
            "Set PROFILE=local|colab or pass --profile."
        )
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data or {}
