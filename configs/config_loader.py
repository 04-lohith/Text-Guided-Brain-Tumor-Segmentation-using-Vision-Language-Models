"""
configs/config_loader.py
Utility to load and merge YAML config with runtime overrides.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml


class DotDict(dict):
    """Dict subclass that supports attribute-style access."""
    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
            return DotDict(v) if isinstance(v, dict) else v
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def load_config(config_path: str | Path | None = None) -> DotDict:
    """
    Load the YAML config file and return a DotDict.

    Parameters
    ----------
    config_path : str | Path | None
        Path to config.yaml. If None, uses the default
        configs/config.yaml relative to this file.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = _to_dotdict(raw)

    # Ensure all output directories exist
    for key in ["checkpoints", "logs", "results"]:
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)

    return cfg


def _to_dotdict(d: Any) -> Any:
    if isinstance(d, dict):
        return DotDict({k: _to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_to_dotdict(item) for item in d]
    return d


def override_config(cfg: DotDict, overrides: dict) -> DotDict:
    """
    Apply flat key=value overrides using dot notation.
    E.g. overrides={'training.lr': 5e-5, 'dataset.binary_mask': True}
    """
    for dotkey, value in overrides.items():
        keys = dotkey.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg
