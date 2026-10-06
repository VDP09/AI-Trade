"""Loading predictor settings from the TOML files in ``config/``.

``config/common.toml`` holds shared settings; ``config/<name>.toml`` holds one
predictor's settings and is deep-merged on top, so it can override any shared
key. Alpaca credentials are never stored in config files — they come from the
``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` environment variables.
"""

import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_SOURCES = ("alpaca", "yfinance")


def _read_toml(path):
    """Parse one TOML file.

    Args:
        path (pathlib.Path): File to read.

    Returns:
        dict: Parsed contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base, override):
    """Recursively merge two dicts; ``override`` wins on conflicts.

    Args:
        base (dict): Default values.
        override (dict): Values that replace or extend ``base``.

    Returns:
        dict: A new merged dict (inputs are not modified).
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(name, config_dir=CONFIG_DIR):
    """Load a predictor's settings merged over the shared settings.

    Args:
        name (str): Predictor name, e.g. ``"daily"`` → ``config/daily.toml``.
        config_dir (pathlib.Path): Directory holding the TOML files.

    Returns:
        dict: Merged settings. ``csv_file`` is resolved to an absolute
        :class:`pathlib.Path` (relative paths are taken from the repo root),
        and ``alpaca.api_key`` / ``alpaca.secret_key`` are filled from the
        environment.

    Raises:
        FileNotFoundError: If either TOML file is missing.
        ValueError: If ``data.source`` is not a supported data source.
    """
    cfg = _deep_merge(_read_toml(config_dir / "common.toml"), _read_toml(config_dir / f"{name}.toml"))

    source = cfg["data"]["source"]
    if source not in DATA_SOURCES:
        raise ValueError(f"config: data.source must be one of {DATA_SOURCES}, got {source!r}")

    cfg["csv_file"] = REPO_ROOT / cfg["csv_file"]
    cfg["alpaca"]["api_key"] = os.environ.get("ALPACA_API_KEY", "")
    cfg["alpaca"]["secret_key"] = os.environ.get("ALPACA_SECRET_KEY", "")
    return cfg
