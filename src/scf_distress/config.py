"""Load and hash the run configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(path: str | Path) -> str:
    """SHA-256 of the config file bytes; record this before scoring the test wave."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
