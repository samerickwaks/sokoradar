"""Paths and settings shared by every stage of the pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("SOKORADAR_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STORE_DIR = Path(os.environ.get("SOKORADAR_STORE", DATA_DIR / "store"))
MODELS_DIR = Path(os.environ.get("SOKORADAR_MODELS", ROOT / "models"))
DOCS_DIR = ROOT / "docs"

PHASES = ["Normal", "Stress", "Alert", "Crisis"]
PHASES_SW = {"Normal": "Kawaida", "Stress": "Shinikizo", "Alert": "Tahadhari", "Crisis": "Hatari"}


@lru_cache(maxsize=1)
def settings() -> dict:
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, STORE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
