"""Download every open data source into data/raw.

Each source is fetched from its canonical location (HDX resources are looked up through
the HDX API so renamed files keep working). A failed download keeps the previous copy,
so one flaky provider never stops the monthly refresh.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import RAW_DIR, ensure_dirs, settings

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "SokoRadar/1.0 (open-data early warning research; github.com/sokoradar)"}


def _get(url: str, timeout: int = 300, retries: int = 3) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # pragma: no cover - network dependent
            last = e
            time.sleep(2 ** attempt)
    raise last


def hdx_resource_url(dataset: str, resource_name: str) -> str:
    meta = _get(settings()["hdx_api"] + dataset, timeout=60).json()["result"]
    for res in meta["resources"]:
        if res["name"] == resource_name:
            return res["url"]
    raise KeyError(f"{resource_name} not found in HDX dataset {dataset}")


def pink_sheet_url() -> str:
    cfg = settings()["sources"]["pink_sheet"]
    page = _get(cfg["landing_page"], timeout=60).text
    m = re.search(cfg["link_pattern"], page)
    if not m:
        raise RuntimeError("Pink Sheet link not found on the World Bank commodity page")
    return m.group(0)


def _save(content: bytes, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(dest)


def fetch_source(key: str) -> Path:
    cfg = settings()["sources"][key]
    dest = RAW_DIR / cfg["file"]
    if "hdx_dataset" in cfg:
        url = hdx_resource_url(cfg["hdx_dataset"], cfg["resource_name"])
    elif key == "pink_sheet":
        url = pink_sheet_url()
    else:
        url = cfg["url"]
    log.info("downloading %s from %s", key, url)
    _save(_get(url).content, dest)
    return dest


def fetch_all(skip: tuple[str, ...] = ()) -> dict:
    """Download all sources; returns a manifest of what was refreshed."""
    ensure_dirs()
    manifest = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {}}
    for key in settings()["sources"]:
        if key in skip:
            continue
        try:
            path = fetch_source(key)
            manifest["sources"][key] = {"status": "ok", "bytes": path.stat().st_size}
        except Exception as e:  # pragma: no cover - network dependent
            dest = RAW_DIR / settings()["sources"][key]["file"]
            status = "kept previous copy" if dest.exists() else "missing"
            log.warning("could not refresh %s (%s); %s", key, e, status)
            manifest["sources"][key] = {"status": status, "error": str(e)[:200]}
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = [k for k, v in manifest["sources"].items() if v["status"] == "missing"]
    core = {"wfp_prices", "wfp_markets"}
    if core & set(missing):
        raise RuntimeError(f"core sources unavailable: {sorted(core & set(missing))}")
    return manifest
