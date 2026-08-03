"""YAML config loader: per-industry checkpoints + global flags, plus a stable
config hash recorded per run.

The frozen DETECTION PROMPT does not live here — it is a measurement instrument
and lives in a single dedicated module (added in Phase 2), hashed separately and
recorded as runs.detection_prompt_hash. Config here is operational knobs only.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # imported lazily so importing this module never requires PyYAML

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping, got {type(data).__name__}")
    return data


@lru_cache(maxsize=None)
def load_global() -> Dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "global.yaml")


@lru_cache(maxsize=None)
def load_industry(name: str) -> Dict[str, Any]:
    """Load config/<name>.yaml (e.g. 'vape', 'alcohol', 'gambling')."""
    return _load_yaml(CONFIG_DIR / f"{name}.yaml")


def load(industry: str) -> Dict[str, Any]:
    """Resolved view = global knobs + the industry's checkpoints.

    Returns a dict: {**global, "industry_name": <name>, "industry": <industry cfg>}.
    """
    g = load_global()
    ind = load_industry(industry)
    return {**g, "industry_name": industry, "industry": ind}


def checkpoints(industry: str) -> List[Dict[str, Any]]:
    """Ordered checkpoint dicts for an industry. Each has at least name/goal/terminal."""
    return list(load_industry(industry).get("checkpoints", []))


def terminal_checkpoints(industry: str) -> List[str]:
    """Names of checkpoints marked terminal:true — the structural no-action wall."""
    return [c["name"] for c in checkpoints(industry) if c.get("terminal")]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(industry: str) -> str:
    """sha256 of the resolved (global + industry) config — recorded as runs.config_hash."""
    return hashlib.sha256(_canonical(load(industry)).encode("utf-8")).hexdigest()
