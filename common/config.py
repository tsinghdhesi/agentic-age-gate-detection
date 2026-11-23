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


# sites.industry (what 01_build_queue.py stamps, inferred from the seed
# filename) and the config/<name>.yaml FILENAME are two different namespaces
# that happen to collide for alcohol/gambling but not for nicotine: the
# default seed is "nicotine-industry-sites.csv" -> sites.industry="nicotine",
# but the checkpoint config is "vape.yaml" -> industry declares "vape". No
# existing script has had to bridge this before — every checkpoint-reading
# script so far (spike_navigation.py) just hardcodes "vape" directly. A
# collection script filtering `sites` by industry (a DB value) and also
# loading checkpoints by industry (a config filename) needs both directions
# reconciled explicitly, or it silently loads the wrong checkpoints / finds
# zero sites depending on which string it was given.
_SITES_INDUSTRY_TO_CONFIG_NAME = {
    "nicotine": "vape",
    # alcohol/gambling: sites.industry and the config filename already match
    # (both come from an "<name>-industry-sites.csv" seed convention), so no
    # entry is needed — the identity fallback below covers them.
}


def config_name_for_sites_industry(industry: str) -> str:
    """sites.industry value -> config/<name>.yaml filename to load
    checkpoints from. Identity for any industry not explicitly remapped
    above (alcohol, gambling — and nicotine's OWN config filename, should a
    future seed ever be built with --industry nicotine instead of relying
    on inference from the default filename)."""
    return _SITES_INDUSTRY_TO_CONFIG_NAME.get(industry, industry)


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
