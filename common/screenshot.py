"""Screenshot capture. The capture STRATEGY is fixed and must not be flipped:

  * Age-gate DETECTION  -> full_page by default (a gate can be below the fold;
                           a viewport crop risks a false negative — the worst
                           error for the core measurement).
  * Navigation GROUNDING -> viewport always (the model grounds on viewport-shaped
                           images; you can only click what is currently visible).

During COLLECTION we always capture BOTH at every checkpoint and store both, so
the detection-capture choice is a re-classification, never a re-crawl. Which
stored PNG the detector reads is the single `detection_capture` toggle in
config/global.yaml (read by 03_classify.py via detection_capture_kind()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from common import config


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def viewport(page: Any, out_path: Path | str) -> Path:
    """Capture only the current viewport (what's visible). For navigation."""
    p = _ensure_parent(out_path)
    page.screenshot(path=str(p), full_page=False)
    return p


def full_page(page: Any, out_path: Path | str) -> Path:
    """Capture the entire scrollable page. Default for age-gate detection."""
    p = _ensure_parent(out_path)
    page.screenshot(path=str(p), full_page=True)
    return p


def detection_capture_kind() -> str:
    """'full_page' | 'viewport' — the configured detector input. 03_classify uses
    this to pick which stored PNG to feed; collection ignores it (stores both)."""
    kind = config.load_global()["capture"]["detection_capture"]
    if kind not in ("full_page", "viewport"):
        raise ValueError(f"detection_capture must be full_page|viewport, got {kind!r}")
    return kind


def capture_for(
    page: Any,
    purpose: str,
    *,
    fullpage_path: Path | str | None = None,
    viewport_path: Path | str | None = None,
) -> Dict[str, Path]:
    """Capture the screenshot(s) appropriate for `purpose`.

    purpose:
      'navigation' -> {'viewport': ...}                 (viewport only)
      'collection' -> {'fullpage': ..., 'viewport': ...} (BOTH — the default at
                       every checkpoint, so detection stays a re-classification)
      'detection'  -> the single configured capture     (live one-off use)
    """
    if purpose == "navigation":
        if viewport_path is None:
            raise ValueError("navigation capture requires viewport_path")
        return {"viewport": viewport(page, viewport_path)}

    if purpose == "collection":
        if fullpage_path is None or viewport_path is None:
            raise ValueError("collection capture requires both paths")
        return {
            "fullpage": full_page(page, fullpage_path),
            "viewport": viewport(page, viewport_path),
        }

    if purpose == "detection":
        kind = detection_capture_kind()
        if kind == "full_page":
            if fullpage_path is None:
                raise ValueError("detection (full_page) requires fullpage_path")
            return {"fullpage": full_page(page, fullpage_path)}
        if viewport_path is None:
            raise ValueError("detection (viewport) requires viewport_path")
        return {"viewport": viewport(page, viewport_path)}

    raise ValueError(f"unknown capture purpose: {purpose!r}")
