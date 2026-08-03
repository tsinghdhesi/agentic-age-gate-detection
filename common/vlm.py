"""Local Ollama client — the ONLY inference surface. No cloud, ever.

Two calls to the same Qwen3-VL model:
  * detect_age_gate()    — frozen prompt, returns a binarized 1/0 plus the raw
                           output. Used by 03_classify.py.
  * decide_navigation()  — viewport screenshot + checkpoint goal, returns a
                           structured action on the 1000x1000 canvas. Used live
                           by 02_collect.py.

Qwen3-VL has a "thinking" mode; we disable it (think=False) so detection returns
a bare character and navigation returns bare JSON, not chain-of-thought.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ollama

from common import config
from common.prompts import DETECTION_PROMPT, NAV_FORMAT, navigation_prompt

ImageInput = Union[str, Path, bytes]

_VALID_ACTIONS = {"click", "type", "scroll", "done", "stuck"}


# --------------------------------------------------------------------------- #
# Client / helpers
# --------------------------------------------------------------------------- #
def _client() -> ollama.Client:
    host = config.load_global()["ollama"]["host"]
    return ollama.Client(host=host)


def _image_bytes(image: ImageInput) -> bytes:
    if isinstance(image, bytes):
        return image
    return Path(image).read_bytes()


def _resp_text(resp: Any) -> str:
    """Text of a generate() response, with a fallback to the `thinking` channel.

    Quirk: on this Ollama/qwen3-vl build, a `format=`-constrained generation
    routes the schema JSON into `thinking` and leaves `response` empty. So prefer
    `response`, but fall back to `thinking` when it is blank. Unconstrained
    detection (no format) returns a clean `response` and never hits the fallback.
    """
    if isinstance(resp, dict):
        text = resp.get("response") or ""
        thinking = resp.get("thinking") or ""
    else:
        text = getattr(resp, "response", "") or ""
        thinking = getattr(resp, "thinking", "") or ""
    return text if text.strip() else thinking


def _binarize(raw: str) -> Optional[int]:
    """First 0/1 character wins. Returns None if neither appears (logged raw)."""
    for ch in raw:
        if ch == "1":
            return 1
        if ch == "0":
            return 0
    return None


# --------------------------------------------------------------------------- #
# Detection (frozen instrument)
# --------------------------------------------------------------------------- #
def detect_age_gate(
    image: ImageInput, model: Optional[str] = None
) -> Dict[str, Any]:
    """Run the frozen detection prompt on one screenshot.

    Returns: {label: 1|0|None, raw: str, latency_ms: int, model: str}.
    `label` is binarized; `raw` is always preserved for the dataset.
    """
    model = model or config.load_global()["models"]["detection_default"]
    t0 = time.monotonic()
    resp = _client().generate(
        model=model,
        prompt=DETECTION_PROMPT,
        images=[_image_bytes(image)],
        think=False,
        options={"temperature": 0},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = _resp_text(resp).strip()
    return {"label": _binarize(raw), "raw": raw, "latency_ms": latency_ms, "model": model}


# --------------------------------------------------------------------------- #
# Navigation (operational)
# --------------------------------------------------------------------------- #
def decide_navigation(
    viewport_image: ImageInput, goal: str, model: Optional[str] = None
) -> Dict[str, Any]:
    """Ask the model for the next browser action toward `goal`.

    Returns: {action, box, text, raw, latency_ms, model}. `box` is the raw
    1000x1000 box (or None); the loop rescales via common.coords. An unparseable
    or malformed response is normalized to action="stuck".
    """
    model = model or config.load_global()["models"]["navigation"]
    t0 = time.monotonic()
    resp = _client().generate(
        model=model,
        prompt=navigation_prompt(goal),
        images=[_image_bytes(viewport_image)],
        format=NAV_FORMAT,
        think=False,
        options={"temperature": 0},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = _resp_text(resp).strip()
    action, box, text = _parse_nav(raw)
    return {
        "action": action,
        "box": box,
        "text": text,
        "raw": raw,
        "latency_ms": latency_ms,
        "model": model,
    }


def _parse_nav(raw: str):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ("stuck", None, "")
    action = obj.get("action")
    if action not in _VALID_ACTIONS:
        return ("stuck", None, "")
    box = obj.get("box")
    if not (isinstance(box, list) and len(box) == 4):
        box = None
    text = obj.get("text") or ""
    # click/type without a usable target can't be executed → treat as stuck.
    if action in ("click", "type") and box is None:
        return ("stuck", None, text)
    return (action, box, text)
