"""Local Ollama client — the ONLY inference surface. No cloud, ever.

Three calls:
  * detect_age_gate()     — Qwen3-VL, frozen prompt + screenshot, returns a
                            binarized 1/0 plus the raw output. Used by
                            03_classify.py.
  * decide_navigation()   — Qwen3-VL, viewport screenshot + checkpoint goal,
                            returns a structured action on the 1000x1000
                            canvas. Used live by 02_collect.py.
  * classify_relevance()  — a TEXT model (no image), frozen prompt + a
                            page's extracted text, returns a structured
                            category. Used by 01b_screen_relevance.py to
                            build the sampling frame.

Both Qwen3-VL and the text model support a "thinking" mode; we disable it
(think=False) everywhere so responses are bare output, not chain-of-thought.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ollama

from common import config
from common.prompts import (
    DETECTION_PROMPT,
    NAV_FORMAT,
    RELEVANCE_FORMAT,
    RELEVANCE_PROMPT,
    navigation_prompt,
)

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
    if not isinstance(obj, dict):
        # json.loads succeeds on any valid JSON scalar (null, "text", 42,
        # [1,2,3]) — none of those have .get(), so this must be checked
        # explicitly or a single degenerate model response crashes the loop.
        return ("stuck", None, "")
    action = obj.get("action")
    if action not in _VALID_ACTIONS:
        return ("stuck", None, "")
    box = _normalize_box(obj.get("box"))
    text = obj.get("text") or ""
    # click/type without a usable target can't be executed → treat as stuck.
    if action in ("click", "type") and box is None:
        return ("stuck", None, text)
    return (action, box, text)


def _normalize_box(box: Any) -> Optional[list]:
    """Accept both box shapes the model actually emits, return a 4-element box.

    Qwen3-VL frequently answers with a 2-element POINT [x, y] rather than the
    4-element [x1, y1, x2, y2] the prompt asks for — observed live in the
    navigation spike, where it correctly located an age-gate button but
    returned a point. Rejecting those silently downgraded a perfectly good
    click to "stuck", which would have stalled the crawl on any site with a
    blocking overlay.

    A point is widened to the degenerate box [x, y, x, y], whose centre is the
    point itself — so common.coords.box_center_to_pixel() keeps working
    unchanged and every downstream caller keeps its 4-element contract.
    """
    if not isinstance(box, list):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
        return None
    if len(box) == 4:
        return box
    if len(box) == 2:
        x, y = box
        return [x, y, x, y]
    return None


_VALID_RELEVANCE_CATEGORIES = {
    "retailer", "brand_manufacturer", "marketplace", "informational", "unrelated",
}


# --------------------------------------------------------------------------- #
# Frame relevance (frozen instrument) — TEXT model, no image.
# --------------------------------------------------------------------------- #
def classify_relevance(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Run the frozen relevance prompt on one page's extracted text (title +
    meta description + h1 + a body excerpt — see common.relevance.build_relevance_text).

    No image is sent; `model` should be a text-capable Ollama tag (default:
    config models.relevance, currently qwen3.6:27b — a dedicated text model,
    not a VLM used text-only).

    Returns: {category: str|None, raw: str, latency_ms: int, model: str}.
    `category` is None if the response was unparseable or not one of the five
    valid categories — never silently defaulted to "unrelated", so a caller
    can distinguish "the model said unrelated" from "the response was junk".
    """
    model = model or config.load_global()["models"]["relevance"]
    t0 = time.monotonic()
    resp = _client().generate(
        model=model,
        prompt=f"{RELEVANCE_PROMPT}\n\nPAGE TEXT:\n{text}",
        format=RELEVANCE_FORMAT,
        think=False,
        options={"temperature": 0},
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = _resp_text(resp).strip()
    category = _parse_relevance(raw)
    return {"category": category, "raw": raw, "latency_ms": latency_ms, "model": model}


def _parse_relevance(raw: str) -> Optional[str]:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        # json.loads succeeds on any valid JSON scalar (null, "text", 42,
        # [1,2,3]) — none of those have .get(), so this must be checked
        # explicitly or a single degenerate model response crashes the screen.
        return None
    category = obj.get("category")
    return category if category in _VALID_RELEVANCE_CATEGORIES else None
