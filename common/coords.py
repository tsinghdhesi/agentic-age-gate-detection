"""Qwen3-VL coordinate rescaling — the ONE place this conversion lives.

Qwen3-VL emits spatial coordinates on a normalized 1000x1000 canvas.
They are NOT pixels and NOT 0-1 floats. Before Playwright can act on any
coordinate the model returns, it MUST be rescaled to the real viewport:

    actual_x = (model_x / 1000) * viewport_width_px
    actual_y = (model_y / 1000) * viewport_height_px

Getting this wrong silently lands every click in the wrong place, so this
helper is the single source of truth. Import from here everywhere; never
re-derive the math inline. See tests/test_coords.py for the contract.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Qwen3-VL normalized canvas extent, per axis (0..CANVAS inclusive).
CANVAS: int = 1000


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def model_to_pixel(
    model_x: float,
    model_y: float,
    viewport_w: int,
    viewport_h: int,
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Convert one point on the 1000x1000 canvas to viewport pixels.

    Args:
        model_x, model_y: coordinate on the 1000x1000 canvas.
        viewport_w, viewport_h: real viewport size in pixels.
        clamp: if True (default), out-of-range canvas coords are clamped into
            [0, CANVAS] so a stray model value can never click off-screen.

    Returns:
        (px, py) as floats. Callers may round if they need ints.
    """
    if viewport_w <= 0 or viewport_h <= 0:
        raise ValueError(f"viewport must be positive, got {viewport_w}x{viewport_h}")
    if clamp:
        model_x = _clamp(model_x, 0, CANVAS)
        model_y = _clamp(model_y, 0, CANVAS)
    px = (model_x / CANVAS) * viewport_w
    py = (model_y / CANVAS) * viewport_h
    return (px, py)


def box_center_to_pixel(
    box: Sequence[float],
    viewport_w: int,
    viewport_h: int,
    *,
    clamp: bool = True,
) -> Tuple[float, float]:
    """Convert a [x1, y1, x2, y2] box on the 1000x1000 canvas to its pixel center.

    The navigation model returns a bounding box; we click its center. This is
    the function the action loop should call.
    """
    if len(box) != 4:
        raise ValueError(f"box must be [x1,y1,x2,y2], got {box!r}")
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return model_to_pixel(cx, cy, viewport_w, viewport_h, clamp=clamp)
