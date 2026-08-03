"""Unit test for the 1000x1000 -> pixel rescale. This is the load-bearing test:
if coords.py is wrong, every navigation click is wrong. Run: pytest -q
"""

import math

import pytest

from common.coords import CANVAS, box_center_to_pixel, model_to_pixel

# A deliberately non-square viewport so a transposed width/height bug shows up.
VW, VH = 1280, 800


def approx(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)


def test_origin_maps_to_origin():
    assert model_to_pixel(0, 0, VW, VH) == (0.0, 0.0)


def test_full_canvas_maps_to_full_viewport():
    px, py = model_to_pixel(CANVAS, CANVAS, VW, VH)
    assert approx(px, VW) and approx(py, VH)


def test_center_maps_to_center():
    px, py = model_to_pixel(500, 500, VW, VH)
    assert approx(px, VW / 2) and approx(py, VH / 2)


def test_axes_are_not_swapped():
    # x uses width, y uses height — a 100x900 point must respect each axis.
    px, py = model_to_pixel(100, 900, VW, VH)
    assert approx(px, 0.1 * VW)   # 128.0
    assert approx(py, 0.9 * VH)   # 720.0


def test_box_center():
    # Box spanning (200,400)->(600,800) has center (400,600) on the canvas.
    px, py = box_center_to_pixel([200, 400, 600, 800], VW, VH)
    assert approx(px, (400 / CANVAS) * VW)
    assert approx(py, (600 / CANVAS) * VH)


def test_out_of_range_is_clamped_by_default():
    # A stray model value beyond the canvas must not click off-screen.
    px, py = model_to_pixel(1200, -50, VW, VH)
    assert approx(px, VW) and approx(py, 0.0)


def test_clamp_can_be_disabled():
    px, py = model_to_pixel(1200, -50, VW, VH, clamp=False)
    assert approx(px, (1200 / CANVAS) * VW)
    assert approx(py, (-50 / CANVAS) * VH)


def test_zero_viewport_rejected():
    with pytest.raises(ValueError):
        model_to_pixel(500, 500, 0, 800)


def test_bad_box_rejected():
    with pytest.raises(ValueError):
        box_center_to_pixel([1, 2, 3], VW, VH)
