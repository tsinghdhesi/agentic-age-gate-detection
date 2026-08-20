"""Unit tests for common/vlm.py — currently just classify_relevance(), the
LLM-based relevance judgment added alongside the switch away from a keyword
classifier. Mocks common.vlm._client() so the suite never touches a live
Ollama server; actual classification QUALITY can only be validated live
(see spike/spike_relevance.py) — these tests only exercise the wiring:
prompt/format construction and response parsing.
"""

from unittest.mock import MagicMock, patch

from common.vlm import _normalize_box, _parse_nav, _parse_relevance, classify_relevance


# --------------------------------------------------------------------------- #
# _parse_nav / _normalize_box — box-shape tolerance.
# --------------------------------------------------------------------------- #


def test_parse_nav_accepts_four_element_box():
    action, box, text = _parse_nav('{"action": "click", "box": [10, 20, 30, 40]}')
    assert action == "click"
    assert box == [10, 20, 30, 40]


def test_parse_nav_accepts_two_element_point():
    # Regression: Qwen3-VL frequently returns a POINT [x,y] instead of a
    # 4-element box. Observed live in the navigation spike — the model
    # correctly located an age-gate button but the point was rejected,
    # silently downgrading a good click to "stuck" and stalling the crawl.
    action, box, text = _parse_nav('{"action": "click", "box": [498, 501]}')
    assert action == "click"
    assert box == [498, 501, 498, 501]  # degenerate box; centre == the point


def test_normalize_box_point_centre_is_the_point():
    from common.coords import box_center_to_pixel

    box = _normalize_box([500, 500])
    # Centre of the widened box must be the original point, so a point-style
    # answer clicks exactly where the model pointed.
    assert box_center_to_pixel(box, 1000, 1000) == (500.0, 500.0)


def test_parse_nav_rejects_malformed_boxes():
    for bad in ("[1,2,3]", "[1,2,3,4,5]", '["a","b","c","d"]', "null", '"nope"', "{}"):
        _action, box, _text = _parse_nav(f'{{"action": "scroll", "box": {bad}}}')
        assert box is None, bad


def test_parse_nav_click_without_box_is_stuck():
    action, box, _ = _parse_nav('{"action": "click"}')
    assert action == "stuck"
    assert box is None


def test_parse_nav_booleans_are_not_valid_coordinates():
    # bool is a subclass of int in Python — must not slip through as a coord.
    _action, box, _text = _parse_nav('{"action": "click", "box": [true, false]}')
    assert box is None


def _fake_client(response_text: str, thinking_text: str = ""):
    client = MagicMock()
    client.generate.return_value = {"response": response_text, "thinking": thinking_text}
    return client


def test_classify_relevance_parses_valid_response():
    with patch("common.vlm._client", return_value=_fake_client('{"category": "retailer"}')):
        r = classify_relevance("some page text")
    assert r["category"] == "retailer"
    assert r["raw"] == '{"category": "retailer"}'
    assert r["model"] == "qwen3.6:27b"


def test_classify_relevance_falls_back_to_thinking_field():
    # Documented quirk (see vlm._resp_text): format=-constrained generations
    # sometimes route the JSON into `thinking` and leave `response` empty.
    with patch(
        "common.vlm._client",
        return_value=_fake_client("", thinking_text='{"category": "unrelated"}'),
    ):
        r = classify_relevance("some page text")
    assert r["category"] == "unrelated"


def test_classify_relevance_unparseable_json_returns_none_category():
    with patch("common.vlm._client", return_value=_fake_client("not json at all")):
        r = classify_relevance("some page text")
    assert r["category"] is None
    assert r["raw"] == "not json at all"


def test_classify_relevance_invalid_category_returns_none():
    # A category outside the 5 valid values must not be silently accepted.
    with patch("common.vlm._client", return_value=_fake_client('{"category": "definitely_not_valid"}')):
        r = classify_relevance("some page text")
    assert r["category"] is None


def test_classify_relevance_missing_category_key_returns_none():
    with patch("common.vlm._client", return_value=_fake_client('{"foo": "bar"}')):
        r = classify_relevance("some page text")
    assert r["category"] is None


def test_classify_relevance_respects_explicit_model_override():
    client = _fake_client('{"category": "retailer"}')
    with patch("common.vlm._client", return_value=client):
        r = classify_relevance("some page text", model="qwen3-vl:8b")
    assert r["model"] == "qwen3-vl:8b"
    assert client.generate.call_args.kwargs["model"] == "qwen3-vl:8b"


def test_classify_relevance_sends_no_images_kwarg():
    # This is a TEXT-only call — unlike detect_age_gate/decide_navigation,
    # no image bytes should ever be sent.
    client = _fake_client('{"category": "retailer"}')
    with patch("common.vlm._client", return_value=client):
        classify_relevance("some page text")
    assert "images" not in client.generate.call_args.kwargs


def test_classify_relevance_disables_thinking_and_uses_zero_temperature():
    client = _fake_client('{"category": "retailer"}')
    with patch("common.vlm._client", return_value=client):
        classify_relevance("some page text")
    kwargs = client.generate.call_args.kwargs
    assert kwargs["think"] is False
    assert kwargs["options"]["temperature"] == 0


def test_classify_relevance_includes_page_text_in_prompt():
    client = _fake_client('{"category": "retailer"}')
    with patch("common.vlm._client", return_value=client):
        classify_relevance("UNIQUE_MARKER_TEXT_12345")
    assert "UNIQUE_MARKER_TEXT_12345" in client.generate.call_args.kwargs["prompt"]


def test_parse_relevance_valid_categories():
    for cat in ("retailer", "brand_manufacturer", "marketplace", "informational", "unrelated"):
        assert _parse_relevance(f'{{"category": "{cat}"}}') == cat


def test_parse_relevance_none_on_garbage():
    assert _parse_relevance("") is None
    assert _parse_relevance("{") is None
    assert _parse_relevance("null") is None
