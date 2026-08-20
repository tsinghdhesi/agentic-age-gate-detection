"""Unit tests for common/relevance.py.

classify_html() (the ORIGINAL deterministic keyword classifier, no longer on
the production path — see the module docstring) is still tested below since
it's still real, correct code. fetch() tests mock urlopen; classify_page()
tests mock common.vlm.classify_relevance() — this file must never depend on
live internet access OR a live Ollama server. classify_page()'s actual
classification QUALITY can only be validated live (spike/spike_relevance.py).
"""

import socket
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from common.relevance import ClassifyResult, FetchResult, build_relevance_text, classify_html, classify_page, fetch


# --------------------------------------------------------------------------- #
# classify_html — English
# --------------------------------------------------------------------------- #


def test_classify_english_retailer():
    html = """
    <html><head><title>Best Vape Shop - Buy E-Liquid Online</title></head>
    <body><h1>Vape Shop</h1><p>Shop our e-liquid range.</p>
    <button>Add to Cart</button><span>$19.99</span></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"
    assert r.nicotine_hits > 0
    assert r.commerce_hits > 0


def test_classify_unrelated_no_nicotine_terms():
    html = "<html><head><title>Joe's BBQ &amp; Grill</title></head><body><p>Best ribs in town.</p></body></html>"
    r = classify_html(html)
    assert r.category == "unrelated"
    assert r.nicotine_hits == 0


def test_classify_informational_blog_about_vaping():
    html = """
    <html><head><title>Vape News Blog</title></head>
    <body><p>Powered by WordPress. Latest news about e-cigarette regulation
    and nicotine policy debates.</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "informational"
    assert r.commerce_hits == 0


def test_classify_brand_manufacturer():
    html = """
    <html><head><title>Acme Vapes - Official Website</title></head>
    <body><p>Acme is a leading e-cigarette brand. Find a retailer near you.</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "brand_manufacturer"


def test_classify_marketplace():
    html = """
    <html><head><title>VapeMarket</title></head>
    <body><p>The largest vape marketplace. Add to cart to buy from any vendor.
    Become a seller today.</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "marketplace"


def test_classify_block_page_takes_priority_over_nicotine_terms():
    # A block/CAPTCHA page might incidentally quote the site's own vape
    # content in a meta tag; the block marker must still win.
    html = """
    <html><head><meta name="description" content="vape shop nicotine e-liquid"></head>
    <body><h1>Sorry, you have been blocked</h1><p>Ray ID: abc123</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "error"


def test_classify_perfume_site_tobacco_note_is_not_a_false_positive():
    # Regression test for a real false positive found against 1111perfume.ae:
    # "Tobacco" is a common perfume fragrance NOTE name (e.g. Tom Ford's
    # "Tobacco Vanille"), so a perfume shop selling a scent literally named
    # "11 11 Tobacco" must not be classified as a nicotine retailer.
    html = """
    <html><head><title>11 11 Tobacco Eau de Parfum</title></head>
    <body><p>Oriental spicy unisex fragrance. Tobacco, cacao and vanilla notes.</p>
    <button>Add to cart</button></body></html>
    """
    r = classify_html(html)
    assert r.category == "unrelated"


def test_classify_genuine_tobacco_retailer_not_suppressed_by_fragrance_guard():
    # The fragrance-context guard must only fire when EVERY matched term is
    # tobacco-ambiguous. A genuine tobacco shop that also matches an
    # unambiguous term (e.g. "cigar") must still classify as a retailer even
    # if "fragrance" appears somewhere on the page (e.g. a cigar humidor
    # accessory description).
    html = """
    <html><head><title>Premium Cigar & Tobacco Shop</title></head>
    <body><p>Fine cigars, pipe tobacco, and humidor fragrance beads.</p>
    <button>Add to cart</button></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"


def test_classify_bot_check_page():
    html = "<html><body><p>Please confirm you are human. I am not a robot.</p></body></html>"
    r = classify_html(html)
    assert r.category == "error"


# --------------------------------------------------------------------------- #
# classify_html — non-English (the classifier's main precision/recall risk)
# --------------------------------------------------------------------------- #


def test_classify_german_retailer():
    html = """
    <html><head><title>E-Zigaretten kaufen</title></head>
    <body><p>Nikotin Liquid im Sortiment. In den Warenkorb legen.</p>
    <span>19,99&euro;</span></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"


def test_classify_bulgarian_retailer():
    html = """
    <html><head><title>Вейп магазин</title></head>
    <body><p>Никотинови соли и течности за вейп.</p>
    <button>Добави в количката</button></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"
    assert r.nicotine_hits > 0


def test_classify_polish_retailer():
    html = """
    <html><head><title>Sklep z e-papierosami</title></head>
    <body><p>Liquid nikotyna. Dodaj do koszyka.</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"


def test_classify_dutch_retailer():
    html = """
    <html><head><title>E-sigaret webshop</title></head>
    <body><p>Nicotine vloeistof kopen.</p><button>Toevoegen aan winkelwagen</button></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"


def test_all_language_groups_have_terms():
    from common.relevance import _NICOTINE_TERMS_BY_LANG

    for lang in ("en", "de", "fr", "it", "es", "pt", "ru", "pl", "nl", "bg"):
        assert lang in _NICOTINE_TERMS_BY_LANG
        assert len(_NICOTINE_TERMS_BY_LANG[lang]) >= 5


# --------------------------------------------------------------------------- #
# Extraction correctness
# --------------------------------------------------------------------------- #


def test_script_and_style_content_is_stripped():
    # A nicotine term ONLY inside <script>/<style> must not be counted —
    # otherwise analytics/ad script boilerplate could false-positive a site.
    html = """
    <html><head><style>.vape-nicotine-class { color: red; }</style>
    <script>var nicotine_tracker = "vape";</script></head>
    <body><p>Welcome to our store.</p></body></html>
    """
    r = classify_html(html)
    assert r.category == "unrelated"
    assert r.nicotine_hits == 0


def test_title_mentions_weighted_higher_than_body_mentions():
    title_html = "<html><head><title>Vape Shop</title></head><body><p>Welcome.</p></body></html>"
    body_html = "<html><head><title>Welcome</title></head><body><p>We sell vape products.</p></body></html>"
    title_result = classify_html(title_html)
    body_result = classify_html(body_html)
    assert title_result.nicotine_hits > body_result.nicotine_hits


def test_meta_description_is_read():
    html = '<html><head><meta name="description" content="Buy nicotine e-liquid online"></head><body></body></html>'
    r = classify_html(html)
    assert r.nicotine_hits > 0


def test_currency_digit_signal():
    html = "<html><body><p>vape e-liquid</p><p>R$50</p></body></html>"
    r = classify_html(html)
    assert r.commerce_hits > 0


def test_cart_link_in_body_counts_even_when_button_text_is_js_templated():
    # Regression test for a real miss found against holy-vape.fr: modern
    # JS-rendered storefronts often template the "Add to cart" BUTTON text
    # from a JS i18n object inside <script> (correctly stripped by the
    # extractor), while the nav-bar cart LINK label is still plain body
    # HTML. A bare "panier" in the nav must be enough on its own.
    html = """
    <html><head><script>var i18n = {ADD_CART: 'Ajouter au panier'};</script></head>
    <body><p>Boutique de vape et e-liquide.</p>
    <a class="header__cart"><span>Panier</span></a></body></html>
    """
    r = classify_html(html)
    assert r.category == "retailer"
    assert r.commerce_hits > 0


def test_shopify_platform_fingerprint_boosts_commerce():
    html = '<html><head><script src="https://cdn.shopify.com/s/foo.js"></script></head><body><p>vape shop</p></body></html>'
    r = classify_html(html)
    assert r.commerce_hits >= 2


def test_classify_html_never_raises_on_malformed_input():
    for garbage in ["<html", "<<<>>>", "", "not html at all", "<div><span></div>"]:
        r = classify_html(garbage)
        assert isinstance(r, ClassifyResult)
        assert r.category in (
            "retailer", "brand_manufacturer", "marketplace",
            "informational", "unrelated", "error",
        )


# --------------------------------------------------------------------------- #
# fetch() — mocked urlopen, never touches the real network.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, url: str = "https://example.com/", content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.status = status
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self._body

    def geturl(self):
        return self._url


def test_fetch_success():
    with patch("common.relevance.urlopen", return_value=_FakeResponse(b"<html>hi</html>")):
        r = fetch("https://example.com/")
    assert r.failure_reason is None
    assert r.html == "<html>hi</html>"
    assert r.http_status == 200


def test_fetch_403_is_blocked_bot():
    err = HTTPError("https://example.com/", 403, "Forbidden", Message(), None)
    with patch("common.relevance.urlopen", side_effect=err):
        r = fetch("https://example.com/")
    assert r.failure_reason == "blocked_bot"
    assert r.html is None


def test_fetch_429_is_blocked_bot():
    err = HTTPError("https://example.com/", 429, "Too Many Requests", Message(), None)
    with patch("common.relevance.urlopen", side_effect=err):
        r = fetch("https://example.com/")
    assert r.failure_reason == "blocked_bot"


def test_fetch_500_is_error():
    err = HTTPError("https://example.com/", 500, "Server Error", Message(), None)
    with patch("common.relevance.urlopen", side_effect=err):
        r = fetch("https://example.com/")
    assert r.failure_reason == "error"


def test_fetch_timeout():
    with patch("common.relevance.urlopen", side_effect=socket.timeout("timed out")):
        r = fetch("https://example.com/")
    assert r.failure_reason == "timeout"


def test_fetch_dns_failure_is_error():
    with patch("common.relevance.urlopen", side_effect=URLError("DNS lookup failed")):
        r = fetch("https://example.com/")
    assert r.failure_reason == "error"


def test_fetch_never_raises_on_unexpected_exception():
    with patch("common.relevance.urlopen", side_effect=RuntimeError("boom")):
        r = fetch("https://example.com/")
    assert r.failure_reason == "error"
    assert r.html is None


def test_fetch_truncates_at_max_bytes():
    body = b"x" * 1000
    with patch("common.relevance.urlopen", return_value=_FakeResponse(body)):
        r = fetch("https://example.com/", max_bytes=100)
    assert r.truncated is True
    assert len(r.html) == 100


def test_fetch_decodes_charset_from_content_type():
    body = "café".encode("latin-1")
    with patch(
        "common.relevance.urlopen",
        return_value=_FakeResponse(body, content_type="text/html; charset=latin-1"),
    ):
        r = fetch("https://example.com/")
    assert r.html == "café"


# --------------------------------------------------------------------------- #
# build_relevance_text — pure, deterministic.
# --------------------------------------------------------------------------- #


def test_build_relevance_text_includes_all_fields():
    text = build_relevance_text("My Title", "My meta", "My H1", "body content")
    assert "My Title" in text
    assert "My meta" in text
    assert "My H1" in text
    assert "body content" in text


def test_build_relevance_text_truncates_body():
    long_body = "x" * 10000
    text = build_relevance_text("t", "m", "h", long_body, max_body_chars=100)
    # The body excerpt portion must be capped; the whole assembled text can
    # exceed 100 chars (title/meta/h1/labels), so check the excerpt itself.
    body_line = [l for l in text.splitlines() if l.startswith("Body text")][0]
    excerpt = body_line.split(": ", 1)[1]
    assert len(excerpt) == 100


def test_build_relevance_text_strips_whitespace():
    text = build_relevance_text("  spaced title  ", "", "", "")
    assert "Title: spaced title" in text


# --------------------------------------------------------------------------- #
# classify_page — orchestration: fetch failure -> block check -> LLM.
# Mocks common.vlm.classify_relevance(); never touches a live Ollama server.
# --------------------------------------------------------------------------- #


def test_classify_page_fetch_failure_passes_through_no_llm_call():
    fr = FetchResult(html=None, http_status=403, final_url=None, failure_reason="blocked_bot")
    with patch("common.relevance.classify_relevance") as mock_llm:
        result = classify_page(fr)
    mock_llm.assert_not_called()
    assert result["category"] == "blocked"
    assert result["raw"] is None
    assert result["model"] is None


def test_classify_page_timeout_maps_to_unreachable():
    fr = FetchResult(html=None, http_status=None, final_url=None, failure_reason="timeout")
    result = classify_page(fr)
    assert result["category"] == "unreachable"


def test_classify_page_block_marker_short_circuits_no_llm_call():
    html = "<html><body><h1>Sorry, you have been blocked</h1></body></html>"
    fr = FetchResult(html=html, http_status=200, final_url=None, failure_reason=None)
    with patch("common.relevance.classify_relevance") as mock_llm:
        result = classify_page(fr)
    mock_llm.assert_not_called()
    assert result["category"] == "error"


def test_classify_page_calls_llm_on_normal_page():
    html = "<html><head><title>Vape Shop</title></head><body><p>Buy vapes here.</p></body></html>"
    fr = FetchResult(html=html, http_status=200, final_url=None, failure_reason=None)
    with patch(
        "common.relevance.classify_relevance",
        return_value={"category": "retailer", "raw": '{"category": "retailer"}', "latency_ms": 100, "model": "qwen3.6:27b"},
    ) as mock_llm:
        result = classify_page(fr)
    mock_llm.assert_called_once()
    assert result["category"] == "retailer"
    assert result["model"] == "qwen3.6:27b"
    # The text the LLM was given must be built from the extracted fields.
    called_text = mock_llm.call_args[0][0]
    assert "Vape Shop" in called_text
    assert "Buy vapes here" in called_text


def test_classify_page_passes_through_none_category_from_llm():
    # An unparseable LLM response (category=None) must not be silently
    # coerced to "unrelated" by classify_page — that's the caller's call.
    html = "<html><body><p>vape shop</p></body></html>"
    fr = FetchResult(html=html, http_status=200, final_url=None, failure_reason=None)
    with patch(
        "common.relevance.classify_relevance",
        return_value={"category": None, "raw": "garbage", "latency_ms": 50, "model": "qwen3.6:27b"},
    ):
        result = classify_page(fr)
    assert result["category"] is None
