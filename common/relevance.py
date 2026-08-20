"""Frame-relevance screening for `scripts/01b_screen_relevance.py`.

Two layers, deliberately kept apart:

- `fetch()` — the only function that touches the network. Plain HTTP GET, no
  browser, no screenshots, no Playwright. Talks to exactly one host, once,
  and never raises; failures come back as a FetchResult with a
  failure_reason drawn from the same vocabulary as db.FAILURE_REASONS. This
  is what makes screening the ENTIRE ~116k-site frame affordable at all:
  page load (~1-2s) is far cheaper than a Playwright goto (~7-9s), so
  dropping the browser lets the whole frame get screened instead of a
  sample — sampling moves downstream, to already-confirmed relevant sites.
- The JUDGMENT — is this page a nicotine retailer — is an LLM call
  (`common.vlm.classify_relevance()`, frozen `RELEVANCE_PROMPT`), not a
  keyword list. `classify_page()` is the orchestration entry point: fetch
  failure -> passthrough; block/bot-check page -> deterministic "error"
  (mechanical, not a judgment call, no LLM call needed); otherwise -> LLM.
  This is the single function both `01b_screen_relevance.py` and
  `spike/spike_relevance.py` call, so they can't drift out of sync.

`classify_html()` below is the ORIGINAL deterministic keyword/pattern
classifier. It is NOT on the production path anymore — a keyword list
structurally can't tell a perfume site's "Tobacco Vanille" fragrance note
from an actual tobacco retailer, which is exactly the failure mode that
motivated switching to an LLM judgment. Kept in the file because it's fully
tested and useful for fast local iteration without a live Ollama call, not
because anything currently calls it in production.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common.vlm import classify_relevance

RULESET_VERSION = 1  # classify_html()'s own version marker; unrelated to
                     # sites.relevance_prompt_hash, which tracks RELEVANCE_PROMPT.

CATEGORIES = (
    "retailer",
    "brand_manufacturer",
    "marketplace",
    "informational",
    "unrelated",
    "error",
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# fetch() — the only network-touching function in this module.
# --------------------------------------------------------------------------- #


@dataclass
class FetchResult:
    html: Optional[str]
    http_status: Optional[int]
    final_url: Optional[str]
    failure_reason: Optional[str]  # None on success; else db.FAILURE_REASONS vocabulary
    truncated: bool = False


def _extract_charset(content_type: str) -> Optional[str]:
    m = re.search(r"charset=([\w\-]+)", content_type, re.IGNORECASE)
    return m.group(1) if m else None


def fetch(
    url: str,
    *,
    timeout_s: float = 10,
    max_bytes: int = 5_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """Plain HTTP(S) GET, no browser, no JS execution. urllib follows
    redirects for GET by default. Reads at most `max_bytes` of body — never
    unboundedly buffers a response. Never raises: every failure mode comes
    back as a FetchResult with failure_reason set, using the same vocabulary
    db.FAILURE_REASONS uses, so callers can write it straight into a row.

    Deliberately does NOT send "Accept-Encoding: gzip" — most servers then
    respond with plain text, avoiding the need to handle compressed-body
    decoding here at the cost of somewhat higher bandwidth. A reasonable
    trade for a screen that touches ~116k hosts exactly once each.
    """
    req = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "*",
        },
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            status = resp.status
            final_url = resp.geturl()
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_bytes + 1)
    except HTTPError as e:
        reason = "blocked_bot" if e.code in (403, 429) else "error"
        return FetchResult(None, e.code, None, reason)
    except (socket.timeout, TimeoutError):
        return FetchResult(None, None, None, "timeout")
    except URLError as e:
        reason = "timeout" if isinstance(e.reason, (socket.timeout, TimeoutError)) else "error"
        return FetchResult(None, None, None, reason)
    except Exception:  # noqa: BLE001 — this is a best-effort frame screen
        return FetchResult(None, None, None, "error")

    truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    charset = _extract_charset(content_type) or "utf-8"
    try:
        html = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        html = raw.decode("utf-8", errors="replace")

    return FetchResult(
        html=html,
        http_status=status,
        final_url=final_url,
        failure_reason=None,
        truncated=truncated,
    )


# --------------------------------------------------------------------------- #
# Lightweight HTML text extraction — no external dependency (BeautifulSoup
# is not in requirements.txt; stdlib html.parser is sufficient here since we
# only need title/meta/h1/body text, not a DOM).
# --------------------------------------------------------------------------- #


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.h1: list[str] = []
        self.body_text: list[str] = []
        self.meta_description: Optional[str] = None
        self._in_title = False
        self._in_h1 = False
        self._skip_depth = 0  # inside script/style/noscript

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "h1":
            self._in_h1 = True
        elif tag == "meta":
            attrs_d = dict(attrs)
            name = (attrs_d.get("name") or attrs_d.get("property") or "").lower()
            if name in ("description", "og:description") and self.meta_description is None:
                self.meta_description = attrs_d.get("content") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title.append(data)
        if self._in_h1:
            self.h1.append(data)
        self.body_text.append(data)


# --------------------------------------------------------------------------- #
# Term lists. Organized per-language so coverage is auditable and testable —
# see tests/test_relevance.py for a test asserting every language group has
# at least N terms and at least one is exercised.
# --------------------------------------------------------------------------- #

_NICOTINE_TERMS_BY_LANG: dict[str, list[str]] = {
    "en": [
        "vape", "vaping", "vapor", "e-cigarette", "ecig", "e-liquid",
        "eliquid", "e juice", "ejuice", "nicotine", "disposable vape",
        "vape pod", "vape kit", "vape shop", "cigarette", "cigar",
        "tobacco", "snus", "nicotine pouch", "hookah", "shisha",
    ],
    "de": [
        "dampfen", "e-zigarette", "e-zigaretten", "liquid", "nikotin",
        "tabak", "zigarette", "verdampfer", "dampfgerät", "shisha",
        "snus", "vapen",
    ],
    "fr": [
        "cigarette électronique", "vapoteur", "vapotage", "e-liquide",
        "nicotine", "tabac", "cigare", "chicha", "tabagisme", "vapoter",
    ],
    "it": [
        "sigaretta elettronica", "sigarette elettroniche", "svapo",
        "svapare", "liquido", "nicotina", "tabacco", "sigaro", "narghilè",
    ],
    "es": [
        "cigarrillo electrónico", "cigarrillos electrónicos", "vapeo",
        "vapear", "líquido", "nicotina", "tabaco", "cigarro", "cachimba",
        "shisha",
    ],
    "pt": [
        "cigarro eletrônico", "cigarros eletrônicos", "vape", "vapor",
        "líquido", "nicotina", "tabaco", "charuto", "narguilé",
    ],
    "ru": [
        "сигарета", "электронная сигарета", "жидкость", "никотин",
        "табак", "вейп", "парение", "испаритель",
    ],
    "pl": [
        "papieros elektroniczny", "e-papieros", "liquid", "nikotyna",
        "tytoń", "cygaro", "waporyzator",
    ],
    "nl": [
        "e-sigaret", "damp", "vloeistof", "nicotine", "tabak", "sigaar",
        "verdampen", "vapen",
    ],
    "bg": [
        "цигара", "никотин", "тютюн", "пара", "вейп", "течност",
    ],
}

_NICOTINE_TERMS: frozenset[str] = frozenset(
    term.lower() for terms in _NICOTINE_TERMS_BY_LANG.values() for term in terms
)

# Commerce markers — mostly language-specific "add to cart" / "checkout"
# phrases plus a few English defaults for pages that fall back to English UI.
_COMMERCE_MARKERS = frozenset(
    t.lower()
    for t in [
        "add to cart", "add to basket", "buy now", "checkout", "shopping cart",
        "in den warenkorb", "zur kasse", "jetzt kaufen",
        "ajouter au panier", "commander", "acheter",
        "aggiungi al carrello", "acquista ora",
        "añadir al carrito", "agregar al carrito", "comprar ahora",
        "adicionar ao carrinho", "comprar agora",
        "добавить в корзину", "купить",
        "dodaj do koszyka", "kup teraz",
        "toevoegen aan winkelwagen", "in winkelwagen", "nu kopen",
        "добави в количката", "купи сега",
    ]
)

# Bare "shopping cart" nouns, one per language — a supplementary, weaker but
# more robust signal than the phrases above. Modern JS-rendered storefronts
# (Shopify/Vue/React themes) often template their "Add to cart" BUTTON text
# from a JS i18n object inside <script> (which is deliberately stripped, see
# _Extractor), while the nav-bar cart LINK label ("Panier", "Warenkorb", ...)
# is still plain body HTML. Verified against a real miss: holy-vape.fr's
# "Ajouter au panier" only existed inside a <script> JS template string, but
# its header nav had a plain <span>Panier</span> cart link. Only ever
# consulted after nicotine_hits > 0, so the genericness of e.g. "boutique" is
# low-risk — it can't promote an unrelated site on its own.
_COMMERCE_WORDS_SINGLE = frozenset(
    t.lower()
    for t in [
        "panier", "boutique",                      # fr
        "warenkorb",                                 # de
        "carrello",                                  # it
        "carrito", "cesta",                          # es
        "carrinho",                                  # pt
        "koszyk",                                    # pl
        "winkelwagen",                               # nl
        "корзина",                                   # ru
        "количка",                                   # bg
    ]
)

_PLATFORM_RE = re.compile(
    r"cdn\.shopify\.com|myshopify\.com|woocommerce|wp-content/plugins/woocommerce"
    r"|magento|prestashop|/cart\.php|shopify\.com",
    re.IGNORECASE,
)

_CURRENCY_DIGIT_RE = re.compile(r"[€$£]\s?\d|\d\s?(zł|лв|kr|₽|r\$)", re.IGNORECASE)

_BRAND_PHRASES = frozenset(
    t.lower()
    for t in [
        "official website", "official site", "official brand",
        "find a retailer", "where to buy", "find a store near you",
        "authorized retailers",
    ]
)

_MARKETPLACE_PHRASES = frozenset(
    t.lower()
    for t in ["marketplace", "multi-vendor", "sell on", "become a seller", "vendor login"]
)

_NEGATIVE_MARKERS = frozenset(
    t.lower()
    for t in [
        "powered by wordpress", "powered by blogger", "blogspot",
        "wikipedia", "powered by mediawiki", "powered by phpbb",
        "powered by vbulletin", "powered by discourse", "forum index",
        "domain may be for sale", "this domain is for sale",
        "parked domain", "buy this domain", "under construction",
        "coming soon",
    ]
)

# "tobacco" (and its de/es/fr/pl cognates) is one of the most common
# FRAGRANCE NOTE names in perfumery (e.g. Tom Ford's "Tobacco Vanille") —
# found as a real false positive on 1111perfume.ae during live testing: a
# perfume retailer selling products literally named "11 11 Tobacco" and "Red
# Tobacco" matched the nicotine-term gate on "tobacco" alone. If a fragrance-
# context word is present AND every matched nicotine term is one of these
# ambiguous tobacco-family words, the page is a perfume site, not a nicotine
# retailer — see the override in classify_html() below.
_TOBACCO_AMBIGUOUS_TERMS = frozenset({"tobacco", "tabak", "tabaco", "tabac", "tytoń"})
_FRAGRANCE_CONTEXT = frozenset(
    t.lower() for t in ["eau de parfum", "eau de toilette", "fragrance", "cologne", "perfume", "parfum"]
)

# Bot-check / block interstitials — fetched successfully (often HTTP 200) but
# not a real page. These map to category="error", distinct from "unrelated",
# mirroring the funnel crawl's blocked_bot classification (invariant: a CAPTCHA
# or block page is never miscounted as ordinary page content).
_BLOCK_MARKERS = frozenset(
    t.lower()
    for t in [
        "checking your browser", "verify you are human", "i am not a robot",
        "sorry, you have been blocked", "access denied", "attention required",
        "please enable javascript and cookies", "ray id",
        "verifique que es un ser humano", "verificación de seguridad en curso",
        "bestätigen sie, dass sie ein mensch sind",
        "request has been denied",  # found live: vapegourmet.de's WAF block page
    ]
)


@dataclass
class ClassifyResult:
    category: str
    score: float
    nicotine_hits: int
    commerce_hits: int
    negative_hits: int
    matched_terms: list[str] = field(default_factory=list)


def classify_html(html: str) -> ClassifyResult:
    """Pure, offline, deterministic. Never touches the network or the
    filesystem — safe to call repeatedly against stored HTML with a revised
    ruleset (--reclassify).
    """
    extractor = _Extractor()
    try:
        extractor.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML must not crash the screen
        pass

    title = "".join(extractor.title).strip()
    h1 = "".join(extractor.h1).strip()
    meta_desc = extractor.meta_description or ""
    body = "".join(extractor.body_text)

    priority_lower = f"{title} {meta_desc} {h1}".lower()
    full_lower = f"{priority_lower} {body}".lower()

    matched: list[str] = []
    nicotine_hits = 0
    for term in _NICOTINE_TERMS:
        hit_full = term in full_lower
        hit_priority = term in priority_lower
        if hit_full:
            nicotine_hits += 1
            matched.append(term)
        if hit_priority:
            # Title/meta/h1 mentions are a much stronger identity signal than
            # an incidental body mention — weight them double.
            nicotine_hits += 1

    if any(marker in full_lower for marker in _BLOCK_MARKERS):
        return ClassifyResult("error", 0.0, nicotine_hits, 0, 0, matched)

    matched_set = set(matched)
    if (
        matched_set
        and matched_set <= _TOBACCO_AMBIGUOUS_TERMS
        and any(m in full_lower for m in _FRAGRANCE_CONTEXT)
    ):
        # Every nicotine-term hit is an ambiguous tobacco-family word AND the
        # page talks about fragrance/perfume — a "Tobacco" or "Tabac" product
        # name here almost certainly means a scent note, not nicotine.
        return ClassifyResult("unrelated", 0.0, 0, 0, 0, matched)

    commerce_hits = sum(1 for m in _COMMERCE_MARKERS if m in full_lower)
    commerce_hits += sum(1 for w in _COMMERCE_WORDS_SINGLE if w in full_lower)
    if _CURRENCY_DIGIT_RE.search(full_lower):
        commerce_hits += 1
    if _PLATFORM_RE.search(html.lower()):
        commerce_hits += 2

    negative_hits = sum(1 for m in _NEGATIVE_MARKERS if m in full_lower)

    if nicotine_hits == 0:
        return ClassifyResult("unrelated", 0.0, 0, commerce_hits, negative_hits, matched)

    if commerce_hits > 0:
        if any(p in full_lower for p in _BRAND_PHRASES):
            category = "brand_manufacturer"
        elif any(p in full_lower for p in _MARKETPLACE_PHRASES):
            category = "marketplace"
        else:
            category = "retailer"
    elif any(p in full_lower for p in _BRAND_PHRASES):
        category = "brand_manufacturer"
    else:
        # Nicotine-related, no commerce, no brand phrasing: blog/news/forum/
        # review content about vaping rather than a place to buy it.
        category = "informational"

    score = float(nicotine_hits + commerce_hits - negative_hits)
    return ClassifyResult(category, score, nicotine_hits, commerce_hits, negative_hits, matched)


# --------------------------------------------------------------------------- #
# LLM-based relevance judgment — the PRODUCTION path. classify_html() above
# is the deterministic classifier, kept for reference/fast local iteration
# but not called by 01b_screen_relevance.py.
# --------------------------------------------------------------------------- #

# Maps a fetch()-level failure_reason to the relevance_label a site gets when
# it could never even be classified. Distinct from a page that fetched fine
# but is a bot-check/block interstitial (_BLOCK_MARKERS below) — both
# eventually resolve toward category="error", but only the fetch-failure
# case has no HTML to inspect at all.
_FETCH_FAILURE_TO_LABEL = {"blocked_bot": "blocked", "timeout": "unreachable", "error": "error"}


def build_relevance_text(title: str, meta_desc: str, h1: str, body: str, *, max_body_chars: int = 3000) -> str:
    """Assemble the LLM's input from extracted page fields. Title/meta/h1
    are kept in full — they carry most of a site's identity signal; body is
    truncated, both for LLM call speed at ~116k-site scale and because a
    home page's purpose is usually established well within the first few
    thousand characters.
    """
    body_excerpt = body.strip()[:max_body_chars]
    parts = [
        f"Title: {title.strip()}",
        f"Meta description: {meta_desc.strip()}",
        f"Main heading (h1): {h1.strip()}",
        f"Body text (excerpt): {body_excerpt}",
    ]
    return "\n".join(parts)


def classify_page(
    fetch_result: FetchResult, *, model: Optional[str] = None, max_body_chars: int = 3000
) -> dict:
    """Full relevance judgment for one already-fetched page: fetch-failure
    passthrough -> deterministic block-page pre-check -> LLM
    classify_relevance() on the extracted text. This is the single function
    01b_screen_relevance.py and spike/spike_relevance.py both call, so the
    two can't drift out of sync.

    Returns: {category, raw, latency_ms, model}. `raw`/`latency_ms`/`model`
    are None for the two paths that never reach the LLM (fetch failure,
    block page) — there was no LLM call to report on. `category` can also
    be None if the LLM's response was unparseable (see vlm._parse_relevance)
    — never silently coerced to "unrelated"; the caller decides how to
    record that.
    """
    if fetch_result.failure_reason:
        label = _FETCH_FAILURE_TO_LABEL.get(fetch_result.failure_reason, "error")
        return {"category": label, "raw": None, "latency_ms": None, "model": None}

    extractor = _Extractor()
    try:
        extractor.feed(fetch_result.html)
    except Exception:  # noqa: BLE001 — malformed HTML must not crash the screen
        pass
    title = "".join(extractor.title).strip()
    h1 = "".join(extractor.h1).strip()
    meta_desc = extractor.meta_description or ""
    body = "".join(extractor.body_text)
    full_lower = f"{title} {meta_desc} {h1} {body}".lower()

    if any(marker in full_lower for marker in _BLOCK_MARKERS):
        return {"category": "error", "raw": None, "latency_ms": None, "model": None}

    text = build_relevance_text(title, meta_desc, h1, body, max_body_chars=max_body_chars)
    result = classify_relevance(text, model=model)
    return {
        "category": result["category"],
        "raw": result["raw"],
        "latency_ms": result["latency_ms"],
        "model": result["model"],
    }
