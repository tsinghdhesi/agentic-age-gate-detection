"""Browser context creation with playwright-stealth (v2 API), plus per-ccTLD
locale/timezone so a site sees a plausible local visitor.

Serial-friendly by construction: build one stealthed Playwright, launch one
browser, and create a FRESH context per site. No async, no pools.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

from common import config

# ccTLD (exact public suffix from sites.cctld) -> plausible locale/timezone.
# Generic suffixes (com/org/net/app/...) and unknowns fall back to _DEFAULT.
_DEFAULT = {"locale": "en-US", "timezone_id": "America/New_York"}
_CCTLD_PROFILE: Dict[str, Dict[str, str]] = {
    "us": {"locale": "en-US", "timezone_id": "America/New_York"},
    "uk": {"locale": "en-GB", "timezone_id": "Europe/London"},
    "ie": {"locale": "en-IE", "timezone_id": "Europe/Dublin"},
    "de": {"locale": "de-DE", "timezone_id": "Europe/Berlin"},
    "fr": {"locale": "fr-FR", "timezone_id": "Europe/Paris"},
    "it": {"locale": "it-IT", "timezone_id": "Europe/Rome"},
    "es": {"locale": "es-ES", "timezone_id": "Europe/Madrid"},
    "nl": {"locale": "nl-NL", "timezone_id": "Europe/Amsterdam"},
    "pl": {"locale": "pl-PL", "timezone_id": "Europe/Warsaw"},
    "ru": {"locale": "ru-RU", "timezone_id": "Europe/Moscow"},
    "ua": {"locale": "uk-UA", "timezone_id": "Europe/Kyiv"},
    "br": {"locale": "pt-BR", "timezone_id": "America/Sao_Paulo"},
    "pt": {"locale": "pt-PT", "timezone_id": "Europe/Lisbon"},
    "mx": {"locale": "es-MX", "timezone_id": "America/Mexico_City"},
    "ca": {"locale": "en-CA", "timezone_id": "America/Toronto"},
    "au": {"locale": "en-AU", "timezone_id": "Australia/Sydney"},
    "nz": {"locale": "en-NZ", "timezone_id": "Pacific/Auckland"},
    "jp": {"locale": "ja-JP", "timezone_id": "Asia/Tokyo"},
    "kr": {"locale": "ko-KR", "timezone_id": "Asia/Seoul"},
    "cn": {"locale": "zh-CN", "timezone_id": "Asia/Shanghai"},
    "tw": {"locale": "zh-TW", "timezone_id": "Asia/Taipei"},
    "in": {"locale": "en-IN", "timezone_id": "Asia/Kolkata"},
    "tn": {"locale": "fr-TN", "timezone_id": "Africa/Tunis"},
    "se": {"locale": "sv-SE", "timezone_id": "Europe/Stockholm"},
    "no": {"locale": "nb-NO", "timezone_id": "Europe/Oslo"},
    "fi": {"locale": "fi-FI", "timezone_id": "Europe/Helsinki"},
    "dk": {"locale": "da-DK", "timezone_id": "Europe/Copenhagen"},
    "ch": {"locale": "de-CH", "timezone_id": "Europe/Zurich"},
    "at": {"locale": "de-AT", "timezone_id": "Europe/Vienna"},
}


def _country_label(cctld: Optional[str]) -> Optional[str]:
    """Last label of the suffix: 'co.uk' -> 'uk', 'com.br' -> 'br', 'com' -> 'com'."""
    if not cctld:
        return None
    return cctld.rsplit(".", 1)[-1].lower()


def profile_for_cctld(cctld: Optional[str]) -> Dict[str, str]:
    """{'locale', 'timezone_id'} for a ccTLD; default for generic/unknown."""
    return dict(_CCTLD_PROFILE.get(_country_label(cctld), _DEFAULT))


@contextmanager
def stealth_playwright():
    """Yield a sync Playwright with stealth auto-applied to every context/page.

    Usage:
        with stealth_playwright() as p:
            browser = launch_browser(p)
            ctx = new_context(browser, cctld="co.uk")
            page = ctx.new_page()
    """
    with Stealth().use_sync(sync_playwright()) as p:
        yield p


def launch_browser(p: Any, headless: bool = True) -> Any:
    return p.chromium.launch(headless=headless)


def new_context(browser: Any, cctld: Optional[str] = None) -> Any:
    """Fresh context: configured viewport + per-ccTLD locale/timezone. Stealth is
    injected automatically because the browser came from stealth_playwright()."""
    vp = config.load_global()["viewport"]
    prof = profile_for_cctld(cctld)
    return browser.new_context(
        viewport={"width": vp["width"], "height": vp["height"]},
        locale=prof["locale"],
        timezone_id=prof["timezone_id"],
    )
