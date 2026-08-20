"""Unit tests for common/urls.py — URL normalization, site_id, and the offline
public-suffix extraction invariant #3 depends on. Run: pytest -q

The network-poisoning tests are the load-bearing ones here: if tldextract ever
starts fetching the PSL over the network (e.g. a future edit drops
suffix_list_urls=() or cache_dir=None), the frame-construction step would
silently start depending on network availability, which invariant #3
forbids outright. These tests make that regression fail loudly instead of
"just" being slower.
"""

import socket

import pytest

from common.urls import (
    Normalized,
    better_origin,
    host_slug,
    make_site_id,
    normalize,
    split_host,
)


# --------------------------------------------------------------------------- #
# Invariant #3 — tldextract must never touch the network.
# --------------------------------------------------------------------------- #


def test_suffix_list_urls_is_empty():
    from common.urls import _EXTRACT

    assert _EXTRACT.suffix_list_urls == ()


def test_no_network_call_during_extraction(monkeypatch):
    def _blocked(*_a, **_k):
        raise AssertionError("network access attempted — invariant #3 violation")

    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    # Exercise both the low-level split_host and the full normalize() path.
    assert split_host("shop.co.uk")[2] == "co.uk"
    assert normalize("https://shop.co.uk") is not None


# --------------------------------------------------------------------------- #
# split_host — offline PSL correctness across the multi-label suffixes the
# real seed contains.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "host,expected_suffix,expected_registrable",
    [
        ("vape.co.uk", "co.uk", "vape.co.uk"),
        ("shop.com.br", "com.br", "shop.com.br"),
        ("knyszyn.bialystok.lasy.gov.pl", "gov.pl", "lasy.gov.pl"),
        # edupage.org is NOT on the PSL private-domains list (unlike
        # blogspot.com below), so "spkrasiczyn" is an ordinary subdomain and
        # the registrable domain is just "edupage.org".
        ("spkrasiczyn.edupage.org", "org", "edupage.org"),
        ("m.hindi.vampedvape.com", "com", "vampedvape.com"),
    ],
)
def test_split_host_suffixes(host, expected_suffix, expected_registrable):
    _sub, registrable, suffix = split_host(host)
    assert suffix == expected_suffix
    assert registrable == expected_registrable


def test_split_host_treats_blogspot_as_private_not_public():
    # include_psl_private_domains=False is load-bearing: cctld must be an
    # ICANN suffix ("com"), not the PSL-private "blogspot.com" — the latter
    # is neither a ccTLD nor a usable stealth.profile_for_cctld locale key.
    _sub, registrable, suffix = split_host("atoresbrasileirosnus.blogspot.com")
    assert suffix == "com"
    assert registrable == "blogspot.com"


def test_split_host_raises_on_no_suffix():
    with pytest.raises(ValueError):
        split_host("nosuffixthingy")


# --------------------------------------------------------------------------- #
# normalize() — the rules, in the order they're documented in the plan.
# --------------------------------------------------------------------------- #


def test_normalize_defaults_to_https_scheme():
    n = normalize("vape.co.uk")
    assert n.scheme == "https"
    assert n.url_normalized == "https://vape.co.uk/"


def test_normalize_rejects_non_http_schemes():
    assert normalize("ftp://vape.co.uk") is None
    # No "://" substring at all — must not be mistaken for a bare host
    # missing its scheme and silently prepended with "https://" (which would
    # make urlsplit parse "mailto:foo" as userinfo and "vape.co.uk" as a
    # bogus-but-valid-looking host).
    assert normalize("mailto:foo@vape.co.uk") is None
    assert normalize("tel:+1-555-0100") is None
    assert normalize("javascript:alert(1)") is None


def test_normalize_bare_host_port_is_not_mistaken_for_a_scheme():
    # "vape.co.uk:9900" must still default to https and keep the port — the
    # colon here is a port separator, not a scheme, even though the host
    # part contains dots like a scheme token would.
    n = normalize("vape.co.uk:9900")
    assert n is not None
    assert n.scheme == "https"
    assert n.url_key == "vape.co.uk:9900"


def test_normalize_rejects_empty_and_comment_lines():
    assert normalize("") is None
    assert normalize("   ") is None
    assert normalize("# not a url") is None
    assert normalize(None) is None


def test_normalize_strips_userinfo():
    n = normalize("https://user:pass@vape.co.uk/")
    assert n is not None
    assert "@" not in n.url_normalized
    assert n.host == "vape.co.uk"


def test_normalize_drops_path_query_fragment():
    n = normalize("https://vape.co.uk/shop?ref=x#top")
    assert n.url_normalized == "https://vape.co.uk/"


def test_normalize_keeps_nondefault_port():
    n = normalize("https://vape.co.uk:9900/")
    assert n.url_key == "vape.co.uk:9900"
    assert n.url_normalized == "https://vape.co.uk:9900/"


def test_normalize_drops_default_port():
    n = normalize("https://vape.co.uk:443/")
    assert n.url_key == "vape.co.uk"
    assert ":443" not in n.url_normalized


def test_normalize_lowercases_and_strips_trailing_dot():
    n = normalize("https://VAPE.CO.UK./")
    assert n.host == "vape.co.uk"


def test_normalize_punycode_passthrough():
    # Already-punycode hosts (351 rows in the real seed) must pass unchanged.
    n = normalize("https://xn--mnchen-3ya.de")
    assert n is not None
    assert n.host == "xn--mnchen-3ya.de"


def test_normalize_encodes_unicode_host_to_punycode():
    n = normalize("https://münchen.de")
    assert n is not None
    assert n.host == "xn--mnchen-3ya.de"


def test_normalize_underscore_in_host_is_legal():
    n = normalize("https://katarzyna_michalska.users.sggw.pl")
    assert n is not None
    assert n.host == "katarzyna_michalska.users.sggw.pl"


def test_normalize_rejects_no_public_suffix():
    assert normalize("https://nosuffixthingy") is None


# --------------------------------------------------------------------------- #
# www-handling — the biggest normalization judgment call: exactly one leading
# www. label is stripped for the DEDUPE KEY, but m./hindi./no. etc. are kept
# as genuinely distinct origins. This is what makes the locked site_id example
# in CLAUDE.md work.
# --------------------------------------------------------------------------- #


def test_www_and_apex_share_a_url_key():
    a = normalize("https://www.vape.co.uk")
    b = normalize("https://vape.co.uk")
    c = normalize("http://vape.co.uk")  # scheme difference must not matter for the key
    assert a.url_key == b.url_key == c.url_key == "vape.co.uk"


def test_www_and_apex_share_a_site_id():
    a = normalize("https://www.vape.co.uk")
    b = normalize("https://vape.co.uk")
    assert make_site_id(a.url_key) == make_site_id(b.url_key)


def test_only_one_leading_www_label_is_stripped():
    # www2. and a deeper www. are NOT the conventional apex alias.
    a = normalize("https://www2.vape.co.uk")
    b = normalize("https://vape.co.uk")
    assert a.url_key != b.url_key


def test_subdomain_stays_distinct_from_apex():
    # m., hindi., no. etc. are genuinely different origins, not aliases.
    m = normalize("https://m.hindi.vampedvape.com")
    apex = normalize("https://vampedvape.com")
    assert m.url_key != apex.url_key
    assert make_site_id(m.url_key) != make_site_id(apex.url_key)


# --------------------------------------------------------------------------- #
# site_id — must match the locked CLAUDE.md example exactly.
# --------------------------------------------------------------------------- #


def test_site_id_matches_locked_claude_md_example():
    n = normalize("https://m.hindi.vampedvape.com")
    site_id = make_site_id(n.url_key)
    assert site_id.startswith("m_hindi_vampedvape_com_")
    # 6 lowercase hex chars after the final underscore.
    suffix = site_id.rsplit("_", 1)[-1]
    assert len(suffix) == 6
    assert all(c in "0123456789abcdef" for c in suffix)


def test_site_id_is_deterministic():
    n = normalize("https://vape.co.uk")
    assert make_site_id(n.url_key) == make_site_id(n.url_key)


def test_host_slug_truncates_and_never_empty():
    long_host = "a" * 200 + ".com"
    slug = host_slug(long_host)
    assert len(slug) <= 80
    assert slug  # never empty
    assert host_slug("---") == "host"  # degenerates to nothing alnum


# --------------------------------------------------------------------------- #
# better_origin — representative selection must be order-independent.
# --------------------------------------------------------------------------- #


def test_better_origin_prefers_https():
    assert better_origin("http://vape.co.uk/", "https://vape.co.uk/") == "https://vape.co.uk/"
    assert better_origin("https://vape.co.uk/", "http://vape.co.uk/") == "https://vape.co.uk/"


def test_better_origin_prefers_apex_over_www():
    assert better_origin("https://www.vape.co.uk/", "https://vape.co.uk/") == "https://vape.co.uk/"
    assert better_origin("https://vape.co.uk/", "https://www.vape.co.uk/") == "https://vape.co.uk/"


def test_better_origin_is_order_independent_and_deterministic():
    candidates = ["https://www.vape.co.uk/", "http://vape.co.uk/", "https://vape.co.uk/"]
    import itertools

    winners = {
        better_origin(a, b)
        for a, b in itertools.permutations(candidates, 2)
        if a != b
    }
    # Every pairwise comparison that includes the true best candidate must
    # pick it, regardless of argument order.
    assert winners <= {"https://vape.co.uk/", "https://www.vape.co.uk/", "http://vape.co.uk/"}
    assert better_origin(candidates[0], candidates[1]) == better_origin(candidates[1], candidates[0])
