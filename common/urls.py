"""URL normalization, deterministic site_id derivation, and offline public-suffix
extraction — the missing piece invariant #3 and the locked site_id/cctld
decisions in CLAUDE.md depend on.

Two ideas kept deliberately separate:

- `url_key` — the canonical (host[:port]) GROUPING key. www-stripped, default
  ports dropped, punycode. Never stored directly; it is what `site_id` is a
  pure function of, and what dedupe/representative-selection operate on.
- `url_normalized` — the REPRESENTATIVE origin actually stored and navigated
  to (`scheme://host/`), which may still carry `www.` if that was the only
  form observed. `site_id = f(url_key)`, never `f(url_normalized)` — this is
  what makes `01_build_queue.py` re-ingest-safe: a later seed contributing
  `https://www.x.com` when only `http://x.com` was known before updates the
  existing row's `url_normalized` in place, rather than minting a duplicate
  site_id for the same site (see `better_origin`).

tldextract is configured strictly OFFLINE (bundled PSL snapshot, no network,
no cache) per invariant #3 — see `tests/test_urls.py` for the executable proof
(network calls are poisoned and the test asserts none happen).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import tldextract

# NEVER fetch the Public Suffix List over the network, ever (invariant #3).
# include_psl_private_domains=False is load-bearing: with it, a host like
# "foo.blogspot.com" resolves to suffix "com" (an ICANN suffix — which is what
# `cctld` means throughout this project and what stealth.profile_for_cctld
# consumes). With include_psl_private_domains=True it would resolve to suffix
# "blogspot.com", which is neither a ccTLD nor a usable locale key.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
    include_psl_private_domains=False,
)

_DEFAULT_PORTS = {"http": 80, "https": 443}
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SCHEME_TOKEN_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):(.*)$", re.DOTALL)
_MAX_SLUG_LEN = 80  # longest observed seed host is 84 chars; site_id stays a
                    # safe filesystem path component (APFS/most FS limit 255).


@dataclass(frozen=True)
class Normalized:
    """Result of normalizing one raw seed URL."""

    url_key: str             # canonical host[:port], www-stripped — dedupe/site_id key
    url_normalized: str      # scheme://host/ — this row's own observed origin
    host: str                # host as used in url_normalized (may retain www.)
    scheme: str               # "http" | "https"
    registrable_domain: str  # e.g. "vampedvape.com"
    cctld: str                 # exact public suffix, e.g. "com", "co.uk", "com.br" — never ""


def split_host(host: str) -> tuple[str, str, str]:
    """Return (subdomain, registrable_domain, cctld) for a bare host via the
    offline PSL. Raises ValueError if no public suffix is found — cctld must
    NEVER be silently empty (locked invariant); callers should catch this and
    route the row to a rejects log rather than writing a bad row.
    """
    r = _EXTRACT(host)
    if not r.suffix:
        raise ValueError(f"no public suffix found for host: {host!r}")
    registrable_domain = getattr(r, "top_domain_under_public_suffix", None) or r.registered_domain
    return r.subdomain, registrable_domain, r.suffix


def host_slug(host_or_key: str) -> str:
    """Lowercase, [a-z0-9] + underscore, collapsed, truncated. Port suffixes
    (":9900") are stripped first since they are not part of the host slug.
    """
    host = host_or_key.split(":", 1)[0]
    slug = _SLUG_RE.sub("_", host.lower()).strip("_")
    slug = slug[:_MAX_SLUG_LEN]
    return slug or "host"


def make_site_id(url_key: str) -> str:
    """site_id = <host_slug>_<6-hex sha256(url_key)>. Matches the locked
    CLAUDE.md example: 'm.hindi.vampedvape.com' -> 'm_hindi_vampedvape_com_3f9a1c'.
    A pure function of url_key alone — deterministic and re-derivable by
    anyone with the salt-free (site_id has no salt) key, unlike sample_u.
    """
    digest = hashlib.sha256(url_key.encode("utf-8")).hexdigest()[:6]
    return f"{host_slug(url_key)}_{digest}"


def _explicit_scheme(s: str) -> Optional[str]:
    """If `s` starts with `<token>:`, return the lowercased token — UNLESS
    that colon is actually a bare host:port separator ('vape.co.uk:9900' is
    host:port, not a "vape.co.uk" scheme: the token contains a dot, as a
    hostname would, and what follows the colon is purely digits). Returns
    None when there is no colon-prefix at all, or when it's a port.

    This exists so a raw line with a real non-http scheme (e.g.
    'mailto:foo@vape.co.uk', which contains no "://" substring at all) is
    NOT silently mistaken for a bare host missing its scheme and prepended
    with "https://" — urlsplit would then parse "mailto:foo" as userinfo and
    "vape.co.uk" as a bogus-but-valid-looking host.
    """
    m = _SCHEME_TOKEN_RE.match(s)
    if not m:
        return None
    token, rest = m.group(1), m.group(2)
    if rest.isdigit() and "." in token:
        return None
    return token.lower()


def normalize(raw: str) -> Optional[Normalized]:
    """Normalize one raw seed line. Returns None (never raises) for anything
    unparseable/unusable — callers should treat None as "reject, log it."
    """
    if raw is None:
        return None
    s = raw.strip().lstrip("\ufeff")
    if not s or s.startswith("#"):
        return None

    scheme_token = _explicit_scheme(s)
    if scheme_token is None:
        if "://" not in s:
            s = "https://" + s
    elif scheme_token not in ("http", "https"):
        return None

    try:
        parts = urlsplit(s)
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return None

    hostname = parts.hostname
    if not hostname:
        return None
    host = hostname.lower().rstrip(".")
    if not host:
        return None

    # IDN -> punycode A-label always, so there is exactly one representation.
    # Python's built-in "idna" codec handles multi-label hosts (applied per
    # dot-separated label) and is a no-op for already-ASCII hosts, which is
    # every row in the current seed (0 non-ASCII, 351 already punycode).
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None

    try:
        port = parts.port
    except ValueError:
        return None
    default_port = _DEFAULT_PORTS.get(scheme)
    keep_port = port is not None and port != default_port
    port_suffix = f":{port}" if keep_port else ""

    host_nowww = _WWW_RE.sub("", host, count=1)
    url_key = f"{host_nowww}{port_suffix}"
    url_normalized = f"{scheme}://{host}{port_suffix}/"

    try:
        _subdomain, registrable_domain, cctld = split_host(host_nowww)
    except ValueError:
        return None

    return Normalized(
        url_key=url_key,
        url_normalized=url_normalized,
        host=host,
        scheme=scheme,
        registrable_domain=registrable_domain,
        cctld=cctld,
    )


def better_origin(a: str, b: str) -> str:
    """Given two url_normalized candidates that share a url_key, deterministically
    pick the representative to store: prefer https, then the apex (non-www)
    form, then lexicographically smallest. Order-independent — the same pair
    always yields the same winner regardless of which one is "a" vs "b", which
    is what makes representative selection safe to run over an unordered
    dict-reduce of 133,982 seed rows.

    Preferring the apex form is a default, not a universal truth: some hosts
    only serve on www, and better_origin never discards a candidate that was
    never offered — it just prefers apex when both are known to resolve to
    the same site.
    """

    def rank(u: str) -> tuple[bool, bool, str]:
        parts = urlsplit(u)
        host = (parts.hostname or "").lower()
        return (parts.scheme != "https", host.startswith("www."), u)

    return min(a, b, key=rank)
