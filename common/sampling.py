"""Pure, DB-free sampling math for the stratified probability sample drawn
from relevant sites (`scripts/01c_draw_sample.py`). Nothing here touches the
network, the filesystem, or SQLite — everything is a deterministic function
of its inputs, which is what makes the design reproducible from
`(frame_salt, site_id)` alone and independently testable.

Stratification variable: a 9-way regulatory-region bucket derived from
`sites.cctld` (never null — a locked invariant). ccTLD is chosen over raw
country because age-gating is fundamentally a *regulatory* phenomenon, and
cctld is fully observed on the frame and already drives locale/timezone via
`stealth.profile_for_cctld`.

Region map is necessarily a set of documented judgment calls, not a precise
taxonomy — see `_GENERIC_SUFFIXES` and `_CC_TO_REGION` below. Anything not
recognized falls into the "other" catch-all stratum rather than raising, so
an incomplete map degrades to a coarser stratum instead of crashing frame
construction.
"""

from __future__ import annotations

import hashlib
from typing import Optional

STRATA = (
    "gtld",
    "uk_ie",
    "eu_west",
    "eu_east",
    "latam",
    "apac",
    "mena_af",
    "na",
    "other",
)

# Suffixes with no country signal at all — either truly generic (com, org),
# or a ccTLD that is overwhelmingly used generically rather than nationally.
# ".co" (Colombia), ".me" (Montenegro), ".io" (British Indian Ocean Territory),
# ".tv" (Tuvalu), ".cc" (Cocos Islands), and ".us" are all real ccTLDs folded
# in here deliberately — a pre-registered choice, not a post-hoc one.
#
# The second block (top/fun/click/... through xn--6frz82g) was added after
# inspecting what actually landed in the "other" catch-all against the real
# seed: these "new gTLD" suffixes are cheap, geography-free registrations —
# together they accounted for roughly half of everything "other" caught, and
# leaving them there would have diluted the stratum's intended meaning
# (unmapped geography, not "no geography at all"). xn--6frz82g is the
# punycode form of the Chinese-script ".网站" ("website") gTLD — same idea,
# different script. ".ws" (Samoa) and ".asia"/".tokyo"/".to" were judgment
# calls resolved by usage pattern, documented in _CC_TO_REGION instead where
# they carry at least regional signal.
_GENERIC_SUFFIXES = frozenset(
    {
        "com", "net", "org", "info", "biz", "name", "pro", "mobi",
        "xyz", "online", "shop", "store", "site", "club", "app", "dev",
        "tech", "io", "co", "me", "tv", "cc", "us",
        "top", "fun", "click", "vip", "life", "website", "uno", "world",
        "live", "space", "today", "sbs", "blog", "icu", "link", "games",
        "cfd", "ws", "xn--6frz82g",
        "bond", "lol", "one", "cloud", "buzz", "help", "page", "pw",
        "digital", "work", "red",
        # institutional TLDs — not regionally distinguishing beyond the
        # already-folded ".us", same logic applied.
        "gov", "edu", "mil",
    }
)

# ISO-3166-ish trailing country label -> regulatory-region stratum. Applied
# to the LAST dot-segment of cctld, so this covers both bare ccTLDs ("de")
# and compound ones ("co.uk" -> "uk", "com.br" -> "br"). Not exhaustive by
# design — anything missing falls into "other" rather than misclassifying.
_CC_TO_REGION: dict[str, str] = {
    # --- UK & Ireland ---
    "uk": "uk_ie", "ie": "uk_ie",
    # --- Western Europe (incl. non-EU EEA / GDPR-adjacent) ---
    "de": "eu_west", "fr": "eu_west", "nl": "eu_west", "be": "eu_west",
    "at": "eu_west", "ch": "eu_west", "es": "eu_west", "pt": "eu_west",
    "it": "eu_west", "lu": "eu_west", "dk": "eu_west", "se": "eu_west",
    "no": "eu_west", "fi": "eu_west", "is": "eu_west", "eu": "eu_west",
    "gr": "eu_west", "mt": "eu_west", "cy": "eu_west",
    # --- Eastern Europe / post-Soviet / Balkans ---
    "pl": "eu_east", "cz": "eu_east", "sk": "eu_east", "hu": "eu_east",
    "ro": "eu_east", "bg": "eu_east", "hr": "eu_east", "si": "eu_east",
    "lt": "eu_east", "lv": "eu_east", "ee": "eu_east", "ua": "eu_east",
    "by": "eu_east", "md": "eu_east", "rs": "eu_east", "ba": "eu_east",
    "mk": "eu_east", "al": "eu_east", "ru": "eu_east", "ge": "eu_east",
    "am": "eu_east", "az": "eu_east", "kz": "eu_east", "uz": "eu_east",
    "su": "eu_east",  # legacy USSR domain, still used mostly by RU-affiliated sites
    "tr": "eu_east",  # Turkey — judgment call: grouped with CEE, not mena_af
    "pm": "eu_west",  # Saint Pierre and Miquelon — French overseas territory
    # --- Latin America & Caribbean ---
    "br": "latam", "mx": "latam", "ar": "latam", "cl": "latam",
    "pe": "latam", "ve": "latam", "ec": "latam", "uy": "latam",
    "py": "latam", "bo": "latam", "cr": "latam", "pa": "latam",
    "gt": "latam", "do": "latam", "hn": "latam", "sv": "latam",
    "ni": "latam", "cu": "latam", "pr": "latam", "ai": "latam",
    "bz": "latam", "jm": "latam", "tt": "latam", "bs": "latam", "bb": "latam",
    "lat": "latam",  # regional Latin America gTLD, not a country code
    # --- Asia-Pacific ---
    "cn": "apac", "jp": "apac", "kr": "apac", "in": "apac", "id": "apac",
    "my": "apac", "sg": "apac", "th": "apac", "vn": "apac", "ph": "apac",
    "tw": "apac", "hk": "apac", "au": "apac", "nz": "apac", "pk": "apac",
    "bd": "apac", "lk": "apac", "mn": "apac", "kh": "apac", "la": "apac",
    "mm": "apac", "np": "apac", "bn": "apac", "mo": "apac",
    "asia": "apac", "tokyo": "apac", "to": "apac",  # to = Tonga (Oceania)
    # --- Middle East & Africa ---
    "sa": "mena_af", "ae": "mena_af", "eg": "mena_af", "ma": "mena_af",
    "dz": "mena_af", "tn": "mena_af", "jo": "mena_af", "lb": "mena_af",
    "qa": "mena_af", "kw": "mena_af", "bh": "mena_af", "om": "mena_af",
    "il": "mena_af", "ir": "mena_af", "iq": "mena_af", "ng": "mena_af",
    "za": "mena_af", "ke": "mena_af", "gh": "mena_af", "tz": "mena_af",
    "ug": "mena_af", "et": "mena_af", "ly": "mena_af", "sd": "mena_af",
    "sy": "mena_af", "ye": "mena_af", "ps": "mena_af", "cm": "mena_af",
    "ci": "mena_af", "sn": "mena_af", "zm": "mena_af", "zw": "mena_af",
    "mz": "mena_af", "ao": "mena_af", "na": "mena_af",  # na = Namibia here
    # --- North America (".us" folds into gtld above; this is just Canada) ---
    "ca": "na",
}


def stratum_for_cctld(cctld: Optional[str]) -> str:
    """Map an exact cctld string (as stored in sites.cctld — 'com', 'co.uk',
    'com.br', 'de', ...) to one of the 9 STRATA. Never raises; unmapped or
    empty input returns 'other'.
    """
    cc = (cctld or "").strip().lower()
    if not cc:
        return "other"
    if cc in _GENERIC_SUFFIXES:
        return "gtld"
    last = cc.rsplit(".", 1)[-1]
    if last in _GENERIC_SUFFIXES:
        return "gtld"
    return _CC_TO_REGION.get(last, "other")


def sample_u(site_id: str, salt: str) -> float:
    """Deterministic draw variate in [0, 1) for `site_id` under `frame_salt`.
    Reproducible from (salt, site_id) alone — no RNG state to persist or
    corrupt. Changing `salt` invalidates and silently redraws the entire
    sample, so treat it as immutable once wave 1 has been drawn.
    """
    digest = hashlib.sha256(f"{salt}|{site_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 2**64


def allocate(
    frame_counts: dict[str, int],
    n_total: int,
    *,
    power: float = 0.5,
    floor: int = 150,
) -> dict[str, int]:
    """Power allocation (n_h ∝ N_h**power) across strata: power=1.0 is
    proportional allocation (favors big strata, can starve small ones to
    single digits); power=0.0 is equal allocation (wastes budget on tiny
    strata, undersamples big ones badly). power=0.5 is the standard
    compromise (Bankier 1988; Kish's compromise allocation) between
    national-level and per-stratum precision.

    Each stratum is floored (never below `floor` while room allows) and
    capped (never above its own N_h), then largest-remainder rounded so
    sum(result.values()) == min(n_total, sum(frame_counts.values())) exactly.

    Caveat: if sum(floor for every stratum) exceeds n_total, the floor and
    the total budget are in direct tension and the floor will not be fully
    honored everywhere — this only matters for a tiny n_total relative to
    floor * len(strata), which the configured defaults (target_n=2500,
    floor=30, 9 strata) are well clear of.
    """
    if n_total < 0:
        raise ValueError("n_total must be non-negative")
    strata = list(frame_counts.keys())
    if not strata:
        return {}

    total_pop = sum(frame_counts.values())
    n_total = min(n_total, total_pop)

    weights = {h: frame_counts[h] ** power for h in strata}
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return {h: 0 for h in strata}

    raw = {h: n_total * weights[h] / weight_sum for h in strata}

    alloc_f: dict[str, float] = {}
    for h in strata:
        n_h = max(raw[h], min(float(floor), float(frame_counts[h])))
        n_h = min(n_h, float(frame_counts[h]))
        alloc_f[h] = n_h

    int_alloc = {h: int(alloc_f[h]) for h in strata}
    remainders = {h: alloc_f[h] - int_alloc[h] for h in strata}
    min_alloc = {h: min(floor, frame_counts[h]) for h in strata}
    shortfall = n_total - sum(int_alloc.values())

    if shortfall > 0:
        # Give +1 to strata with the largest fractional remainder that still
        # have room to grow (n_h < N_h), re-filtered each step since a
        # stratum drops out once it hits its cap.
        while shortfall > 0:
            eligible = [h for h in strata if int_alloc[h] < frame_counts[h]]
            if not eligible:
                break
            h = max(eligible, key=lambda h: remainders[h])
            int_alloc[h] += 1
            shortfall -= 1
    elif shortfall < 0:
        # Remove from strata with the smallest fractional remainder, but
        # protect the floor: only shrink a stratum below its floor as a last
        # resort, once every above-floor stratum has been exhausted (which
        # only happens if floor * len(strata) > n_total — floor and the
        # total budget are then in direct, unresolvable tension).
        for respect_floor in (True, False):
            while shortfall < 0:
                minimum = min_alloc if respect_floor else {h: 0 for h in strata}
                eligible = [h for h in strata if int_alloc[h] > minimum[h]]
                if not eligible:
                    break
                h = min(eligible, key=lambda h: remainders[h])
                int_alloc[h] -= 1
                shortfall += 1
            if shortfall == 0:
                break

    return int_alloc


def design_summary(frame_counts: dict[str, int], alloc: dict[str, int]) -> dict:
    """Per-stratum sampling weight w_h = N_h/n_h and rate f_h = n_h/N_h, plus
    overall n_eff and DEFF (design effect) for the stratified design. Print
    this verbatim in `01c_draw_sample.py --dry-run` — it goes straight into
    the methods chapter.

    n_eff = (sum_h N_h)^2 / sum_h(N_h^2 / n_h); DEFF = n_total / n_eff.
    DEFF = 1.0 exactly under proportional allocation (verified in tests) —
    that is the reference point every other allocation is measured against.
    """
    strata_rows: dict[str, dict] = {}
    sum_N = 0
    sum_N2_over_n = 0.0
    n_total = 0
    for h, N_h in frame_counts.items():
        n_h = alloc.get(h, 0)
        f_h = (n_h / N_h) if N_h else 0.0
        w_h = (N_h / n_h) if n_h else None
        strata_rows[h] = {"N_h": N_h, "n_h": n_h, "f_h": f_h, "w_h": w_h}
        if n_h:
            sum_N += N_h
            sum_N2_over_n += (N_h**2) / n_h
            n_total += n_h

    n_eff = (sum_N**2) / sum_N2_over_n if sum_N2_over_n else 0.0
    deff = (n_total / n_eff) if n_eff else float("inf")
    return {"strata": strata_rows, "n_total": n_total, "n_eff": n_eff, "deff": deff}
