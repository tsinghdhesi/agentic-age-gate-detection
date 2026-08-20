#!/usr/bin/env python3
"""Relevance-instrument spike — validates RELEVANCE_PROMPT + classify_relevance()
(qwen3.6:27b) against real seed URLs before freezing the prompt for the full
115,792-site frame screen. Mirrors spike_detection.py's methodology: real
URLs, hand-verified labels, eyeball mismatches, fix the prompt, then freeze.

Reuses several domains with ALREADY-ESTABLISHED ground truth from last
session's live testing (spike_detection.py / manual verification), plus new
candidates for under-tested categories (brand_manufacturer, marketplace,
informational) and the known-hard case the old keyword classifier got wrong
(1111perfume.ae's "Tobacco Vanille" fragrance note).

Two entries are expected to occasionally mismatch for reasons that are NOT
classifier bugs — don't chase these if they reappear:
  - smoketown.com.br fetches to a near-empty JS app shell (no real content
    ever reaches a no-browser HTTP GET) — a structural limitation of
    screening without a browser, not something a prompt/model fix can solve.
  - holy-vape.fr can start failing to fetch if this spike (or the earlier
    live classify_relevance() smoke test) hits it repeatedly in one
    session — plausibly the site's own bot defenses reacting to repeated
    automated requests from one source. A real full-frame run hits each
    site exactly once, so this shouldn't occur there.

Standalone + throwaway. Run:  ./.venv/bin/python spike/spike_relevance.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.relevance import build_relevance_text, classify_page  # noqa: E402

# (url, gold_label_or_None). None = no prior ground truth; eyeball live.
LABELED = [
    # --- confirmed real retailers (established via last session's live testing) ---
    ("https://vape.bg", "retailer"),                    # BG
    ("https://www.dutchvapeshop.nl", "retailer"),        # NL
    ("https://www.holy-vape.fr", "retailer"),            # FR
    ("https://boogivape.pl", "retailer"),                # PL
    ("https://www.smoketown.com.br", "retailer"),        # PT-BR
    ("https://www.vapesmoke.it", "retailer"),            # IT
    ("https://www.vkvape.com", "retailer"),              # EN
    ("https://e-vaping.co.uk", "retailer"),              # EN
    ("https://puffbar.com", "retailer"),                 # EN
    # --- confirmed unrelated ---
    ("https://bbqsmoke.ru", "unrelated"),                # BBQ restaurant
    ("https://spkrasiczyn.edupage.org", "unrelated"),    # school
    ("https://atoresbrasileirosnus.blogspot.com", "unrelated"),
    # --- the known-hard false-positive case (tobacco fragrance note) ---
    ("https://1111perfume.ae", "unrelated"),
    # --- block/bot-check pages: must short-circuit to "error" WITHOUT an LLM call ---
    ("https://vapegourmet.de", "error"),
    ("https://bonvape.es", "error"),
    # --- new, untested candidates: category unknown, eyeball live ---
    ("https://www.1vapereviews.com", None),
    ("https://vapeforums.lv", None),
    ("https://blog.vaporessobrasil.com", None),
    ("https://thaivapemall.bentoweb.com", None),
    ("https://vapemarket2.com", None),
    ("https://www.hongkongvapemall.com", None),
    ("https://www.vapebarmarket.de", None),
    ("https://vape89.it", None),
]


def main() -> None:
    from common.relevance import fetch

    print(f"relevance spike — model=qwen3.6:27b\n")
    print(f"{'url':42} {'gold':16} {'pred':16}  {'s':>5}  raw")

    rows = []
    for url, gold in LABELED:
        t0 = time.time()
        fr = fetch(url, timeout_s=10)
        result = classify_page(fr)
        dt = time.time() - t0
        match = "" if gold is None else ("OK" if result["category"] == gold else "MISMATCH")
        rows.append((url, gold, result["category"], dt, result.get("raw", ""), match))
        gold_s = gold or "?"
        print(f"  {url:40} {gold_s:16} {result['category']!s:16}  {dt:5.1f}  {match}")

    labeled_rows = [r for r in rows if r[1] is not None]
    correct = sum(1 for r in labeled_rows if r[1] == r[2])
    print(f"\naccuracy on labeled subset: {correct}/{len(labeled_rows)}")
    total_s = sum(r[3] for r in rows)
    print(f"total time: {total_s:.1f}s  avg: {total_s/len(rows):.2f}s/site")


if __name__ == "__main__":
    main()
