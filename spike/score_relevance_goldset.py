#!/usr/bin/env python3
"""Score the hand-labeled relevance gold set against the frozen
RELEVANCE_PROMPT / classify_relevance() — the reproducible replacement for
the ad-hoc scratchpad scoring run that originally produced the P/R/F1
figures recorded in CLAUDE.md. Makes NO prompt changes: this measures the
CURRENT frozen instrument, it never edits or re-derives it.

Reads data/labels/relevance_goldset_claude-handlabeled_n150.txt (despite the
filename, the file now holds n=190 after the 2026-08-20 extension — rows
151-190 were added later; the original n=150 filename was kept for
continuity since it's referenced elsewhere). Runs the exact same
classify_page() orchestration 01b_screen_relevance.py uses in production
(same fetch -> block-page short-circuit -> optional browser re-render for
thin pages -> LLM judgment path), so this measures the real pipeline, not a
simplified stand-in.

Reports overall precision/recall/F1, a confusion matrix, per-language and
per-stratum recall breakdowns, and every mismatch for manual inspection —
mirroring the breakdown CLAUDE.md already cites for the original n=150 run,
so a re-run is directly comparable.

Run:
  ./.venv/bin/python spike/score_relevance_goldset.py
  ./.venv/bin/python spike/score_relevance_goldset.py --limit 150   # just the original set
  ./.venv/bin/python spike/score_relevance_goldset.py --no-browser-fallback
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from common import config  # noqa: E402
from common.prompts import RELEVANCE_PROMPT_HASH  # noqa: E402
from common.relevance import classify_page, fetch  # noqa: E402
from common.urls import normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDSET_PATH = REPO_ROOT / "data" / "labels" / "relevance_goldset_claude-handlabeled_n150.txt"

# Crude language guess from the ccTLD, ONLY for the reporting breakdown
# (matches the "English vs non-English-market" split CLAUDE.md already
# cites for the n=150 run) — not used anywhere in the actual judgment path.
_ENGLISH_MARKET_TLDS = frozenset({"com", "net", "org", "uk", "co.uk", "ca", "us", "au", "nz", "ie", "in", "io"})


def _load_goldset(path: Path, limit: Optional[int]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        url, label = parts[0], parts[1]
        note = parts[2] if len(parts) > 2 else ""
        rows.append((url, int(label), note))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _tld_of(url: str) -> str:
    n = normalize(url)
    return (n.cctld if n else None) or "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goldset", type=Path, default=GOLDSET_PATH)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows (e.g. 150 for the original set)")
    ap.add_argument("--no-browser-fallback", action="store_true")
    args = ap.parse_args()

    relevance_cfg = dict(config.load_global()["relevance"])
    include_categories = set(relevance_cfg["include_categories"])
    browser_fallback = relevance_cfg.get("browser_fallback") and not args.no_browser_fallback

    renderer = None
    if browser_fallback:
        from importlib import import_module

        mod = import_module("01b_screen_relevance")
        renderer = mod._BrowserRenderer(relevance_cfg)  # noqa: SLF001 — reusing production's exact renderer

    rows = _load_goldset(args.goldset, args.limit)
    print(f"score_relevance_goldset — n={len(rows)} prompt_hash={RELEVANCE_PROMPT_HASH[:12]} "
          f"model={config.load_global()['models']['relevance']} browser_fallback={'on' if browser_fallback else 'off'}")

    tp = fp = tn = fn = 0
    mismatches: list[tuple[str, int, str, str]] = []
    by_tld_correct: dict[str, list[bool]] = {}
    t0 = time.time()

    for i, (url, gold, note) in enumerate(rows, start=1):
        fr = fetch(url, timeout_s=relevance_cfg["timeout_s"],
                   max_bytes=relevance_cfg["max_response_bytes"],
                   user_agent=relevance_cfg["user_agent"])
        result = classify_page(
            fr, url=url,
            max_body_chars=relevance_cfg["max_body_chars"],
            thin_text_threshold=relevance_cfg["thin_text_threshold"] if browser_fallback else 0,
            render_fn=renderer.render if renderer is not None else None,
        )
        category = result["category"] or "unparseable"
        pred = 1 if category in include_categories else 0

        correct = pred == gold
        tld = _tld_of(url)
        by_tld_correct.setdefault(tld, []).append(correct)

        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1

        if not correct:
            mismatches.append((url, gold, category, note))

        if i % 25 == 0 or i == len(rows):
            elapsed = time.time() - t0
            print(f"  {i}/{len(rows)}  {elapsed/i:.2f}s/site  TP={tp} FP={fp} TN={tn} FN={fn}")

    if renderer is not None:
        renderer.close()

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    print(f"\nn={len(rows)}  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")

    english = [c for tld, cs in by_tld_correct.items() if tld in _ENGLISH_MARKET_TLDS for c in cs]
    non_english = [c for tld, cs in by_tld_correct.items() if tld not in _ENGLISH_MARKET_TLDS for c in cs]
    print(f"\nEnglish-market ccTLDs:     {sum(english)}/{len(english)} correct")
    print(f"Non-English-market ccTLDs: {sum(non_english)}/{len(non_english)} correct")

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es):")
        for url, gold, pred_cat, note in mismatches:
            print(f"  {url}")
            print(f"    gold={gold}  pred_category={pred_cat!r}")
            print(f"    note: {note}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s ({dt/max(1, len(rows)):.2f}s/site avg)")


if __name__ == "__main__":
    main()
