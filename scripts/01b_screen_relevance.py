#!/usr/bin/env python3
"""01b — Screen the WHOLE frame for relevance. Fetch is plain HTTP GET, no
browser, no screenshots — one request per site, once. This is what makes
screening the entire ~116k-site frame affordable in the first place: page
load (~1-2s) is far cheaper than a Playwright goto (~7-9s), so dropping the
browser lets the whole frame get screened instead of a sample; sampling
(01c_draw_sample.py) moves downstream to sites already confirmed relevant.

The JUDGMENT — is this page a nicotine retailer — is an LLM call
(common.vlm.classify_relevance(), frozen RELEVANCE_PROMPT), not a keyword
list. A deterministic keyword classifier structurally can't tell a perfume
site's "Tobacco Vanille" fragrance note from an actual tobacco retailer,
which is exactly the failure mode that motivated the switch (see
common/relevance.py's module docstring — the old classify_html() path is
kept there for reference, not used here). A block/bot-check page still
short-circuits to "error" deterministically, with no LLM call — recognizing
a Cloudflare interstitial is a mechanical string match, not a judgment call.

Fetched HTML is stored gzipped to data/html/<site_id>.html.gz — the source
of truth for this step, mirroring the funnel crawl's "screenshots are the
source of truth" invariant. --reclassify re-scores that stored HTML with NO
NETWORK calls, but still costs an LLM inference call per site — it is not
free, just not network-bound.

Serial: one request at a time, one host, once, with a polite inter-request
delay. No async pool — matches invariant #4's spirit (the funnel crawl's
serial-execution rule exists for anti-detection reasons that don't apply to
a single homepage GET, but there is still no reason for this step to be the
one that introduces parallelism into the codebase).

Resumable with no cursor file: the work queue is "frame rows whose
relevance_prompt_hash lags the live RELEVANCE_PROMPT_HASH" (db.iter_pending_relevance).
Run it in --limit-ed sessions; killing it mid-run and re-running picks up
exactly where it left off. A --limit-ed or interrupted run still balances
across strata (round-robin interleave below) rather than exhausting one
stratum before touching the next.

Run:
  ./.venv/bin/python scripts/01b_screen_relevance.py --limit 100
  ./.venv/bin/python scripts/01b_screen_relevance.py --retry-failed
  ./.venv/bin/python scripts/01b_screen_relevance.py --reclassify   # after a RELEVANCE_PROMPT change
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, db  # noqa: E402
from common.prompts import RELEVANCE_PROMPT_HASH  # noqa: E402
from common.relevance import FetchResult, classify_page, fetch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_DIR = REPO_ROOT / "data" / "html"


def _html_path(site_id: str) -> Path:
    return HTML_DIR / f"{site_id}.html.gz"


def _store_html(site_id: str, html: str) -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(_html_path(site_id), "wt", encoding="utf-8") as f:
        f.write(html)


def _load_html(site_id: str) -> Optional[str]:
    path = _html_path(site_id)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read()


def _interleave_by_stratum(rows: Iterable) -> Iterator:
    """Round-robin across sample_stratum so a --limit-ed or interrupted run
    covers every stratum a little rather than exhausting one alphabetically-
    first stratum before touching the next (iter_pending_relevance's SQL
    orders by stratum, site_id, which would otherwise front-load one bucket).
    """
    groups: dict = defaultdict(deque)
    order: list = []
    for r in rows:
        s = r["sample_stratum"]
        if s not in groups:
            order.append(s)
        groups[s].append(r)
    while any(groups[s] for s in order):
        for s in order:
            if groups[s]:
                yield groups[s].popleft()


def _to_row(result: dict, relevance_cfg: dict, http_status: Optional[int]) -> dict:
    """Shared result -> db.set_site_relevance kwargs mapping for both the
    live-fetch and --reclassify paths. `category=None` (LLM response
    unparseable — see vlm._parse_relevance) is recorded as "unparseable",
    never silently coerced to "unrelated": a junk response and a genuine
    negative judgment must stay distinguishable in the data.

    relevance_prompt_hash is stamped even on fetch-failure/block-page rows
    (no LLM was actually called for those) — it marks "processed under this
    01b pass," which is what makes a fetch-failed site correctly drop out of
    the default pending queue while still being reachable via --retry-failed.
    """
    category = result["category"] or "unparseable"
    included = 1 if category in relevance_cfg["include_categories"] else 0
    return dict(
        relevance_label=category,
        included=included,
        relevance_model=result["model"],
        relevance_prompt_hash=RELEVANCE_PROMPT_HASH,
        http_status=http_status,
    )


def _screen_one(site: object, relevance_cfg: dict) -> dict:
    """Fetch + classify one site. Returns kwargs for db.set_site_relevance."""
    fr = fetch(
        site["url_normalized"],
        timeout_s=relevance_cfg["timeout_s"],
        max_bytes=relevance_cfg["max_response_bytes"],
        user_agent=relevance_cfg["user_agent"],
    )
    if fr.html is not None:
        _store_html(site["site_id"], fr.html)

    result = classify_page(fr, max_body_chars=relevance_cfg["max_body_chars"])
    return _to_row(result, relevance_cfg, fr.http_status)


def _reclassify_one(site: object, relevance_cfg: dict) -> Optional[dict]:
    """Re-score already-stored HTML — no network. None if nothing is stored
    (the site previously failed to fetch; nothing to reclassify yet)."""
    html = _load_html(site["site_id"])
    if html is None:
        return None
    fr = FetchResult(html=html, http_status=site["http_status"], final_url=None, failure_reason=None)
    result = classify_page(fr, max_body_chars=relevance_cfg["max_body_chars"])
    return _to_row(result, relevance_cfg, site["http_status"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    ap.add_argument("--industry", default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after N sites this invocation")
    ap.add_argument("--retry-failed", action="store_true", help="also re-screen unreachable/blocked/error sites")
    ap.add_argument("--reclassify", action="store_true", help="re-score stored HTML only — NO network calls (still an LLM call per site)")
    ap.add_argument("--dry-run", action="store_true", help="report the work queue size and exit")
    args = ap.parse_args()

    relevance_cfg = config.load_global()["relevance"]

    conn = db.connect(args.db)
    db.init_db(conn)

    pending = list(db.iter_pending_relevance(
        conn, prompt_hash=RELEVANCE_PROMPT_HASH, industry=args.industry,
        retry_failed=args.retry_failed,
    ))
    pending = list(_interleave_by_stratum(pending))

    print(f"01b_screen_relevance — model={config.load_global()['models']['relevance']} "
          f"prompt_hash={RELEVANCE_PROMPT_HASH[:12]} pending={len(pending)}"
          f"{'  [RECLASSIFY, no network]' if args.reclassify else ''}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    if args.dry_run:
        conn.close()
        return

    if args.limit is not None:
        pending = pending[: args.limit]

    processed = skipped = 0
    label_counts: dict = {}
    include_categories = set(relevance_cfg["include_categories"])
    t0 = time.time()

    for i, site in enumerate(pending, start=1):
        if args.reclassify:
            result = _reclassify_one(site, relevance_cfg)
            if result is None:
                skipped += 1
                continue
        else:
            result = _screen_one(site, relevance_cfg)
            time.sleep(relevance_cfg["inter_request_delay_s"])

        db.set_site_relevance(conn, site["site_id"], **result)
        processed += 1
        label = result["relevance_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

        if i % 100 == 0 or i == len(pending):
            elapsed = time.time() - t0
            rate = elapsed / i
            included_so_far = sum(v for k, v in label_counts.items() if k in include_categories)
            yield_pct = 100 * included_so_far / max(1, processed)
            print(f"  {i}/{len(pending)}  {rate:.2f}s/site  included={included_so_far} ({yield_pct:.1f}% yield)")

    conn.close()

    dt = time.time() - t0
    print(f"\nprocessed {processed} sites in {dt:.1f}s ({dt/max(1, processed):.2f}s/site avg)"
          + (f", skipped {skipped} (no stored HTML)" if skipped else ""))
    print("label breakdown:")
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:20} {count}")


if __name__ == "__main__":
    main()
