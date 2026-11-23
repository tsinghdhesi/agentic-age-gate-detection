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
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, db, stealth  # noqa: E402
from common.prompts import RELEVANCE_PROMPT_HASH  # noqa: E402
from common.relevance import FetchResult, classify_page, fetch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HTML_DIR = REPO_ROOT / "data" / "html"
CHUNK_LOG_DIR = REPO_ROOT / "data" / "logs" / "chunks"
CONSUMER = "01b_screen_relevance"


class _BrowserRenderer:
    """Lazily-started stealth browser used ONLY to re-render pages whose plain
    HTTP fetch came back with almost no text (fully JS-rendered storefronts).

    Injected into common.relevance.classify_page() as `render_fn` so that
    relevance.py — the deliberately browser-free HTTP layer — never imports
    Playwright. The browser is started on first actual need and reused, so a
    run where nothing is thin never pays to launch Chromium at all.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._cm = None
        self._browser = None
        self.renders = 0

    def _ensure(self) -> None:
        if self._browser is None:
            self._cm = stealth.stealth_playwright()
            p = self._cm.__enter__()
            self._browser = stealth.launch_browser(p, headless=True)

    def render(self, url: str) -> Optional[str]:
        """Return rendered HTML, or None on any failure — a browser problem
        must degrade to 'classify what HTTP gave us', never kill the screen."""
        try:
            self._ensure()
            ctx = self._browser.new_context()
            try:
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded",
                          timeout=self._cfg["browser_timeout_ms"])
                page.wait_for_timeout(self._cfg["browser_settle_ms"])
                html = page.content()
                self.renders += 1
                return html
            finally:
                ctx.close()
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
            if self._cm is not None:
                self._cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


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


def _classify_kwargs(relevance_cfg: dict, renderer: Optional[_BrowserRenderer]) -> dict:
    """Shared classify_page() options for both the live and --reclassify paths."""
    enabled = relevance_cfg.get("browser_fallback") and renderer is not None
    return dict(
        max_body_chars=relevance_cfg["max_body_chars"],
        thin_text_threshold=relevance_cfg["thin_text_threshold"] if enabled else 0,
        render_fn=renderer.render if enabled else None,
    )


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

    The returned dict carries one extra key, `_latency_ms`, that is NOT a
    db.set_site_relevance() parameter — it's per-chunk JSONL telemetry only.
    The caller pops it before the DB write; see main()'s chunked logging path.
    """
    category = result["category"] or "unparseable"
    included = 1 if category in relevance_cfg["include_categories"] else 0
    return dict(
        relevance_label=category,
        included=included,
        relevance_model=result["model"],
        relevance_prompt_hash=RELEVANCE_PROMPT_HASH,
        relevance_fetch_mode=result.get("fetch_mode"),
        http_status=http_status,
        _latency_ms=result.get("latency_ms"),
    )


def _screen_one(site: object, relevance_cfg: dict,
                renderer: Optional[_BrowserRenderer] = None) -> dict:
    """Fetch + classify one site. Returns kwargs for db.set_site_relevance."""
    fr = fetch(
        site["url_normalized"],
        timeout_s=relevance_cfg["timeout_s"],
        max_bytes=relevance_cfg["max_response_bytes"],
        user_agent=relevance_cfg["user_agent"],
    )
    if fr.html is not None:
        _store_html(site["site_id"], fr.html)

    result = classify_page(fr, url=site["url_normalized"],
                           **_classify_kwargs(relevance_cfg, renderer))
    return _to_row(result, relevance_cfg, fr.http_status)


def _reclassify_one(site: object, relevance_cfg: dict,
                    renderer: Optional[_BrowserRenderer] = None) -> Optional[dict]:
    """Re-score already-stored HTML. None if nothing is stored (the site
    previously failed to fetch; nothing to reclassify yet).

    NOTE: with browser_fallback on, a thin stored page WILL trigger a live
    re-render, so --reclassify is not strictly network-free for that subset.
    That is deliberate — re-scoring a JS shell against a revised prompt would
    otherwise just reproduce the original miss.
    """
    html = _load_html(site["site_id"])
    if html is None:
        return None
    fr = FetchResult(html=html, http_status=site["http_status"],
                     final_url=site["url_normalized"], failure_reason=None)
    result = classify_page(fr, url=site["url_normalized"],
                           **_classify_kwargs(relevance_cfg, renderer))
    return _to_row(result, relevance_cfg, site["http_status"])


def _git_sha() -> Optional[str]:
    """Best-effort git HEAD sha for chunk JSONL provenance. None on any
    failure (no git installed, not a repo, detached weirdness) — this is
    telemetry, never allowed to break a multi-day screen."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _chunk_jsonl_path(chunk_set_id: str, chunk_index: int, scope: str) -> Path:
    """data/logs/chunks/<set>/c<idx:05d>__<consumer>__<scope[:12]>__<ts>.jsonl —
    one file PER ATTEMPT (the ts makes each start_chunk_run call get its own
    file), so a crash and its resume are both preserved rather than one
    overwriting the other."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"c{chunk_index:05d}__{CONSUMER}__{scope[:12]}__{ts}.jsonl"
    return CHUNK_LOG_DIR / chunk_set_id / name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    ap.add_argument("--industry", default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after N sites this invocation")
    ap.add_argument("--retry-failed", action="store_true", help="also re-screen unreachable/blocked/error sites")
    ap.add_argument("--reclassify", action="store_true", help="re-score stored HTML (still an LLM call per site; thin pages may re-render)")
    ap.add_argument("--no-browser-fallback", action="store_true", help="disable the JS re-render for thin pages")
    ap.add_argument("--dry-run", action="store_true", help="report the work queue size and exit")
    ap.add_argument("--chunk-set", dest="chunk_set", default=None,
                    help="chunk set id (see scripts/chunk_work.py create) — restricts work to one chunk of it")
    ap.add_argument("--chunk", default=None,
                    help="chunk index within --chunk-set, or 'next' for the lowest chunk with pending work")
    args = ap.parse_args()

    if (args.chunk_set is None) != (args.chunk is None):
        ap.error("--chunk-set and --chunk must be given together")

    relevance_cfg = dict(config.load_global()["relevance"])
    if args.no_browser_fallback:
        relevance_cfg["browser_fallback"] = False
    # Caps --retry-failed re-attempts on unreachable/blocked/error sites (see
    # schema v5 / sites.relevance_attempts) — applies whether or not --chunk
    # is used; a fresh or stale-hash site is unaffected either way.
    max_attempts = relevance_cfg.get("max_attempts")

    conn = db.connect(args.db)
    db.init_db(conn)

    chunk_set_id = args.chunk_set
    chunk_index: Optional[int] = None
    if chunk_set_id is not None:
        if db.get_chunk_set(conn, chunk_set_id) is None:
            print(f"ERROR: no such chunk set {chunk_set_id!r} — create it with "
                  "scripts/chunk_work.py create", file=sys.stderr)
            conn.close()
            sys.exit(1)
        if args.chunk == "next":
            chunk_index = db.next_pending_chunk_index(
                conn, chunk_set_id, prompt_hash=RELEVANCE_PROMPT_HASH, industry=args.industry,
                retry_failed=args.retry_failed, max_attempts=max_attempts,
            )
            if chunk_index is None:
                print(f"01b_screen_relevance — chunk set {chunk_set_id!r}: every chunk is "
                      "already fully processed under the current queue definition. Nothing to do.")
                conn.close()
                return
        else:
            try:
                chunk_index = int(args.chunk)
            except ValueError:
                ap.error("--chunk must be an integer or 'next'")

    pending = list(db.iter_pending_relevance(
        conn, prompt_hash=RELEVANCE_PROMPT_HASH, industry=args.industry,
        retry_failed=args.retry_failed, max_attempts=max_attempts,
        chunk_set=chunk_set_id, chunk_index=chunk_index,
    ))
    if chunk_set_id is None:
        pending = list(_interleave_by_stratum(pending))
    # else: chunk membership (common.chunking.partition) is already a
    # uniformly random permutation of the frame, and iter_pending_relevance
    # already ordered this by member_rank — round-robin interleaving is
    # neither needed (any prefix is already an unbiased subsample) nor
    # correct here (it would destroy that ordering).

    # A completed chunk re-run is a LOUD NO-OP that exits 0, so a
    # `for i in $(seq 0 46); do ...; done` driver is idempotent and
    # restartable without special-casing "already done" at the shell level.
    if chunk_set_id is not None and not pending:
        print(f"01b_screen_relevance — chunk {chunk_index} of {chunk_set_id!r}: "
              "nothing pending for this invocation's flags. Already done — no-op.")
        conn.close()
        return

    fb = "on" if relevance_cfg.get("browser_fallback") else "off"
    chunk_info = f" chunk_set={chunk_set_id!r} chunk={chunk_index}" if chunk_set_id is not None else ""
    print(f"01b_screen_relevance — model={config.load_global()['models']['relevance']} "
          f"prompt_hash={RELEVANCE_PROMPT_HASH[:12]} pending={len(pending)}{chunk_info} "
          f"browser_fallback={fb}"
          f"{'  [RECLASSIFY]' if args.reclassify else ''}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    if args.dry_run:
        conn.close()
        return

    n_pending_start = len(pending)  # BEFORE --limit truncation — the true queue size this chunk saw
    if args.limit is not None:
        pending = pending[: args.limit]

    # Chunked-run bookkeeping: JSONL log + chunk_runs lifecycle. None of this
    # runs on the non-chunked path, which stays exactly as before (zero files
    # written, no chunk_runs row).
    jsonl_file = None
    chunk_scope = RELEVANCE_PROMPT_HASH
    if chunk_set_id is not None:
        prior_run = db.get_chunk_run(
            conn, chunk_set_id=chunk_set_id, chunk_index=chunk_index, consumer=CONSUMER, scope=chunk_scope,
        )
        if prior_run is not None and prior_run["status"] == "running":
            print(f"WARNING: chunk {chunk_index} of {chunk_set_id!r} has a stale 'running' row "
                  f"(attempt {prior_run['attempts']}, started {prior_run['started']}, "
                  f"host={prior_run['host']} pid={prior_run['pid']}) — a prior process likely "
                  "crashed. chunk_runs is not a lock; overwriting it, NOT blocking.")

        n_members = len(list(db.iter_chunk_members(conn, chunk_set_id, chunk_index)))
        jsonl_path = _chunk_jsonl_path(chunk_set_id, chunk_index, chunk_scope)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(jsonl_path, "w", encoding="utf-8")

        db.start_chunk_run(
            conn, chunk_set_id=chunk_set_id, chunk_index=chunk_index, consumer=CONSUMER,
            scope=chunk_scope, n_members=n_members, n_pending_start=n_pending_start,
            host=socket.gethostname(), pid=os.getpid(),
            jsonl_path=str(jsonl_path.relative_to(REPO_ROOT)),
        )
        jsonl_file.write(json.dumps({
            "type": "meta", "chunk_set": chunk_set_id, "chunk_index": chunk_index,
            "consumer": CONSUMER, "scope": chunk_scope, "n_members": n_members,
            "n_pending_start": n_pending_start, "limit": args.limit,
            "retry_failed": args.retry_failed, "reclassify": args.reclassify,
            "industry": args.industry, "no_browser_fallback": args.no_browser_fallback,
            "max_attempts": max_attempts, "model": config.load_global()["models"]["relevance"],
            "prompt_hash": RELEVANCE_PROMPT_HASH, "git_sha": _git_sha(), "ts": db.now_iso(),
        }) + "\n")
        jsonl_file.flush()

    processed = skipped = skipped_no_html = 0
    label_counts: dict = {}
    include_categories = set(relevance_cfg["include_categories"])
    renderer = _BrowserRenderer(relevance_cfg) if relevance_cfg.get("browser_fallback") else None
    t0 = time.time()
    last_error: Optional[str] = None

    try:
        for i, site in enumerate(pending, start=1):
            if args.reclassify:
                result = _reclassify_one(site, relevance_cfg, renderer)
                if result is None:
                    skipped += 1
                    skipped_no_html += 1
                    continue
            else:
                result = _screen_one(site, relevance_cfg, renderer)
                time.sleep(relevance_cfg["inter_request_delay_s"])

            latency_ms = result.pop("_latency_ms", None)
            db.set_site_relevance(conn, site["site_id"], **result)
            processed += 1
            label = result["relevance_label"]
            label_counts[label] = label_counts.get(label, 0) + 1

            if jsonl_file is not None:
                jsonl_file.write(json.dumps({
                    "type": "site", "site_id": site["site_id"], "url": site["url_normalized"],
                    "relevance_label": label, "included": result["included"],
                    "http_status": result["http_status"], "fetch_mode": result["relevance_fetch_mode"],
                    "latency_ms": latency_ms, "ts": db.now_iso(),
                }) + "\n")
                jsonl_file.flush()

            if i % 100 == 0 or i == len(pending):
                elapsed = time.time() - t0
                rate = elapsed / i
                included_so_far = sum(v for k, v in label_counts.items() if k in include_categories)
                yield_pct = 100 * included_so_far / max(1, processed)
                print(f"  {i}/{len(pending)}  {rate:.2f}s/site  included={included_so_far} ({yield_pct:.1f}% yield)")
    except KeyboardInterrupt:
        last_error = "KeyboardInterrupt"
        raise
    except Exception as e:  # noqa: BLE001 — must still finish the chunk_run record before propagating
        last_error = f"{type(e).__name__}: {e}"
        raise
    finally:
        if renderer is not None:
            renderer.close()
        if chunk_set_id is not None:
            # Status is ALWAYS derived from the BASE queue definition (prompt
            # hash only — no retry_failed/industry), regardless of which
            # flags THIS invocation used to select its own work. Otherwise a
            # --retry-failed or --industry run could regress an
            # already-"done" chunk back to "partial" just because it looked
            # at a wider or narrower slice than the default.
            remaining = db.count_pending_relevance(
                conn, prompt_hash=RELEVANCE_PROMPT_HASH, chunk_set=chunk_set_id, chunk_index=chunk_index,
            )
            if skipped_no_html and remaining:
                note = (f"{skipped_no_html} site(s) skipped: no stored HTML "
                        "(a --reclassify-only pass can't process them — run without "
                        "--reclassify at least once first)")
                last_error = f"{last_error}; {note}" if last_error else note
            if last_error is not None:
                status = "interrupted"
            elif remaining == 0:
                status = "done"
            else:
                status = "partial"
            db.finish_chunk_run(
                conn, chunk_set_id=chunk_set_id, chunk_index=chunk_index, consumer=CONSUMER,
                scope=chunk_scope, status=status, n_pending_end=remaining,
                n_processed=processed, last_error=last_error,
            )
            if jsonl_file is not None:
                jsonl_file.write(json.dumps({
                    "type": "summary", "status": status, "n_processed": processed,
                    "n_pending_end": remaining, "skipped_no_html": skipped_no_html,
                    "last_error": last_error, "ts": db.now_iso(),
                }) + "\n")
                jsonl_file.close()
            print(f"\nchunk {chunk_index} of {chunk_set_id!r}: status={status} "
                  f"processed={processed} pending_remaining={remaining}")

    conn.close()

    dt = time.time() - t0
    print(f"\nprocessed {processed} sites in {dt:.1f}s ({dt/max(1, processed):.2f}s/site avg)"
          + (f", skipped {skipped} (no stored HTML)" if skipped else ""))
    if renderer is not None:
        print(f"browser re-renders (thin pages): {renderer.renders}")
    print("label breakdown:")
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:20} {count}")


if __name__ == "__main__":
    main()
