#!/usr/bin/env python3
"""01 — Build the crawl queue from a seed CSV.

Ingests ALL rows from one or more seed files: normalizes each URL (common.urls),
dedupes on the canonical host (www/port-collapsed — see common.urls.normalize),
derives a deterministic site_id + ccTLD-derived sample_stratum (common.sampling),
and upserts into `sites` with included=0 — NOT yet screened for relevance. That
happens in 01b_screen_relevance.py, a deterministic HTML classifier (no VLM, no
browser), run over the WHOLE frame this script produces. Sampling then happens
in 01c_draw_sample.py, over the sites 01b confirms relevant — sampling AFTER
relevance screening, not before, so the funnel-crawl budget (02_collect.py)
is spent on confirmed retailers rather than raw seed noise.

Idempotent and safe to re-run: db.upsert_site() COALESCE-protects included,
relevance_label, sample_u, and sample_wave, so a re-run of this script can
NEVER wipe out prior relevance-screening or sample-draw work. Frame facts
(host, cctld, registrable_domain, sample_stratum, ...) ARE refreshed on every
re-run — they're deterministic recomputations, not destructive.

Standalone step, run BEFORE 02_collect.py. Not folded into a "phase" — see
CLAUDE.md's build-status list.

Run:
  ./.venv/bin/python scripts/01_build_queue.py
  ./.venv/bin/python scripts/01_build_queue.py --seed data/seed/x.csv --industry vape
  ./.venv/bin/python scripts/01_build_queue.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import db  # noqa: E402
from common.sampling import stratum_for_cctld  # noqa: E402
from common.urls import Normalized, better_origin, make_site_id, normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = REPO_ROOT / "data" / "seed" / "nicotine-industry-sites.csv"
LOG_DIR = REPO_ROOT / "data" / "logs"

# The known seed's exact row count (verified against the actual file: no
# trailing newline, so a naive split('\n') that drops the tail silently
# undercounts by 1 — this assertion catches that regression immediately).
KNOWN_SEED_ROW_COUNTS = {DEFAULT_SEED.name: 133982}


def _infer_industry(seed_path: Path) -> str:
    """'nicotine-industry-sites.csv' -> 'nicotine' (first hyphen segment)."""
    return seed_path.stem.split("-")[0]


def _read_seed_rows(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for line in f:
            raw = line.strip()
            if raw:
                yield raw


class _Record:
    __slots__ = ("raw", "n", "source_file")

    def __init__(self, raw: str, n: Normalized, source_file: str):
        self.raw = raw
        self.n = n
        self.source_file = source_file


def reduce_seeds(seed_paths: list[Path]) -> tuple[dict[str, _Record], list[tuple[str, str, str]], Counter]:
    """Stream + normalize + dedupe every seed row. Returns (by_key, rejects,
    row_counts_per_file). Pure/offline — no DB, no network, safe to call in
    --dry-run.
    """
    by_key: dict[str, _Record] = {}
    rejects: list[tuple[str, str, str]] = []  # (raw_line, reason, source_file)
    row_counts: Counter = Counter()

    for seed_path in seed_paths:
        name = seed_path.name
        for raw in _read_seed_rows(seed_path):
            row_counts[name] += 1
            n = normalize(raw)
            if n is None:
                rejects.append((raw, "unparseable", name))
                continue
            existing = by_key.get(n.url_key)
            if existing is None:
                by_key[n.url_key] = _Record(raw, n, name)
            else:
                winner = better_origin(existing.n.url_normalized, n.url_normalized)
                if winner != existing.n.url_normalized:
                    by_key[n.url_key] = _Record(raw, n, name)
        expected = KNOWN_SEED_ROW_COUNTS.get(name)
        if expected is not None and row_counts[name] != expected:
            raise AssertionError(
                f"{name}: read {row_counts[name]} rows, expected exactly {expected}. "
                "This seed's row count is verified and pinned — an unexpected count "
                "means the read path silently dropped or duplicated rows (check for "
                "a trailing-newline / split('\\n') regression)."
            )

    return by_key, rejects, row_counts


def assign_site_ids(
    by_key: dict[str, _Record], industry: str
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Derive site_id + sample_stratum for each surviving key. Returns
    (rows_ready_for_db, additional_rejects).
    """
    rows: list[dict] = []
    rejects: list[tuple[str, str, str]] = []
    seen_site_ids: set[str] = set()

    for url_key, rec in by_key.items():
        site_id = make_site_id(url_key)
        if site_id in seen_site_ids:
            # Structurally shouldn't happen: site_id is a function of url_key,
            # and url_key is the dict key itself (so collisions here would mean
            # two DIFFERENT url_keys hashed to the same 6-hex site_id suffix —
            # a 24-bit collision). Loud rejection, never silent data loss.
            rejects.append((rec.raw, f"site_id collision: {site_id}", rec.source_file))
            continue
        seen_site_ids.add(site_id)
        rows.append(
            dict(
                site_id=site_id,
                url_original=rec.raw,
                url_normalized=rec.n.url_normalized,
                host=rec.n.host,
                registrable_domain=rec.n.registrable_domain,
                cctld=rec.n.cctld,
                industry=industry,
                source_file=rec.source_file,
                sample_stratum=stratum_for_cctld(rec.n.cctld),
            )
        )
    return rows, rejects


def write_rows(
    db_path: Path, rows: list[dict], *, commit_every: int = 10000
) -> list[tuple[str, str, str]]:
    """Upsert every row; returns any (url, reason, source_file) IntegrityError
    rejects. Never silently drops a row on conflict — every reject is logged.
    """
    rejects: list[tuple[str, str, str]] = []
    conn = db.connect(db_path)
    db.init_db(conn)
    try:
        for i, row in enumerate(rows, start=1):
            try:
                db.upsert_site(conn, **row)
            except sqlite3.IntegrityError as e:
                rejects.append((row["url_original"], f"IntegrityError: {e}", row["source_file"]))
                continue
            if i % commit_every == 0:
                conn.commit()
        conn.commit()
    finally:
        conn.close()
    return rejects


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--seed", action="append", type=Path, default=None,
        help=f"seed CSV, repeatable (default: {DEFAULT_SEED.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--industry", default=None,
        help="industry label for every ingested row (default: inferred from the first --seed filename)",
    )
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB, help="SQLite DB path")
    ap.add_argument("--dry-run", action="store_true", help="normalize + report only, write nothing")
    args = ap.parse_args()

    seed_paths = args.seed or [DEFAULT_SEED]
    for p in seed_paths:
        if not p.exists():
            print(f"ERROR: seed file not found: {p}", file=sys.stderr)
            sys.exit(1)
    industry = args.industry or _infer_industry(seed_paths[0])

    print(f"01_build_queue — seeds={[str(p) for p in seed_paths]} industry={industry!r} "
          f"db={args.db}{'  [DRY RUN]' if args.dry_run else ''}")

    t0 = time.time()
    by_key, rejects, row_counts = reduce_seeds(seed_paths)
    rows, more_rejects = assign_site_ids(by_key, industry)
    rejects.extend(more_rejects)

    total_read = sum(row_counts.values())
    print(f"\nread {total_read} rows across {len(seed_paths)} file(s)")
    for name, count in row_counts.items():
        print(f"  {name}: {count}")
    print(f"frame size (distinct sites): {len(rows)}")
    print(f"rejects: {len(rejects)}")

    stratum_counts = Counter(r["sample_stratum"] for r in rows)
    print("\nstratum distribution:")
    for stratum, count in stratum_counts.most_common():
        pct = 100 * count / len(rows) if rows else 0.0
        print(f"  {stratum:10} {count:7}  ({pct:.1f}%)")

    write_rejects = None
    if not args.dry_run:
        db_rejects = write_rows(args.db, rows)
        rejects.extend(db_rejects)
        print(f"\nwrote {len(rows) - len(db_rejects)} rows to {args.db}")
        if db_rejects:
            print(f"  ({len(db_rejects)} additional DB-level rejects — see log)")

    if rejects:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        write_rejects = LOG_DIR / f"01_build_queue_rejects_{ts}.csv"
        with open(write_rejects, "w", encoding="utf-8", newline="") as f:
            f.write("raw_line,reason,source_file\n")
            for raw, reason, source_file in rejects:
                safe_raw = raw.replace('"', '""')
                safe_reason = reason.replace('"', '""')
                f.write(f'"{safe_raw}","{safe_reason}","{source_file}"\n')
        print(f"\nrejects written to {write_rejects}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.2f}s{'  [DRY RUN — nothing written]' if args.dry_run else ''}")

    if rejects and not args.dry_run:
        print(
            "\nNOTE: non-zero rejects on a real run usually means a normalization "
            "rule has a bug — inspect the rejects log before trusting the frame.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
