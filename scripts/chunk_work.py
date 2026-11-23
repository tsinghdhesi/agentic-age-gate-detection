#!/usr/bin/env python3
"""chunk_work — cross-cutting utility for addressable, resumable work chunks.

Deliberately UNNUMBERED: unlike 01/01b/01c/02/03, this is not a pipeline
stage. It creates and inspects the `chunk_sets`/`chunk_members`/`chunk_runs`
partition (schema v5, common/chunking.py) that 01b_screen_relevance.py's
--chunk-set/--chunk flags consume, and that a future 02_collect.py /
03_classify.py will consume the same way with a different `scope`.

RULE OF ONE TRUTH (see CLAUDE.md): a site's completion state lives in the
site's own row (sites.relevance_prompt_hash / site_outcomes / detections).
chunk_members decides which sites an invocation may TOUCH. chunk_runs
records what HAPPENED. This script writes both tables but never uses
chunk_runs to compute "pending" — `status` always re-derives pending counts
live from the same pending ⋈ chunk_members join the work scripts use
(db.count_pending_relevance), and displays the recorded chunk_runs.status
alongside it so any disagreement between "what we last recorded" and "what
is actually still pending" (e.g. after a prompt revision) is visible, not
hidden.

Subcommands:
  create  --set ID --preset frame|drawn-sample --chunk-size N --consumer C
          (or --where SQL / --from-file PATH instead of --preset)
  extend  --set ID          (chunks only the complement, as a new generation)
  verify  --set ID          (re-derives every generation's partition, asserts match)
  status  --set ID          (per-chunk membership + recorded run + live pending)
  unstamp --set ID --chunk N --yes   (auditable redo: dumps labels, clears the hash)

Run:
  ./.venv/bin/python scripts/chunk_work.py create --set frame-v1 --preset frame \\
      --chunk-size 2500 --consumer 01b_screen_relevance --dry-run
  ./.venv/bin/python scripts/chunk_work.py create --set frame-v1 --preset frame \\
      --chunk-size 2500 --consumer 01b_screen_relevance
  ./.venv/bin/python scripts/chunk_work.py verify --set frame-v1
  ./.venv/bin/python scripts/chunk_work.py status --set frame-v1 --failed
  ./.venv/bin/python scripts/chunk_work.py unstamp --set frame-v1 --chunk 3 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import chunking, config, db  # noqa: E402
from common.prompts import RELEVANCE_PROMPT_HASH  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "data" / "logs" / "chunks"

_PRESETS: dict[str, Optional[str]] = {
    # "frame": every site 01_build_queue.py ingested — 01b's consumer.
    "frame": None,
    # "drawn-sample": sites 01c_draw_sample.py has drawn into a wave — the
    # population 02_collect.py will eventually consume (scope=run_id).
    "drawn-sample": "included = 1 AND sample_wave IS NOT NULL",
}

# Default chunk sizes per consumer (config/global.yaml: chunking.*), used
# when --chunk-size is omitted for a recognized consumer.
_DEFAULT_CHUNK_SIZE_KEY = {
    "01b_screen_relevance": "chunk_size_01b",
    "02_collect": "chunk_size_02",
}

# --where must be a read-only predicate: reject anything that could mutate
# the DB, and reject statement-separators outright. This is a denylist, not
# a parser — it exists to catch mistakes and casual misuse, not to be an
# airtight sandbox (the same trust boundary as chunk_work.py's operator).
_BANNED_SQL_TOKENS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|TRUNCATE|EXEC|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


def _validate_where(conn, where_sql: str) -> str:
    if ";" in where_sql:
        raise SystemExit("ERROR: --where must not contain ';'")
    if _BANNED_SQL_TOKENS.search(where_sql):
        raise SystemExit("ERROR: --where must be a read-only predicate (no DDL/DML keywords)")
    try:
        conn.execute(f"SELECT COUNT(*) FROM sites WHERE {where_sql}")
    except Exception as e:  # noqa: BLE001 — surface as a clean CLI error, not a traceback
        raise SystemExit(f"ERROR: --where does not parse against sites: {e}") from e
    return where_sql


def _resolve_where(conn, args) -> tuple[Optional[str], list]:
    """One of --preset / --where / --from-file -> (where_sql, params) for
    iter_sites_with_stratum(), or (where_sql, params, explicit_ids) when
    --from-file supplies a literal site_id list instead of a predicate.
    """
    picked = [x for x in (args.preset, args.where, args.from_file) if x]
    if len(picked) != 1:
        raise SystemExit("ERROR: exactly one of --preset / --where / --from-file is required")

    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            raise SystemExit(f"ERROR: --from-file not found: {path}")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return None, [], ids

    where_sql = _PRESETS[args.preset] if args.preset else _validate_where(conn, args.where)
    if args.industry is not None:
        industry_clause = "industry = ?"
        where_sql = f"({where_sql}) AND {industry_clause}" if where_sql else industry_clause
        return where_sql, [args.industry], None
    return where_sql, [], None


def _chunk_size_for(consumer: str, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    key = _DEFAULT_CHUNK_SIZE_KEY.get(consumer)
    if key is None:
        raise SystemExit(
            f"ERROR: --chunk-size is required for consumer {consumer!r} "
            "(no default in config/global.yaml: chunking.*)"
        )
    return config.load_global()["chunking"][key]


def _print_stratum_table(assignments: list[tuple[str, int, int, int]], stratum_by_id: dict[str, str]) -> None:
    """Per-chunk stratum composition, matching 01c_draw_sample.py's --dry-run
    table convention. Prints every chunk — at chunk_size~2500 that's ~47
    rows, small enough to read in full rather than sampling a subset.
    """
    from collections import Counter, defaultdict

    by_chunk: dict[int, Counter] = defaultdict(Counter)
    sizes: dict[int, int] = {}
    for site_id, chunk_index, _rank, _gen in assignments:
        by_chunk[chunk_index][stratum_by_id.get(site_id, "?")] += 1
        sizes[chunk_index] = sizes.get(chunk_index, 0) + 1

    all_strata = sorted({s for c in by_chunk.values() for s in c})
    header = f"  {'chunk':>6} {'n':>6}  " + "  ".join(f"{s:>9}" for s in all_strata)
    print(header)
    for idx in sorted(by_chunk):
        row = f"  {idx:6} {sizes[idx]:6}  " + "  ".join(f"{by_chunk[idx].get(s, 0):9}" for s in all_strata)
        print(row)


def cmd_create(args) -> None:
    conn = db.connect(args.db)
    db.init_db(conn)

    if db.get_chunk_set(conn, args.set) is not None:
        raise SystemExit(
            f"ERROR: chunk set {args.set!r} already exists — use `extend` to grow it, "
            "chunk_work.py does not redefine an existing set."
        )

    where_sql, params, explicit_ids = _resolve_where(conn, args)
    seed = args.seed or config.load_global()["chunking"]["chunk_seed"]
    chunk_size = _chunk_size_for(args.consumer, args.chunk_size)

    if explicit_ids is not None:
        site_ids = explicit_ids
        stratum_by_id = {row["site_id"]: row["sample_stratum"] for row in db.iter_sites_with_stratum(conn)}
    else:
        rows = list(db.iter_sites_with_stratum(conn, where_sql=where_sql, params=params))
        site_ids = [r["site_id"] for r in rows]
        stratum_by_id = {r["site_id"]: r["sample_stratum"] for r in rows}

    assignments = chunking.partition(site_ids, seed=seed, chunk_size=chunk_size)
    n_chunks = chunking.n_chunks(len(site_ids), chunk_size)

    print(f"chunk_work create — set={args.set!r} consumer={args.consumer!r} "
          f"n_sites={len(site_ids)} chunk_size={chunk_size} seed={seed!r} "
          f"n_chunks={n_chunks}{'  [DRY RUN]' if args.dry_run else ''}")
    _print_stratum_table(assignments, stratum_by_id)

    if args.dry_run:
        conn.close()
        return

    db.create_chunk_set(
        conn, chunk_set_id=args.set, intended_consumer=args.consumer, seed=seed,
        chunk_size=chunk_size, selector_sql=where_sql, notes=args.notes,
    )
    db.insert_chunk_members(conn, args.set, assignments)
    print(f"\ncreated chunk set {args.set!r}: {len(site_ids)} members across {n_chunks} chunks")
    conn.close()


def cmd_extend(args) -> None:
    conn = db.connect(args.db)
    db.init_db(conn)

    chunk_set = db.get_chunk_set(conn, args.set)
    if chunk_set is None:
        raise SystemExit(f"ERROR: no such chunk set {args.set!r} — use `create` first")

    where_sql = chunk_set["selector_sql"]
    rows = list(db.iter_sites_with_stratum(conn, where_sql=where_sql))
    current_ids = {r["site_id"] for r in rows}
    stratum_by_id = {r["site_id"]: r["sample_stratum"] for r in rows}

    existing = db.existing_member_site_ids(conn, args.set)
    new_ids = sorted(current_ids - existing)  # sorted: deterministic input order (partition() re-sorts anyway)

    next_gen = (db.max_generation(conn, args.set) or 0) + 1
    next_idx = (db.max_chunk_index(conn, args.set) or -1) + 1

    print(f"chunk_work extend — set={args.set!r} new_members={len(new_ids)} "
          f"generation={next_gen} first_chunk_index={next_idx}{'  [DRY RUN]' if args.dry_run else ''}")

    if not new_ids:
        print("nothing to extend — the eligible population has no new members")
        conn.close()
        return

    assignments = chunking.partition(
        new_ids, seed=chunk_set["seed"], chunk_size=chunk_set["chunk_size"],
        generation=next_gen, first_chunk_index=next_idx,
    )
    _print_stratum_table(assignments, stratum_by_id)

    if args.dry_run:
        conn.close()
        return

    db.insert_chunk_members(conn, args.set, assignments)
    print(f"\nextended {args.set!r}: +{len(new_ids)} members, generation {next_gen}")
    conn.close()


def cmd_verify(args) -> None:
    """Re-derive every generation's partition from (seed, generation,
    recorded members) and assert every (chunk_index, member_rank) matches —
    the reproducibility guarantee a reviewer can actually run, rather than a
    seed column they must trust.
    """
    conn = db.connect(args.db)
    db.init_db(conn)

    chunk_set = db.get_chunk_set(conn, args.set)
    if chunk_set is None:
        raise SystemExit(f"ERROR: no such chunk set {args.set!r}")

    all_members = list(db.iter_chunk_members(conn, args.set))
    from collections import defaultdict

    by_generation: dict[int, list] = defaultdict(list)
    for m in all_members:
        by_generation[m["generation"]].append(m)

    ok = True
    for generation in sorted(by_generation):
        members = by_generation[generation]
        site_ids = [m["site_id"] for m in members]
        recorded = {(m["site_id"], m["chunk_index"], m["member_rank"]) for m in members}
        first_chunk_index = min(m["chunk_index"] for m in members)

        rederived = chunking.partition(
            site_ids, seed=chunk_set["seed"], chunk_size=chunk_set["chunk_size"],
            generation=generation, first_chunk_index=first_chunk_index,
        )
        rederived_set = {(sid, ci, mr) for sid, ci, mr, _g in rederived}

        matched = recorded == rederived_set
        ok = ok and matched
        status = "OK" if matched else "MISMATCH"
        print(f"generation {generation}: {len(members)} members, "
              f"{len(set(m['chunk_index'] for m in members))} chunks — {status}")
        if not matched:
            print(f"  recorded-only: {len(recorded - rederived_set)}  rederived-only: {len(rederived_set - recorded)}")

    print(f"\n{'PASS' if ok else 'FAIL'} — chunk set {args.set!r}, "
          f"{len(all_members)} total members across {len(by_generation)} generation(s)")
    conn.close()
    if not ok:
        sys.exit(1)


def cmd_status(args) -> None:
    conn = db.connect(args.db)
    db.init_db(conn)

    chunk_set = db.get_chunk_set(conn, args.set)
    if chunk_set is None:
        raise SystemExit(f"ERROR: no such chunk set {args.set!r}")

    consumer = args.consumer or chunk_set["intended_consumer"]
    scope = args.scope or (RELEVANCE_PROMPT_HASH if consumer == "01b_screen_relevance" else None)
    if scope is None and not args.all_scopes:
        raise SystemExit(
            f"ERROR: no default scope known for consumer {consumer!r} — pass --scope explicitly "
            "or use --all-scopes to see every recorded scope"
        )

    print(f"chunk_work status — set={args.set!r} consumer={consumer!r} "
          f"scope={scope[:12] + '…' if scope else '(all)'}")

    indices = db.chunk_indices(conn, args.set)
    runs_by_key = {}
    for run in db.iter_chunk_runs(conn, args.set, consumer=consumer if not args.all_scopes else None):
        runs_by_key.setdefault(run["chunk_index"], []).append(run)

    print(f"\n  {'chunk':>6} {'n_mem':>6} {'live_pending':>12} {'recorded':>10} {'finished':>21}")
    any_failed = False
    for idx in indices:
        n_members = len(list(db.iter_chunk_members(conn, args.set, idx)))
        live_pending = None
        if scope is not None:
            # Dispatch on consumer: 01b's "pending" is prompt-hash-scoped
            # (scope=RELEVANCE_PROMPT_HASH); 02_collect's is run_id-scoped
            # against site_outcomes. Calling count_pending_relevance with a
            # run_id in place of a prompt hash wouldn't error — it would
            # just silently print a meaningless number, since `status` never
            # decides work either way (Rule of One Truth), it would just lie
            # on screen.
            if consumer == "02_collect":
                live_pending = db.count_pending_collect(conn, run_id=scope, chunk_set=args.set, chunk_index=idx)
            else:
                live_pending = db.count_pending_relevance(conn, prompt_hash=scope, chunk_set=args.set, chunk_index=idx)
        runs = runs_by_key.get(idx, [])
        latest = runs[-1] if runs else None
        recorded_status = latest["status"] if latest else "—"
        finished = (latest["finished"] or "—") if latest else "—"

        is_failed = (live_pending not in (None, 0)) or (recorded_status in ("partial", "interrupted"))
        any_failed = any_failed or is_failed
        if args.failed and not is_failed:
            continue

        pending_disp = "—" if live_pending is None else str(live_pending)
        print(f"  {idx:6} {n_members:6} {pending_disp:>12} {recorded_status:>10} {finished:>21}")

        if args.site_failures and live_pending:
            if consumer == "02_collect":
                failing = list(db.iter_pending_collect(conn, run_id=scope, chunk_set=args.set, chunk_index=idx))
                for r in failing[:20]:
                    print(f"      {r['site_id']:40} (no site_outcomes row yet for run {scope[:12]}…)")
            else:
                failing = list(db.iter_pending_relevance(conn, prompt_hash=scope, chunk_set=args.set, chunk_index=idx))
                for r in failing[:20]:
                    print(f"      {r['site_id']:40} label={r['relevance_label']!r} attempts={r['relevance_attempts']}")
            if len(failing) > 20:
                print(f"      ... and {len(failing) - 20} more")

        if args.all_scopes:
            for run in runs:
                print(f"      scope={run['scope'][:16]!r} consumer={run['consumer']!r} "
                      f"status={run['status']} attempts={run['attempts']}")

    if args.failed and not any_failed:
        print("\n  (no chunks currently failed/pending)")
    conn.close()


def cmd_unstamp(args) -> None:
    """Auditable redo path: dump every member's PRE-CLEAR relevance row to
    JSONL, then clear relevance_prompt_hash for the whole chunk so it
    re-enters the pending queue under the current instrument. There is no
    --force on any work script (see 01b_screen_relevance.py) — this is the
    only sanctioned way to make a "done" chunk pending again, and it always
    leaves an audit trail of what it overwrote.
    """
    if not args.yes:
        raise SystemExit("ERROR: unstamp is destructive to a chunk's 'done' status — pass --yes to confirm")

    conn = db.connect(args.db)
    db.init_db(conn)

    chunk_set = db.get_chunk_set(conn, args.set)
    if chunk_set is None:
        raise SystemExit(f"ERROR: no such chunk set {args.set!r}")

    members = list(db.iter_chunk_members(conn, args.set, args.chunk))
    if not members:
        raise SystemExit(f"ERROR: chunk {args.chunk} of {args.set!r} has no members")

    site_ids = [m["site_id"] for m in members]
    rows = [dict(db.get_site(conn, sid)) for sid in site_ids]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_dir = LOG_DIR / args.set
    set_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = set_dir / f"unstamp_c{args.chunk:05d}_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")

    db.unstamp_relevance(conn, site_ids)
    print(f"unstamped chunk {args.chunk} of {args.set!r}: {len(site_ids)} sites, "
          f"pre-clear labels dumped to {out_path}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="partition an eligible population into a new chunk set")
    p_create.add_argument("--set", required=True, help="chunk set id, e.g. frame-v1")
    p_create.add_argument("--consumer", required=True, help="e.g. 01b_screen_relevance, 02_collect")
    p_create.add_argument("--preset", choices=sorted(_PRESETS), default=None)
    p_create.add_argument("--where", default=None, help="read-only predicate against sites, e.g. \"industry = 'nicotine'\"")
    p_create.add_argument("--from-file", default=None, help="path to a newline-separated site_id list")
    p_create.add_argument("--industry", default=None)
    p_create.add_argument("--chunk-size", type=int, default=None)
    p_create.add_argument("--seed", default=None, help="default: config chunking.chunk_seed")
    p_create.add_argument("--notes", default=None)
    p_create.add_argument("--dry-run", action="store_true")
    p_create.set_defaults(func=cmd_create)

    p_extend = sub.add_parser("extend", help="partition only the complement, as a new generation")
    p_extend.add_argument("--set", required=True)
    p_extend.add_argument("--dry-run", action="store_true")
    p_extend.set_defaults(func=cmd_extend)

    p_verify = sub.add_parser("verify", help="re-derive every generation's partition and assert it matches")
    p_verify.add_argument("--set", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_status = sub.add_parser("status", help="per-chunk membership + recorded run + live pending count")
    p_status.add_argument("--set", required=True)
    p_status.add_argument("--consumer", default=None, help="default: the set's intended_consumer")
    p_status.add_argument("--scope", default=None, help="default: current RELEVANCE_PROMPT_HASH for 01b")
    p_status.add_argument("--failed", action="store_true", help="only show chunks with pending work or a non-done last run")
    p_status.add_argument("--site-failures", action="store_true", help="list the individual pending sites per failed chunk")
    p_status.add_argument("--all-scopes", action="store_true", help="show every recorded (consumer, scope) run per chunk")
    p_status.set_defaults(func=cmd_status)

    p_unstamp = sub.add_parser("unstamp", help="auditable redo: dump labels, clear the hash for one chunk")
    p_unstamp.add_argument("--set", required=True)
    p_unstamp.add_argument("--chunk", type=int, required=True)
    p_unstamp.add_argument("--yes", action="store_true")
    p_unstamp.set_defaults(func=cmd_unstamp)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
