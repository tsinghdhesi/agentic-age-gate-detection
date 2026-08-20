#!/usr/bin/env python3
"""01c — Draw a stratified probability sample from sites 01b confirmed
relevant (included=1). This is the LAST step before 02_collect.py; sampling
happens AFTER relevance screening, not before, so the funnel-crawl budget is
spent entirely on confirmed retailers rather than raw seed noise.

Draw mechanism: sample_u = sha256(salt|site_id) is a deterministic variate in
[0,1), computed once and never changed (write-once — see db.set_site_sample_u).
Within each ccTLD-derived stratum, the wave draws the n_h sites with the
SMALLEST sample_u. This makes the design reproducible from (salt, site_id)
alone, and — because sample_u also becomes the funnel-crawl iteration order
(02_collect.py should call iter_included_sites(order_by="sample_u")) — ANY
truncation of the funnel crawl remains a valid stratified sample of whatever
it covered. See common/sampling.py for the allocation math (Bankier/Kish
compromise allocation, power=0.5 by default) and design_summary() (n_eff,
DEFF) — the --dry-run table belongs directly in the methods chapter.

ALWAYS run --dry-run first and review the allocation table before drawing.

Run:
  ./.venv/bin/python scripts/01c_draw_sample.py --dry-run
  ./.venv/bin/python scripts/01c_draw_sample.py --wave 1
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config, db  # noqa: E402
from common.sampling import allocate, design_summary, sample_u  # noqa: E402


def _next_wave(conn) -> int:
    row = conn.execute("SELECT MAX(sample_wave) AS m FROM sites").fetchone()
    return (row["m"] or 0) + 1


def _backfill_sample_u(conn, salt: str, industry) -> int:
    """Compute sample_u for every relevant (included=1) site that doesn't
    have one yet. Write-once under the hood (db.set_site_sample_u), so this
    is always safe to call — a re-run never perturbs an existing variate.
    """
    clauses = ["included = 1", "sample_u IS NULL"]
    params: list = []
    if industry is not None:
        clauses.append("industry = ?")
        params.append(industry)
    cur = conn.execute(f"SELECT site_id FROM sites WHERE {' AND '.join(clauses)}", params)
    rows = [(row["site_id"], sample_u(row["site_id"], salt)) for row in cur]
    if rows:
        db.set_site_sample_u(conn, rows)
    return len(rows)


def main() -> None:
    cfg = config.load_global()["sampling"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=db.DEFAULT_DB)
    ap.add_argument("--industry", default=None)
    ap.add_argument("--wave", type=int, default=None, help="wave number (default: next unused)")
    ap.add_argument(
        "--n", type=int, default=None,
        help=f"sites to draw this wave (default: target_n/waves = {cfg['target_n']}/{cfg['waves']})",
    )
    ap.add_argument("--power", type=float, default=cfg["power"])
    ap.add_argument("--floor", type=int, default=cfg["stratum_floor"])
    ap.add_argument("--salt", default=cfg["frame_salt"])
    ap.add_argument("--dry-run", action="store_true", help="print the allocation + DEFF table and exit")
    args = ap.parse_args()

    n_total = args.n if args.n is not None else cfg["target_n"] // cfg["waves"]

    conn = db.connect(args.db)
    db.init_db(conn)

    wave = args.wave if args.wave is not None else _next_wave(conn)

    backfilled = _backfill_sample_u(conn, args.salt, args.industry)
    if backfilled:
        print(f"backfilled sample_u for {backfilled} relevant sites (salt={args.salt!r})")

    frame_counts = db.frame_counts_by_stratum(conn, args.industry, included_only=True, undrawn_only=True)
    total_available = sum(frame_counts.values())
    print(f"01c_draw_sample — wave={wave} n_total={n_total} power={args.power} floor={args.floor}")
    print(f"undrawn relevant population: {total_available}")

    if total_available == 0:
        print("\nNothing to draw — no relevant, undrawn sites. Has 01b_screen_relevance.py run yet?")
        conn.close()
        return

    alloc = allocate(frame_counts, n_total, power=args.power, floor=args.floor)
    summary = design_summary(frame_counts, alloc)

    print("\nallocation:")
    print(f"  {'stratum':10} {'N_h':>8} {'n_h':>6} {'f_h':>7} {'w_h':>8}")
    for stratum, row in sorted(summary["strata"].items(), key=lambda kv: -kv[1]["n_h"]):
        w_h = f"{row['w_h']:.2f}" if row["w_h"] is not None else "-"
        print(f"  {stratum:10} {row['N_h']:8} {row['n_h']:6} {row['f_h']:7.3f} {w_h:>8}")
    print(f"\n  n_total={summary['n_total']}  n_eff={summary['n_eff']:.1f}  DEFF={summary['deff']:.3f}")

    if args.dry_run:
        print("\n[DRY RUN — nothing drawn]")
        conn.close()
        return

    # A wave is draw-once PER STRATUM, not per invocation: re-running the
    # same --wave must top up each stratum to its allocation target, never
    # draw past it. Without this, calling this script twice with the same
    # --wave would keep pulling MORE sites into "wave 1" every time (since
    # mark_sites_drawn only protects individual site_ids from being
    # re-drawn, not the wave's total size) — silently growing a wave beyond
    # its designed size and breaking the fixed-size-per-wave guarantee the
    # sampling design depends on.
    already_drawn: dict = {}
    for row in conn.execute(
        "SELECT sample_stratum, COUNT(*) AS n FROM sites WHERE sample_wave = ? GROUP BY sample_stratum",
        (wave,),
    ):
        already_drawn[row["sample_stratum"]] = row["n"]
    if already_drawn:
        print(f"wave {wave} already has {sum(already_drawn.values())} site(s) drawn — topping up to target per stratum")

    # iter_undrawn_relevant_sites is already ordered by (stratum, sample_u)
    # and NULL-safe, so a per-stratum draw is just a groupby + slice — one
    # query instead of one per stratum.
    candidates = db.iter_undrawn_relevant_sites(conn, args.industry)
    by_stratum = {
        stratum: list(rows)
        for stratum, rows in itertools.groupby(candidates, key=lambda r: r["sample_stratum"])
    }

    drawn_total = 0
    for stratum, n_h in alloc.items():
        deficit = n_h - already_drawn.get(stratum, 0)
        if deficit <= 0:
            continue
        site_ids = [r["site_id"] for r in by_stratum.get(stratum, [])[:deficit]]
        db.mark_sites_drawn(conn, site_ids, wave)
        drawn_total += len(site_ids)

    conn.close()
    wave_total = sum(already_drawn.values()) + drawn_total
    print(f"\ndrew {drawn_total} new site(s) into wave {wave} ({wave_total} total in this wave)")


if __name__ == "__main__":
    main()
