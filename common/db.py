"""SQLite schema + typed read/write helpers — the single data model shared by
the collection script and every classification script. They must never fork it.

Design split (deliberate, do not merge):
  * COLLECTION facts live in `observations` (and `site_outcomes`). No age-gate
    label is ever written here.
  * CLASSIFICATION labels live in `detections` (age gate) and `block_checks`
    (block pages), one row per model. This is what makes re-classification and
    the three-model (8b/30b/32b) comparison clean and crawl-free.

All writes are idempotent upserts keyed on natural primary keys, so killing a
run mid-flight and restarting never produces duplicate rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "agegate.sqlite"

SCHEMA_VERSION = 3  # v3: sites.relevance_ruleset (INTEGER, a deterministic-
                    # classifier concept) replaced by relevance_model +
                    # relevance_prompt_hash — mirrors detections' model +
                    # detection_prompt_hash now that 01b's judgment is an LLM
                    # call (RELEVANCE_PROMPT), not a keyword ruleset. v2 added
                    # relevance_score/relevance_ts/http_status/sample_u/
                    # sample_wave — see 01b/01c.

# Allowed enums (documented here; enforced by callers, kept as TEXT in SQLite).
FAILURE_REASONS = ("blocked_bot", "nav_stuck", "no_such_stage", "timeout", "error")
TERMINAL_STATUSES = (
    "completed",        # reached the terminal checkpoint cleanly
    "dropped_blocked",  # blocked_bot (retryable with --retry-blocked)
    "dropped_stuck",    # nav_stuck
    "no_such_stage",    # site genuinely lacks the funnel stage (NOT a failure)
    "error",            # timeout/error
)


def now_iso() -> str:
    """UTC timestamp, ISO-8601 with 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Connection + schema
# --------------------------------------------------------------------------- #
def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open a connection with sane pragmas (FKs on, WAL, row access by name)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    started               TEXT,
    finished              TEXT,
    config_hash           TEXT,
    detection_prompt_hash TEXT,
    stealth_profile       TEXT,
    notes                 TEXT
);

-- Crawl queue. 01_build_queue.py ingests ALL seed URLs (normalize + dedupe,
-- full provenance) AND assigns sample_stratum immediately — it is a pure
-- function of cctld (common.sampling.stratum_for_cctld), not something that
-- needs a separate pass. cctld stores the EXACT public suffix (e.g. com,
-- org, co.uk, com.br, ru) and is NEVER null for a parseable host.
--
-- included/relevance_label/relevance_score/relevance_model/relevance_prompt_hash/
-- relevance_ts/http_status are set by 01b_screen_relevance.py. The fetch
-- (common.relevance.fetch()) is still plain HTTP GET, no browser — but the
-- JUDGMENT is an LLM call (common.vlm.classify_relevance(), frozen
-- RELEVANCE_PROMPT), not a keyword ruleset, over the WHOLE frame.
-- sample_u/sample_wave are set by 01c_draw_sample.py, which draws a
-- stratified probability sample from the sites 01b confirmed relevant —
-- sampling happens AFTER relevance screening, not before, so the funnel
-- crawl budget is spent on confirmed retailers rather than raw seed noise.
CREATE TABLE IF NOT EXISTS sites (
    site_id            TEXT PRIMARY KEY,          -- hostslug_<6char url_key hash>
    url_original       TEXT NOT NULL,
    url_normalized     TEXT NOT NULL,
    host               TEXT,
    registrable_domain TEXT,
    cctld              TEXT,                      -- exact eTLD, never null
    industry           TEXT NOT NULL,
    source_file        TEXT,
    added_ts           TEXT,
    included           INTEGER NOT NULL DEFAULT 0,-- crawler reads only included=1
    relevance_label    TEXT,                      -- NULL until 01b classifies it
    sample_stratum     TEXT,                      -- ccTLD-derived region, set at ingest
    relevance_score      REAL,                    -- classify_html() score IF that legacy path is ever used
    relevance_model      TEXT,                    -- which Ollama model produced this label
    relevance_prompt_hash TEXT,                   -- which frozen RELEVANCE_PROMPT produced this
    relevance_ts        TEXT,
    http_status          INTEGER,                 -- from the relevance-screen fetch
    sample_u             REAL,                    -- deterministic draw variate; write-once
    sample_wave          INTEGER,                 -- NULL = undrawn; else which wave drew it
    UNIQUE (url_normalized)
);

-- One row per (run, site, checkpoint). COLLECTION facts only. NO age-gate label.
CREATE TABLE IF NOT EXISTS observations (
    run_id         TEXT NOT NULL,
    site_id        TEXT NOT NULL,
    checkpoint     TEXT NOT NULL,
    reached        INTEGER NOT NULL DEFAULT 0,
    fullpage_path  TEXT,
    viewport_path  TEXT,
    dom_path       TEXT,
    nav_method     TEXT,
    nav_steps      INTEGER,
    failure_reason TEXT,                          -- see FAILURE_REASONS, or NULL
    ts             TEXT,
    PRIMARY KEY (run_id, site_id, checkpoint),
    FOREIGN KEY (run_id)  REFERENCES runs(run_id),
    FOREIGN KEY (site_id) REFERENCES sites(site_id)
);

-- Filled by 03_classify.py. One row per (run, site, checkpoint, model).
CREATE TABLE IF NOT EXISTS detections (
    run_id                TEXT NOT NULL,
    site_id               TEXT NOT NULL,
    checkpoint            TEXT NOT NULL,
    model                 TEXT NOT NULL,          -- enables the 3-model comparison
    age_gate_detected     INTEGER,                -- binarized 1/0
    raw_vlm_output        TEXT,                   -- logged verbatim, not just binary
    age_gate_type         TEXT,                   -- NULL until manual labeling
    detection_prompt_hash TEXT,                   -- which frozen instrument produced this
    capture               TEXT,                   -- full_page | viewport actually fed
    vlm_latency_ms        INTEGER,
    ts                    TEXT,
    PRIMARY KEY (run_id, site_id, checkpoint, model),
    FOREIGN KEY (run_id, site_id, checkpoint)
        REFERENCES observations(run_id, site_id, checkpoint)
);

-- Filled by 04_classify_blocks.py. Kept separate so a block page can never
-- pollute an age-gate label. One row per (run, site, checkpoint, model).
CREATE TABLE IF NOT EXISTS block_checks (
    run_id         TEXT NOT NULL,
    site_id        TEXT NOT NULL,
    checkpoint     TEXT NOT NULL,
    model          TEXT NOT NULL,
    is_block_page  INTEGER,
    http_status    INTEGER,
    raw_vlm_output TEXT,
    ts             TEXT,
    PRIMARY KEY (run_id, site_id, checkpoint, model),
    FOREIGN KEY (run_id, site_id, checkpoint)
        REFERENCES observations(run_id, site_id, checkpoint)
);

-- One row per (run, site).
CREATE TABLE IF NOT EXISTS site_outcomes (
    run_id                 TEXT NOT NULL,
    site_id                TEXT NOT NULL,
    max_checkpoint_reached TEXT,
    terminal_status        TEXT,                  -- see TERMINAL_STATUSES
    drop_checkpoint        TEXT,
    drop_reason            TEXT,
    ts                     TEXT,
    PRIMARY KEY (run_id, site_id),
    FOREIGN KEY (run_id)  REFERENCES runs(run_id),
    FOREIGN KEY (site_id) REFERENCES sites(site_id)
);

CREATE INDEX IF NOT EXISTS idx_sites_included      ON sites(included);
CREATE INDEX IF NOT EXISTS idx_sites_industry      ON sites(industry);
CREATE INDEX IF NOT EXISTS idx_sites_cctld         ON sites(cctld);
CREATE INDEX IF NOT EXISTS idx_sites_sample        ON sites(sample_stratum, sample_u);
CREATE INDEX IF NOT EXISTS idx_sites_wave          ON sites(sample_wave);
CREATE INDEX IF NOT EXISTS idx_obs_run             ON observations(run_id);
CREATE INDEX IF NOT EXISTS idx_obs_failure         ON observations(failure_reason);
CREATE INDEX IF NOT EXISTS idx_det_model           ON detections(model);
CREATE INDEX IF NOT EXISTS idx_det_gate            ON detections(age_gate_detected);
CREATE INDEX IF NOT EXISTS idx_outcome_status      ON site_outcomes(terminal_status);
"""


class SchemaVersionMismatch(RuntimeError):
    """Raised when an existing DB's user_version doesn't match SCHEMA_VERSION.
    init_db() must never silently re-stamp an incompatible database — a
    version bump adds columns the running code depends on, and a mismatch
    here usually means an old DB file needs a real migration, not a stamp.
    """


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables + indexes (idempotent) and stamp the schema version.
    Raises SchemaVersionMismatch if the DB already exists with a DIFFERENT,
    non-zero user_version — refuses to silently paper over a stale schema.
    """
    existing = conn.execute("PRAGMA user_version;").fetchone()[0]
    if existing not in (0, SCHEMA_VERSION):
        raise SchemaVersionMismatch(
            f"DB user_version={existing} does not match SCHEMA_VERSION="
            f"{SCHEMA_VERSION}. This DB predates a schema change — inspect it "
            "and migrate deliberately rather than re-stamping."
        )
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.commit()


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def start_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    config_hash: Optional[str] = None,
    detection_prompt_hash: Optional[str] = None,
    stealth_profile: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_id, started, config_hash, detection_prompt_hash,
                          stealth_profile, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            config_hash           = COALESCE(excluded.config_hash, runs.config_hash),
            detection_prompt_hash = COALESCE(excluded.detection_prompt_hash, runs.detection_prompt_hash),
            stealth_profile       = COALESCE(excluded.stealth_profile, runs.stealth_profile),
            notes                 = COALESCE(excluded.notes, runs.notes)
        """,
        (run_id, now_iso(), config_hash, detection_prompt_hash, stealth_profile, notes),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute("UPDATE runs SET finished = ? WHERE run_id = ?", (now_iso(), run_id))
    conn.commit()


# --------------------------------------------------------------------------- #
# sites (queue)
# --------------------------------------------------------------------------- #
def upsert_site(
    conn: sqlite3.Connection,
    *,
    site_id: str,
    url_original: str,
    url_normalized: str,
    host: Optional[str],
    registrable_domain: Optional[str],
    cctld: Optional[str],
    industry: str,
    source_file: Optional[str] = None,
    included: Optional[int] = None,
    relevance_label: Optional[str] = None,
    sample_stratum: Optional[str] = None,
) -> None:
    """The FRAME writer — called by 01_build_queue.py only. `included` and
    `relevance_label` are set by 01b_screen_relevance.py's narrower
    set_site_relevance() in normal operation; they're accepted here too
    (COALESCE-protected) for flexibility, but a plain re-ingest that omits
    them (leaves them None) must NEVER wipe out prior relevance-screening
    work — that was a real bug: the original unconditional `excluded.*`
    overwrite meant re-running 01_build_queue after a multi-hour screen
    reset every site back to included=0. sample_stratum stays a plain
    overwrite: it's a pure function of cctld computed fresh at every ingest,
    so recomputing it is correct, not destructive (e.g. if the ccTLD->region
    map in common.sampling is later refined). Does NOT commit — callers
    batch-commit across many rows.
    """
    # `included` is NOT NULL DEFAULT 0 in the schema, but DEFAULT only kicks
    # in when a column is OMITTED from the INSERT, not when it's explicitly
    # bound to NULL — which is what happens on first-insert whenever the
    # caller leaves included=None. COALESCE(?, 0) in the VALUES clause
    # covers first-insert; the raw `?` bound again in the SET clause (NOT
    # excluded.included, which would already have been defaulted to 0 by
    # that same VALUES-clause COALESCE) is what lets a None on re-ingest
    # correctly fall through to preserving sites.included.
    conn.execute(
        """
        INSERT INTO sites (site_id, url_original, url_normalized, host,
                           registrable_domain, cctld, industry, source_file,
                           added_ts, included, relevance_label, sample_stratum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0), ?, ?)
        ON CONFLICT(site_id) DO UPDATE SET
            url_original       = excluded.url_original,
            url_normalized     = excluded.url_normalized,
            host               = excluded.host,
            registrable_domain = excluded.registrable_domain,
            cctld              = excluded.cctld,
            industry           = excluded.industry,
            source_file        = excluded.source_file,
            included           = COALESCE(?, sites.included),
            relevance_label    = COALESCE(excluded.relevance_label, sites.relevance_label),
            sample_stratum     = excluded.sample_stratum
        """,
        (site_id, url_original, url_normalized, host, registrable_domain, cctld,
         industry, source_file, now_iso(), included, relevance_label, sample_stratum,
         included),
    )


def get_site(conn: sqlite3.Connection, site_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,))
    return cur.fetchone()


_ORDER_BY_COLUMNS = {"site_id": "site_id", "sample_u": "sample_u"}


def iter_included_sites(
    conn: sqlite3.Connection,
    industry: Optional[str] = None,
    *,
    order_by: str = "site_id",
) -> Iterator[sqlite3.Row]:
    """Crawl queue: sites flagged included=1, optionally filtered by industry.
    order_by="sample_u" is what 02_collect.py should use: since sample_u is
    the deterministic draw variate, iterating in that order means ANY
    truncation of the funnel crawl (it's the slowest, most likely-to-be-cut-
    short step) is itself still a valid stratified sample of what it covered.
    """
    if order_by not in _ORDER_BY_COLUMNS:
        raise ValueError(f"order_by must be one of {sorted(_ORDER_BY_COLUMNS)}, got {order_by!r}")
    column = _ORDER_BY_COLUMNS[order_by]
    if industry is None:
        cur = conn.execute(f"SELECT * FROM sites WHERE included = 1 ORDER BY {column}")
    else:
        cur = conn.execute(
            f"SELECT * FROM sites WHERE included = 1 AND industry = ? ORDER BY {column}",
            (industry,),
        )
    yield from cur


# --------------------------------------------------------------------------- #
# sites — relevance screening (01b_screen_relevance.py, common.relevance)
# --------------------------------------------------------------------------- #
def set_site_relevance(
    conn: sqlite3.Connection,
    site_id: str,
    *,
    relevance_label: str,
    included: int,
    relevance_score: Optional[float] = None,
    relevance_model: Optional[str] = None,
    relevance_prompt_hash: Optional[str] = None,
    http_status: Optional[int] = None,
) -> None:
    """The narrow writer for 01b_screen_relevance.py — keeps relevance
    concerns out of upsert_site's frame-writing responsibility. Commits.
    """
    conn.execute(
        """
        UPDATE sites SET
            relevance_label        = ?,
            included               = ?,
            relevance_score        = ?,
            relevance_model        = ?,
            relevance_prompt_hash  = ?,
            relevance_ts           = ?,
            http_status            = ?
        WHERE site_id = ?
        """,
        (relevance_label, included, relevance_score, relevance_model,
         relevance_prompt_hash, now_iso(), http_status, site_id),
    )
    conn.commit()


def iter_pending_relevance(
    conn: sqlite3.Connection,
    *,
    prompt_hash: str,
    industry: Optional[str] = None,
    retry_failed: bool = False,
) -> Iterator[sqlite3.Row]:
    """Work queue for 01b: frame rows not yet classified under the CURRENT
    RELEVANCE_PROMPT. No cursor file needed — resumability comes from the
    data itself (relevance_prompt_hash lags the live prompt's hash, e.g.
    after a deliberate prompt revision — see common.prompts.RELEVANCE_PROMPT_HASH).
    retry_failed also re-queues rows whose previous attempt was
    unreachable/blocked/error, up to config relevance.max_attempts (enforced
    by the caller, not here).
    """
    clauses = ["(relevance_prompt_hash IS NULL OR relevance_prompt_hash != ?)"]
    params: list = [prompt_hash]
    if retry_failed:
        clauses[-1] = (
            "(relevance_prompt_hash IS NULL OR relevance_prompt_hash != ? "
            "OR relevance_label IN ('unreachable', 'blocked', 'error'))"
        )
    if industry is not None:
        clauses.append("industry = ?")
        params.append(industry)
    where = " AND ".join(clauses)
    cur = conn.execute(
        f"SELECT * FROM sites WHERE {where} ORDER BY sample_stratum, site_id", params
    )
    yield from cur


# --------------------------------------------------------------------------- #
# sites — stratified sample draw (01c_draw_sample.py, common.sampling)
# --------------------------------------------------------------------------- #
def set_site_sample_u(conn: sqlite3.Connection, rows: list[tuple[str, float]]) -> None:
    """Write-once: only fills sample_u where it is currently NULL, so a
    re-run never perturbs a variate that a prior wave's draw already
    depended on. rows = [(site_id, u), ...]. Commits.
    """
    conn.executemany(
        "UPDATE sites SET sample_u = COALESCE(sample_u, ?) WHERE site_id = ?",
        [(u, site_id) for site_id, u in rows],
    )
    conn.commit()


def mark_sites_drawn(conn: sqlite3.Connection, site_ids: list[str], wave: int) -> None:
    """Write-once: only fills sample_wave where it is currently NULL — a
    site already drawn in an earlier wave keeps that wave number. Commits.
    """
    conn.executemany(
        "UPDATE sites SET sample_wave = COALESCE(sample_wave, ?) WHERE site_id = ?",
        [(wave, site_id) for site_id in site_ids],
    )
    conn.commit()


def frame_counts_by_stratum(
    conn: sqlite3.Connection,
    industry: Optional[str] = None,
    *,
    included_only: bool = False,
    undrawn_only: bool = False,
) -> dict[str, int]:
    """N_h per stratum for common.sampling.allocate(). included_only=True
    restricts to sites 01b confirmed relevant (included=1) — this is the
    population 01c actually samples from, per the finalized pipeline order
    (relevance screening happens over the WHOLE frame before sampling).
    """
    clauses = []
    params: list = []
    if industry is not None:
        clauses.append("industry = ?")
        params.append(industry)
    if included_only:
        clauses.append("included = 1")
    if undrawn_only:
        clauses.append("sample_wave IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cur = conn.execute(
        f"SELECT sample_stratum, COUNT(*) AS n FROM sites {where} GROUP BY sample_stratum",
        params,
    )
    return {row["sample_stratum"]: row["n"] for row in cur}


def iter_undrawn_relevant_sites(
    conn: sqlite3.Connection,
    industry: Optional[str] = None,
) -> Iterator[sqlite3.Row]:
    """Population 01c draws from: relevant (included=1), not yet assigned to
    any wave, ordered by (stratum, sample_u) so per-stratum lowest-u
    selection is a simple LIMIT/groupby slice.

    sample_u IS NOT NULL is defensive, not redundant: SQLite sorts NULL
    FIRST in ASC order, so a row 01c's backfill step hasn't reached yet
    (e.g. 01b_screen_relevance.py still writing concurrently) would
    otherwise be wrongly prioritized ahead of every real low-u candidate.
    """
    clauses = ["included = 1", "sample_wave IS NULL", "sample_u IS NOT NULL"]
    params: list = []
    if industry is not None:
        clauses.append("industry = ?")
        params.append(industry)
    cur = conn.execute(
        f"SELECT * FROM sites WHERE {' AND '.join(clauses)} ORDER BY sample_stratum, sample_u",
        params,
    )
    yield from cur


# --------------------------------------------------------------------------- #
# observations  (collection facts — NO age-gate label)
# --------------------------------------------------------------------------- #
def upsert_observation(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    site_id: str,
    checkpoint: str,
    reached: int,
    fullpage_path: Optional[str] = None,
    viewport_path: Optional[str] = None,
    dom_path: Optional[str] = None,
    nav_method: Optional[str] = None,
    nav_steps: Optional[int] = None,
    failure_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO observations (run_id, site_id, checkpoint, reached,
                                  fullpage_path, viewport_path, dom_path,
                                  nav_method, nav_steps, failure_reason, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, site_id, checkpoint) DO UPDATE SET
            reached        = excluded.reached,
            fullpage_path  = excluded.fullpage_path,
            viewport_path  = excluded.viewport_path,
            dom_path       = excluded.dom_path,
            nav_method     = excluded.nav_method,
            nav_steps      = excluded.nav_steps,
            failure_reason = excluded.failure_reason,
            ts             = excluded.ts
        """,
        (run_id, site_id, checkpoint, reached, fullpage_path, viewport_path,
         dom_path, nav_method, nav_steps, failure_reason, now_iso()),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# detections  (age-gate labels — filled by 03_classify.py)
# --------------------------------------------------------------------------- #
def upsert_detection(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    site_id: str,
    checkpoint: str,
    model: str,
    age_gate_detected: Optional[int],
    raw_vlm_output: Optional[str],
    detection_prompt_hash: Optional[str] = None,
    capture: Optional[str] = None,
    age_gate_type: Optional[str] = None,
    vlm_latency_ms: Optional[int] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO detections (run_id, site_id, checkpoint, model,
                                age_gate_detected, raw_vlm_output, age_gate_type,
                                detection_prompt_hash, capture, vlm_latency_ms, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, site_id, checkpoint, model) DO UPDATE SET
            age_gate_detected     = excluded.age_gate_detected,
            raw_vlm_output        = excluded.raw_vlm_output,
            detection_prompt_hash = excluded.detection_prompt_hash,
            capture               = excluded.capture,
            vlm_latency_ms        = excluded.vlm_latency_ms,
            ts                    = excluded.ts
            -- age_gate_type is intentionally NOT overwritten (manual label)
        """,
        (run_id, site_id, checkpoint, model, age_gate_detected, raw_vlm_output,
         age_gate_type, detection_prompt_hash, capture, vlm_latency_ms, now_iso()),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# block_checks  (block-page labels — filled by 04_classify_blocks.py)
# --------------------------------------------------------------------------- #
def upsert_block_check(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    site_id: str,
    checkpoint: str,
    model: str,
    is_block_page: Optional[int],
    http_status: Optional[int] = None,
    raw_vlm_output: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO block_checks (run_id, site_id, checkpoint, model,
                                  is_block_page, http_status, raw_vlm_output, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, site_id, checkpoint, model) DO UPDATE SET
            is_block_page  = excluded.is_block_page,
            http_status    = excluded.http_status,
            raw_vlm_output = excluded.raw_vlm_output,
            ts             = excluded.ts
        """,
        (run_id, site_id, checkpoint, model, is_block_page, http_status,
         raw_vlm_output, now_iso()),
    )
    conn.commit()


# --------------------------------------------------------------------------- #
# site_outcomes
# --------------------------------------------------------------------------- #
def upsert_site_outcome(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    site_id: str,
    max_checkpoint_reached: Optional[str],
    terminal_status: Optional[str],
    drop_checkpoint: Optional[str] = None,
    drop_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO site_outcomes (run_id, site_id, max_checkpoint_reached,
                                   terminal_status, drop_checkpoint, drop_reason, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, site_id) DO UPDATE SET
            max_checkpoint_reached = excluded.max_checkpoint_reached,
            terminal_status        = excluded.terminal_status,
            drop_checkpoint        = excluded.drop_checkpoint,
            drop_reason            = excluded.drop_reason,
            ts                     = excluded.ts
        """,
        (run_id, site_id, max_checkpoint_reached, terminal_status,
         drop_checkpoint, drop_reason, now_iso()),
    )
    conn.commit()
