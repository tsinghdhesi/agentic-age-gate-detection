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

SCHEMA_VERSION = 5  # v5: chunk_sets/chunk_members/chunk_runs — addressable,
                    # resumable work chunks shared by 01b/02/03 (see
                    # common/chunking.py for the partition math and
                    # scripts/chunk_work.py for the CLI). chunk_members is
                    # write-once (trigger-enforced). Also v5:
                    # sites.relevance_attempts, backing a real
                    # relevance.max_attempts config cap — previously
                    # documented on iter_pending_relevance() but not enforced
                    # anywhere, so unbounded --retry-failed could hammer a
                    # permanently-dead host forever.
                    # v4: sites.relevance_fetch_mode ('http'|'browser') records
                    # whether a label came from the plain HTTP GET or from the
                    # JS-render browser fallback (see common/relevance.py).
                    # v3: sites.relevance_ruleset (INTEGER, a deterministic-
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
    relevance_fetch_mode TEXT,                    -- 'http' | 'browser' (JS-render fallback)
    relevance_ts        TEXT,
    http_status          INTEGER,                 -- from the relevance-screen fetch
    sample_u             REAL,                    -- deterministic draw variate; write-once
    sample_wave          INTEGER,                 -- NULL = undrawn; else which wave drew it
    relevance_attempts   INTEGER NOT NULL DEFAULT 0, -- v5: incremented by set_site_relevance();
                                                    -- backs relevance.max_attempts (config)
    UNIQUE (url_normalized)
);

-- v5 — addressable, resumable work chunks shared by 01b_screen_relevance.py
-- and (later) 02_collect.py/03_classify.py. Three tables for three distinct
-- lifetimes: set definition (written once), membership (immutable), and
-- execution record (rewritten per attempt). See common/chunking.py for the
-- partition math and scripts/chunk_work.py for the CLI.
--
-- RULE OF ONE TRUTH: a site's completion state is a property of the site's
-- OWN row (sites.relevance_prompt_hash / site_outcomes / detections).
-- chunk_members decides which sites an invocation may TOUCH. chunk_runs
-- records what HAPPENED. No script may ever read chunk_runs to decide
-- whether to do work — work selection is always the indexed join
-- `pending ⋈ chunk_members` (see _pending_relevance_sql below).
CREATE TABLE IF NOT EXISTS chunk_sets (
    chunk_set_id      TEXT PRIMARY KEY,
    intended_consumer TEXT NOT NULL,
    selector_sql      TEXT,             -- the (validated) predicate chunk_work.py partitioned
    seed              TEXT NOT NULL,    -- distinct from sampling.frame_salt — see chunking.py
    chunk_size        INTEGER NOT NULL,
    created_ts        TEXT,
    notes             TEXT
);

-- Write-once: (chunk_set_id, site_id) -> (chunk_index, member_rank,
-- generation). A partition a later pass could silently re-assign is not
-- reproducible, so the trigger below aborts any UPDATE outright.
-- `extend` only ever INSERTs new site_ids as a new generation; existing rows
-- are never touched.
CREATE TABLE IF NOT EXISTS chunk_members (
    chunk_set_id  TEXT NOT NULL,
    site_id       TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    member_rank   INTEGER NOT NULL,   -- 0-based rank WITHIN this chunk, not global
    generation    INTEGER NOT NULL DEFAULT 0,
    assigned_ts   TEXT,
    PRIMARY KEY (chunk_set_id, site_id),
    FOREIGN KEY (chunk_set_id) REFERENCES chunk_sets(chunk_set_id),
    FOREIGN KEY (site_id) REFERENCES sites(site_id)
);

CREATE TRIGGER IF NOT EXISTS trg_chunk_members_immutable
BEFORE UPDATE ON chunk_members
BEGIN
    SELECT RAISE(ABORT, 'chunk_members is write-once; membership must never be reassigned');
END;

-- Latest-attempt execution summary per (chunk, consumer, scope). `scope` is
-- what makes this generic across consumers — it holds whatever namespaces
-- "done" for that consumer (RELEVANCE_PROMPT_HASH for 01b; run_id for 02;
-- "run_id|model" for 03). A prompt/instrument revision reconciles for free:
-- the old scope's rows stay "done" forever (a true historical statement
-- about the old instrument); under the new hash the chunk simply has no row
-- and reports pending — the PARTITION itself is untouched by the revision.
-- Invariant #4a: no lease/claim/owner/heartbeat column, and never will be —
-- this is a scheduling record, not a work-distribution primitive. A stale
-- `running` row (from a kill -9) is the crash signature, not a lock; the
-- next run simply overwrites it.
CREATE TABLE IF NOT EXISTS chunk_runs (
    chunk_set_id    TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    consumer        TEXT NOT NULL,
    scope           TEXT NOT NULL,
    status          TEXT,              -- running | done | partial | interrupted
    attempts        INTEGER NOT NULL DEFAULT 0,
    n_members       INTEGER,
    n_pending_start INTEGER,
    n_pending_end   INTEGER,
    n_processed     INTEGER,
    started         TEXT,
    finished        TEXT,
    host            TEXT,
    pid             INTEGER,
    jsonl_path      TEXT,
    last_error      TEXT,
    PRIMARY KEY (chunk_set_id, chunk_index, consumer, scope),
    FOREIGN KEY (chunk_set_id) REFERENCES chunk_sets(chunk_set_id)
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
CREATE INDEX IF NOT EXISTS idx_chunk_members_lookup ON chunk_members(chunk_set_id, chunk_index, member_rank);
CREATE INDEX IF NOT EXISTS idx_chunk_runs_status    ON chunk_runs(chunk_set_id, status);
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
    relevance_fetch_mode: Optional[str] = None,
    http_status: Optional[int] = None,
) -> None:
    """The narrow writer for 01b_screen_relevance.py — keeps relevance
    concerns out of upsert_site's frame-writing responsibility. Commits.
    Increments relevance_attempts unconditionally (every call here IS an
    attempt — live screen or --reclassify), which is what lets
    relevance.max_attempts cap a permanently-dead host's retry count.
    """
    conn.execute(
        """
        UPDATE sites SET
            relevance_label        = ?,
            included               = ?,
            relevance_score        = ?,
            relevance_model        = ?,
            relevance_prompt_hash  = ?,
            relevance_fetch_mode   = ?,
            relevance_ts           = ?,
            http_status            = ?,
            relevance_attempts     = relevance_attempts + 1
        WHERE site_id = ?
        """,
        (relevance_label, included, relevance_score, relevance_model,
         relevance_prompt_hash, relevance_fetch_mode, now_iso(), http_status, site_id),
    )
    conn.commit()


def _pending_relevance_sql(
    *,
    prompt_hash: str,
    industry: Optional[str] = None,
    retry_failed: bool = False,
    max_attempts: Optional[int] = None,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> tuple[str, str, list, str]:
    """The SINGLE definition of "pending" for relevance screening, shared by
    iter_pending_relevance() and count_pending_relevance() so the queue can
    never drift into two different definitions. Returns
    (from_sql, where_sql, params, order_sql) — callers compose their own
    SELECT list around it.

    chunk_set/chunk_index restrict to one chunk's membership via an indexed
    join on chunk_members — this IS the "pending ⋈ chunk_members" join the
    Rule of One Truth (CLAUDE.md) requires; chunk_runs is never consulted
    here or anywhere in the pending-queue definition.

    retry_failed re-queues rows whose previous attempt was
    unreachable/blocked/error; max_attempts (when given) additionally caps
    those re-queued rows to sites.relevance_attempts < max_attempts, so a
    permanently-dead host stops being re-queued instead of blocking a chunk
    from ever reaching "done" forever.
    """
    from_sql = "sites"
    params: list = [prompt_hash]
    base_clause = "(sites.relevance_prompt_hash IS NULL OR sites.relevance_prompt_hash != ?)"
    if retry_failed:
        failed_clause = "sites.relevance_label IN ('unreachable', 'blocked', 'error')"
        if max_attempts is not None:
            failed_clause += " AND sites.relevance_attempts < ?"
        base_clause = f"({base_clause} OR ({failed_clause}))"
        if max_attempts is not None:
            params.append(max_attempts)
    clauses = [base_clause]

    if industry is not None:
        clauses.append("sites.industry = ?")
        params.append(industry)

    order_sql = "sites.sample_stratum, sites.site_id"
    if chunk_set is not None:
        from_sql = "sites JOIN chunk_members ON chunk_members.site_id = sites.site_id"
        clauses.append("chunk_members.chunk_set_id = ?")
        params.append(chunk_set)
        if chunk_index is not None:
            clauses.append("chunk_members.chunk_index = ?")
            params.append(chunk_index)
        order_sql = "chunk_members.chunk_index, chunk_members.member_rank"

    where_sql = " AND ".join(clauses)
    return from_sql, where_sql, params, order_sql


def iter_pending_relevance(
    conn: sqlite3.Connection,
    *,
    prompt_hash: str,
    industry: Optional[str] = None,
    retry_failed: bool = False,
    max_attempts: Optional[int] = None,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Iterator[sqlite3.Row]:
    """Work queue for 01b: frame rows not yet classified under the CURRENT
    RELEVANCE_PROMPT. No cursor file needed — resumability comes from the
    data itself (relevance_prompt_hash lags the live prompt's hash, e.g.
    after a deliberate prompt revision — see common.prompts.RELEVANCE_PROMPT_HASH).
    See _pending_relevance_sql for retry_failed/max_attempts/chunk_set semantics.

    Without chunk_set, ordered by (sample_stratum, site_id) — 01b's own
    _interleave_by_stratum then round-robins this into a per-stratum-balanced
    sequence. WITH chunk_set, ordered by (chunk_index, member_rank) instead:
    a chunk's membership is already a uniformly random permutation of the
    frame (see common.chunking.partition), so ANY prefix of it is an unbiased
    subsample — round-robin interleaving is not needed and is bypassed by
    the chunked caller.
    """
    from_sql, where_sql, params, order_sql = _pending_relevance_sql(
        prompt_hash=prompt_hash, industry=industry, retry_failed=retry_failed,
        max_attempts=max_attempts, chunk_set=chunk_set, chunk_index=chunk_index,
    )
    cur = conn.execute(f"SELECT sites.* FROM {from_sql} WHERE {where_sql} ORDER BY {order_sql}", params)
    yield from cur


def count_pending_relevance(
    conn: sqlite3.Connection,
    *,
    prompt_hash: str,
    industry: Optional[str] = None,
    retry_failed: bool = False,
    max_attempts: Optional[int] = None,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> int:
    """COUNT(*) form of iter_pending_relevance() — same _pending_relevance_sql
    builder, so the two definitions of "pending" can never disagree. Used for
    chunk_runs.n_pending_start/n_pending_end and chunk_work.py status, which
    must derive chunk status by re-running this count, never by reading
    chunk_runs itself (Rule of One Truth).
    """
    from_sql, where_sql, params, _ = _pending_relevance_sql(
        prompt_hash=prompt_hash, industry=industry, retry_failed=retry_failed,
        max_attempts=max_attempts, chunk_set=chunk_set, chunk_index=chunk_index,
    )
    cur = conn.execute(f"SELECT COUNT(*) AS n FROM {from_sql} WHERE {where_sql}", params)
    return cur.fetchone()["n"]


def next_pending_chunk_index(
    conn: sqlite3.Connection,
    chunk_set_id: str,
    *,
    prompt_hash: str,
    industry: Optional[str] = None,
    retry_failed: bool = False,
    max_attempts: Optional[int] = None,
) -> Optional[int]:
    """What `--chunk next` resolves to: the LOWEST chunk_index in this set
    that has at least one pending member under the current default queue
    definition. Derived from the same pending ⋈ chunk_members join as
    iter_pending_relevance() / count_pending_relevance() — deliberately does
    NOT consult chunk_runs, which would make the convenience flag a second,
    disagreeable source of truth (Rule of One Truth). Returns None if every
    chunk in the set is fully processed.
    """
    from_sql, where_sql, params, _ = _pending_relevance_sql(
        prompt_hash=prompt_hash, industry=industry, retry_failed=retry_failed,
        max_attempts=max_attempts, chunk_set=chunk_set_id,
    )
    cur = conn.execute(f"SELECT MIN(chunk_members.chunk_index) AS c FROM {from_sql} WHERE {where_sql}", params)
    row = cur.fetchone()
    return row["c"] if row is not None else None


# --------------------------------------------------------------------------- #
# collection pending queue (02_collect*.py) — scope=run_id, doneness lives in
# site_outcomes(run_id, site_id). Mirrors _pending_relevance_sql's shape
# exactly so the two consumers stay structurally consistent, but is a
# genuinely separate builder (not a dispatching wrapper) because the
# underlying join is different in kind: relevance checks a column ON the
# sites row itself (relevance_prompt_hash); collection checks whether a
# SEPARATE table's row exists at all for this run.
# --------------------------------------------------------------------------- #
def _pending_collect_sql(
    *,
    run_id: str,
    industry: Optional[str] = None,
    retry_blocked: bool = False,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> tuple[str, str, list, str]:
    """The SINGLE definition of "pending" for the funnel crawl. "Pending" =
    no site_outcomes row yet for THIS run_id; retry_blocked widens that to
    also re-queue sites whose prior outcome for this run was
    dropped_blocked (mirrors --retry-failed's widening in
    _pending_relevance_sql exactly).

    Unlike 01b, there is no "whole frame" population for collection — the
    drawn-sample restriction (included=1 AND sample_wave IS NOT NULL) is
    baked into chunk MEMBERSHIP at chunk-creation time by
    `chunk_work.py create --preset drawn-sample`, not enforced here. A call
    without chunk_set therefore iterates the WHOLE sites table, which is
    only useful for tests — scripts/02_collect_browseruse.py's own contract
    is to always pass chunk_set/chunk_index in real use.
    """
    from_sql = (
        "sites LEFT JOIN site_outcomes "
        "ON site_outcomes.run_id = ? AND site_outcomes.site_id = sites.site_id"
    )
    params: list = [run_id]
    base_clause = "site_outcomes.site_id IS NULL"
    if retry_blocked:
        base_clause = f"({base_clause} OR site_outcomes.terminal_status = 'dropped_blocked')"
    clauses = [base_clause]

    if industry is not None:
        clauses.append("sites.industry = ?")
        params.append(industry)

    order_sql = "sites.site_id"
    if chunk_set is not None:
        from_sql += " JOIN chunk_members ON chunk_members.site_id = sites.site_id"
        clauses.append("chunk_members.chunk_set_id = ?")
        params.append(chunk_set)
        if chunk_index is not None:
            clauses.append("chunk_members.chunk_index = ?")
            params.append(chunk_index)
        order_sql = "chunk_members.chunk_index, chunk_members.member_rank"

    where_sql = " AND ".join(clauses)
    return from_sql, where_sql, params, order_sql


def iter_pending_collect(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    industry: Optional[str] = None,
    retry_blocked: bool = False,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Iterator[sqlite3.Row]:
    """Work queue for the funnel crawl: sites in the given chunk with no
    site_outcomes row yet for run_id. See _pending_collect_sql for
    retry_blocked/chunk_set semantics."""
    from_sql, where_sql, params, order_sql = _pending_collect_sql(
        run_id=run_id, industry=industry, retry_blocked=retry_blocked,
        chunk_set=chunk_set, chunk_index=chunk_index,
    )
    cur = conn.execute(f"SELECT sites.* FROM {from_sql} WHERE {where_sql} ORDER BY {order_sql}", params)
    yield from cur


def count_pending_collect(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    industry: Optional[str] = None,
    retry_blocked: bool = False,
    chunk_set: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> int:
    """COUNT(*) form of iter_pending_collect() — same builder, so the two
    definitions of "pending" can never disagree. Used for chunk_runs status
    derivation and chunk_work.py status for consumer=02_collect."""
    from_sql, where_sql, params, _ = _pending_collect_sql(
        run_id=run_id, industry=industry, retry_blocked=retry_blocked,
        chunk_set=chunk_set, chunk_index=chunk_index,
    )
    cur = conn.execute(f"SELECT COUNT(*) AS n FROM {from_sql} WHERE {where_sql}", params)
    return cur.fetchone()["n"]


def next_pending_collect_chunk_index(
    conn: sqlite3.Connection,
    chunk_set_id: str,
    *,
    run_id: str,
    industry: Optional[str] = None,
    retry_blocked: bool = False,
) -> Optional[int]:
    """`--chunk next` for the collection consumer: the lowest chunk_index
    with at least one pending member for this run_id. Sibling of
    next_pending_chunk_index (relevance) rather than a dispatching wrapper —
    each stays a straight-line read of its own SQL, matching the existing
    per-consumer builder split. Never consults chunk_runs (Rule of One Truth).
    """
    from_sql, where_sql, params, _ = _pending_collect_sql(
        run_id=run_id, industry=industry, retry_blocked=retry_blocked, chunk_set=chunk_set_id,
    )
    cur = conn.execute(f"SELECT MIN(chunk_members.chunk_index) AS c FROM {from_sql} WHERE {where_sql}", params)
    row = cur.fetchone()
    return row["c"] if row is not None else None


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
# chunking (schema v5, common/chunking.py, scripts/chunk_work.py)
#
# RULE OF ONE TRUTH: chunk_members decides which sites an invocation may
# TOUCH (via the pending ⋈ chunk_members join above). chunk_runs records what
# HAPPENED. No function below is ever consulted by a work loop to decide
# whether to do work — only iter_pending_relevance/count_pending_relevance/
# next_pending_chunk_index (which join chunk_members, never chunk_runs) are.
# --------------------------------------------------------------------------- #
def create_chunk_set(
    conn: sqlite3.Connection,
    *,
    chunk_set_id: str,
    intended_consumer: str,
    seed: str,
    chunk_size: int,
    selector_sql: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Register a new chunk set. Raises sqlite3.IntegrityError on a duplicate
    chunk_set_id — sets are not meant to be silently redefined; use `extend`
    (a new generation, same id) to grow one instead.
    """
    conn.execute(
        """
        INSERT INTO chunk_sets (chunk_set_id, intended_consumer, selector_sql,
                                seed, chunk_size, created_ts, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (chunk_set_id, intended_consumer, selector_sql, seed, chunk_size, now_iso(), notes),
    )
    conn.commit()


def get_chunk_set(conn: sqlite3.Connection, chunk_set_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM chunk_sets WHERE chunk_set_id = ?", (chunk_set_id,))
    return cur.fetchone()


def insert_chunk_members(
    conn: sqlite3.Connection,
    chunk_set_id: str,
    assignments: list[tuple[str, int, int, int]],
) -> None:
    """assignments = [(site_id, chunk_index, member_rank, generation), ...],
    exactly common.chunking.partition()'s return shape. Relies on the
    write-once trigger (trg_chunk_members_immutable) to abort outright if a
    site_id already belongs to this set — a normal `create`/`extend` only
    ever inserts site_ids that were excluded from the partition input
    (existing_member_site_ids), so this should never fire in practice; if it
    does, that's a real bug in the caller, not something to swallow. Commits.
    """
    conn.executemany(
        """
        INSERT INTO chunk_members (chunk_set_id, site_id, chunk_index,
                                   member_rank, generation, assigned_ts)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(chunk_set_id, site_id, chunk_index, member_rank, generation, now_iso())
         for site_id, chunk_index, member_rank, generation in assignments],
    )
    conn.commit()


def iter_chunk_members(
    conn: sqlite3.Connection, chunk_set_id: str, chunk_index: Optional[int] = None
) -> Iterator[sqlite3.Row]:
    """All members of a set, or just one chunk, ordered by member_rank.
    Used by chunk_work.py verify/status, not by the 01b work loop itself
    (which reads via iter_pending_relevance's join instead)."""
    if chunk_index is None:
        cur = conn.execute(
            "SELECT * FROM chunk_members WHERE chunk_set_id = ? ORDER BY chunk_index, member_rank",
            (chunk_set_id,),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM chunk_members WHERE chunk_set_id = ? AND chunk_index = ? ORDER BY member_rank",
            (chunk_set_id, chunk_index),
        )
    yield from cur


def chunk_indices(conn: sqlite3.Connection, chunk_set_id: str) -> list[int]:
    cur = conn.execute(
        "SELECT DISTINCT chunk_index FROM chunk_members WHERE chunk_set_id = ? ORDER BY chunk_index",
        (chunk_set_id,),
    )
    return [row["chunk_index"] for row in cur]


def max_chunk_index(conn: sqlite3.Connection, chunk_set_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT MAX(chunk_index) AS m FROM chunk_members WHERE chunk_set_id = ?", (chunk_set_id,)
    ).fetchone()
    return row["m"]


def max_generation(conn: sqlite3.Connection, chunk_set_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT MAX(generation) AS g FROM chunk_members WHERE chunk_set_id = ?", (chunk_set_id,)
    ).fetchone()
    return row["g"]


def existing_member_site_ids(conn: sqlite3.Connection, chunk_set_id: str) -> set[str]:
    """Every site_id already assigned in this set, across all generations —
    what `chunk_work.py extend` must exclude before partitioning the
    complement as a new generation."""
    cur = conn.execute("SELECT site_id FROM chunk_members WHERE chunk_set_id = ?", (chunk_set_id,))
    return {row["site_id"] for row in cur}


def unstamp_relevance(conn: sqlite3.Connection, site_ids: list[str]) -> None:
    """The deliberate, auditable redo path for `chunk_work.py unstamp`: clears
    relevance_prompt_hash for these sites so they re-enter the pending queue
    under the CURRENT hash, without touching relevance_label/score/etc — the
    old label stays visible as a historical record until the site is
    actually re-screened. Callers MUST dump the pre-clear row to JSONL before
    calling this (chunk_work.py does); this function itself does not log.
    Commits.
    """
    conn.executemany(
        "UPDATE sites SET relevance_prompt_hash = NULL WHERE site_id = ?",
        [(site_id,) for site_id in site_ids],
    )
    conn.commit()


def iter_sites_with_stratum(
    conn: sqlite3.Connection, *, where_sql: Optional[str] = None, params: Optional[list] = None
) -> Iterator[sqlite3.Row]:
    """(site_id, sample_stratum) for chunk_work.py's partition input and its
    --dry-run per-chunk stratum table. `where_sql` is a caller-VALIDATED
    predicate fragment (chunk_work.py rejects `;` and DDL/DML keywords before
    it ever reaches here, mirroring the delete-everything plan's `--where`
    safety note) — this function trusts its caller, it does not re-validate.
    """
    sql = "SELECT site_id, sample_stratum FROM sites"
    if where_sql:
        sql += f" WHERE {where_sql}"
    cur = conn.execute(sql, params or [])
    yield from cur


def start_chunk_run(
    conn: sqlite3.Connection,
    *,
    chunk_set_id: str,
    chunk_index: int,
    consumer: str,
    scope: str,
    n_members: int,
    n_pending_start: int,
    host: Optional[str] = None,
    pid: Optional[int] = None,
    jsonl_path: Optional[str] = None,
) -> None:
    """Mark (chunk, consumer, scope) as 'running', bumping `attempts`. A
    stale 'running' row left by a `kill -9` is simply overwritten here — per
    invariant #4a, chunk_runs has no lease/claim/heartbeat and this call must
    NEVER block on one; printing a loud warning about a stale row (if found)
    is the caller's job, not this function's.
    """
    conn.execute(
        """
        INSERT INTO chunk_runs (chunk_set_id, chunk_index, consumer, scope, status,
                                attempts, n_members, n_pending_start, n_pending_end,
                                n_processed, started, finished, host, pid, jsonl_path, last_error)
        VALUES (?, ?, ?, ?, 'running', 1, ?, ?, NULL, 0, ?, NULL, ?, ?, ?, NULL)
        ON CONFLICT(chunk_set_id, chunk_index, consumer, scope) DO UPDATE SET
            status          = 'running',
            attempts        = chunk_runs.attempts + 1,
            n_members       = excluded.n_members,
            n_pending_start = excluded.n_pending_start,
            n_pending_end   = NULL,
            n_processed     = 0,
            started         = excluded.started,
            finished        = NULL,
            host            = excluded.host,
            pid             = excluded.pid,
            jsonl_path      = excluded.jsonl_path,
            last_error      = NULL
        """,
        (chunk_set_id, chunk_index, consumer, scope, n_members, n_pending_start,
         now_iso(), host, pid, jsonl_path),
    )
    conn.commit()


def finish_chunk_run(
    conn: sqlite3.Connection,
    *,
    chunk_set_id: str,
    chunk_index: int,
    consumer: str,
    scope: str,
    status: str,
    n_pending_end: int,
    n_processed: int,
    last_error: Optional[str] = None,
) -> None:
    """status must be derived by the CALLER from a fresh count_pending_relevance
    call ("done" if n_pending_end == 0 else "partial"/"interrupted") — never
    inferred here from "the loop reached the end" (Rule of One Truth)."""
    conn.execute(
        """
        UPDATE chunk_runs SET
            status        = ?,
            n_pending_end = ?,
            n_processed   = ?,
            finished      = ?,
            last_error    = ?
        WHERE chunk_set_id = ? AND chunk_index = ? AND consumer = ? AND scope = ?
        """,
        (status, n_pending_end, n_processed, now_iso(), last_error,
         chunk_set_id, chunk_index, consumer, scope),
    )
    conn.commit()


def get_chunk_run(
    conn: sqlite3.Connection, *, chunk_set_id: str, chunk_index: int, consumer: str, scope: str
) -> Optional[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM chunk_runs WHERE chunk_set_id = ? AND chunk_index = ? AND consumer = ? AND scope = ?",
        (chunk_set_id, chunk_index, consumer, scope),
    )
    return cur.fetchone()


def iter_chunk_runs(
    conn: sqlite3.Connection, chunk_set_id: str, *, consumer: Optional[str] = None
) -> Iterator[sqlite3.Row]:
    """Every recorded run for a set, for chunk_work.py status — display only,
    never consulted to decide pending work (Rule of One Truth)."""
    if consumer is None:
        cur = conn.execute(
            "SELECT * FROM chunk_runs WHERE chunk_set_id = ? ORDER BY chunk_index, scope", (chunk_set_id,)
        )
    else:
        cur = conn.execute(
            "SELECT * FROM chunk_runs WHERE chunk_set_id = ? AND consumer = ? ORDER BY chunk_index, scope",
            (chunk_set_id, consumer),
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
