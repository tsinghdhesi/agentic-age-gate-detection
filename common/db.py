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

SCHEMA_VERSION = 1

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

-- Crawl queue. Ingests ALL seed URLs (full provenance); a later pass populates
-- inclusion/sampling. cctld stores the EXACT public suffix (e.g. com, org,
-- co.uk, com.br, ru) and is NEVER null for a parseable host.
CREATE TABLE IF NOT EXISTS sites (
    site_id            TEXT PRIMARY KEY,          -- hostslug_<6char url hash>
    url_original       TEXT NOT NULL,
    url_normalized     TEXT NOT NULL,
    host               TEXT,
    registrable_domain TEXT,
    cctld              TEXT,                      -- exact eTLD, never null
    industry           TEXT NOT NULL,
    source_file        TEXT,
    added_ts           TEXT,
    included           INTEGER NOT NULL DEFAULT 0,-- crawler reads only included=1
    relevance_label    TEXT,                      -- NULL until classified
    sample_stratum     TEXT,
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
CREATE INDEX IF NOT EXISTS idx_obs_run             ON observations(run_id);
CREATE INDEX IF NOT EXISTS idx_obs_failure         ON observations(failure_reason);
CREATE INDEX IF NOT EXISTS idx_det_model           ON detections(model);
CREATE INDEX IF NOT EXISTS idx_det_gate            ON detections(age_gate_detected);
CREATE INDEX IF NOT EXISTS idx_outcome_status      ON site_outcomes(terminal_status);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables + indexes (idempotent) and stamp the schema version."""
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
    included: int = 0,
    relevance_label: Optional[str] = None,
    sample_stratum: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sites (site_id, url_original, url_normalized, host,
                           registrable_domain, cctld, industry, source_file,
                           added_ts, included, relevance_label, sample_stratum)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(site_id) DO UPDATE SET
            url_original       = excluded.url_original,
            url_normalized     = excluded.url_normalized,
            host               = excluded.host,
            registrable_domain = excluded.registrable_domain,
            cctld              = excluded.cctld,
            industry           = excluded.industry,
            source_file        = excluded.source_file,
            included           = excluded.included,
            relevance_label    = excluded.relevance_label,
            sample_stratum     = excluded.sample_stratum
        """,
        (site_id, url_original, url_normalized, host, registrable_domain, cctld,
         industry, source_file, now_iso(), included, relevance_label, sample_stratum),
    )


def get_site(conn: sqlite3.Connection, site_id: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM sites WHERE site_id = ?", (site_id,))
    return cur.fetchone()


def iter_included_sites(
    conn: sqlite3.Connection, industry: Optional[str] = None
) -> Iterator[sqlite3.Row]:
    """Crawl queue: sites flagged included=1, optionally filtered by industry."""
    if industry is None:
        cur = conn.execute("SELECT * FROM sites WHERE included = 1 ORDER BY site_id")
    else:
        cur = conn.execute(
            "SELECT * FROM sites WHERE included = 1 AND industry = ? ORDER BY site_id",
            (industry,),
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
