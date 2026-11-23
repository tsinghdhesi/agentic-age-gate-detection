"""Unit tests for common/db.py — schema, upserts, and the sampling/relevance
writers added for frame construction. Run: pytest -q

Every test uses a fresh in-memory SQLite connection (db.connect(":memory:")),
so the suite is fast and never touches disk. test_upsert_site_preserves_*
tests are the load-bearing ones here: they are the regression tests that
protect a 24h+ relevance screen (or a stratified sample draw) from being
silently wiped out by a re-run of 01_build_queue — see the docstring on
upsert_site() for the bug this guards against.
"""

import sqlite3

import pytest

from common import db


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def _upsert_frame_site(conn, site_id="vape_co_uk_abc123", **overrides):
    # url_normalized defaults to something derived from site_id so distinct
    # site_ids used across a test don't collide on the real UNIQUE(url_normalized)
    # constraint unless a test explicitly wants that collision.
    kwargs = dict(
        site_id=site_id,
        url_original=f"https://www.{site_id}.example",
        url_normalized=f"https://{site_id}.example/",
        host=f"{site_id}.example",
        registrable_domain=f"{site_id}.example",
        cctld="co.uk",
        industry="nicotine",
        source_file="seed.csv",
        sample_stratum="uk_ie",
    )
    kwargs.update(overrides)
    db.upsert_site(conn, **kwargs)
    conn.commit()


# --------------------------------------------------------------------------- #
# init_db / schema version guard
# --------------------------------------------------------------------------- #


def test_init_db_creates_all_tables(conn):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur}
    assert tables >= {
        "runs", "sites", "observations", "detections", "block_checks", "site_outcomes",
    }


def test_init_db_stamps_schema_version(conn):
    version = conn.execute("PRAGMA user_version;").fetchone()[0]
    assert version == db.SCHEMA_VERSION


def test_init_db_is_idempotent():
    c = db.connect(":memory:")
    db.init_db(c)
    db.init_db(c)  # must not raise
    assert conn_version(c) == db.SCHEMA_VERSION


def conn_version(c):
    return c.execute("PRAGMA user_version;").fetchone()[0]


def test_init_db_raises_on_mismatched_existing_version():
    c = db.connect(":memory:")
    c.execute("PRAGMA user_version = 999;")
    c.commit()
    with pytest.raises(db.SchemaVersionMismatch):
        db.init_db(c)


# --------------------------------------------------------------------------- #
# sites — new v2 columns exist and are NULL by default
# --------------------------------------------------------------------------- #


def test_sites_table_has_v3_columns(conn):
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(sites)")}
    assert cols >= {
        "relevance_score", "relevance_model", "relevance_prompt_hash", "relevance_ts",
        "http_status", "sample_u", "sample_wave",
    }


def test_upsert_site_leaves_v2_columns_null_on_first_insert(conn):
    _upsert_frame_site(conn)
    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["sample_u"] is None
    assert row["sample_wave"] is None
    assert row["relevance_score"] is None


# --------------------------------------------------------------------------- #
# upsert_site idempotency — the load-bearing regression tests.
# --------------------------------------------------------------------------- #


def test_upsert_site_preserves_inclusion_on_reingest(conn):
    _upsert_frame_site(conn)
    db.set_site_relevance(
        conn, "vape_co_uk_abc123",
        relevance_label="retailer", included=1, relevance_score=5.0, relevance_prompt_hash="hash_v1",
    )

    # Re-run 01_build_queue-style: a plain frame re-ingest with no relevance
    # info supplied (the real call site never passes included/relevance_label).
    _upsert_frame_site(conn)

    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["included"] == 1
    assert row["relevance_label"] == "retailer"
    assert row["relevance_score"] == 5.0


def test_upsert_site_preserves_sample_draw_on_reingest(conn):
    _upsert_frame_site(conn)
    db.set_site_sample_u(conn, [("vape_co_uk_abc123", 0.1234)])
    db.mark_sites_drawn(conn, ["vape_co_uk_abc123"], wave=1)

    _upsert_frame_site(conn)  # re-ingest, as if 01_build_queue ran again

    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["sample_u"] == pytest.approx(0.1234)
    assert row["sample_wave"] == 1


def test_upsert_site_updates_frame_facts_on_reingest(conn):
    # Frame facts (host, cctld, registrable_domain, sample_stratum, ...) ARE
    # meant to be refreshed by a re-ingest — only inclusion/sampling state is
    # protected.
    _upsert_frame_site(conn, source_file="seed_v1.csv")
    _upsert_frame_site(conn, source_file="seed_v2.csv", url_original="https://vape.co.uk/new")
    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["source_file"] == "seed_v2.csv"
    assert row["url_original"] == "https://vape.co.uk/new"


def test_upsert_site_explicit_included_overrides(conn):
    # A caller that DOES pass included=1 explicitly (e.g. a one-off manual
    # correction) must still be able to set it directly.
    _upsert_frame_site(conn, included=1, relevance_label="retailer")
    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["included"] == 1
    assert row["relevance_label"] == "retailer"


def test_upsert_site_url_normalized_unique_constraint(conn):
    _upsert_frame_site(conn, site_id="site_a", url_normalized="https://shared.example/")
    with pytest.raises(sqlite3.IntegrityError):
        db.upsert_site(
            conn,
            site_id="site_b",  # different site_id, SAME url_normalized
            url_original="https://shared.example",
            url_normalized="https://shared.example/",
            host="shared.example",
            registrable_domain="shared.example",
            cctld="com",
            industry="nicotine",
        )


def test_upsert_site_does_not_autocommit(conn):
    # upsert_site deliberately does NOT commit — the caller batches. Verify
    # a rollback undoes an uncommitted upsert_site call.
    _upsert_frame_site(conn, site_id="committed_site")
    db.upsert_site(
        conn, site_id="uncommitted_site", url_original="https://x.com",
        url_normalized="https://x.com/", host="x.com", registrable_domain="x.com",
        cctld="com", industry="nicotine",
    )
    conn.rollback()
    assert db.get_site(conn, "committed_site") is not None
    assert db.get_site(conn, "uncommitted_site") is None


# --------------------------------------------------------------------------- #
# set_site_relevance / iter_pending_relevance
# --------------------------------------------------------------------------- #


def test_iter_pending_relevance_returns_unclassified_sites(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1, relevance_prompt_hash="hash_v1")

    pending = [row["site_id"] for row in db.iter_pending_relevance(conn, prompt_hash="hash_v1")]
    assert pending == ["b"]


def test_iter_pending_relevance_requeues_on_prompt_change(conn):
    _upsert_frame_site(conn, site_id="a")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1, relevance_prompt_hash="hash_v1")

    pending_v1 = list(db.iter_pending_relevance(conn, prompt_hash="hash_v1"))
    pending_v2 = list(db.iter_pending_relevance(conn, prompt_hash="hash_v2"))
    assert pending_v1 == []
    assert [r["site_id"] for r in pending_v2] == ["a"]


def test_iter_pending_relevance_retry_failed(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1, relevance_prompt_hash="hash_v1")
    db.set_site_relevance(conn, "b", relevance_label="blocked", included=0, relevance_prompt_hash="hash_v1")

    without_retry = [r["site_id"] for r in db.iter_pending_relevance(conn, prompt_hash="hash_v1")]
    with_retry = [
        r["site_id"]
        for r in db.iter_pending_relevance(conn, prompt_hash="hash_v1", retry_failed=True)
    ]
    assert without_retry == []
    assert with_retry == ["b"]


def test_iter_pending_relevance_filters_by_industry(conn):
    _upsert_frame_site(conn, site_id="a", industry="nicotine")
    _upsert_frame_site(conn, site_id="b", industry="alcohol")
    pending = [r["site_id"] for r in db.iter_pending_relevance(conn, prompt_hash="hash_v1", industry="alcohol")]
    assert pending == ["b"]


def test_set_site_relevance_commits(conn):
    _upsert_frame_site(conn)
    db.set_site_relevance(conn, "vape_co_uk_abc123", relevance_label="unrelated", included=0)
    conn2_row = conn.execute(
        "SELECT relevance_label FROM sites WHERE site_id = ?", ("vape_co_uk_abc123",)
    ).fetchone()
    assert conn2_row["relevance_label"] == "unrelated"


def test_set_site_relevance_increments_attempts(conn):
    _upsert_frame_site(conn)
    assert db.get_site(conn, "vape_co_uk_abc123")["relevance_attempts"] == 0
    db.set_site_relevance(conn, "vape_co_uk_abc123", relevance_label="unrelated", included=0)
    assert db.get_site(conn, "vape_co_uk_abc123")["relevance_attempts"] == 1
    db.set_site_relevance(conn, "vape_co_uk_abc123", relevance_label="unrelated", included=0)
    assert db.get_site(conn, "vape_co_uk_abc123")["relevance_attempts"] == 2


def test_iter_pending_relevance_retry_failed_respects_max_attempts(conn):
    _upsert_frame_site(conn, site_id="a")
    # 3 attempts, still failing every time
    for _ in range(3):
        db.set_site_relevance(conn, "a", relevance_label="blocked", included=0, relevance_prompt_hash="hash_v1")

    under_cap = [
        r["site_id"] for r in db.iter_pending_relevance(
            conn, prompt_hash="hash_v1", retry_failed=True, max_attempts=5,
        )
    ]
    at_cap = [
        r["site_id"] for r in db.iter_pending_relevance(
            conn, prompt_hash="hash_v1", retry_failed=True, max_attempts=3,
        )
    ]
    assert under_cap == ["a"]
    assert at_cap == []  # a permanently-dead host must stop being re-queued


def test_count_pending_relevance_agrees_with_iter(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.set_site_relevance(conn, "a", relevance_label="blocked", included=0, relevance_prompt_hash="hash_v1")

    for kwargs in (
        dict(prompt_hash="hash_v1"),
        dict(prompt_hash="hash_v1", retry_failed=True),
        dict(prompt_hash="hash_v1", retry_failed=True, max_attempts=1),
        dict(prompt_hash="hash_v1", industry="nicotine"),
        dict(prompt_hash="hash_v2"),
    ):
        n_iter = len(list(db.iter_pending_relevance(conn, **kwargs)))
        n_count = db.count_pending_relevance(conn, **kwargs)
        assert n_iter == n_count, kwargs


# --------------------------------------------------------------------------- #
# set_site_sample_u / mark_sites_drawn — write-once semantics
# --------------------------------------------------------------------------- #


def test_set_site_sample_u_write_once(conn):
    _upsert_frame_site(conn)
    db.set_site_sample_u(conn, [("vape_co_uk_abc123", 0.111)])
    db.set_site_sample_u(conn, [("vape_co_uk_abc123", 0.999)])  # must NOT overwrite
    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["sample_u"] == pytest.approx(0.111)


def test_mark_sites_drawn_write_once(conn):
    _upsert_frame_site(conn)
    db.mark_sites_drawn(conn, ["vape_co_uk_abc123"], wave=1)
    db.mark_sites_drawn(conn, ["vape_co_uk_abc123"], wave=2)  # must NOT overwrite
    row = db.get_site(conn, "vape_co_uk_abc123")
    assert row["sample_wave"] == 1


def test_set_site_sample_u_batch(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.set_site_sample_u(conn, [("a", 0.1), ("b", 0.2)])
    assert db.get_site(conn, "a")["sample_u"] == pytest.approx(0.1)
    assert db.get_site(conn, "b")["sample_u"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# frame_counts_by_stratum / iter_undrawn_relevant_sites
# --------------------------------------------------------------------------- #


def test_frame_counts_by_stratum(conn):
    _upsert_frame_site(conn, site_id="a", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="b", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="c", sample_stratum="eu_west")
    counts = db.frame_counts_by_stratum(conn)
    assert counts == {"uk_ie": 2, "eu_west": 1}


def test_frame_counts_by_stratum_included_only(conn):
    _upsert_frame_site(conn, site_id="a", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="b", sample_stratum="uk_ie")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1)
    db.set_site_relevance(conn, "b", relevance_label="unrelated", included=0)
    counts = db.frame_counts_by_stratum(conn, included_only=True)
    assert counts == {"uk_ie": 1}


def test_frame_counts_by_stratum_undrawn_only(conn):
    _upsert_frame_site(conn, site_id="a", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="b", sample_stratum="uk_ie")
    db.mark_sites_drawn(conn, ["a"], wave=1)
    counts = db.frame_counts_by_stratum(conn, undrawn_only=True)
    assert counts == {"uk_ie": 1}


def test_iter_undrawn_relevant_sites_orders_by_stratum_then_u(conn):
    _upsert_frame_site(conn, site_id="a", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="b", sample_stratum="uk_ie")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1)
    db.set_site_relevance(conn, "b", relevance_label="retailer", included=1)
    db.set_site_sample_u(conn, [("a", 0.9), ("b", 0.1)])

    rows = list(db.iter_undrawn_relevant_sites(conn))
    assert [r["site_id"] for r in rows] == ["b", "a"]


def test_iter_undrawn_relevant_sites_excludes_drawn(conn):
    _upsert_frame_site(conn, site_id="a")
    db.set_site_relevance(conn, "a", relevance_label="retailer", included=1)
    db.mark_sites_drawn(conn, ["a"], wave=1)
    assert list(db.iter_undrawn_relevant_sites(conn)) == []


def test_iter_undrawn_relevant_sites_excludes_unincluded(conn):
    _upsert_frame_site(conn, site_id="a")
    db.set_site_relevance(conn, "a", relevance_label="unrelated", included=0)
    assert list(db.iter_undrawn_relevant_sites(conn)) == []


# --------------------------------------------------------------------------- #
# iter_included_sites — order_by
# --------------------------------------------------------------------------- #


def test_iter_included_sites_order_by_sample_u(conn):
    _upsert_frame_site(conn, site_id="a", included=1)
    _upsert_frame_site(conn, site_id="b", included=1)
    db.set_site_sample_u(conn, [("a", 0.9), ("b", 0.1)])
    rows = list(db.iter_included_sites(conn, order_by="sample_u"))
    assert [r["site_id"] for r in rows] == ["b", "a"]


def test_iter_included_sites_rejects_bad_order_by(conn):
    with pytest.raises(ValueError):
        list(db.iter_included_sites(conn, order_by="'; DROP TABLE sites; --"))


def test_iter_included_sites_only_returns_included(conn):
    _upsert_frame_site(conn, site_id="a", included=1)
    _upsert_frame_site(conn, site_id="b", included=0)
    rows = list(db.iter_included_sites(conn))
    assert [r["site_id"] for r in rows] == ["a"]


# --------------------------------------------------------------------------- #
# Pre-existing functions — smoke coverage (no test file existed before this).
# --------------------------------------------------------------------------- #


def test_start_run_and_finish_run(conn):
    db.start_run(conn, "run1", config_hash="abc", detection_prompt_hash="def")
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert row["config_hash"] == "abc"
    assert row["finished"] is None
    db.finish_run(conn, "run1")
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert row["finished"] is not None


def test_start_run_coalesces_on_reupsert(conn):
    db.start_run(conn, "run1", config_hash="abc")
    db.start_run(conn, "run1")  # re-call with no config_hash must not wipe it
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert row["config_hash"] == "abc"


def test_upsert_observation_roundtrip(conn):
    db.start_run(conn, "run1")
    _upsert_frame_site(conn)
    db.upsert_observation(
        conn, run_id="run1", site_id="vape_co_uk_abc123", checkpoint="site_arrival",
        reached=1, fullpage_path="a.png", viewport_path="b.png",
    )
    row = conn.execute(
        "SELECT * FROM observations WHERE run_id=? AND site_id=? AND checkpoint=?",
        ("run1", "vape_co_uk_abc123", "site_arrival"),
    ).fetchone()
    assert row["reached"] == 1
    assert row["fullpage_path"] == "a.png"


def test_upsert_detection_preserves_age_gate_type(conn):
    db.start_run(conn, "run1")
    _upsert_frame_site(conn)
    db.upsert_observation(conn, run_id="run1", site_id="vape_co_uk_abc123", checkpoint="site_arrival", reached=1)
    db.upsert_detection(
        conn, run_id="run1", site_id="vape_co_uk_abc123", checkpoint="site_arrival",
        model="qwen3-vl:30b", age_gate_detected=1, raw_vlm_output="1",
        age_gate_type="interstitial",
    )
    # Re-upsert without age_gate_type — must not be wiped (manual label).
    db.upsert_detection(
        conn, run_id="run1", site_id="vape_co_uk_abc123", checkpoint="site_arrival",
        model="qwen3-vl:30b", age_gate_detected=1, raw_vlm_output="1",
    )
    row = conn.execute(
        "SELECT * FROM detections WHERE run_id=? AND site_id=? AND checkpoint=? AND model=?",
        ("run1", "vape_co_uk_abc123", "site_arrival", "qwen3-vl:30b"),
    ).fetchone()
    assert row["age_gate_type"] == "interstitial"


def test_upsert_site_outcome_roundtrip(conn):
    db.start_run(conn, "run1")
    _upsert_frame_site(conn)
    db.upsert_site_outcome(
        conn, run_id="run1", site_id="vape_co_uk_abc123",
        max_checkpoint_reached="checkout_initiation", terminal_status="completed",
    )
    row = conn.execute(
        "SELECT * FROM site_outcomes WHERE run_id=? AND site_id=?",
        ("run1", "vape_co_uk_abc123"),
    ).fetchone()
    assert row["terminal_status"] == "completed"


# --------------------------------------------------------------------------- #
# chunking (schema v5): chunk_sets / chunk_members / chunk_runs
# --------------------------------------------------------------------------- #


def _seed_sites(conn, n, prefix="site"):
    ids = [f"{prefix}{i}" for i in range(n)]
    for site_id in ids:
        _upsert_frame_site(conn, site_id=site_id)
    conn.commit()
    return ids


def _make_chunk_set(conn, chunk_set_id="cs1", n_sites=10, chunk_size=4, seed="seed1"):
    ids = _seed_sites(conn, n_sites)
    db.create_chunk_set(conn, chunk_set_id=chunk_set_id, intended_consumer="01b_screen_relevance",
                        seed=seed, chunk_size=chunk_size)
    from common.chunking import partition

    db.insert_chunk_members(conn, chunk_set_id, partition(ids, seed=seed, chunk_size=chunk_size))
    return ids


def test_create_chunk_set_and_get(conn):
    db.create_chunk_set(conn, chunk_set_id="cs1", intended_consumer="01b_screen_relevance",
                        seed="seed1", chunk_size=100, notes="pilot")
    row = db.get_chunk_set(conn, "cs1")
    assert row["intended_consumer"] == "01b_screen_relevance"
    assert row["chunk_size"] == 100
    assert row["notes"] == "pilot"


def test_create_chunk_set_duplicate_id_raises(conn):
    db.create_chunk_set(conn, chunk_set_id="cs1", intended_consumer="01b_screen_relevance",
                        seed="seed1", chunk_size=100)
    with pytest.raises(sqlite3.IntegrityError):
        db.create_chunk_set(conn, chunk_set_id="cs1", intended_consumer="01b_screen_relevance",
                            seed="seed1", chunk_size=100)


def test_insert_and_iter_chunk_members(conn):
    ids = _make_chunk_set(conn, n_sites=10, chunk_size=4)
    all_members = [r["site_id"] for r in db.iter_chunk_members(conn, "cs1")]
    assert sorted(all_members) == sorted(ids)
    assert db.chunk_indices(conn, "cs1") == [0, 1, 2]
    assert db.max_chunk_index(conn, "cs1") == 2
    assert db.max_generation(conn, "cs1") == 0


def test_iter_chunk_members_single_chunk_ordered_by_rank(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    chunk0 = list(db.iter_chunk_members(conn, "cs1", 0))
    ranks = [r["member_rank"] for r in chunk0]
    assert ranks == sorted(ranks) == list(range(len(chunk0)))


def test_chunk_members_write_once_trigger_aborts_update(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    site_id = db.chunk_indices(conn, "cs1") and list(db.iter_chunk_members(conn, "cs1", 0))[0]["site_id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE chunk_members SET member_rank = 999 WHERE chunk_set_id = ? AND site_id = ?",
            ("cs1", site_id),
        )


def test_existing_member_site_ids(conn):
    ids = _make_chunk_set(conn, n_sites=10, chunk_size=4)
    assert db.existing_member_site_ids(conn, "cs1") == set(ids)


def test_extend_adds_new_generation_without_touching_existing(conn):
    ids = _make_chunk_set(conn, n_sites=10, chunk_size=4)
    before = {(r["site_id"], r["chunk_index"], r["member_rank"], r["generation"])
              for r in db.iter_chunk_members(conn, "cs1")}

    new_ids = _seed_sites(conn, 6, prefix="new")
    from common.chunking import partition

    next_gen = db.max_generation(conn, "cs1") + 1
    next_idx = db.max_chunk_index(conn, "cs1") + 1
    assignments = partition(new_ids, seed="seed1", chunk_size=4, generation=next_gen, first_chunk_index=next_idx)
    db.insert_chunk_members(conn, "cs1", assignments)

    after = {(r["site_id"], r["chunk_index"], r["member_rank"], r["generation"])
             for r in db.iter_chunk_members(conn, "cs1")}
    assert before <= after  # existing membership untouched
    assert db.existing_member_site_ids(conn, "cs1") == set(ids) | set(new_ids)
    assert db.max_generation(conn, "cs1") == next_gen


def test_iter_sites_with_stratum(conn):
    _upsert_frame_site(conn, site_id="a", sample_stratum="uk_ie")
    _upsert_frame_site(conn, site_id="b", sample_stratum="gtld")
    rows = {r["site_id"]: r["sample_stratum"] for r in db.iter_sites_with_stratum(conn)}
    assert rows == {"a": "uk_ie", "b": "gtld"}


def test_iter_sites_with_stratum_where_filter(conn):
    _upsert_frame_site(conn, site_id="a", industry="nicotine")
    _upsert_frame_site(conn, site_id="b", industry="alcohol")
    rows = list(db.iter_sites_with_stratum(conn, where_sql="industry = ?", params=["alcohol"]))
    assert [r["site_id"] for r in rows] == ["b"]


def test_pending_relevance_chunk_join_restricts_to_one_chunk(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    chunk0_ids = {r["site_id"] for r in db.iter_chunk_members(conn, "cs1", 0)}

    pending = [r["site_id"] for r in db.iter_pending_relevance(
        conn, prompt_hash="hash_v1", chunk_set="cs1", chunk_index=0,
    )]
    assert set(pending) == chunk0_ids
    assert db.count_pending_relevance(conn, prompt_hash="hash_v1", chunk_set="cs1", chunk_index=0) == len(chunk0_ids)


def test_pending_relevance_chunk_join_orders_by_member_rank(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    pending = list(db.iter_pending_relevance(conn, prompt_hash="hash_v1", chunk_set="cs1", chunk_index=0))
    members_by_rank = {r["site_id"]: r["member_rank"] for r in db.iter_chunk_members(conn, "cs1", 0)}
    ranks = [members_by_rank[r["site_id"]] for r in pending]
    assert ranks == sorted(ranks)


def test_next_pending_chunk_index_advances_as_chunks_complete(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    assert db.next_pending_chunk_index(conn, "cs1", prompt_hash="hash_v1") == 0

    for r in db.iter_chunk_members(conn, "cs1", 0):
        db.set_site_relevance(conn, r["site_id"], relevance_label="unrelated", included=0,
                              relevance_prompt_hash="hash_v1")
    assert db.next_pending_chunk_index(conn, "cs1", prompt_hash="hash_v1") == 1

    for idx in (1, 2):
        for r in db.iter_chunk_members(conn, "cs1", idx):
            db.set_site_relevance(conn, r["site_id"], relevance_label="unrelated", included=0,
                                  relevance_prompt_hash="hash_v1")
    assert db.next_pending_chunk_index(conn, "cs1", prompt_hash="hash_v1") is None


def test_next_pending_chunk_index_does_not_consult_chunk_runs(conn):
    """The Rule of One Truth: --chunk next must resolve purely from the
    pending ⋈ chunk_members join, never from chunk_runs.status. Simulate a
    chunk marked 'done' in chunk_runs whose sites were never actually
    screened (e.g. a bogus/forged row) — it must still be reported pending.
    """
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    db.start_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                       scope="hash_v1", n_members=4, n_pending_start=4)
    db.finish_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                        scope="hash_v1", status="done", n_pending_end=0, n_processed=4)
    # No sites were actually screened (relevance_prompt_hash still NULL).
    assert db.next_pending_chunk_index(conn, "cs1", prompt_hash="hash_v1") == 0


def test_start_and_finish_chunk_run_lifecycle(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    db.start_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                       scope="hash_v1", n_members=4, n_pending_start=4, host="mac", pid=42,
                       jsonl_path="c00000.jsonl")
    row = db.get_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance", scope="hash_v1")
    assert row["status"] == "running"
    assert row["attempts"] == 1
    assert row["finished"] is None

    db.finish_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                        scope="hash_v1", status="partial", n_pending_end=2, n_processed=2,
                        last_error="interrupted")
    row = db.get_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance", scope="hash_v1")
    assert row["status"] == "partial"
    assert row["n_pending_end"] == 2
    assert row["finished"] is not None
    assert row["last_error"] == "interrupted"


def test_start_chunk_run_overwrites_stale_running_row(conn):
    """A kill -9 leaves a 'running' row behind. It is NOT a lock — the next
    invocation must simply overwrite it, bumping attempts, never block."""
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    db.start_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                       scope="hash_v1", n_members=4, n_pending_start=4)
    # never finished (simulated crash) — start again
    db.start_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                       scope="hash_v1", n_members=4, n_pending_start=4)
    row = db.get_chunk_run(conn, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance", scope="hash_v1")
    assert row["status"] == "running"
    assert row["attempts"] == 2


def test_iter_chunk_runs(conn):
    _make_chunk_set(conn, n_sites=10, chunk_size=4)
    for idx in (0, 1):
        db.start_chunk_run(conn, chunk_set_id="cs1", chunk_index=idx, consumer="01b_screen_relevance",
                           scope="hash_v1", n_members=4, n_pending_start=4)
        db.finish_chunk_run(conn, chunk_set_id="cs1", chunk_index=idx, consumer="01b_screen_relevance",
                            scope="hash_v1", status="done", n_pending_end=0, n_processed=4)
    runs = list(db.iter_chunk_runs(conn, "cs1"))
    assert [r["chunk_index"] for r in runs] == [0, 1]
    assert all(r["status"] == "done" for r in runs)


def test_chunk_runs_scope_reconciles_prompt_revision():
    """A frozen-prompt revision must reconcile with zero cleanup: the OLD
    scope's chunk_runs row stays 'done' forever (a true historical statement
    about the old instrument); under the NEW hash the same chunk reports
    pending again, entirely from the sites-table join — chunk_runs for the
    new scope simply doesn't exist yet, and that absence IS "pending"."""
    c = db.connect(":memory:")
    db.init_db(c)
    ids = _make_chunk_set(c, n_sites=4, chunk_size=4, seed="seed1")
    for site_id in ids:
        db.set_site_relevance(c, site_id, relevance_label="unrelated", included=0,
                              relevance_prompt_hash="hash_v1")
    db.start_chunk_run(c, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                       scope="hash_v1", n_members=4, n_pending_start=4)
    db.finish_chunk_run(c, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance",
                        scope="hash_v1", status="done", n_pending_end=0, n_processed=4)

    # Old scope: done, and genuinely 0 pending.
    old_run = db.get_chunk_run(c, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance", scope="hash_v1")
    assert old_run["status"] == "done"
    assert db.count_pending_relevance(c, prompt_hash="hash_v1", chunk_set="cs1", chunk_index=0) == 0

    # New scope (revised prompt hash): no chunk_runs row at all, and the
    # SAME partition (chunk_members untouched) reports fully pending again.
    new_run = db.get_chunk_run(c, chunk_set_id="cs1", chunk_index=0, consumer="01b_screen_relevance", scope="hash_v2")
    assert new_run is None
    assert db.count_pending_relevance(c, prompt_hash="hash_v2", chunk_set="cs1", chunk_index=0) == 4
    c.close()


# --------------------------------------------------------------------------- #
# collection pending queue (scope=run_id) — iter_pending_collect /
# count_pending_collect / next_pending_collect_chunk_index
# --------------------------------------------------------------------------- #


def test_iter_pending_collect_returns_sites_with_no_outcome(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.start_run(conn, "run1")
    db.upsert_site_outcome(conn, run_id="run1", site_id="a",
                           max_checkpoint_reached="cart", terminal_status="completed")

    pending = [r["site_id"] for r in db.iter_pending_collect(conn, run_id="run1")]
    assert pending == ["b"]


def test_iter_pending_collect_is_scoped_per_run(conn):
    _upsert_frame_site(conn, site_id="a")
    db.start_run(conn, "run1")
    db.start_run(conn, "run2")
    db.upsert_site_outcome(conn, run_id="run1", site_id="a",
                           max_checkpoint_reached="cart", terminal_status="completed")

    assert list(db.iter_pending_collect(conn, run_id="run1")) == []
    assert [r["site_id"] for r in db.iter_pending_collect(conn, run_id="run2")] == ["a"]


def test_iter_pending_collect_retry_blocked(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.start_run(conn, "run1")
    db.upsert_site_outcome(conn, run_id="run1", site_id="a",
                           max_checkpoint_reached="cart", terminal_status="completed")
    db.upsert_site_outcome(conn, run_id="run1", site_id="b",
                           max_checkpoint_reached="store", terminal_status="dropped_blocked",
                           drop_checkpoint="store", drop_reason="blocked_bot")

    without_retry = [r["site_id"] for r in db.iter_pending_collect(conn, run_id="run1")]
    with_retry = [r["site_id"] for r in db.iter_pending_collect(conn, run_id="run1", retry_blocked=True)]
    assert without_retry == []
    assert with_retry == ["b"]


def test_iter_pending_collect_filters_by_industry(conn):
    _upsert_frame_site(conn, site_id="a", industry="nicotine")
    _upsert_frame_site(conn, site_id="b", industry="alcohol")
    db.start_run(conn, "run1")
    pending = [r["site_id"] for r in db.iter_pending_collect(conn, run_id="run1", industry="alcohol")]
    assert pending == ["b"]


def test_count_pending_collect_agrees_with_iter(conn):
    _upsert_frame_site(conn, site_id="a")
    _upsert_frame_site(conn, site_id="b")
    db.start_run(conn, "run1")
    db.upsert_site_outcome(conn, run_id="run1", site_id="a",
                           max_checkpoint_reached="store", terminal_status="dropped_blocked",
                           drop_checkpoint="store", drop_reason="blocked_bot")

    for kwargs in (
        dict(run_id="run1"),
        dict(run_id="run1", retry_blocked=True),
        dict(run_id="run1", industry="nicotine"),
    ):
        assert db.count_pending_collect(conn, **kwargs) == len(list(db.iter_pending_collect(conn, **kwargs)))


def test_pending_collect_chunk_join_restricts_to_one_chunk(conn):
    ids = _make_chunk_set(conn, chunk_set_id="collect1", n_sites=10, chunk_size=4)
    db.start_run(conn, "run1")
    chunk0_ids = {r["site_id"] for r in db.iter_chunk_members(conn, "collect1", 0)}

    pending = [r["site_id"] for r in db.iter_pending_collect(
        conn, run_id="run1", chunk_set="collect1", chunk_index=0,
    )]
    assert set(pending) == chunk0_ids
    assert db.count_pending_collect(conn, run_id="run1", chunk_set="collect1", chunk_index=0) == len(chunk0_ids)
    assert set(ids) >= chunk0_ids  # sanity: chunk0 is a subset of the full set


def test_next_pending_collect_chunk_index_advances_as_chunks_complete(conn):
    _make_chunk_set(conn, chunk_set_id="collect1", n_sites=10, chunk_size=4)
    db.start_run(conn, "run1")
    assert db.next_pending_collect_chunk_index(conn, "collect1", run_id="run1") == 0

    for r in db.iter_chunk_members(conn, "collect1", 0):
        db.upsert_site_outcome(conn, run_id="run1", site_id=r["site_id"],
                               max_checkpoint_reached="cart", terminal_status="completed")
    assert db.next_pending_collect_chunk_index(conn, "collect1", run_id="run1") == 1

    for idx in (1, 2):
        for r in db.iter_chunk_members(conn, "collect1", idx):
            db.upsert_site_outcome(conn, run_id="run1", site_id=r["site_id"],
                                   max_checkpoint_reached="cart", terminal_status="completed")
    assert db.next_pending_collect_chunk_index(conn, "collect1", run_id="run1") is None


def test_next_pending_collect_chunk_index_does_not_consult_chunk_runs(conn):
    """Rule of One Truth: a bogus 'done' chunk_runs row for this run must not
    hide genuinely pending sites — only site_outcomes existence decides."""
    _make_chunk_set(conn, chunk_set_id="collect1", n_sites=10, chunk_size=4)
    db.start_run(conn, "run1")
    db.start_chunk_run(conn, chunk_set_id="collect1", chunk_index=0, consumer="02_collect",
                       scope="run1", n_members=4, n_pending_start=4)
    db.finish_chunk_run(conn, chunk_set_id="collect1", chunk_index=0, consumer="02_collect",
                        scope="run1", status="done", n_pending_end=0, n_processed=4)
    # No site_outcomes rows were actually written for run1.
    assert db.next_pending_collect_chunk_index(conn, "collect1", run_id="run1") == 0
