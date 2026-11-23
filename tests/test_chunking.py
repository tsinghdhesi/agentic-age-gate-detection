"""Unit tests for common/chunking.py — pure partition math for addressable
work chunks. Run: pytest -q

test_partition_matches_naive_drain_the_pool is the load-bearing test: it
implements the ORIGINAL stated spec ("duplicate the pool, repeatedly draw n
at random without replacement until empty") as a literal executable
reference — using partition_key() as the deterministic "random" draw value
instead of Python's random module — and asserts partition() (sort-then-slice)
produces the IDENTICAL assignment. This is what proves the two are the same
algorithm rather than merely similar-looking ones.
"""

from __future__ import annotations

from common import chunking


def _naive_drain_the_pool(site_ids, *, seed, chunk_size, generation=0, first_chunk_index=0):
    """Literal reference implementation of the original spec: repeatedly draw
    the "random" (here: lowest partition_key) item from the remaining pool
    without replacement, filling one chunk at a time. O(n^2) by design — it
    exists to be obviously correct, not fast.
    """
    pool = list(site_ids)
    out = []
    chunk_index = first_chunk_index
    while pool:
        chunk = []
        for _ in range(min(chunk_size, len(pool))):
            best = min(pool, key=lambda sid: chunking.partition_key(seed, generation, sid))
            pool.remove(best)
            chunk.append(best)
        for rank, sid in enumerate(chunk):
            out.append((sid, chunk_index, rank, generation))
        chunk_index += 1
    return out


def _site_ids(n, prefix="site"):
    return [f"{prefix}_{i}" for i in range(n)]


def test_partition_matches_naive_drain_the_pool():
    site_ids = _site_ids(137)
    fast = chunking.partition(site_ids, seed="s1", chunk_size=10)
    naive = _naive_drain_the_pool(site_ids, seed="s1", chunk_size=10)
    assert fast == naive


def test_partition_matches_naive_drain_the_pool_with_generation_and_offset():
    site_ids = _site_ids(53, prefix="extra")
    fast = chunking.partition(site_ids, seed="chunk-seed", chunk_size=7, generation=1, first_chunk_index=8)
    naive = _naive_drain_the_pool(site_ids, seed="chunk-seed", chunk_size=7, generation=1, first_chunk_index=8)
    assert fast == naive


def test_partition_key_deterministic():
    assert chunking.partition_key("s", 0, "a") == chunking.partition_key("s", 0, "a")


def test_partition_key_seed_sensitive():
    assert chunking.partition_key("s1", 0, "a") != chunking.partition_key("s2", 0, "a")


def test_partition_key_generation_sensitive():
    assert chunking.partition_key("s", 0, "a") != chunking.partition_key("s", 1, "a")


def test_partition_key_site_sensitive():
    assert chunking.partition_key("s", 0, "a") != chunking.partition_key("s", 0, "b")


def test_partition_every_site_exactly_once():
    site_ids = _site_ids(1000)
    result = chunking.partition(site_ids, seed="s", chunk_size=64)
    assigned = [sid for sid, _, _, _ in result]
    assert sorted(assigned) == sorted(site_ids)
    assert len(assigned) == len(set(assigned)) == 1000


def test_partition_chunk_sizes_bar_a_short_tail():
    site_ids = _site_ids(1000)
    result = chunking.partition(site_ids, seed="s", chunk_size=64)
    from collections import Counter

    sizes = Counter(chunk_index for _, chunk_index, _, _ in result)
    full_chunks = [n for idx, n in sorted(sizes.items()) if idx != max(sizes)]
    assert all(n == 64 for n in full_chunks)
    assert sizes[max(sizes)] == 1000 - 64 * (len(sizes) - 1)
    assert sizes[max(sizes)] <= 64


def test_partition_member_rank_is_0_based_within_chunk():
    site_ids = _site_ids(20)
    result = chunking.partition(site_ids, seed="s", chunk_size=6)
    from collections import defaultdict

    by_chunk = defaultdict(list)
    for sid, chunk_index, member_rank, _ in result:
        by_chunk[chunk_index].append(member_rank)
    for chunk_index, ranks in by_chunk.items():
        assert sorted(ranks) == list(range(len(ranks)))


def test_partition_deterministic_across_calls():
    site_ids = _site_ids(200)
    a = chunking.partition(site_ids, seed="fixed", chunk_size=17)
    b = chunking.partition(site_ids, seed="fixed", chunk_size=17)
    assert a == b


def test_partition_deterministic_regardless_of_input_order():
    site_ids = _site_ids(200)
    import random

    shuffled = list(site_ids)
    random.Random(42).shuffle(shuffled)
    a = chunking.partition(site_ids, seed="fixed", chunk_size=17)
    b = chunking.partition(shuffled, seed="fixed", chunk_size=17)
    assert sorted(a) == sorted(b)


def test_partition_different_seed_gives_different_assignment():
    site_ids = _site_ids(200)
    a = chunking.partition(site_ids, seed="seed-a", chunk_size=17)
    b = chunking.partition(site_ids, seed="seed-b", chunk_size=17)
    assert a != b


def test_partition_generation_offsets_chunk_indices():
    site_ids = _site_ids(30)
    gen0 = chunking.partition(site_ids, seed="s", chunk_size=10, generation=0, first_chunk_index=0)
    gen1 = chunking.partition(site_ids, seed="s", chunk_size=10, generation=1, first_chunk_index=3)
    gen0_indices = {c for _, c, _, _ in gen0}
    gen1_indices = {c for _, c, _, _ in gen1}
    assert gen0_indices == {0, 1, 2}
    assert gen1_indices == {3, 4, 5}
    # a different generation is a genuinely different permutation, not a
    # relabeling of the same one
    gen0_order = [sid for sid, _, _, _ in sorted(gen0, key=lambda t: (t[1], t[2]))]
    gen1_order = [sid for sid, _, _, _ in sorted(gen1, key=lambda t: (t[1], t[2]))]
    assert gen0_order != gen1_order


def test_partition_rejects_nonpositive_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunking.partition(["a", "b"], seed="s", chunk_size=0)
    with pytest.raises(ValueError):
        chunking.partition(["a", "b"], seed="s", chunk_size=-1)


def test_partition_stratum_mix_tracks_frame():
    """A chunk of a uniformly random partition should approximate the
    frame's overall stratum composition — not exactly, but within a
    generous tolerance for a few thousand sites split across a few dozen
    chunks. This is the property the CLAUDE.md methods-chapter claim (\"any
    prefix of any chunk is an unbiased subsample of the frame\") rests on.
    """
    import random

    rng = random.Random(7)
    strata = ["gtld"] * 655 + ["eu_east"] * 127 + ["eu_west"] * 79 + ["apac"] * 48 + ["other"] * 91
    site_ids = [f"site_{i}" for i in range(len(strata))]
    stratum_by_id = dict(zip(site_ids, strata))
    rng.shuffle(site_ids)  # partition() must not care about input order

    frame_frac = {s: strata.count(s) / len(strata) for s in set(strata)}

    result = chunking.partition(site_ids, seed="strata-check", chunk_size=100)
    from collections import Counter, defaultdict

    by_chunk = defaultdict(list)
    for sid, chunk_index, _, _ in result:
        by_chunk[chunk_index].append(stratum_by_id[sid])

    # Only check FULL chunks (the tail chunk is short and noisier by nature).
    full_chunk_indices = [i for i, members in by_chunk.items() if len(members) == 100]
    assert full_chunk_indices, "expected at least one full chunk"
    for idx in full_chunk_indices:
        counts = Counter(by_chunk[idx])
        chunk_frac = {s: counts.get(s, 0) / 100 for s in frame_frac}
        for s, frac in frame_frac.items():
            assert abs(chunk_frac[s] - frac) < 0.12, (
                f"chunk {idx} stratum {s}: chunk={chunk_frac[s]:.3f} frame={frac:.3f}"
            )


def test_n_chunks():
    assert chunking.n_chunks(0, 10) == 0
    assert chunking.n_chunks(10, 10) == 1
    assert chunking.n_chunks(11, 10) == 2
    assert chunking.n_chunks(2500 * 46 + 292, 2500) == 47
