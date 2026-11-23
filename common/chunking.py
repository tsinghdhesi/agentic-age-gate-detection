"""Pure, DB-free partition math for addressable work chunks (schema v5's
`chunk_sets`/`chunk_members`/`chunk_runs` in `common/db.py`). Mirrors
`common/sampling.py`'s role: nothing here touches the network, the
filesystem, or SQLite — the partition is a deterministic function of its
inputs, which is what makes it reproducible from `(seed, generation,
site_ids)` alone and independently unit-testable (see `tests/test_chunking.py`).

Use a chunk seed DISTINCT from `sampling.frame_salt` (see
`config/global.yaml: chunking.chunk_seed`). Reusing the sampling salt would
entangle the sampling design with operational scheduling and make chunk 0
exactly the sites `01c_draw_sample.py` draws first.

Why hashing instead of `random.shuffle` + draw-without-replacement (the
originally stated spec: "duplicate the pool, repeatedly draw n at random
without replacement until empty")? `partition()` below is distributionally
identical to that spec, but stateless — no RNG stream position to get wrong
on resume — and stable across Python versions, since `random.shuffle`'s
output sequence is a CPython implementation detail, which is bad for a
thesis a reviewer might want to re-derive independently (e.g. in R).
`tests/test_chunking.py` contains the literal drain-the-pool loop as an
executable reference and asserts it produces the IDENTICAL assignment (not
just a statistically similar one) — because "draw the item with the smallest
deterministic key repeatedly without replacement" and "sort by that key once,
then slice" are the same algorithm, not merely similar ones.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def partition_key(seed: str, generation: int, site_id: str) -> str:
    """Deterministic ordering key for one site_id under (seed, generation).
    Reproducible from its inputs alone — nothing else feeds it.
    """
    digest = hashlib.sha256(f"{seed}|{generation}|{site_id}".encode("utf-8"))
    return digest.hexdigest()


def partition(
    site_ids: Sequence[str],
    *,
    seed: str,
    chunk_size: int,
    generation: int = 0,
    first_chunk_index: int = 0,
) -> list[tuple[str, int, int, int]]:
    """Partition `site_ids` into chunks of `chunk_size`, sorted by
    `partition_key`. Returns [(site_id, chunk_index, member_rank,
    generation), ...] — one tuple per site, in chunk/rank order.

    `member_rank` is 0-based WITHIN its chunk (not the global rank), since
    that's what a chunked consumer (e.g. 01b_screen_relevance.py --chunk N)
    actually iterates by. Deterministic and idempotent: calling this twice
    with identical arguments produces an identical result, which is what
    `chunk_work.py verify` re-derives and checks.

    Every input site_id appears exactly once; the last chunk may be shorter
    than `chunk_size` if len(site_ids) is not an exact multiple.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    ordered = sorted(site_ids, key=lambda sid: partition_key(seed, generation, sid))

    out: list[tuple[str, int, int, int]] = []
    for i, site_id in enumerate(ordered):
        chunk_index = first_chunk_index + i // chunk_size
        member_rank = i % chunk_size
        out.append((site_id, chunk_index, member_rank, generation))
    return out


def n_chunks(n_sites: int, chunk_size: int) -> int:
    """How many chunks `partition()` would produce for `n_sites` items."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if n_sites <= 0:
        return 0
    return -(-n_sites // chunk_size)  # ceil division without importing math
