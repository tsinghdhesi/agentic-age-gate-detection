"""Unit tests for common/sampling.py — the stratified probability sample math
for scripts/01c_draw_sample.py. Run: pytest -q

Pure functions, no DB — these tests exercise the arithmetic in isolation so a
bug in the draw is caught before it ever touches 24h+ of relevance-screening
work.
"""

import pytest

from common.sampling import STRATA, allocate, design_summary, sample_u, stratum_for_cctld


# --------------------------------------------------------------------------- #
# stratum_for_cctld
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "cctld,expected",
    [
        ("com", "gtld"),
        ("org", "gtld"),
        ("io", "gtld"),        # British Indian Ocean Territory, used generically
        ("us", "gtld"),        # deliberate fold per the plan's judgment call
        ("co.uk", "uk_ie"),
        ("uk", "uk_ie"),
        ("de", "eu_west"),
        ("com.br", "latam"),
        ("gov.pl", "eu_east"),
        ("cn", "apac"),
        ("sa", "mena_af"),
        ("ca", "na"),
        ("zz", "other"),       # not a real code — must not raise
        ("", "other"),
        (None, "other"),
    ],
)
def test_stratum_for_cctld(cctld, expected):
    assert stratum_for_cctld(cctld) == expected


def test_stratum_for_cctld_never_raises_on_garbage():
    for garbage in ["", ".", "..", "a.b.c.d.e", "🇬🇧", "com.br.extra"]:
        assert stratum_for_cctld(garbage) in STRATA


# --------------------------------------------------------------------------- #
# sample_u — deterministic, reproducible, well-distributed.
# --------------------------------------------------------------------------- #


def test_sample_u_deterministic():
    a = sample_u("vape_co_uk_abc123", "salt-v1")
    b = sample_u("vape_co_uk_abc123", "salt-v1")
    assert a == b


def test_sample_u_in_unit_interval():
    for site_id in ["a_1", "b_2", "some_longer_site_id_ffffff"]:
        u = sample_u(site_id, "salt-v1")
        assert 0.0 <= u < 1.0


def test_sample_u_varies_with_salt():
    a = sample_u("vape_co_uk_abc123", "salt-v1")
    b = sample_u("vape_co_uk_abc123", "salt-v2")
    assert a != b


def test_sample_u_varies_with_site_id():
    a = sample_u("site_a_111111", "salt-v1")
    b = sample_u("site_b_222222", "salt-v1")
    assert a != b


def test_sample_u_reasonably_spread_out():
    # Not a rigorous uniformity test — just a smoke check that 500 distinct
    # ids don't collapse onto a handful of values.
    us = {sample_u(f"site_{i}", "salt-v1") for i in range(500)}
    assert len(us) == 500
    assert min(us) < 0.1
    assert max(us) > 0.9


# --------------------------------------------------------------------------- #
# allocate
# --------------------------------------------------------------------------- #


def _frame():
    # Roughly the shape reported for the real frame (order of magnitude only).
    return {
        "gtld": 74605,
        "eu_east": 14319,
        "eu_west": 9072,
        "apac": 5359,
        "latam": 3665,
        "other": 2949,
        "uk_ie": 2918,
        "mena_af": 1872,
        "na": 1012,
    }


def test_allocate_sums_to_n_exactly():
    alloc = allocate(_frame(), 2500, power=0.5, floor=30)
    assert sum(alloc.values()) == 2500


def test_allocate_never_exceeds_stratum_population():
    frame = _frame()
    alloc = allocate(frame, 2500, power=0.5, floor=30)
    for h, n_h in alloc.items():
        assert n_h <= frame[h]


def test_allocate_respects_floor_when_room_allows():
    frame = _frame()
    alloc = allocate(frame, 2500, power=0.5, floor=150)
    for h, n_h in alloc.items():
        assert n_h >= min(150, frame[h])


def test_allocate_caps_at_population_when_n_total_exceeds_frame():
    small_frame = {"a": 10, "b": 5}
    alloc = allocate(small_frame, 1000, power=0.5, floor=0)
    assert alloc == {"a": 10, "b": 5}
    assert sum(alloc.values()) == 15


def test_allocate_empty_frame_returns_empty():
    assert allocate({}, 100) == {}


def test_allocate_zero_n_total():
    alloc = allocate(_frame(), 0, power=0.5, floor=0)
    assert sum(alloc.values()) == 0


def test_allocate_proportional_power_matches_shares_roughly():
    # power=1.0, floor=0 should track population shares closely (allocate()
    # still floors at 0 and caps at N_h, so this isn't exact, but the largest
    # stratum should get the largest allocation and roughly its share).
    frame = _frame()
    alloc = allocate(frame, 10000, power=1.0, floor=0)
    total_pop = sum(frame.values())
    for h in frame:
        expected_share = frame[h] / total_pop
        actual_share = alloc[h] / 10000
        assert abs(expected_share - actual_share) < 0.01


def test_allocate_negative_n_total_raises():
    with pytest.raises(ValueError):
        allocate(_frame(), -1)


# --------------------------------------------------------------------------- #
# design_summary
# --------------------------------------------------------------------------- #


def test_design_summary_deff_is_one_under_proportional_allocation():
    frame = {"a": 8000, "b": 2000}
    n_total = 1000
    # Exact proportional split (integers, no rounding slop) to isolate the
    # DEFF=1.0 claim from allocate()'s own rounding behavior.
    alloc = {"a": 800, "b": 200}
    summary = design_summary(frame, alloc)
    assert summary["n_total"] == n_total
    assert summary["deff"] == pytest.approx(1.0, abs=1e-9)


def test_design_summary_deff_exceeds_one_under_power_allocation():
    frame = _frame()
    alloc = allocate(frame, 2500, power=0.5, floor=30)
    summary = design_summary(frame, alloc)
    assert summary["deff"] > 1.0
    assert summary["n_eff"] < summary["n_total"]


def test_design_summary_weights_are_consistent():
    frame = {"a": 8000, "b": 2000}
    alloc = {"a": 800, "b": 200}
    summary = design_summary(frame, alloc)
    assert summary["strata"]["a"]["w_h"] == pytest.approx(10.0)
    assert summary["strata"]["b"]["w_h"] == pytest.approx(10.0)
    assert summary["strata"]["a"]["f_h"] == pytest.approx(0.1)


def test_design_summary_handles_zero_allocation_stratum():
    frame = {"a": 100, "b": 5}
    alloc = {"a": 50, "b": 0}
    summary = design_summary(frame, alloc)
    assert summary["strata"]["b"]["w_h"] is None
    assert summary["n_total"] == 50
