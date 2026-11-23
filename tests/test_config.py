"""Unit tests for common/config.py's sites.industry <-> config-filename
bridge. Run: pytest -q

config_name_for_sites_industry() exists because sites.industry (a DB value,
inferred by 01_build_queue.py from the seed filename) and config/<name>.yaml
(a checkpoint-definition filename) are two different namespaces that happen
to collide for alcohol/gambling but not for nicotine/vape.
"""

from common.config import config_name_for_sites_industry


def test_nicotine_maps_to_vape_config():
    assert config_name_for_sites_industry("nicotine") == "vape"


def test_alcohol_is_identity():
    assert config_name_for_sites_industry("alcohol") == "alcohol"


def test_gambling_is_identity():
    assert config_name_for_sites_industry("gambling") == "gambling"


def test_unknown_industry_falls_through_as_identity():
    """An unmapped/unexpected industry string must not raise — it degrades
    to "try this filename verbatim" rather than crashing the caller."""
    assert config_name_for_sites_industry("something_new") == "something_new"
