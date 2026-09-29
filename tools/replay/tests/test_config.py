import pytest

from tools.replay import config


def test_defaults_then_overrides():
    cfg, layers = config.resolve("no-such-vehicle", ["ttc_warn_s=2.5", "confirm_hits=4"])
    assert cfg["ttc_warn_s"] == 2.5 and cfg["confirm_hits"] == 4
    assert layers == ["configs/fcw_default.json", "--set ttc_warn_s=2.5", "--set confirm_hits=4"]


def test_bad_override():
    with pytest.raises(ValueError):
        config.resolve("x", ["ttc_warn_s"])
