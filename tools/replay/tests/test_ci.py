from tools.replay import ci
from tools.replay.bugs import check


def ev(t, kind="fcw_warning_on", ttc=2.0):
    return {"t_us": int(t * 1e6), "t_s": t, "kind": kind, "ttc": ttc}


def test_events_match_within_tolerance():
    assert ci.compare_events([ev(1.0)], [ev(1.05)]) == []


def test_shifted_onset_is_a_regression():
    d = ci.compare_events([ev(9.84)], [ev(9.99)])
    assert d == ["missing fcw_warning_on at 9.84 s", "new fcw_warning_on at 9.99 s (TTC 2.00 s)"]


def test_kind_must_match():
    assert ci.compare_events([ev(1.0)], [ev(1.0, "brake_request_on")])


def test_markdown_and_junit_render():
    rep = {"suite": "s", "drop": "d", "drop_version": "0.2.0", "drop_commit": "abc", "wall_s": 1.0, "verdict": "FAIL",
           "recordings": [{"scene": "scene-1", "verdict": "FAIL", "notes": [],
                           "checks": {"runs": ("PASS", ""), "regression": ("FAIL", 'a "quoted" <diff>')}}]}
    assert "| scene-1 | **FAIL** | PASS | - | - | **FAIL** |" in ci.markdown(rep)
    x = ci.junit(rep)
    assert 'failures="1"' in x and "&quot;quoted&quot; &lt;diff&gt;" in x


def result(events, warnings=(), missed=()):
    return {"exit_code": 0, "events": events,
            "oracle": {"warnings": list(warnings), "missed": list(missed)}}


def test_ticket_expectations():
    false_w = {"verdict": "false", "t_s": 1.0, "track": 4, "stack_ttc_s": 2.0}
    assert check({"kind": "no_false_warning"}, result([ev(1.0)], [false_w]))[0] is False
    assert check({"kind": "no_false_warning"}, result([]))[0] is True
    assert check({"kind": "no_warning"}, result([ev(1.0)]))[0] is False
    assert check({"kind": "warning", "max_ttc_s": 2.5}, result([ev(1.0, ttc=2.2)]))[0] is True
    assert check({"kind": "no_warning"}, {"exit_code": 3, "error": "boom"})[0] is False


def test_version_resolves_to_the_tagged_drop(tmp_path, monkeypatch):
    import json
    import subprocess

    from tools.replay import drops, paths

    monkeypatch.setattr(paths, "DROPS", tmp_path)
    for name, commit, t in [("v0.2.0-aaaaaaaaaaaa", "aaaaaaaaaaaa", "1"), ("v0.2.0-bbbbbbbbbbbb", "bbbbbbbbbbbb", "2")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "drop.json").write_text(json.dumps({"version": "0.2.0", "commit": commit, "sha256": "x",
                                                               "built_at": t}))

    class Out:
        stdout = "aaaaaaaaaaaa1234\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Out())
    assert drops.resolve("v0.2.0").commit == "aaaaaaaaaaaa"  # the tag, not the newest build
    assert drops.resolve("bbbbbbb").commit == "bbbbbbbbbbbb"
    assert drops.resolve("latest").commit == "bbbbbbbbbbbb"
    Out.stdout = ""
    import pytest

    with pytest.raises(drops.ResolveError, match="ambiguous"):
        drops.resolve("0.2.0")
