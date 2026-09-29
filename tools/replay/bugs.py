"""Bug replay: reproduce a reported issue straight from its recording, then prove the fix on a newer drop.

A ticket (bugs/BUG-xxxx.toml) names the recording, the time window, the drop it was reported against and an
executable expectation. `replay bug BUG-0001` runs the window (with pre-roll) on the reported drop and must
see the expectation FAIL (reproduced); on the fix drop the same window must PASS, and the whole recording
must still meet its requirements.
"""
from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tools.replay import ci, drops, paths, runner
from tools.replay.catalog import ResolveError


@dataclass
class Ticket:
    id: str
    title: str
    recording: str
    window_s: list[float]
    reported_on: str
    expect: dict
    preroll_s: float | None = None
    fixed_in: str | None = None
    notes: str = ""


def load(bug_id: str) -> Ticket:
    path = paths.BUGS / f"{bug_id.upper()}.toml"
    if not path.exists():
        known = ", ".join(p.stem for p in sorted(paths.BUGS.glob("BUG-*.toml")))
        raise ResolveError(f"no ticket {bug_id}. Known: {known}")
    d = tomllib.loads(path.read_text())
    return Ticket(**d)


def check(expect: dict, result: dict) -> tuple[bool, str]:
    """Evaluates a ticket expectation on a run result. Returns (holds, explanation)."""
    if result.get("exit_code") != 0:
        return False, f"stack exited with {result.get('exit_code')}: {result.get('error', '')[:200]}"
    kind = expect["kind"]
    ev = result["events"]
    o = result["oracle"]
    onsets = [e for e in ev if e["kind"] == "fcw_warning_on"]
    if kind == "no_false_warning":
        bad = [w for w in o["warnings"] if w["verdict"] == "false"]
        return not bad, (f"{len(bad)} false warning(s): " + ", ".join(
            f"t={w['t_s']} s track {w['track']} stack TTC {w['stack_ttc_s']:.2f} s" for w in bad)) if bad else \
            f"no false warnings ({len(onsets)} warning(s), all justified by ground truth)"
    if kind == "no_missed_threat":
        m = o["missed"]
        return not m, f"{len(m)} missed threat(s)" if m else "no missed threats"
    if kind == "no_warning":
        return not onsets, f"{len(onsets)} warning(s)" if onsets else "no warning"
    if kind == "warning":
        max_ttc = expect.get("max_ttc_s", 99)
        ok = [e for e in onsets if e["ttc"] <= max_ttc]
        return bool(ok), f"{len(ok)} warning(s) with TTC <= {max_ttc} s"
    raise ValueError(f"unknown expectation kind '{kind}'")


def ensure_drop(version: str) -> drops.Drop:
    """Finds the drop for a version; builds it from the git tag when it is not there yet."""
    try:
        return drops.resolve(version)
    except ResolveError:
        # a release version maps to its tag; anything else may be a commit
        for ref in (version if version.startswith("v") else f"v{version}", version):
            if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=paths.ROOT,
                              capture_output=True, check=False).returncode == 0:
                print(f"[bug] no drop for {version} yet, building {ref} ...", flush=True)
                return drops.build(ref)
        raise


def replay_bug(bug_id: str, fix_drop: str | None = None, preroll_s: float | None = None) -> dict:
    t = load(bug_id)
    pre = t.preroll_s if preroll_s is None else preroll_s
    reported = ensure_drop(t.reported_on)
    report: dict = {"ticket": t.__dict__, "steps": []}

    def step(label: str, drop: drops.Drop, window: list[float] | None, expect: dict | None) -> dict:
        plan = runner.make_plan(t.recording, drop.name, window, pre, tag=f"{t.id}-{label}")
        res = runner.execute(plan)
        if expect is None:
            holds = res.get("exit_code") == 0 and res["oracle"]["requirements_met"]
            why = (f"{res['oracle']['false_warnings']} false warning(s), {res['oracle']['missed_threats']} missed"
                   if res.get("exit_code") == 0 else "stack failed")
        else:
            holds, why = check(expect, res)
        s = {"label": label, "drop": drop.name, "window_s": window, "holds": holds, "why": why,
             "run_dir": plan.out_dir, "event_digest": res.get("summary", {}).get("event_digest")}
        report["steps"].append(s)
        return s

    r = step("reported", reported, t.window_s, t.expect)
    report["reproduced"] = not r["holds"]
    report["fidelity"] = window_fidelity(t, reported, r["run_dir"])
    fix = drops.resolve(fix_drop or t.fixed_in or "latest") if (fix_drop or t.fixed_in) else None
    if fix is None:
        latest = drops.resolve("latest")
        fix = latest if latest.name != reported.name else None
    if fix is not None:
        f = step("fix-window", fix, t.window_s, t.expect)
        w = step("fix-whole-recording", fix, None, None)
        report["fix_confirmed"] = report["reproduced"] and f["holds"] and w["holds"]
    report["verdict"] = _verdict(report)
    return report


def window_fidelity(t: Ticket, drop: drops.Drop, window_run_dir: str) -> dict:
    """Does the window replay show the same events as the whole drive replayed on the same drop?

    A window starts the stack cold. If the pre-roll is too short, the tracker and the warning debounce are
    in a different state and the bug may appear at another time, or not at all, for the wrong reason.
    """
    whole = runner.execute(runner.make_plan(t.recording, drop.name, tag=f"{t.id}-whole"))
    win = json.loads((Path(window_run_dir) / "manifest.json").read_text())
    lo, hi = win["plan"]["report_from_us"], win["plan"]["end_us"]
    ref = [e for e in whole["events"] if lo <= e["t_us"] <= hi]
    start = whole["summary"]["first_us"]
    rel = lambda es: [{**e, "t_s": round((e["t_us"] - start) * 1e-6, 3)} for e in es]
    diffs = ci.compare_events(rel(ref), rel(win["events"]), tol_s=0.001)
    return {"identical": not diffs, "diffs": diffs, "whole_run_dir": whole["plan"]["out_dir"]}


def _verdict(r: dict) -> str:
    if r["reproduced"] and not r["fidelity"]["identical"]:
        return "REPRODUCED ONLY APPROXIMATELY: the window diverges from the whole-drive replay, raise the pre-roll"
    if not r["reproduced"]:
        return "NOT REPRODUCED on the reported drop - check the window, pre-roll and drop"
    if "fix_confirmed" not in r:
        return "REPRODUCED (no newer drop to check a fix against)"
    return "REPRODUCED and FIX CONFIRMED" if r["fix_confirmed"] else "REPRODUCED, fix NOT confirmed"


def render(r: dict) -> str:
    t = r["ticket"]
    lines = [f"# {t['id']}: {t['title']}", "",
             f"- recording: `{t['recording']}`, window {t['window_s'][0]}-{t['window_s'][1]} s",
             f"- reported against: stack {t['reported_on']}",
             f"- expectation: `{t['expect']}`", "", "| step | drop | expectation holds | detail |", "|---|---|---|---|"]
    for s in r["steps"]:
        lines.append(f"| {s['label']} | {s['drop']} | {'yes' if s['holds'] else 'NO'} | {s['why']} |")
    f = r["fidelity"]
    lines += ["", "Window fidelity (window replay vs whole-drive replay on the reported drop): " +
              ("identical events" if f["identical"] else "DIVERGED: " + "; ".join(f["diffs"]))]
    lines += ["", f"**Verdict: {r['verdict']}**", ""]
    if t.get("notes"):
        lines += [t["notes"].strip(), ""]
    return "\n".join(lines)


def write_report(r: dict, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"{r['ticket']['id']}.md"
    p.write_text(render(r))
    return p
