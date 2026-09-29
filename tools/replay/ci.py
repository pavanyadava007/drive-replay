"""CI replay suite: every recording through a build drop, with a clear PASS / FAIL verdict.

Checks per recording:
  runs          the stack exits 0 on the whole recording
  deterministic a second run gives the same event and state digests
  requirements  the ground-truth oracle finds no false warning and no missed threat (a ticketed known issue
                is reported as WAIVED, and a waiver whose bug is fixed is flagged so it gets removed)
  regression    the warning events match the accepted golden baseline (onset times within tolerance);
                a behaviour change fails until someone reviews it and runs `replay golden accept`
Cycle timing is reported but never fails the suite: x86 replay timing says nothing about the target.
"""
from __future__ import annotations

import json
import os
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

from tools.replay import drops, paths, runner

TOL_S = 0.10


def load_suite(path: Path | None = None) -> dict:
    return tomllib.loads((path or paths.SUITE).read_text())


def golden_path(scene: str) -> Path:
    return paths.GOLDEN / f"{scene}.json"


def compare_events(golden: list[dict], got: list[dict], tol_s: float = TOL_S) -> list[str]:
    """Matches events by kind and time (track ids may be renumbered by harmless changes)."""
    diffs = []
    left = list(got)
    for g in golden:
        m = next((e for e in left if e["kind"] == g["kind"] and abs(e["t_us"] - g["t_us"]) <= tol_s * 1e6), None)
        if m is None:
            diffs.append(f"missing {g['kind']} at {g['t_s']:.2f} s")
        else:
            left.remove(m)
    diffs += [f"new {e['kind']} at {e['t_s']:.2f} s (TTC {e['ttc']:.2f} s)" for e in left]
    return diffs


def _rel(events: list[dict], start_us: int) -> list[dict]:
    return [{**e, "t_s": round((e["t_us"] - start_us) * 1e-6, 3)} for e in events]


def check_recording(scene: str, drop: drops.Drop, out_root: Path, suite: dict) -> dict:
    waivers = {w["scene"]: w for w in suite.get("known_issue", [])}
    r: dict = {"scene": scene, "checks": {}, "notes": []}
    t0 = time.monotonic()
    a = runner.execute(runner.make_plan(scene, drop.name, out_dir=str(out_root / scene / "run1")))
    if a["exit_code"] != 0:
        r["checks"]["runs"] = ("FAIL", a.get("error", "")[-300:])
        r["verdict"] = "FAIL"
        return r
    r["checks"]["runs"] = ("PASS", f"{a['summary']['cycles']} cycles")
    start_us = json.loads(json.dumps(a["summary"]))["first_us"]
    if suite["suite"].get("determinism_runs", 2) > 1:
        b = runner.execute(runner.make_plan(scene, drop.name, out_dir=str(out_root / scene / "run2")), judge=False)
        same = b["exit_code"] == 0 and all(a["summary"][k] == b["summary"][k] for k in ("event_digest", "state_digest"))
        r["checks"]["deterministic"] = ("PASS" if same else "FAIL",
                                        f"digest {a['summary']['state_digest']}" if same else "digests differ")
    o = a["oracle"]
    req_ok = o["requirements_met"]
    detail = f"{len(o['warnings'])} warning(s), {o['false_warnings']} false, {o['missed_threats']} missed"
    if not req_ok and scene in waivers:
        r["checks"]["requirements"] = ("WAIVED", f"{detail}; known issue {waivers[scene]['bug']}")
    elif req_ok and scene in waivers:
        r["checks"]["requirements"] = ("PASS", detail)
        r["notes"].append(f"known issue {waivers[scene]['bug']} no longer reproduces: remove the waiver")
    else:
        r["checks"]["requirements"] = ("PASS" if req_ok else "FAIL", detail)
    gp = golden_path(scene)
    got = _rel(a["events"], start_us)
    if gp.exists():
        g = json.loads(gp.read_text())
        diffs = compare_events(g["events"], got)
        if diffs:
            r["checks"]["regression"] = ("FAIL", f"vs golden from {g['stack_version']}: " + "; ".join(diffs[:6]))
        else:
            same_state = g.get("state_digest") == a["summary"]["state_digest"]
            r["checks"]["regression"] = ("PASS", "events match golden" +
                                         ("" if same_state else " (internal state changed, events did not)"))
    else:
        r["checks"]["regression"] = ("FAIL", "no golden baseline; review and run `replay golden accept`")
    r["events"] = got
    r["summary"] = a["summary"]
    r["oracle"] = o
    r["wall_s"] = round(time.monotonic() - t0, 2)
    r["verdict"] = "FAIL" if any(v == "FAIL" for v, _ in r["checks"].values()) else "PASS"
    return r


def run_suite(drop_spec: str | None, out_dir: str | None = None, jobs: int | None = None,
              scenes: list[str] | None = None) -> dict:
    suite = load_suite()
    drop = drops.resolve(drop_spec)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(out_dir) if out_dir else paths.RUNS / f"ci-{stamp}-{drop.name}"
    todo = scenes or suite["suite"]["recordings"]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs or min(len(todo), os.cpu_count() or 4)) as ex:
        results = list(ex.map(lambda s: check_recording(s, drop, out_root, suite), todo))
    report = {"suite": suite["suite"]["name"], "drop": drop.name, "drop_version": drop.version,
              "drop_commit": drop.commit, "recordings": results, "wall_s": round(time.monotonic() - t0, 2),
              "verdict": "FAIL" if any(r["verdict"] == "FAIL" for r in results) else "PASS"}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ci_report.json").write_text(json.dumps(report, indent=1) + "\n")
    (out_root / "junit.xml").write_text(junit(report))
    (out_root / "summary.md").write_text(markdown(report))
    report["out_dir"] = str(out_root)
    return report


def markdown(rep: dict) -> str:
    icon = {"PASS": "PASS", "FAIL": "**FAIL**", "WAIVED": "waived"}
    lines = [f"## Replay CI: {rep['verdict']}", "",
             (f"Drop `{rep['drop']}` (stack {rep['drop_version']}, commit {rep['drop_commit']}), "
              f"{len(rep['recordings'])} recordings, {rep['wall_s']} s"), "",
             "| recording | verdict | runs | deterministic | requirements | regression |", "|---|---|---|---|---|---|"]
    for r in rep["recordings"]:
        cells = [icon.get(r["checks"][k][0], r["checks"][k][0]) if k in r["checks"] else "-"
                 for k in ("runs", "deterministic", "requirements", "regression")]
        lines.append(f"| {r['scene']} | {icon[r['verdict']]} | " + " | ".join(cells) + " |")
    lines.append("")
    for r in rep["recordings"]:
        for k, (v, d) in r["checks"].items():
            if v != "PASS":
                lines.append(f"- {r['scene']} / {k}: {v} - {d}")
        for n in r.get("notes", []):
            lines.append(f"- {r['scene']}: {n}")
    return "\n".join(lines) + "\n"


def junit(rep: dict) -> str:
    cases, fails = [], 0
    for r in rep["recordings"]:
        for k, (v, d) in r["checks"].items():
            name = f'{r["scene"]}.{k}'
            msg = escape(d, {'"': "&quot;"})
            if v == "FAIL":
                fails += 1
                cases.append(f'  <testcase classname="replay" name="{name}"><failure message="{msg}"/></testcase>')
            elif v == "WAIVED":
                cases.append(f'  <testcase classname="replay" name="{name}"><skipped message="{msg}"/></testcase>')
            else:
                cases.append(f'  <testcase classname="replay" name="{name}"/>')
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="{rep["suite"]}" tests="{len(cases)}" '
            f'failures="{fails}">\n' + "\n".join(cases) + "\n</testsuite>\n")


def accept_golden(ci_dir: str, scenes: list[str] | None, reason: str) -> list[str]:
    rep = json.loads((Path(ci_dir) / "ci_report.json").read_text())
    done = []
    for r in rep["recordings"]:
        if scenes and r["scene"] not in scenes:
            continue
        if r["checks"]["runs"][0] != "PASS" or r["checks"].get("deterministic", ("PASS",))[0] != "PASS":
            raise RuntimeError(f"{r['scene']}: refusing to accept a golden from a crashed or non-deterministic run")
        paths.GOLDEN.mkdir(parents=True, exist_ok=True)
        golden_path(r["scene"]).write_text(json.dumps({
            "scene": r["scene"], "stack_version": rep["drop_version"], "commit": rep["drop_commit"],
            "reason": reason, "event_digest": r["summary"]["event_digest"],
            "state_digest": r["summary"]["state_digest"],
            "events": [{k: e[k] for k in ("t_us", "t_s", "kind", "track", "ttc", "range", "closing")} for e in r["events"]],
        }, indent=1) + "\n")
        done.append(r["scene"])
    return done
