"""Regenerates every number in docs/RESULTS.md, the bug reports in docs/bugs/ and the demo data in site/data/.

  python scripts/reproduce.py            (needs the recordings in data/recordings and bazel on PATH)

Never edit docs/RESULTS.md by hand. The vehicle-dynamics results (docs/VEHICLE_DYNAMICS.md, site/data/vdyn.json)
come from scripts/reproduce_vdyn.py, which needs the nuScenes CAN bus data (see the README).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.replay import bugs, catalog, ci, drops, paths, runner  # noqa: E402
from tools.replay.oracle import Oracle  # noqa: E402
from tools.replay.recording import load as load_recording  # noqa: E402

OUT = paths.RUNS / "reproduce"
DOCS = ROOT / "docs"
SITE = ROOT / "site" / "data"
DEMO_BRANCH = "demo/silent-regression"
RC_COMMIT = "be3c4e4"  # release candidate the CI blocked (BUG-0003)


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def drop_for(ref: str) -> drops.Drop:
    commit = sh("git", "rev-parse", "--short=12", f"{ref}^{{commit}}")
    for d in drops.list_drops():
        if d.commit == commit:
            return d
    print(f"building drop for {ref} ({commit})", flush=True)
    return drops.build(ref)


def cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def rel(events: list[dict], start_us: int) -> list[dict]:
    return [{**e, "t_s": (e["t_us"] - start_us) * 1e-6} for e in events]


def preroll_study(drop: drops.Drop, cases: list[tuple[str, float]]) -> list[dict]:
    """Window starting 0.1 s before a known warning: does the window replay match the whole-drive replay?"""
    rows = []
    for scene, t_event in cases:
        whole = runner.execute(runner.make_plan(scene, drop.name, out_dir=str(OUT / "preroll" / scene / "whole")))
        start = whole["summary"]["first_us"]
        for pre in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
            w = [t_event - 0.1, t_event + 1.5]
            res = runner.execute(runner.make_plan(scene, drop.name, w, pre,
                                                  out_dir=str(OUT / "preroll" / scene / f"pre{pre}")))
            lo, hi = res["plan"]["report_from_us"], res["plan"]["end_us"]
            ref = [e for e in whole["events"] if lo <= e["t_us"] <= hi]
            diffs = ci.compare_events(rel(ref, start), rel(res["events"], start), tol_s=0.001)
            first = next((e for e in res["events"] if e["kind"] == "fcw_warning_on"), None)
            rows.append({"scene": scene, "event_s": t_event, "preroll_s": pre, "identical": not diffs,
                         "first_warning_s": round((first["t_us"] - start) * 1e-6, 2) if first else None})
    return rows


def throughput(drop: drops.Drop, scenes: list[str], repeats: int, jobs_list: list[int]) -> list[dict]:
    rec_s = sum(e["duration_s"] for e in catalog.entries() if e["scene"] in scenes)
    rows = []
    for jobs in jobs_list:
        plans = [runner.make_plan(s, drop.name, out_dir=str(OUT / "tp" / f"j{jobs}" / f"{s}-{i}"))
                 for i in range(repeats) for s in scenes]
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(lambda p: runner.execute(p, judge=False), plans))
        wall = time.monotonic() - t0
        assert all(r["exit_code"] == 0 for r in res)
        digests = {(r["plan"]["recording"], r["summary"]["state_digest"]) for r in res}
        rows.append({"jobs": jobs, "runs": len(plans), "recorded_s": round(rec_s * repeats, 1), "wall_s": round(wall, 2),
                     "x_realtime": round(rec_s * repeats / wall, 1), "distinct_digests": len(digests),
                     "recordings": len(scenes)})
    return rows


def timelines(drops_by_label: dict[str, drops.Drop], scenes: list[str]) -> dict:
    """Per recording: ego speed, stack TTC per drop, true TTC from the oracle, events and verdicts per drop."""
    data = {}
    for scene in scenes:
        rec = catalog.resolve(scene)
        view = load_recording(rec.path)
        o = Oracle(view)
        t0 = view.start_us
        ego = [[round((e["t_us"] - t0) * 1e-6, 2), e["speed_mps"]] for e in view.ego[::2]]
        truth = []
        for e in view.ego[::4]:
            th = [x for x in o.threats(e["t_us"]) if x["ttc_s"] is not None]
            truth.append([round((e["t_us"] - t0) * 1e-6, 2), min((x["ttc_s"] for x in th), default=None)])
        per_drop = {}
        for label, d in drops_by_label.items():
            res = runner.execute(runner.make_plan(scene, d.name, out_dir=str(OUT / "timeline" / label / scene)))
            trace = [json.loads(line) for line in (Path(res["plan"]["out_dir"]) / "trace.jsonl").read_text().splitlines()]
            ttc = [[round((c["t_us"] - t0) * 1e-6, 2), c["ttc"] if c["ttc"] >= 0 else None] for c in trace[::2]]
            per_drop[label] = {
                "ttc": ttc,
                "events": [{"t": round((e["t_us"] - t0) * 1e-6, 2), "kind": e["kind"], "ttc": e["ttc"],
                            "range": e["range"]} for e in res["events"]],
                "verdicts": [{"t": w["t_s"], "verdict": w["verdict"], "gt": w["gt"][:1]} for w in res["oracle"]["warnings"]],
                "missed": len(res["oracle"]["missed"]),
            }
        entry = next(e for e in catalog.entries() if e["scene"] == scene)
        data[scene] = {"description": entry["description"], "duration_s": entry["duration_s"],
                       "injected": entry.get("injected"), "ego_speed": ego, "true_ttc": truth, "drops": per_drop}
    return data


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("-" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    head = sh("git", "rev-parse", "--short=12", "HEAD")
    d01, drc, d02, ddemo = drop_for("v0.1.0"), drop_for(RC_COMMIT), drop_for("v0.2.0"), drop_for(DEMO_BRANCH)
    suite = ci.load_suite()
    scenes = suite["suite"]["recordings"]

    ci_rows, ci_reports = [], {}
    for label, d in [("v0.1.0", d01), (f"candidate {RC_COMMIT}", drc), ("v0.2.0", d02), (DEMO_BRANCH, ddemo)]:
        rep = ci.run_suite(d.name, str(OUT / f"ci-{d.name}"))
        ci_reports[label] = rep
        fails = [f"{r['scene']}: {k} ({v[1]})" for r in rep["recordings"] for k, v in r["checks"].items() if v[0] == "FAIL"]
        ci_rows.append({"drop": label, "commit": d.commit, "verdict": rep["verdict"],
                        "failed checks": len(fails), "suite wall s": rep["wall_s"],
                        "false warnings": sum(r["oracle"]["false_warnings"] for r in rep["recordings"]),
                        "missed threats": sum(r["oracle"]["missed_threats"] for r in rep["recordings"]),
                        "_fails": fails})

    bug_rows = []
    (DOCS / "bugs").mkdir(parents=True, exist_ok=True)
    for f in sorted(paths.BUGS.glob("BUG-*.toml")):
        rep = bugs.replay_bug(f.stem)
        bugs.write_report(rep, DOCS / "bugs")
        t = rep["ticket"]
        bug_rows.append({"bug": t["id"], "recording": t["recording"], "reported on": t["reported_on"],
                         "fixed in": t.get("fixed_in"), "reproduced": rep["reproduced"],
                         "window fidelity": "identical" if rep["fidelity"]["identical"] else "diverged",
                         "fix confirmed": rep.get("fix_confirmed"), "verdict": rep["verdict"]})

    pre_rows = preroll_study(d01, [("scene-0061", 6.81), ("scene-0103", 1.98)]) + \
        preroll_study(d02, [("scene-0796-stopped-car", 9.84), ("scene-0655-braking-lead", 9.96)])

    # determinism: rerun from manifest; and the same source built twice (v0.2.0 tag vs demo parent) is not needed
    first = runner.execute(runner.make_plan("scene-1077-stopped-car", d02.name, out_dir=str(OUT / "det" / "a")))
    _, _, same = runner.rerun(str(OUT / "det" / "a"), str(OUT / "det" / "b"))

    tp = throughput(d02, scenes, repeats=10, jobs_list=[1, 4, 16, 32])

    SITE.mkdir(parents=True, exist_ok=True)
    tl = timelines({"v0.1.0": d01, RC_COMMIT: drc, "v0.2.0": d02, "silent-regression": ddemo}, scenes)
    (SITE / "timelines.json").write_text(json.dumps(tl, separators=(",", ":")))
    (SITE / "results.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d"), "commit": head, "cpu": cpu(),
        "ci": [{k: v for k, v in r.items()} for r in ci_rows], "bugs": bug_rows, "preroll": pre_rows,
        "throughput": tp, "rerun_identical": same,
        "ci_tables": {k: ci.markdown(v) for k, v in ci_reports.items()},
        "ci_matrix": {k: {r["scene"]: {"verdict": r["verdict"], "checks": {c: list(v) for c, v in r["checks"].items()}}
                          for r in v["recordings"]} for k, v in ci_reports.items()},
        "bug_reports": {p.stem: p.read_text() for p in sorted((DOCS / "bugs").glob("BUG-*.md"))},
    }, indent=1))

    total_s = sum(e["duration_s"] for e in catalog.entries() if e["scene"] in scenes)
    lines = [
        "# Results",
        "",
        f"Generated by `scripts/reproduce.py` on {time.strftime('%Y-%m-%d')} at commit `{head}`. Do not edit by hand.",
        f"Host: {cpu()}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}. "
        "All timings are x86 replay timings and say nothing about an in-car target.",
        "",
        f"Suite: {len(scenes)} recordings, {total_s:.1f} s of driving ({len(scenes) - 3} real nuScenes v1.0-mini drives "
        "and 3 must-warn cases with an injected target). Every drop is checked against the current suite and golden baselines.",
        "",
        "## CI verdict per build drop",
        "",
        md_table(ci_rows, ["drop", "commit", "verdict", "failed checks", "false warnings", "missed threats", "suite wall s"]),
        "",
    ]
    for r in ci_rows:
        if r["_fails"]:
            lines.append(f"- **{r['drop']}**: " + "; ".join(r["_fails"]))
    lines += ["", "## Bug replays", "",
              md_table(bug_rows, ["bug", "recording", "reported on", "fixed in", "reproduced", "window fidelity",
                                  "fix confirmed"]),
              "", "Full reports: [docs/bugs/](bugs/).", "",
              "## Pre-roll study", "",
              "Window starts 0.1 s before a known warning. `identical` = the window replay gives exactly the events of the "
              "whole-drive replay on the same drop.", "",
              md_table(pre_rows, ["scene", "event_s", "preroll_s", "identical", "first_warning_s"]), "",
              "## Determinism", "",
              "- `replay rerun` of a finished run from its manifest: "
              f"{'bit-identical event and state digests' if same else 'DIFFERENT'} "
              f"(state digest {first['summary']['state_digest']}).",
              "- Every CI run replays each recording twice and compares both digests (see the `deterministic` column).",
              "", "## Replay throughput (drop v0.2.0, whole suite x 10)", "",
              md_table(tp, ["jobs", "runs", "recorded_s", "wall_s", "x_realtime", "distinct_digests", "recordings"]), "",
              "`distinct_digests` equal to the number of recordings means every repeat of a recording produced the same "
              "state digest, also under parallel load.", ""]
    (DOCS / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
