"""Replay the suite with the opt-in single-track corridor (path_model 1) ON and OFF, on the CAN recording set.

Needs data/recordings_can (tools/vdyn/add_can_topic.py) and a drop of the current stack. Reports, per recording:
the events and digests with the option OFF and ON, the oracle verdicts, the comparison with the golden
baseline, the bug-ticket expectations (BUG-0001/2/3, window replay as `replay bug` does it) and the warning
onset in the three must-warn cases. It also checks that the extra /vehicle/can topic changes nothing while
the option is OFF (same digests as the published recordings).
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.replay import bugs, catalog, ci, paths, runner

ROOT = Path(__file__).resolve().parents[2]
CAN_CATALOG = ROOT / "recordings" / "catalog_can.json"
CAN_DATA = ROOT / "data" / "recordings_can"
VEHICLE_CONFIG = ROOT / "configs" / "vdyn" / "renault_zoe.json"
BUG_EXTRA = {"BUG-0003": ["scene-1094"]}  # the ticket names scene-0916; CI found the same failure in scene-1094


def on_sets(vehicle: str) -> list[str]:
    return ["path_model=1", f'vehicle_model="{VEHICLE_CONFIG}"', f'vehicle_id="{vehicle}"']


def can_path(scene: str) -> str:
    entry = next(e for e in json.loads(CAN_CATALOG.read_text())["recordings"] if e["scene"] == scene)
    p = CAN_DATA / entry["file"]
    if catalog.sha256_file(p) != entry["sha256"]:
        raise catalog.ResolveError(f"{p}: sha256 does not match recordings/catalog_can.json")
    return str(p)


def _rel(events: list[dict], start_us: int) -> list[dict]:
    return [{**e, "t_s": round((e["t_us"] - start_us) * 1e-6, 3)} for e in events]


def run(drop: str, out: Path) -> dict:
    suite = ci.load_suite()["suite"]["recordings"]
    vehicles = {e["scene"]: e["vehicle"] for e in catalog.entries()}
    rows = []
    for scene in suite:
        veh = vehicles[scene]
        orig = runner.execute(runner.make_plan(scene, drop, out_dir=str(out / scene / "orig")))
        off = runner.execute(runner.make_plan(can_path(scene), drop, out_dir=str(out / scene / "off")))
        on = runner.execute(runner.make_plan(can_path(scene), drop, sets=on_sets(veh), out_dir=str(out / scene / "on")))
        for r in (orig, off, on):
            if r["exit_code"] != 0:
                raise RuntimeError(f"{scene}: {r.get('error')}")
        start = orig["summary"]["first_us"]
        golden = json.loads(ci.golden_path(scene).read_text())
        row = {"scene": scene, "vehicle": veh,
               "off_identical_to_published": all(orig["summary"][k] == off["summary"][k]
                                                 for k in ("event_digest", "state_digest")),
               "golden_warnings": sum(1 for e in golden["events"] if e["kind"] == "fcw_warning_on"),
               "cycles": on["summary"]["cycles"], "single_track_cycles": on["summary"].get("single_track_cycles", 0),
               "state_digest_changed": on["summary"]["state_digest"] != off["summary"]["state_digest"]}
        for label, r in (("off", off), ("on", on)):
            ev = _rel(r["events"], start)
            o = r["oracle"]
            row[label] = {"warnings": sum(1 for e in ev if e["kind"] == "fcw_warning_on"),
                          "first_warning_s": next((e["t_s"] for e in ev if e["kind"] == "fcw_warning_on"), None),
                          "false": o["false_warnings"], "missed": o["missed_threats"],
                          "requirements_met": o["requirements_met"], "golden_diffs": ci.compare_events(golden["events"], ev),
                          "event_digest": r["summary"]["event_digest"]}
        rows.append(row)

    tickets = []
    for f in sorted(paths.BUGS.glob("BUG-*.toml")):
        t = bugs.load(f.stem)
        entry = {"bug": t.id, "recording": t.recording, "window_s": t.window_s, "expect": t.expect}
        for label in ("off", "on"):
            sets = on_sets(vehicles[t.recording]) if label == "on" else None
            res = runner.execute(runner.make_plan(can_path(t.recording), drop, t.window_s, None, sets,
                                                  out_dir=str(out / "bugs" / t.id / label)))
            holds, why = bugs.check(t.expect, res)
            entry[label] = {"expectation_holds": holds, "why": why}
        for extra in BUG_EXTRA.get(t.id, []):
            r = next(x for x in rows if x["scene"] == extra)
            entry[f"{extra}_false_warnings"] = {"off": r["off"]["false"], "on": r["on"]["false"]}
        tickets.append(entry)
    return {"drop": drop, "recordings": rows, "bugs": tickets}
