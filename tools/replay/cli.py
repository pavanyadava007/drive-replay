"""replay: one entry point for replaying recorded drives through the stack.

  replay run scene-0061                     whole drive on the newest drop
  replay run 61 --drop v0.1.0 --window 5 9  a window, with automatic pre-roll
  replay run BUG-0002                       the window of a bug ticket on the newest drop
  replay plan scene-0061                    show what would run, run nothing
  replay rerun runs/<id>                    repeat a run exactly and compare digests
  replay bug BUG-0002 [--fix v0.2.0]        reproduce on the reported drop, prove the fix
  replay ci [--drop v0.2.0]                 CI suite with a PASS/FAIL verdict (exit code 1 on FAIL)
  replay golden accept <ci dir> --reason .. accept reviewed behaviour as the new baseline
  replay drop build <git ref> | list | import <replay_main>
  replay recordings                         catalogue and local status
"""
from __future__ import annotations

import argparse
import sys

from tools.replay import bugs, catalog, ci, drops, paths, runner


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", help="scene name or number, .mcap path, or BUG-xxxx")
    p.add_argument("--drop", help="drop version, commit prefix or directory (default: newest)")
    p.add_argument("--window", nargs=2, type=float, metavar=("FROM_S", "TO_S"), help="seconds from recording start")
    p.add_argument("--preroll", type=float, help=f"seconds replayed before the window (default {runner.DEFAULT_PREROLL_S})")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (JSON value)")
    p.add_argument("--out", help="output directory (default runs/<id>)")


def _plan_from(a: argparse.Namespace) -> runner.Plan:
    target, window, pre = a.target, a.window, a.preroll
    if a.target.upper().startswith("BUG-"):
        t = bugs.load(a.target)
        target, window = t.recording, window or t.window_s
        pre = pre if pre is not None else t.preroll_s
    return runner.make_plan(target, a.drop, window, pre, a.set, a.out)


def _print_run(res: dict) -> None:
    if res["exit_code"] != 0:
        print(f"FAILED: stack exited with {res['exit_code']}\n{res.get('error', '')}")
        return
    s, o = res["summary"], res["oracle"]
    print(f"\n{s['cycles']} cycles in {s['timing']['wall_s']:.2f} s "
          f"(cycle p50 {s['timing']['cycle_ms_p50']:.2f} ms, p99 {s['timing']['cycle_ms_p99']:.2f} ms), "
          f"digests events {s['event_digest']} state {s['state_digest']}")
    for e in res["events"]:
        print(f"  {(e['t_us'] - s['first_us']) * 1e-6:7.2f} s  {e['kind']:<18} track {e['track']:<5} "
              f"TTC {e['ttc']:.2f} s  range {e['range']:.1f} m  closing {e['closing']:.1f} m/s")
    for w in o["warnings"]:
        gt = w["gt"][0] if w["gt"] else None
        why = f"{gt['category']} {gt['gap_m']} m ahead, true TTC {gt['ttc_s']} s" if gt else "nothing in the driven path"
        print(f"  oracle: warning at {w['t_s']} s is {w['verdict'].upper()} ({why})")
    for m in o["missed"]:
        print(f"  oracle: MISSED {m['category']} at {m['t_s']} s, true TTC {m['ttc_s']} s")
    print(f"requirements met: {'yes' if o['requirements_met'] else 'NO'} "
          f"({o['false_warnings']} false, {o['missed_threats']} missed, {o['not_observable']} not radar-observable)")
    print(f"run dir: {res['plan']['out_dir']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="replay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _run_args(sub.add_parser("run", help="replay a recording or bug window"))
    _run_args(sub.add_parser("plan", help="resolve and print the plan only"))
    p = sub.add_parser("rerun", help="repeat a run from its manifest")
    p.add_argument("run_dir")
    p = sub.add_parser("bug", help="reproduce a bug ticket and prove its fix")
    p.add_argument("bug_id", help="BUG-xxxx, or 'list'")
    p.add_argument("--fix", help="drop to prove the fix on (default: fixed_in or newest)")
    p.add_argument("--preroll", type=float)
    p.add_argument("--report-dir", default=None)
    p = sub.add_parser("ci", help="run the replay suite on a drop")
    p.add_argument("--drop")
    p.add_argument("--jobs", type=int)
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--out")
    p.add_argument("--junit", help="also copy junit.xml here")
    p.add_argument("--summary", help="also append the markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    p = sub.add_parser("golden", help="manage golden baselines")
    p.add_argument("action", choices=["accept"])
    p.add_argument("ci_dir")
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("drop", help="build, list or import drops")
    p.add_argument("action", choices=["build", "list", "import"])
    p.add_argument("arg", nargs="?")
    sub.add_parser("recordings", help="list the catalogue")
    a = ap.parse_args(argv)

    try:
        if a.cmd in ("run", "plan"):
            plan = _plan_from(a)
            print(runner.describe(plan))
            if a.cmd == "plan":
                return 0
            res = runner.execute(plan)
            _print_run(res)
            return 0 if res["exit_code"] == 0 else 3
        if a.cmd == "rerun":
            old, new, same = runner.rerun(a.run_dir)
            print(f"original {old['summary']['event_digest']}/{old['summary']['state_digest']}  "
                  f"rerun {new['summary']['event_digest']}/{new['summary']['state_digest']}")
            print("REPRODUCED bit-identical" if same else "DIFFERENT - the run is not reproducible")
            return 0 if same else 1
        if a.cmd == "bug":
            if a.bug_id == "list":
                for f in sorted(paths.BUGS.glob("BUG-*.toml")):
                    t = bugs.load(f.stem)
                    print(f"{t.id}  {t.recording:<11} reported on {t.reported_on:<6} fixed in {t.fixed_in or '-':<6} {t.title}")
                return 0
            rep = bugs.replay_bug(a.bug_id, a.fix, a.preroll)
            print(bugs.render(rep))
            if a.report_dir:
                print(f"report: {bugs.write_report(rep, __import__('pathlib').Path(a.report_dir))}")
            return 0 if rep["reproduced"] and rep.get("fix_confirmed", True) else 1
        if a.cmd == "ci":
            rep = ci.run_suite(a.drop, a.out, a.jobs, a.scenes)
            md = ci.markdown(rep)
            print(md)
            print(f"reports: {rep['out_dir']}/{{ci_report.json,junit.xml,summary.md}}")
            if a.junit:
                __import__("shutil").copy(f"{rep['out_dir']}/junit.xml", a.junit)
            if a.summary:
                with open(a.summary, "a") as f:
                    f.write(md)
            return 0 if rep["verdict"] == "PASS" else 1
        if a.cmd == "golden":
            done = ci.accept_golden(a.ci_dir, a.scenes, a.reason)
            print(f"accepted golden baselines for {len(done)} recording(s): {', '.join(done)}")
            return 0
        if a.cmd == "drop":
            if a.action == "list":
                for d in drops.list_drops():
                    print(f"{d.name:<24} stack {d.version:<7} commit {d.commit:<14} sha256 {d.sha256[:12]}")
                return 0
            if not a.arg:
                ap.error(f"drop {a.action} needs an argument")
            d = drops.build(a.arg) if a.action == "build" else drops.import_binary(__import__("pathlib").Path(a.arg))
            print(f"drop {d.name}: stack {d.version}, commit {d.commit}, sha256 {d.sha256[:12]}")
            return 0
        if a.cmd == "recordings":
            for e in catalog.entries():
                local = (paths.DATA / e["file"]).exists()
                print(f"{e['scene']}  {e['duration_s']:5.1f} s  {e['bytes'] / 1e6:5.2f} MB  {e['vehicle']}  "
                      f"{'local' if local else 'MISSING'}  {e['description']}")
            return 0
    except (catalog.ResolveError, ValueError) as e:
        print(f"replay: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
