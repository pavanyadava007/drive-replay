"""One replay run: resolve everything into a plan, execute it, judge the result, write a manifest.

The plan is the whole point. A user types `replay run scene-0061`; the plan fills in the recording path and
its checksum, the drop, the layered config, the time window and the pre-roll. The manifest stores the plan
next to the outputs, so `replay rerun <run dir>` repeats the run exactly and checks the digests.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tools.replay import catalog, config, drops, paths
from tools.replay.oracle import Oracle
from tools.replay.recording import load as load_recording

DEFAULT_PREROLL_S = 3.0  # tracker confirmation + FCW debounce need about 0.5 s; 3 s also settles the filters


@dataclass
class Plan:
    recording: str
    recording_path: str
    recording_sha256: str
    vehicle: str
    drop: str
    drop_version: str
    drop_commit: str
    drop_sha256: str
    binary: str
    config: dict
    config_layers: list[str]
    window_s: list[float] | None = None  # relative to recording start; None = whole drive
    preroll_s: float = 0.0
    start_us: int = 0
    end_us: int = 0
    report_from_us: int = 0  # events before this (pre-roll) are replayed but not judged
    out_dir: str = ""
    notes: list[str] = field(default_factory=list)

    def command(self) -> list[str]:
        cmd = [self.binary, "--recording", self.recording_path, "--config", str(Path(self.out_dir) / "config.json"),
               "--out", self.out_dir]
        if self.start_us:
            cmd += ["--start_us", str(self.start_us)]
        if self.end_us:
            cmd += ["--end_us", str(self.end_us)]
        return cmd


def make_plan(target: str, drop: str | None = None, window_s: list[float] | None = None,
              preroll_s: float | None = None, sets: list[str] | None = None, out_dir: str | None = None,
              tag: str = "") -> Plan:
    rec = catalog.resolve(target)
    d = drops.resolve(drop)
    cfg, layers = config.resolve(rec.vehicle, sets)
    notes = []
    start_us = end_us = report_from = 0
    pre = 0.0
    if window_s:
        view = load_recording(rec.path)
        pre = DEFAULT_PREROLL_S if preroll_s is None else preroll_s
        report_from = view.start_us + int(window_s[0] * 1e6)
        start_us = max(view.start_us, report_from - int(pre * 1e6))
        end_us = view.start_us + int(window_s[1] * 1e6)
        if start_us == view.start_us and pre > 0:
            notes.append("pre-roll clipped at the start of the recording")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{rec.name}-{d.name}{('-' + tag) if tag else ''}"
    out = Path(out_dir) if out_dir else paths.RUNS / run_id
    return Plan(rec.name, str(rec.path), rec.sha256, rec.vehicle, d.name, d.version, d.commit, d.sha256,
                str(d.binary), cfg, layers, window_s, pre, start_us, end_us, report_from, str(out), notes)


def describe(p: Plan) -> str:
    win = "whole recording" if not p.window_s else (
        f"{p.window_s[0]:.2f}-{p.window_s[1]:.2f} s, pre-roll {p.preroll_s:.1f} s")
    lines = [
        f"recording : {p.recording}  ({p.recording_path}, sha256 {p.recording_sha256[:12]}, vehicle {p.vehicle})",
        f"drop      : {p.drop}  (stack {p.drop_version}, commit {p.drop_commit}, sha256 {p.drop_sha256[:12]})",
        f"config    : {' + '.join(p.config_layers)}",
        f"window    : {win}",
        f"output    : {p.out_dir}",
        "command   : " + " ".join(p.command()),
    ]
    lines += [f"note      : {n}" for n in p.notes]
    return "\n".join(lines)


def execute(p: Plan, judge: bool = True) -> dict:
    out = Path(p.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(p.config, indent=1, sort_keys=True) + "\n")
    t0 = time.monotonic()
    proc = subprocess.run(p.command(), capture_output=True, text=True, check=False)
    wall = time.monotonic() - t0
    result: dict = {"plan": asdict(p), "exit_code": proc.returncode, "wall_s": round(wall, 3),
                    "host": {"machine": platform.machine(), "system": platform.system(),
                             "cpu": _cpu_name(), "python": platform.python_version()}}
    if proc.returncode != 0:
        result["error"] = proc.stderr.strip()[-2000:]
        (out / "manifest.json").write_text(json.dumps(result, indent=1) + "\n")
        return result
    summary = json.loads((out / "summary.json").read_text())
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines() if line]
    judged = [e for e in events if e["t_us"] >= p.report_from_us]
    result["summary"] = summary
    result["events"] = judged
    result["preroll_events"] = len(events) - len(judged)
    if judge:
        verdict = Oracle(load_recording(p.recording_path)).judge(judged)
        if p.window_s:  # misses before the window are pre-roll, not part of the question
            verdict["missed"] = [m for m in verdict["missed"] if m["t_us"] >= p.report_from_us]
            verdict["missed_threats"] = len(verdict["missed"])
            verdict["requirements_met"] = verdict["false_warnings"] == 0 and not verdict["missed"]
        result["oracle"] = verdict
    result["config_sha256"] = hashlib.sha256((out / "config.json").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def rerun(run_dir: str, out_dir: str | None = None) -> tuple[dict, dict, bool]:
    """Repeats a finished run from its manifest and reports whether both digests match."""
    old = json.loads((Path(run_dir) / "manifest.json").read_text())
    p = Plan(**old["plan"])
    rec = catalog.resolve(p.recording_path)
    if rec.sha256 != p.recording_sha256:
        raise catalog.ResolveError("recording changed since the original run")
    d = drops.resolve(p.drop)
    if d.sha256 != p.drop_sha256:
        raise catalog.ResolveError("drop binary changed since the original run")
    p.out_dir = out_dir or str(Path(run_dir).with_name(Path(run_dir).name + "-rerun"))
    new = execute(p)
    same = all(old["summary"][k] == new["summary"][k] for k in ("event_digest", "state_digest", "cycles"))
    return old, new, same


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"
