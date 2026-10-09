"""Second scene-table set for motion prediction: causal (sample-and-hold) inputs plus acceleration signals.

    python tools/mpred/extract.py --can-zip ~/workspace/drive-replay-data/can_dl/can_bus.zip \
        --out ~/workspace/drive-replay-data/can_scenes_v2

The v0.3.0 tables (tools/vdyn/can_extract.py, can_scenes/) interpolate the 100 Hz CAN and IMU signals linearly
onto the 50 Hz pose clock. Linear interpolation reads the next sample, up to 10 ms in the future, which is fine
for model identification but is a small leak for a prediction benchmark. This extractor keeps exactly the same
scenes (same blacklist, same overlap and gap checks, same pose rows, checked against the v0.3.0 tables) and
writes a NEW directory; the old tables
are not touched. Columns, one row per `pose` message:

  t_s          time since the first kept pose sample (s, 0.1 ms resolution; the pose clock is about 49 Hz)
  x, y, yaw    pose position (m) and heading (rad, unwrapped): the reference, as in v0.3.0
  v_pose       pose.vel[0] (m/s): reference speed, never a model input
  drive        1 if the latest vehicle_monitor gear_position is 7 (drive)
  steer_sw     steeranglefeedback.value (rad), latest message at or before t (sample-and-hold, no look-ahead)
  r_imu        ms_imu.rotation_rate[2] (rad/s), latest at or before t
  rpm_rear     mean of zoe_veh_info RL/RR wheel speed (rpm), latest at or before t
  ax_imu       ms_imu.linear_accel[0] (m/s^2, includes gravity on slopes), latest at or before t
  ay_imu       ms_imu.linear_accel[1] (m/s^2), latest at or before t
  ax_can       zoe_veh_info.longitudinal_accel (m/s^2), latest at or before t

The CAN timestamps are reception times (devkit note), so "at or before t" is "received by t". Nothing here
goes into git; the CAN data is CC BY-NC-SA 4.0 (nuScenes terms of use).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.vdyn.can_extract import CAN_BLACKLIST, MAX_GAP_S, MIN_SPAN_S, REQUIRED, max_gap, yaw_of  # noqa: E402

COLUMNS = ["t_s", "x", "y", "yaw", "v_pose", "drive", "steer_sw", "r_imu", "rpm_rear", "ax_imu", "ay_imu", "ax_can"]
FMT = ["%.4f", "%.4f", "%.4f", "%.6f", "%.4f", "%d", "%.6f", "%.6f", "%.3f", "%.5f", "%.5f", "%.4f"]


def hold(t_src: np.ndarray, values: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Latest source value with time <= query time; the first value before the first source sample."""
    idx = np.searchsorted(t_src, t_query, side="right") - 1
    return values[np.clip(idx, 0, None)]


def scene_table(z: zipfile.ZipFile, sid: int) -> tuple[np.ndarray | None, dict]:
    def load(msg: str) -> list[dict]:
        try:
            return json.loads(z.read(f"can_bus/scene-{sid:04d}_{msg}.json"))
        except KeyError:
            return []

    data = {m: load(m) for m in REQUIRED + ("vehicle_monitor",)}
    for m in REQUIRED:
        if len(data[m]) < 10:
            return None, {"reason": f"{m}: {len(data[m])} messages"}
    t = {m: np.array([d["utime"] for d in data[m]], dtype=np.int64) for m in data}
    t0 = int(t["pose"][0])
    ts = {m: (v - t0) * 1e-6 for m, v in t.items()}
    lo = max(ts[m][0] for m in REQUIRED)
    hi = min(ts[m][-1] for m in REQUIRED)
    if hi - lo < MIN_SPAN_S:
        return None, {"reason": f"overlap {hi - lo:.1f} s"}
    bad = {m: g for m in REQUIRED if (g := max_gap(ts[m], lo, hi)) > MAX_GAP_S}
    if bad:
        return None, {"reason": "gap " + ", ".join(f"{m} {g:.2f} s" for m, g in bad.items())}
    pose = data["pose"]
    tp = ts["pose"]
    keep = (tp >= lo) & (tp <= hi)
    tp = tp[keep]
    pos = np.array([p["pos"][:2] for p in pose])[keep]
    yaw = np.unwrap(np.array([yaw_of(p["orientation"]) for p in pose]))[keep]
    v_pose = np.array([p["vel"][0] for p in pose])[keep]
    jump = np.hypot(*np.diff(pos, axis=0).T).max() if len(pos) > 1 else 0.0
    if jump > 2.0:
        return None, {"reason": f"pose jump {jump:.1f} m"}
    imu, zoe, sw = data["ms_imu"], data["zoe_veh_info"], data["steeranglefeedback"]
    steer = hold(ts["steeranglefeedback"], np.array([d["value"] for d in sw]), tp)
    r_imu = hold(ts["ms_imu"], np.array([d["rotation_rate"][2] for d in imu]), tp)
    ax_imu = hold(ts["ms_imu"], np.array([d["linear_accel"][0] for d in imu]), tp)
    ay_imu = hold(ts["ms_imu"], np.array([d["linear_accel"][1] for d in imu]), tp)
    rpm = hold(ts["zoe_veh_info"], np.array([0.5 * (d["RL_wheel_speed"] + d["RR_wheel_speed"]) for d in zoe]), tp)
    ax_can = hold(ts["zoe_veh_info"], np.array([d["longitudinal_accel"] for d in zoe]), tp)
    vm = data["vehicle_monitor"]
    if vm:
        idx = np.searchsorted(ts["vehicle_monitor"], tp, side="right") - 1
        gear = np.array([d["gear_position"] for d in vm])
        drive = np.where(idx < 0, gear[0] == 7, gear[np.clip(idx, 0, None)] == 7).astype(int)
    else:
        drive = np.ones_like(tp, dtype=int)
    tab = np.column_stack([tp - tp[0], pos[:, 0], pos[:, 1], yaw, v_pose, drive, steer, r_imu, rpm, ax_imu, ay_imu, ax_can])
    return tab, {"samples": len(tp), "duration_s": round(float(tp[-1] - tp[0]), 3)}


def write_csv(path: Path, tab: np.ndarray) -> str:
    lines = [",".join(COLUMNS)]
    lines += [",".join(f % v for f, v in zip(FMT, row, strict=True)) for row in tab]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--can-zip", type=Path, required=True)
    ap.add_argument("--v1", type=Path, default=Path.home() / "workspace" / "drive-replay-data" / "can_scenes",
                    help="v0.3.0 tables: the kept scenes and their car/log metadata must match")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    v1 = json.loads((args.v1 / "manifest.json").read_text())["kept"]
    z = zipfile.ZipFile(args.can_zip)
    ids = sorted({int(m.group(1)) for n in z.namelist() if (m := re.match(r"can_bus/scene-(\d{4})_", n))})
    args.out.mkdir(parents=True, exist_ok=True)
    kept = {}
    for sid in ids:
        name = f"scene-{sid:04d}"
        if sid in CAN_BLACKLIST or name not in v1:
            continue
        tab, info = scene_table(z, sid)
        if tab is None:
            raise SystemExit(f"{name}: kept in v0.3.0 but rejected now ({info['reason']})")
        if len(tab) != v1[name]["samples"]:
            raise SystemExit(f"{name}: {len(tab)} pose rows, v0.3.0 has {v1[name]['samples']}")
        info.update({k: v1[name][k] for k in ("log", "vehicle", "location")})
        info["sha256"] = write_csv(args.out / f"{name}.csv", tab)
        # the reference rows must be exactly those of v0.3.0 (same text after formatting): x, y, yaw, v_pose, drive
        old = np.loadtxt(args.v1 / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
        new = np.loadtxt(args.out / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
        if not (np.array_equal(old[:, 1:5], new[:, 1:5]) and np.array_equal(old[:, 8], new[:, 5])):
            raise SystemExit(f"{name}: pose rows differ from the v0.3.0 table")
        kept[name] = info
        if len(kept) % 100 == 0:
            print(f"{len(kept)} scenes", flush=True)
    missing = sorted(set(v1) - set(kept))
    if missing:
        raise SystemExit(f"scenes of v0.3.0 not produced: {missing[:5]}")
    (args.out / "manifest.json").write_text(json.dumps({
        "source": "nuScenes CAN bus expansion (can_bus.zip), CC BY-NC-SA 4.0", "columns": COLUMNS,
        "inputs": "sample-and-hold (latest message at or before the pose time)", "kept": kept}, indent=1) + "\n")
    print(f"wrote {len(kept)} scenes to {args.out}")


if __name__ == "__main__":
    main()
