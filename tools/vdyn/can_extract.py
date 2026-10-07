"""Turn the nuScenes CAN bus expansion into one resampled table per scene, plus the fixed train/val/test split.

    python tools/vdyn/can_extract.py --can-zip ~/workspace/drive-replay-data/can_dl/can_bus.zip \
        --meta ~/workspace/drive-replay-data/meta --out ~/workspace/drive-replay-data/can_scenes

The zip is read in place (never extracted). For every scene with valid CAN data the script writes
<out>/scene-XXXX.csv on the 50 Hz time base of the `pose` message:

  t_s          time since the first kept sample (s)
  x, y, yaw    pose position (global map frame, m) and heading from the pose quaternion (rad, unwrapped)
  v_pose       pose.vel[0], longitudinal speed in the ego frame (m/s); used as the reference, never as model input
  steer_sw     steeranglefeedback.value, steering-wheel angle (rad, left positive), linearly interpolated
  r_imu        ms_imu.rotation_rate[2], yaw rate (rad/s, left positive), linearly interpolated
  rpm_rear     mean of zoe_veh_info RL/RR wheel speeds (rpm), linearly interpolated
  drive        1 if the latest vehicle_monitor gear_position is 7 (drive), else 0 (no monitor data: 1)

Scenes are dropped (and listed in the manifest with the reason) when they are on the devkit's CAN blacklist,
when a required message is missing, or when the messages do not overlap for at least 10 s without a gap
longer than 0.25 s. Nothing here is written into the git tree except the split file; the CAN data is
CC BY-NC-SA 4.0 (nuScenes terms of use) and stays outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path

import numpy as np

# nuscenes-devkit python-sdk/nuscenes/can_bus/can_bus_api.py: scenes without CAN bus data
CAN_BLACKLIST = {161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314}
REQUIRED = ("pose", "steeranglefeedback", "ms_imu", "zoe_veh_info")
MIN_SPAN_S = 10.0
MAX_GAP_S = 0.25
SPLIT_SEED = 20261007
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train, val, test (by scene count, logs kept whole)
COLUMNS = ["t_s", "x", "y", "yaw", "v_pose", "steer_sw", "r_imu", "rpm_rear", "drive"]


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def max_gap(t: np.ndarray, lo: float, hi: float) -> float:
    s = t[(t >= lo) & (t <= hi)]
    if len(s) < 2:
        return math.inf
    return float(max(np.diff(s).max(), s[0] - lo, hi - s[-1]))


def scene_table(z: zipfile.ZipFile, sid: int) -> tuple[np.ndarray | None, dict]:
    def load(msg: str) -> list[dict]:
        name = f"can_bus/scene-{sid:04d}_{msg}.json"
        try:
            return json.loads(z.read(name))
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
    gaps = {m: max_gap(ts[m], lo, hi) for m in REQUIRED}
    bad = {m: g for m, g in gaps.items() if g > MAX_GAP_S}
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
    steer = np.interp(tp, ts["steeranglefeedback"], [d["value"] for d in data["steeranglefeedback"]])
    r_imu = np.interp(tp, ts["ms_imu"], [d["rotation_rate"][2] for d in data["ms_imu"]])
    rpm = np.interp(tp, ts["zoe_veh_info"], [0.5 * (d["RL_wheel_speed"] + d["RR_wheel_speed"]) for d in data["zoe_veh_info"]])
    vm = data["vehicle_monitor"]
    if vm:
        idx = np.searchsorted(ts["vehicle_monitor"], tp, side="right") - 1
        gear = np.array([d["gear_position"] for d in vm])
        drive = np.where(idx < 0, gear[0] == 7, gear[np.clip(idx, 0, None)] == 7).astype(int)
    else:
        drive = np.ones_like(tp, dtype=int)
    tab = np.column_stack([tp - tp[0], pos[:, 0], pos[:, 1], yaw, v_pose, steer, r_imu, rpm, drive])
    info = {"samples": len(tp), "duration_s": round(float(tp[-1] - tp[0]), 2), "t0_utime": int(t0 + round(tp[0] * 1e6)),
            "vehicle_monitor": len(vm), "drive_fraction": round(float(drive.mean()), 3),
            "mean_speed_mps": round(float(v_pose.mean()), 2)}
    if vm:  # sign/scale cross-check: vehicle_monitor.yaw_rate (deg/s, 2 Hz) against the IMU
        vm_t = ts["vehicle_monitor"]
        sel = (vm_t >= lo) & (vm_t <= hi)
        if sel.sum() >= 5:
            a = np.radians([d["yaw_rate"] for d, s in zip(vm, sel, strict=True) if s])
            b = np.interp(vm_t[sel], ts["ms_imu"], [d["rotation_rate"][2] for d in data["ms_imu"]])
            info["vm_imu_yaw_sse"] = float(((a - b) ** 2).sum())
            info["vm_imu_yaw_n"] = int(sel.sum())
            info["vm_imu_yaw_dot"] = float((a * b).sum())
    return tab, info


def write_csv(path: Path, tab: np.ndarray) -> str:
    fmt = ["%.2f", "%.4f", "%.4f", "%.6f", "%.4f", "%.6f", "%.6f", "%.3f", "%d"]
    lines = [",".join(COLUMNS)]
    for row in tab:
        lines.append(",".join(f % v for f, v in zip(fmt, row, strict=True)))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def make_split(scenes: dict[str, dict], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Scene-level split with every log kept whole, stratified by vehicle (Boston n008, Singapore n015)."""
    rng = random.Random(seed)
    parts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    by_vehicle: dict[str, dict[str, list[str]]] = {}
    for name, s in sorted(scenes.items()):
        by_vehicle.setdefault(s["vehicle"], {}).setdefault(s["log"], []).append(name)
    for vehicle in sorted(by_vehicle):
        logs = sorted(by_vehicle[vehicle])
        rng.shuffle(logs)
        total = sum(len(by_vehicle[vehicle][g]) for g in logs)
        done = 0
        for g in logs:
            frac = done / total
            part = "train" if frac < SPLIT_FRACTIONS[0] else "val" if frac < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1] else "test"
            parts[part] += by_vehicle[vehicle][g]
            done += len(by_vehicle[vehicle][g])
    return {k: sorted(v) for k, v in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--can-zip", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="dir with v1.0-trainval/ and v1.0-test/ scene.json + log.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=Path("configs/vdyn/split.json"))
    args = ap.parse_args()

    scenes_meta, logs = {}, {}
    for part in ("v1.0-trainval", "v1.0-test"):
        for lg in json.loads((args.meta / part / "log.json").read_text()):
            logs[lg["token"]] = lg
        for s in json.loads((args.meta / part / "scene.json").read_text()):
            scenes_meta[s["name"]] = s
    z = zipfile.ZipFile(args.can_zip)
    ids = sorted({int(m.group(1)) for n in z.namelist() if (m := re.match(r"can_bus/scene-(\d{4})_", n))})
    args.out.mkdir(parents=True, exist_ok=True)
    kept, dropped = {}, {}
    for sid in ids:
        name = f"scene-{sid:04d}"
        if sid in CAN_BLACKLIST:
            dropped[name] = "devkit CAN blacklist"
            continue
        if name not in scenes_meta:
            dropped[name] = "not in nuScenes scene tables"
            continue
        tab, info = scene_table(z, sid)
        if tab is None:
            dropped[name] = info["reason"]
            continue
        lg = logs[scenes_meta[name]["log_token"]]
        info.update({"log": lg["logfile"], "vehicle": lg["vehicle"], "location": lg["location"],
                     "sha256": write_csv(args.out / f"{name}.csv", tab)})
        kept[name] = info
        if len(kept) % 100 == 0:
            print(f"{len(kept)} scenes", flush=True)
    blacklisted_in_zip = sorted(set(ids) & CAN_BLACKLIST)
    split = make_split(kept)
    manifest = {"source": "nuScenes CAN bus expansion (can_bus.zip), CC BY-NC-SA 4.0",
                "scenes_in_zip": len(ids), "blacklisted_in_zip": blacklisted_in_zip, "kept": kept, "dropped": dropped,
                "columns": COLUMNS}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    args.split.parent.mkdir(parents=True, exist_ok=True)
    args.split.write_text(json.dumps({
        "_comment": "Fixed scene split for vehicle-dynamics identification (train), choices (val) and evaluation (test). "
                    "Logs are kept whole and the split is stratified by vehicle. Generated by tools/vdyn/can_extract.py.",
        "seed": SPLIT_SEED, "fractions": SPLIT_FRACTIONS,
        "counts": {k: len(v) for k, v in split.items()}, **split}, indent=1) + "\n")
    print(f"kept {len(kept)} of {len(ids)} scenes, dropped {len(dropped)}; split " +
          ", ".join(f"{k} {len(v)}" for k, v in split.items()))
    for n, r in sorted(dropped.items()):
        print(f"  dropped {n}: {r}")


if __name__ == "__main__":
    main()
