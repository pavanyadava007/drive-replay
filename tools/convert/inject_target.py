"""Derive a must-warn test case from a real recording by injecting a synthetic target on the driven path.

The nuScenes drives contain no real forward-collision threat, so a stack that never warns would pass every
requirement check. This tool adds one: a stopped car, or a lead car that brakes, placed on the path the
recorded car actually drove. Everything else stays real: ego motion, calibration and all other radar returns.

The replay is open loop. The recorded car does not react to the injected target and drives "through" it,
so these cases test whether and when the warning fires, never braking or evasion. The injected target gets
ARS408-style cluster returns (3 points across its rear, seeded noise) while it is inside the radar's field
of view, and a ground-truth box (id 9000) so the oracle can demand the warning.

Output is deterministic; the catalogue pins it by sha256 and records the base recording and parameters.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

GT_ID = 9000
LENGTH, WIDTH = 4.5, 1.9
FOV_HALF_RAD = math.radians(45.0)
MAX_RANGE_M = 70.0


class Path2D:
    """Arc-length parametrised polyline of the recorded ego positions, extended straight past its end."""

    def __init__(self, ego: list[dict]):
        self.t = [e["t_us"] for e in ego]
        self.xy = [(e["x"], e["y"]) for e in ego]
        self.s = [0.0]
        for (x0, y0), (x1, y1) in zip(self.xy, self.xy[1:], strict=False):
            self.s.append(self.s[-1] + math.hypot(x1 - x0, y1 - y0))

    def s_at_time(self, t_us: int) -> float:
        i = min(max(bisect.bisect_left(self.t, t_us), 1), len(self.t) - 1)
        w = (t_us - self.t[i - 1]) / max(self.t[i] - self.t[i - 1], 1)
        return self.s[i - 1] + min(max(w, 0.0), 1.0) * (self.s[i] - self.s[i - 1])

    def pose_at(self, s: float) -> tuple[float, float, float]:
        i = min(max(bisect.bisect_left(self.s, s), 1), len(self.s) - 1)
        # skip zero-length segments (car standing still)
        j = i
        while j < len(self.s) - 1 and self.s[j] - self.s[j - 1] < 1e-3:
            j += 1
        (x0, y0), (x1, y1) = self.xy[j - 1], self.xy[j]
        seg = max(self.s[j] - self.s[j - 1], 1e-6)
        heading = math.atan2(y1 - y0, x1 - x0)
        d = s - self.s[j - 1]
        return x0 + d * (x1 - x0) / seg, y0 + d * (y1 - y0) / seg, heading


def target_state(kind: str, p: dict, t_us: int, path: Path2D, t0_us: int) -> tuple[float, float] | None:
    """(arc length of the target's rear bumper, its speed) at time t, or None before it exists."""
    t = (t_us - t0_us) * 1e-6
    if kind == "stopped_car":
        return p["s_rear"], 0.0
    if kind == "braking_lead":
        if t < p["t_start_s"]:
            return None
        dt = t - p["t_start_s"]
        v0, a, tb = p["v0"], p["decel"], p["t_brake_s"] - p["t_start_s"]
        if dt <= tb:
            return p["s_rear0"] + v0 * dt, v0
        dtb = dt - tb
        t_stop = v0 / a
        dtb_c = min(dtb, t_stop)
        return p["s_rear0"] + v0 * tb + v0 * dtb_c - 0.5 * a * dtb_c * dtb_c, max(v0 - a * dtb, 0.0)
    raise ValueError(kind)


def inject(src: Path, dst: Path, kind: str, args: argparse.Namespace) -> dict:
    with open(src, "rb") as f:
        msgs = [(c.topic, m.log_time, json.loads(m.data), m.sequence) for _, c, m in make_reader(f).iter_messages()]
    meta = next(d for t, _, d, _ in msgs if t == "/meta")
    calib = next(d for t, _, d, _ in msgs if t == "/calib/radar_front")
    ego = [d for t, _, d, _ in msgs if t == "/ego/state"]
    t0 = meta["t_us"]
    path = Path2D(ego)
    ego_t = [e["t_us"] for e in ego]
    front = 3.7  # rear axle to front bumper, as in the stack and the oracle

    if kind == "stopped_car":
        t_hit = t0 + int(args.t_hit_s * 1e6)
        p = {"s_rear": path.s_at_time(t_hit) + front}
    else:
        ts = t0 + int(args.t_start_s * 1e6)
        i = bisect.bisect_left(ego_t, ts)
        p = {"t_start_s": args.t_start_s, "t_brake_s": args.t_brake_s, "decel": args.decel,
             "v0": ego[i]["speed_mps"], "s_rear0": path.s_at_time(ts) + front + args.gap_m}

    cw, cs = calib["translation"], calib["rotation_wxyz"]
    w, x, y, z = cs
    s_yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rng = random.Random(args.seed)

    def ego_at(t_us: int) -> dict:
        i = min(max(bisect.bisect_left(ego_t, t_us), 0), len(ego) - 1)
        return ego[i]

    def geometry(t_us: int):
        st = target_state(kind, p, t_us, path, t0)
        if st is None:
            return None
        s_rear, v = st
        e = ego_at(t_us)
        if s_rear - path.s_at_time(t_us) - front <= 0.3:  # reached: open loop, the recorded car drives through
            return None
        cx, cy, hd = path.pose_at(s_rear + LENGTH / 2)
        rx, ry, _ = path.pose_at(s_rear)
        return {"center": (cx, cy), "rear": (rx, ry), "heading": hd, "v": v, "ego": e}

    n_radar_by_t: dict[int, int] = {}
    out = []
    for topic, log_time, d, seq in msgs:
        if topic == "/radar/front":
            g = geometry(d["t_us"])
            added = 0
            if g is not None:
                e = g["ego"]
                ce, se = math.cos(e["yaw"]), math.sin(e["yaw"])
                ego_v = (e["speed_mps"] * ce, e["speed_mps"] * se)
                tgt_v = (g["v"] * math.cos(g["heading"]), g["v"] * math.sin(g["heading"]))
                fields = d["fields"]
                for lat in (-0.7, 0.0, 0.7):
                    gx = g["rear"][0] - lat * math.sin(g["heading"])
                    gy = g["rear"][1] + lat * math.cos(g["heading"])
                    # global -> vehicle -> sensor
                    dx, dy = gx - e["x"], gy - e["y"]
                    vx_, vy_ = dx * ce + dy * se, -dx * se + dy * ce
                    sx0, sy0 = vx_ - cw[0], vy_ - cw[1]
                    sx = sx0 * math.cos(s_yaw) + sy0 * math.sin(s_yaw) + rng.gauss(0, 0.1)
                    sy = -sx0 * math.sin(s_yaw) + sy0 * math.cos(s_yaw) + rng.gauss(0, 0.1)
                    if sx <= 0.5 or math.hypot(sx, sy) > MAX_RANGE_M or abs(math.atan2(sy, sx)) > FOV_HALF_RAD:
                        continue
                    rel = (tgt_v[0] - ego_v[0], tgt_v[1] - ego_v[1])
                    rvx = rel[0] * ce + rel[1] * se
                    rvy = -rel[0] * se + rel[1] * ce
                    gvx = tgt_v[0] * ce + tgt_v[1] * se
                    gvy = -tgt_v[0] * se + tgt_v[1] * ce
                    pt = {"x": sx, "y": sy, "vx": rvx + rng.gauss(0, 0.1), "vy": rvy, "vx_comp": gvx, "vy_comp": gvy,
                          "rcs": 12.0, "dyn_prop": 1 if g["v"] < 0.3 else 0, "ambig_state": 3, "invalid_state": 0,
                          "id": 900 + added}
                    d["points"].append([round(pt[k], 4) if isinstance(pt[k], float) else pt[k] for k in fields])
                    added += 1
            n_radar_by_t[d["t_us"]] = added
        out.append((topic, log_time, d, seq))

    radar_t = sorted(n_radar_by_t)
    for topic, _, d, _ in out:
        if topic == "/gt/objects":
            g = geometry(d["t_us"])
            if g is None:
                continue
            j = min(max(bisect.bisect_left(radar_t, d["t_us"]), 0), len(radar_t) - 1)
            d["objects"].append({"id": GT_ID, "category": "vehicle.car", "x": round(g["center"][0], 4),
                                 "y": round(g["center"][1], 4), "yaw": round(g["heading"], 4), "length": LENGTH,
                                 "width": WIDTH, "vx": round(g["v"] * math.cos(g["heading"]), 4),
                                 "vy": round(g["v"] * math.sin(g["heading"]), 4),
                                 "num_radar_pts": n_radar_by_t[radar_t[j]], "num_lidar_pts": 0, "injected": True})
        if topic == "/meta":
            d["scene"] = dst.stem
            d["injected"] = {"kind": kind, "base": src.name, "params": {k: round(v, 4) for k, v in p.items()},
                             "seed": args.seed, "note": "synthetic target on the recorded driven path, open loop"}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        wr = Writer(f, compression=CompressionType.NONE, chunk_size=1 << 20)
        wr.start(profile="", library="drive-replay inject_target 1")
        chans = {}
        for topic in dict.fromkeys(t for t, *_ in out):
            name = {"/meta": "Meta", "/calib/radar_front": "Extrinsic", "/ego/state": "EgoState",
                    "/radar/front": "RadarScan", "/gt/objects": "GroundTruth"}[topic]
            sid = wr.register_schema(name=f"drive_replay.{name}", encoding="jsonschema",
                                     data=json.dumps({"title": f"drive_replay.{name}", "type": "object"}).encode())
            chans[topic] = wr.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for topic, log_time, d, seq in out:
            wr.add_message(chans[topic], log_time=log_time, publish_time=log_time, sequence=seq,
                           data=json.dumps(d, separators=(",", ":")).encode())
        wr.finish()
    return {"scene": dst.stem, "file": dst.name, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            "bytes": dst.stat().st_size, "duration_s": round((ego[-1]["t_us"] - t0) * 1e-6, 3),
            "vehicle": meta["vehicle"], "location": meta["location"],
            "description": f"{meta['description']} + injected {kind.replace('_', ' ')}",
            "derived_from": src.stem, "injected": {"kind": kind, **{k: round(v, 4) for k, v in p.items()}}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--kind", choices=["stopped_car", "braking_lead"], required=True)
    ap.add_argument("--t-hit-s", type=float, default=12.0, help="stopped_car: when the recorded car reaches it")
    ap.add_argument("--t-start-s", type=float, default=3.0, help="braking_lead: when the lead car appears")
    ap.add_argument("--gap-m", type=float, default=18.0, help="braking_lead: initial bumper-to-bumper gap")
    ap.add_argument("--t-brake-s", type=float, default=7.0)
    ap.add_argument("--decel", type=float, default=4.0, help="braking_lead: m/s^2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", type=Path, default=Path("recordings/catalog.json"))
    a = ap.parse_args()
    entry = inject(a.src, a.dst, a.kind, a)
    cat = json.loads(a.catalog.read_text()) if a.catalog.exists() else {"recordings": []}
    cat["recordings"] = [e for e in cat["recordings"] if e["scene"] != entry["scene"]] + [entry]
    a.catalog.write_text(json.dumps(cat, indent=1) + "\n")
    print(f"{entry['scene']}: sha256 {entry['sha256'][:12]}, {entry['injected']}")


if __name__ == "__main__":
    main()
