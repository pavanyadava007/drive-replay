"""Convert nuScenes scenes into drive-replay recordings (one MCAP file per scene).

A recording holds what the car logged, in the order it logged it:

  /meta              scene, log, vehicle, location, licence (one message)
  /calib/radar_front sensor-to-vehicle extrinsic of RADAR_FRONT (one message)
  /ego/state         20 Hz ego pose from the lidar ego poses, speed and yaw rate by central differences
  /radar/front       every RADAR_FRONT sweep (about 13 Hz), points in the SENSOR frame, as recorded
  /gt/objects        2 Hz annotated boxes in the global frame; the stack never subscribes to this topic,
                     only the test oracle reads it

Messages are JSON with a JSON schema, so a recording opens in Foxglove as is. Chunks are not compressed
(the C++ reader then needs no lz4/zstd). Output is byte-for-byte deterministic for the same input, which is
what lets the catalogue pin every recording by sha256.

nuScenes is CC BY-NC-SA 4.0 (https://www.nuscenes.org/terms-of-use). Converted recordings inherit that licence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from mcap.writer import CompressionType, Writer

PCD_TYPES = {("F", 4): "<f4", ("I", 1): "i1", ("I", 2): "<i2", ("U", 1): "u1", ("U", 2): "<u2", ("I", 4): "<i4"}
RADAR_FIELDS = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp",
                "is_quality_valid", "ambig_state", "x_rms", "y_rms", "invalid_state", "pdh0", "vx_rms", "vy_rms"]
KEEP = ["x", "y", "vx", "vy", "vx_comp", "vy_comp", "rcs", "dyn_prop", "ambig_state", "invalid_state", "id"]
GT_CATEGORIES = ("vehicle.", "human.pedestrian", "movable_object.barrier", "movable_object.trafficcone")


def read_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header, _, body = raw.partition(b"DATA binary\n")
    meta = {}
    for line in header.decode().splitlines():
        if line and not line.startswith("#"):
            k, *v = line.split()
            meta[k] = v
    dtype = np.dtype([(n, PCD_TYPES[(t, int(s))]) for n, t, s in zip(meta["FIELDS"], meta["TYPE"], meta["SIZE"], strict=True)])
    n = int(meta["POINTS"][0])
    return np.frombuffer(body[: n * dtype.itemsize], dtype=dtype)


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def r4(v: float) -> float:
    return float(round(float(v), 4))


class NuScenes:
    def __init__(self, root: Path, version: str = "v1.0-mini"):
        self.root = root
        t = lambda name: json.loads((root / version / f"{name}.json").read_text())
        self.scene = t("scene")
        self.sample = {r["token"]: r for r in t("sample")}
        self.sample_data = {r["token"]: r for r in t("sample_data")}
        self.ego_pose = {r["token"]: r for r in t("ego_pose")}
        self.calib = {r["token"]: r for r in t("calibrated_sensor")}
        self.sensor = {r["token"]: r for r in t("sensor")}
        self.log = {r["token"]: r for r in t("log")}
        self.category = {r["token"]: r["name"] for r in t("category")}
        self.instance = {r["token"]: r for r in t("instance")}
        self.ann_by_sample: dict[str, list] = {}
        for a in t("sample_annotation"):
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)
        self.ann = {a["token"]: a for anns in self.ann_by_sample.values() for a in anns}
        for sd in self.sample_data.values():  # the raw tables do not carry sample["data"]; the devkit builds it
            if sd["is_key_frame"]:
                self.sample[sd["sample_token"]].setdefault("data", {})[self.channel(sd)] = sd["token"]

    def channel(self, sd) -> str:
        return self.sensor[self.calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]

    def chain(self, scene, channel: str) -> list[dict]:
        """All sample_data of one channel (key frames and sweeps) inside the scene's time span."""
        first = self.sample[scene["first_sample_token"]]
        last = self.sample[scene["last_sample_token"]]
        sd = self.sample_data[first["data"][channel]]
        while sd["prev"] and self.sample_data[sd["prev"]]["timestamp"] >= first["timestamp"] - 60_000:
            sd = self.sample_data[sd["prev"]]
        out = []
        while True:
            if sd["timestamp"] > last["timestamp"] + 60_000:
                break
            out.append(sd)
            if not sd["next"]:
                break
            sd = self.sample_data[sd["next"]]
        return out

    def samples(self, scene) -> list[dict]:
        s, out = self.sample[scene["first_sample_token"]], []
        while True:
            out.append(s)
            if not s["next"]:
                return out
            s = self.sample[s["next"]]

    def velocity(self, ann) -> tuple[float, float]:
        """Box velocity by finite differences over prev/next annotations (the devkit's box_velocity)."""
        prev = self.ann.get(ann["prev"]) if ann["prev"] else None
        nxt = self.ann.get(ann["next"]) if ann["next"] else None
        a, b = prev or ann, nxt or ann
        if a is b:
            return 0.0, 0.0
        ta = self.sample[a["sample_token"]]["timestamp"]
        tb = self.sample[b["sample_token"]]["timestamp"]
        dt = (tb - ta) * 1e-6
        if dt <= 0 or dt > 1.5:
            return 0.0, 0.0
        return ((b["translation"][0] - a["translation"][0]) / dt, (b["translation"][1] - a["translation"][1]) / dt)


def ego_states(ns: NuScenes, scene) -> list[dict]:
    lidar = ns.chain(scene, "LIDAR_TOP")
    poses = [ns.ego_pose[sd["ego_pose_token"]] for sd in lidar]
    ts = np.array([p["timestamp"] for p in poses], dtype=np.int64)
    xy = np.array([p["translation"][:2] for p in poses])
    yaw = np.unwrap(np.array([yaw_of(p["rotation"]) for p in poses]))
    out = []
    for i in range(len(poses)):
        j0, j1 = max(i - 1, 0), min(i + 1, len(poses) - 1)
        dt = (ts[j1] - ts[j0]) * 1e-6
        vx, vy = (xy[j1] - xy[j0]) / dt
        heading = yaw[i]
        # signed longitudinal speed: projection of the velocity on the heading
        speed = vx * math.cos(heading) + vy * math.sin(heading)
        out.append({"t_us": int(ts[i]), "x": r4(xy[i, 0]), "y": r4(xy[i, 1]), "yaw": r4(wrap(heading)),
                    "speed_mps": r4(speed), "yaw_rate_rps": r4((yaw[j1] - yaw[j0]) / dt)})
    return out


def radar_scans(ns: NuScenes, scene) -> list[dict]:
    out = []
    for sd in ns.chain(scene, "RADAR_FRONT"):
        pts = read_pcd(ns.root / sd["filename"])
        points = [[r4(p[k]) if pts.dtype[k].kind == "f" else int(p[k]) for k in KEEP] for p in pts]
        out.append({"t_us": int(sd["timestamp"]), "fields": KEEP, "points": points})
    return out


def gt_objects(ns: NuScenes, scene) -> list[dict]:
    out = []
    short_ids: dict[str, int] = {}
    for s in ns.samples(scene):
        objs = []
        for a in ns.ann_by_sample.get(s["token"], []):
            cat = ns.category[ns.instance[a["instance_token"]]["category_token"]]
            if not cat.startswith(GT_CATEGORIES):
                continue
            vx, vy = ns.velocity(a)
            sid = short_ids.setdefault(a["instance_token"], len(short_ids))
            objs.append({"id": sid, "category": cat, "x": r4(a["translation"][0]), "y": r4(a["translation"][1]),
                         "yaw": r4(yaw_of(a["rotation"])), "length": r4(a["size"][1]), "width": r4(a["size"][0]),
                         "vx": r4(vx), "vy": r4(vy), "num_radar_pts": a["num_radar_pts"],
                         "num_lidar_pts": a["num_lidar_pts"]})
        out.append({"t_us": int(s["timestamp"]), "objects": objs})
    return out


SCHEMAS = {
    "/meta": "drive_replay.Meta",
    "/calib/radar_front": "drive_replay.Extrinsic",
    "/ego/state": "drive_replay.EgoState",
    "/radar/front": "drive_replay.RadarScan",
    "/gt/objects": "drive_replay.GroundTruth",
}


def write_scene(ns: NuScenes, scene, out_path: Path) -> dict:
    first = ns.sample[scene["first_sample_token"]]
    radar_sd = ns.sample_data[first["data"]["RADAR_FRONT"]]
    lidar_sd = ns.sample_data[first["data"]["LIDAR_TOP"]]
    log = ns.log[scene["log_token"]]
    cal = ns.calib[radar_sd["calibrated_sensor_token"]]
    ego = ego_states(ns, scene)
    radar = radar_scans(ns, scene)
    gt = gt_objects(ns, scene)
    t0 = min(ego[0]["t_us"], radar[0]["t_us"], gt[0]["t_us"])
    meta = {"t_us": t0, "scene": scene["name"], "description": scene["description"], "log": log["logfile"],
            "vehicle": log["vehicle"], "location": log["location"], "date": log["date_captured"],
            "source": "nuScenes v1.0-mini", "licence": "CC BY-NC-SA 4.0 (nuScenes terms of use)",
            "lidar_ego_first_us": int(lidar_sd["timestamp"])}
    calib = {"t_us": t0, "sensor": "RADAR_FRONT", "translation": [r4(v) for v in cal["translation"]],
             "rotation_wxyz": [float(round(v, 9)) for v in cal["rotation"]]}
    msgs = [("/meta", meta), ("/calib/radar_front", calib)]
    msgs += [("/ego/state", m) for m in ego] + [("/radar/front", m) for m in radar] + [("/gt/objects", m) for m in gt]
    order = {"/meta": 0, "/calib/radar_front": 1, "/ego/state": 2, "/gt/objects": 3, "/radar/front": 4}
    msgs.sort(key=lambda tm: (tm[1]["t_us"], order[tm[0]]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        w = Writer(f, compression=CompressionType.NONE, chunk_size=1 << 20)
        w.start(profile="", library="drive-replay nuscenes_to_mcap 1")
        chans = {}
        for topic, name in SCHEMAS.items():
            sid = w.register_schema(name=name, encoding="jsonschema",
                                    data=json.dumps({"title": name, "type": "object"}).encode())
            chans[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for seq, (topic, m) in enumerate(msgs):
            t_ns = m["t_us"] * 1000
            w.add_message(chans[topic], log_time=t_ns, publish_time=t_ns, sequence=seq,
                          data=json.dumps(m, separators=(",", ":")).encode())
        w.finish()
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    counts = {t: sum(1 for x, _ in msgs if x == t) for t in SCHEMAS}
    return {"scene": scene["name"], "file": out_path.name, "sha256": sha, "bytes": out_path.stat().st_size,
            "duration_s": round((msgs[-1][1]["t_us"] - t0) * 1e-6, 3), "messages": counts,
            "vehicle": log["vehicle"], "location": log["location"], "description": scene["description"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nuscenes", type=Path, required=True, help="nuScenes root (contains v1.0-mini/, sweeps/)")
    ap.add_argument("--out", type=Path, default=Path("data/recordings"))
    ap.add_argument("--catalog", type=Path, default=Path("recordings/catalog.json"))
    ap.add_argument("--scenes", nargs="*", help="scene names (default: all)")
    args = ap.parse_args()
    ns = NuScenes(args.nuscenes)
    entries = []
    for scene in sorted(ns.scene, key=lambda s: s["name"]):
        if args.scenes and scene["name"] not in args.scenes:
            continue
        e = write_scene(ns, scene, args.out / f"{scene['name']}.mcap")
        entries.append(e)
        print(f"{e['scene']}: {e['duration_s']:.1f} s, {sum(e['messages'].values())} msgs, "
              f"{e['bytes'] / 1e6:.2f} MB, sha256 {e['sha256'][:12]}")
    if not args.scenes:
        args.catalog.parent.mkdir(parents=True, exist_ok=True)
        args.catalog.write_text(json.dumps({"source": "nuScenes v1.0-mini", "licence": "CC BY-NC-SA 4.0",
                                            "converter": "tools/convert/nuscenes_to_mcap.py",
                                            "recordings": entries}, indent=1) + "\n")


if __name__ == "__main__":
    main()
