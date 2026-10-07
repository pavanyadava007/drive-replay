"""Build the CAN recording set: every suite recording plus a /vehicle/can topic from the nuScenes CAN bus expansion.

    python tools/vdyn/add_can_topic.py --can-zip <can_bus.zip> [--src data/recordings] [--out data/recordings_can]

The published recordings stay untouched: adding a topic would change their sha256, which the catalogue pins
and the golden baselines are tied to. This writes a separate set instead (same messages, same order, plus one
channel) and its own catalogue, recordings/catalog_can.json. Must-warn cases (scene-0796-stopped-car, ...)
get the CAN data of the drive they were built from.

/vehicle/can carries, at the 100 Hz steeranglefeedback timestamps inside the recording's time span:
  t_us            CAN measurement time (us, same clock as the other topics)
  steer_sw_rad    steering-wheel angle (rad, left positive)
  wheel_rpm_rear  mean rear wheel speed from zoe_veh_info (rpm), linearly interpolated to t_us

Output is byte-for-byte deterministic for the same inputs. CAN data: CC BY-NC-SA 4.0 (nuScenes terms of use).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

ROOT = Path(__file__).resolve().parents[2]


def can_messages(z: zipfile.ZipFile, scene: str, t_lo: int, t_hi: int) -> list[dict]:
    steer = json.loads(z.read(f"can_bus/{scene}_steeranglefeedback.json"))
    veh = json.loads(z.read(f"can_bus/{scene}_zoe_veh_info.json"))
    tv = np.array([m["utime"] for m in veh], dtype=np.int64)
    rpm = np.array([0.5 * (m["RL_wheel_speed"] + m["RR_wheel_speed"]) for m in veh])
    out = []
    for m in steer:
        t = int(m["utime"])
        if t_lo <= t <= t_hi:
            out.append({"t_us": t, "steer_sw_rad": round(float(m["value"]), 6),
                        "wheel_rpm_rear": round(float(np.interp(t, tv, rpm)), 3)})
    return out


def augment(src: Path, dst: Path, z: zipfile.ZipFile, base_scene: str) -> dict:
    with open(src, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schemas = {sid: s for sid, s in summary.schemas.items()}
        channels = {cid: c for cid, c in summary.channels.items()}
        msgs = [(ch.topic, m.log_time, m.publish_time, m.data) for _, ch, m in reader.iter_messages(log_time_order=True)]
        profile, library = reader.get_header().profile, reader.get_header().library
    t_lo, t_hi = msgs[0][1] // 1000, msgs[-1][1] // 1000
    can = can_messages(z, base_scene, t_lo, t_hi)
    allm = [(topic, lt, pt, data, 0) for topic, lt, pt, data in msgs]
    allm += [("/vehicle/can", c["t_us"] * 1000, c["t_us"] * 1000, json.dumps(c, separators=(",", ":")).encode(), 1)
             for c in can]
    allm.sort(key=lambda m: (m[1], m[4]))  # stable: original order first at equal log time
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        w = Writer(f, compression=CompressionType.NONE, chunk_size=1 << 20)
        w.start(profile=profile, library=library + " + add_can_topic 1")
        new_ids = {}
        for sid, s in sorted(schemas.items()):
            new_ids[sid] = w.register_schema(name=s.name, encoding=s.encoding, data=s.data)
        chans = {}
        for _cid, c in sorted(channels.items()):
            chans[c.topic] = w.register_channel(topic=c.topic, message_encoding=c.message_encoding,
                                                schema_id=new_ids.get(c.schema_id, 0))
        sid = w.register_schema(name="drive_replay.VehicleCan", encoding="jsonschema",
                                data=json.dumps({"title": "drive_replay.VehicleCan", "type": "object"}).encode())
        chans["/vehicle/can"] = w.register_channel(topic="/vehicle/can", message_encoding="json", schema_id=sid)
        for seq, (topic, lt, pt, data, _) in enumerate(allm):
            w.add_message(chans[topic], log_time=lt, publish_time=pt, sequence=seq, data=data)
        w.finish()
    return {"scene": dst.stem, "file": dst.name, "base_scene": base_scene, "can_messages": len(can),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--can-zip", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "recordings")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "recordings_can")
    ap.add_argument("--catalog", type=Path, default=ROOT / "recordings" / "catalog_can.json")
    args = ap.parse_args()
    base = json.loads((ROOT / "recordings" / "catalog.json").read_text())
    z = zipfile.ZipFile(args.can_zip)
    entries = []
    for e in base["recordings"]:
        name = e["scene"]
        base_scene = name[:10]  # scene-0796-stopped-car -> scene-0796
        src = args.src / e["file"]
        if hashlib.sha256(src.read_bytes()).hexdigest() != e["sha256"]:
            raise SystemExit(f"{src}: sha256 does not match recordings/catalog.json")
        r = augment(src, args.out / e["file"], z, base_scene)
        entries.append(r)
        print(f"{name:<26} +{r['can_messages']} /vehicle/can msgs  sha256 {r['sha256'][:12]}")
    args.catalog.write_text(json.dumps({
        "source": "recordings/catalog.json + nuScenes CAN bus expansion (steeranglefeedback, zoe_veh_info)",
        "licence": "CC BY-NC-SA 4.0", "builder": "tools/vdyn/add_can_topic.py", "recordings": entries}, indent=1) + "\n")


if __name__ == "__main__":
    main()
