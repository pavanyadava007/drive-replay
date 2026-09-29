#!/usr/bin/env bash
# Rebuilds all suite recordings from nuScenes v1.0-mini and checks them against recordings/catalog.json.
#   scripts/make_recordings.sh <nuscenes root> [out dir]
# The catalogue is the lock file: a recording whose sha256 differs is an error, never a silent update.
set -euo pipefail
NUSC="${1:?nuScenes root (contains v1.0-mini/ and sweeps/)}"
OUT="${2:-data/recordings}"
PY="${PYTHON:-python3}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cp recordings/catalog.json "$tmp/lock.json"
"$PY" tools/convert/nuscenes_to_mcap.py --nuscenes "$NUSC" --out "$OUT" --catalog "$tmp/catalog.json"
inject() { "$PY" tools/convert/inject_target.py --catalog "$tmp/catalog.json" "$@"; }
inject --src "$OUT/scene-0796.mcap" --dst "$OUT/scene-0796-stopped-car.mcap" --kind stopped_car --t-hit-s 12 --seed 1
inject --src "$OUT/scene-1077.mcap" --dst "$OUT/scene-1077-stopped-car.mcap" --kind stopped_car --t-hit-s 14 --seed 2
inject --src "$OUT/scene-0655.mcap" --dst "$OUT/scene-0655-braking-lead.mcap" --kind braking_lead --t-start-s 3 \
  --gap-m 18 --t-brake-s 8 --decel 4 --seed 3
"$PY" - "$tmp/lock.json" "$tmp/catalog.json" <<'PY'
import json, sys
lock = {e["scene"]: e["sha256"] for e in json.load(open(sys.argv[1]))["recordings"]}
got = {e["scene"]: e["sha256"] for e in json.load(open(sys.argv[2]))["recordings"]}
bad = [s for s in lock if got.get(s) != lock[s]]
for s in lock:
    print(f"{s:<26} {'ok' if s not in bad else 'MISMATCH'}  {got.get(s, 'missing')[:12]}")
sys.exit(1 if bad else 0)
PY
