"""Recording catalogue: resolves a scene name (or a file) to a verified local recording."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tools.replay import paths


class ResolveError(RuntimeError):
    pass


@dataclass(frozen=True)
class Recording:
    name: str
    path: Path
    sha256: str
    vehicle: str
    duration_s: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entries() -> list[dict]:
    return json.loads(paths.CATALOG.read_text())["recordings"]


def resolve(spec: str) -> Recording:
    """spec: a catalogue scene name ('scene-0061', also '0061' or '61') or a path to an .mcap file."""
    p = Path(spec)
    if p.suffix == ".mcap" and p.exists():
        return Recording(p.stem, p.resolve(), sha256_file(p), "unknown", 0.0)
    wanted = spec if spec.startswith("scene-") else f"scene-{int(spec):04d}" if spec.isdigit() else spec
    for e in entries():
        if e["scene"] == wanted:
            path = paths.DATA / e["file"]
            if not path.exists():
                raise ResolveError(f"{wanted}: {path} is missing. Convert it with "
                                   f"tools/convert/nuscenes_to_mcap.py --nuscenes <root> --out {paths.DATA}")
            got = sha256_file(path)
            if got != e["sha256"]:
                raise ResolveError(f"{wanted}: sha256 {got[:12]} does not match the catalogue ({e['sha256'][:12]}); "
                                   "the recording changed on disk, refusing to replay it")
            return Recording(wanted, path, got, e["vehicle"], e["duration_s"])
    names = ", ".join(e["scene"] for e in entries())
    raise ResolveError(f"unknown recording '{spec}'. Known: {names}")
