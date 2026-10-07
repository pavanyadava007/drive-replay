"""Batch reprocessing: run the C++ model evaluation (vdyn_eval) over many scenes in parallel, with digests.

    python -m tools.vdyn.pipeline --scenes-dir <can_scenes> --split test --workers 16 --out runs/vdyn/test

Scenes are grouped by car (the steering offset is per car) and cut into chunks; each chunk is one vdyn_eval
process, and a thread pool keeps `workers` processes running. Every scene's metrics carry a digest computed by
the C++ side from the metrics alone, so the batch digest (sha256 over the sorted scene digests) is the same for
any number of workers and any chunking. That is checked, not assumed: see docs/VEHICLE_DYNAMICS.md.

Also holds the statistics used by the report and the REST service: pooled errors per scene set and a
bootstrap over scenes (scenes are the unit of resampling, never samples, since samples within a scene are
strongly correlated).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = Path(os.environ.get("VDYN_EVAL", ROOT / "bazel-bin" / "stack" / "vdyn" / "vdyn_eval"))
DEFAULT_CONFIG = ROOT / "configs" / "vdyn" / "renault_zoe.json"
DEFAULT_SCENES = Path(os.environ.get("VDYN_SCENES", Path.home() / "workspace" / "drive-replay-data" / "can_scenes"))
SPLIT = ROOT / "configs" / "vdyn" / "split.json"


@dataclass
class BatchResult:
    scenes: dict[str, dict]
    wall_s: float
    workers: int
    processes: int
    recorded_s: float
    batch_digest: str
    config_sha256: str
    binary_sha256: str
    errors: list[str] = field(default_factory=list)

    @property
    def scenes_per_s(self) -> float:
        return len(self.scenes) / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def x_realtime(self) -> float:
        return self.recorded_s / self.wall_s if self.wall_s > 0 else 0.0

    def summary(self) -> dict:
        return {"scenes": len(self.scenes), "workers": self.workers, "processes": self.processes,
                "wall_s": round(self.wall_s, 3), "recorded_s": round(self.recorded_s, 1),
                "scenes_per_s": round(self.scenes_per_s, 1), "x_realtime": round(self.x_realtime, 1),
                "batch_digest": self.batch_digest, "config_sha256": self.config_sha256[:16],
                "binary_sha256": self.binary_sha256[:16], "errors": self.errors}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(scenes_dir: Path = DEFAULT_SCENES) -> dict[str, dict]:
    return json.loads((scenes_dir / "manifest.json").read_text())["kept"]


def split_scenes(part: str) -> list[str]:
    return json.loads(SPLIT.read_text())[part]


def batch_digest(scenes: dict[str, dict]) -> str:
    h = hashlib.sha256()
    for name in sorted(scenes):
        h.update(f"{name}:{scenes[name]['digest']}\n".encode())
    return h.hexdigest()[:16]


def run_batch(names: list[str], workers: int, scenes_dir: Path = DEFAULT_SCENES, config: Path = DEFAULT_CONFIG,
              binary: Path = DEFAULT_BINARY, sets: list[str] | None = None, chunk: int = 8, yaw_only: bool = False,
              progress: Callable[[int, int], None] | None = None) -> BatchResult:
    vehicle_of = {n: m["vehicle"] for n, m in manifest(scenes_dir).items()}
    unknown = [n for n in names if n not in vehicle_of]
    if unknown:
        raise ValueError(f"unknown scenes: {', '.join(unknown[:5])}")
    jobs = []
    for veh in sorted({vehicle_of[n] for n in names}):
        mine = sorted(n for n in names if vehicle_of[n] == veh)
        jobs += [(veh, mine[i:i + chunk]) for i in range(0, len(mine), chunk)]
    done = 0
    lock = threading.Lock()

    def one(job: tuple[str, list[str]]) -> tuple[list[dict], str]:
        nonlocal done
        veh, part = job
        cmd = [str(binary), "--config", str(config), "--vehicle", veh]
        if yaw_only:
            cmd.append("--yaw-only")
        for s in sets or []:
            cmd += ["--set", s]
        for n in part:
            cmd += ["--scene", str(scenes_dir / f"{n}.csv")]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        with lock:
            done += len(part)
            if progress:
                progress(done, len(names))
        if p.returncode != 0:
            return [], f"{part[0]}..{part[-1]}: {p.stderr.strip()[-300:]}"
        return [json.loads(line) for line in p.stdout.splitlines()], ""

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(one, jobs))
    wall = time.monotonic() - t0
    scenes = {r["scene"]: r for rows, _ in outs for r in rows}
    errors = [e for _, e in outs if e]
    return BatchResult(scenes, wall, workers, len(jobs), sum(r["duration_s"] for r in scenes.values()),
                       batch_digest(scenes), sha256_file(config), sha256_file(binary), errors)


# ---- statistics over scenes -------------------------------------------------------------------------------------

def sums(scenes: list[dict], path: tuple[str, ...]) -> np.ndarray:
    """Per-scene [n, sum |e|, sum e^2] at a path like ("traj", "dynamic", "3", "fde"); zeros where missing."""
    out = np.zeros((len(scenes), 3))
    for i, s in enumerate(scenes):
        node = s
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if node is not None:
            out[i] = node
    return out


def pooled(a: np.ndarray) -> dict:
    n = a[:, 0].sum()
    if n == 0:
        return {"n": 0, "mean_abs": None, "rmse": None}
    return {"n": int(n), "mean_abs": float(a[:, 1].sum() / n), "rmse": float(np.sqrt(a[:, 2].sum() / n))}


def bootstrap_weights(n_scenes: int, b: int = 2000, seed: int = 7) -> np.ndarray:
    """B x S matrix of how often each scene is drawn in each bootstrap resample (scenes with replacement)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_scenes, size=(b, n_scenes))
    w = np.zeros((b, n_scenes))
    np.add.at(w, (np.repeat(np.arange(b), n_scenes), idx.ravel()), 1.0)
    return w


def _stat(w: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    n = w @ a[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ a[:, 1]) / n if kind == "mean_abs" else np.sqrt((w @ a[:, 2]) / n)


def ci(a: np.ndarray, w: np.ndarray, kind: str = "mean_abs") -> dict:
    """Pooled value with a 95 % percentile bootstrap interval over scenes."""
    p = pooled(a)
    if not p["n"]:
        return {**p, "value": None, "lo": None, "hi": None}
    boot = _stat(w, a, kind)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {**p, "value": p[kind], "lo": float(lo), "hi": float(hi)}


def paired(a: np.ndarray, b: np.ndarray, w: np.ndarray, kind: str = "mean_abs") -> dict:
    """A minus B on the same scenes and the same resamples. Negative = A has the smaller error."""
    pa, pb = pooled(a), pooled(b)
    diff = pa[kind] - pb[kind]
    boot = _stat(w, a, kind) - _stat(w, b, kind)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    both = (a[:, 0] > 0) & (b[:, 0] > 0)
    ma = np.divide(a[both, 1], a[both, 0])
    mb = np.divide(b[both, 1], b[both, 0])
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi), "rel": float(diff / pb[kind]) if pb[kind] else None,
            "a_better_scenes": int((ma < mb).sum()), "scenes": int(both.sum()),
            "significant": bool(hi < 0 or lo > 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES)
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = sorted(manifest(args.scenes_dir)) if args.split == "all" else split_scenes(args.split)
    res = run_batch(names, args.workers, args.scenes_dir, args.config, args.binary)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "metrics.jsonl", "w") as f:
        for name in sorted(res.scenes):
            f.write(json.dumps(res.scenes[name], sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"split": args.split, **res.summary()}, indent=1) + "\n")
    print(json.dumps(res.summary(), indent=1))


if __name__ == "__main__":
    main()
