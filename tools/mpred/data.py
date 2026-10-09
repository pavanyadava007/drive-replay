"""Scene loading, start points and ground truth for the motion-prediction benchmark.

All scenes of one split are held as padded (scenes x samples) arrays on the pose clock (about 49 Hz), so the
filters can run over every scene at once and the predictors can gather any sample by (scene, index).

Causality: a prediction made at sample k may read samples 0..k only (the inputs are sample-and-hold values of
messages received by the pose time, see tools/mpred/extract.py). The truth for horizon h is the recorded pose
at t_k + h, linearly interpolated between the two neighbouring pose samples.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = Path(os.environ.get("MPRED_SCENES", Path.home() / "workspace" / "drive-replay-data" / "can_scenes_v2"))
VEHICLE_CONFIG = ROOT / "configs" / "vdyn" / "renault_zoe.json"
SPLIT = ROOT / "configs" / "vdyn" / "split.json"

HORIZONS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00)  # s, display-latency range plus 1 s for context
DT_INT = 0.01  # integration step of every physics rollout (all horizons are multiples of it)
H_STEPS = tuple(int(round(h / DT_INT)) for h in HORIZONS)
HISTORY = 50  # samples (about 1 s) before the first start point: filter warm-up and the longest lag
STRIDE = 5  # evaluation start points every 5 pose samples (about 0.1 s)
MIN_SPEED = 1.0  # m/s wheel speed at the start point
FUTURE_S = 1.2  # recorded time needed after a start point (1 s horizon + up to 0.2 s emulated latency)
SIGNALS = ("t", "x", "y", "yaw", "v_pose", "drive", "steer", "r", "rpm", "ax_imu", "ay_imu", "ax_can")


@dataclass
class Batch:
    """Padded per-sample arrays (S x T) of one scene set. Padding repeats the last row (time keeps running)."""
    names: list[str]
    vehicle: list[str]
    location: list[str]
    n: np.ndarray  # valid samples per scene
    sig: dict[str, np.ndarray]  # SIGNALS plus derived v (wheel speed), d_kin, d_dyn (road-wheel angle, rad)
    extra: dict = field(default_factory=dict)

    @property
    def S(self) -> int:
        return len(self.names)

    @property
    def T(self) -> int:
        return self.sig["t"].shape[1]

    def __getitem__(self, k: str) -> np.ndarray:
        return self.sig[k]

    def with_signals(self, **arrays: np.ndarray) -> Batch:
        return replace(self, sig={**self.sig, **arrays})


def vehicle_config() -> dict:
    return json.loads(VEHICLE_CONFIG.read_text())


def split_names(part: str) -> list[str]:
    return json.loads(SPLIT.read_text())[part]


def derive(sig: dict[str, np.ndarray], vehicle: list[str], cfg: dict) -> None:
    """Wheel speed from rpm and the identified radius; road-wheel angle from the steering wheel (per-car offset)."""
    sig["v"] = sig["rpm"] * (2 * np.pi / 60.0) * cfg["wheel_radius_m"]
    off = cfg["steer_offset_by_vehicle"]
    ok = np.array([off.get(v, {}).get("kinematic", cfg["kinematic"]["steer_offset_rad"]) for v in vehicle])[:, None]
    od = np.array([off.get(v, {}).get("dynamic", cfg["dynamic"]["steer_offset_rad"]) for v in vehicle])[:, None]
    sig["d_kin"] = (sig["steer"] - ok) / cfg["kinematic"]["steering_ratio"]
    sig["d_dyn"] = (sig["steer"] - od) / cfg["dynamic"]["steering_ratio"]


def load(names: list[str], scenes_dir: Path = DEFAULT_SCENES, cfg: dict | None = None) -> Batch:
    cfg = cfg or vehicle_config()
    man = json.loads((scenes_dir / "manifest.json").read_text())["kept"]
    tabs = [np.loadtxt(scenes_dir / f"{n}.csv", delimiter=",", skiprows=1, ndmin=2) for n in names]
    T = max(len(a) for a in tabs)
    S = len(tabs)
    out = np.empty((len(SIGNALS), S, T))
    for i, a in enumerate(tabs):
        m = len(a)
        out[:, i, :m] = a.T
        out[:, i, m:] = a[-1][:, None]
        out[0, i, m:] = a[-1, 0] + 0.02 * np.arange(1, T - m + 1)
    sig = {k: out[j] for j, k in enumerate(SIGNALS)}
    vehicle = [man[n]["vehicle"] for n in names]
    derive(sig, vehicle, cfg)
    return Batch(list(names), vehicle, [man[n]["location"] for n in names], np.array([len(a) for a in tabs]), sig)


def from_arrays(rows: list[np.ndarray], vehicle: list[str], cfg: dict, names: list[str] | None = None) -> Batch:
    """Batch from in-memory tables with the columns of tools/mpred/extract.py (used by the tests)."""
    T = max(len(a) for a in rows)
    out = np.empty((len(SIGNALS), len(rows), T))
    for i, a in enumerate(rows):
        m = len(a)
        out[:, i, :m] = a.T
        out[:, i, m:] = a[-1][:, None]
        out[0, i, m:] = a[-1, 0] + 0.02 * np.arange(1, T - m + 1)
    sig = {k: out[j] for j, k in enumerate(SIGNALS)}
    derive(sig, vehicle, cfg)
    names = names or [f"synthetic-{i}" for i in range(len(rows))]
    return Batch(names, vehicle, ["synthetic"] * len(rows), np.array([len(a) for a in rows]), sig)


def start_points(b: Batch, stride: int = STRIDE, min_speed: float = MIN_SPEED, future_s: float = FUTURE_S,
                 history: int = HISTORY) -> tuple[np.ndarray, np.ndarray]:
    """(scene index, sample index) of every start point: in drive from k to t_k + future_s, moving, warm."""
    si, ki = [], []
    for s in range(b.S):
        n = int(b.n[s])
        t = b["t"][s, :n]
        drive = b["drive"][s, :n] > 0.5
        ks = np.arange(history, n, stride)
        end = np.searchsorted(t, t[ks] + future_s, side="left")
        ok = (end < n) & (b["v"][s, ks] >= min_speed)
        # drive gear from k to the end of the window: cumulative count of non-drive samples
        nd = np.concatenate([[0], np.cumsum(~drive)])
        endc = np.minimum(end, n - 1)
        ok &= (nd[endc + 1] - nd[ks]) == 0
        si.append(np.full(ok.sum(), s))
        ki.append(ks[ok])
    return np.concatenate(si).astype(np.int64), np.concatenate(ki).astype(np.int64)


def interp_pose(b: Batch, si: np.ndarray, times: np.ndarray, keys=("x", "y", "yaw", "v_pose")) -> np.ndarray:
    """Recorded signals at arbitrary times (N x H) by linear interpolation, per scene. Returns (N, H, len(keys))."""
    out = np.empty(times.shape + (len(keys),))
    for s in np.unique(si):
        sel = si == s
        n = int(b.n[s])
        t = b["t"][s, :n]
        for j, k in enumerate(keys):
            out[sel, :, j] = np.interp(times[sel], t, b[k][s, :n])
    return out


def truth(b: Batch, si: np.ndarray, ki: np.ndarray, horizons=HORIZONS, offset_s: np.ndarray | float = 0.0) -> np.ndarray:
    """Recorded x, y, yaw, speed at t_k + offset + h for each horizon h: (N, H, 4)."""
    t0 = b["t"][si, ki] + offset_s
    return interp_pose(b, si, t0[:, None] + np.asarray(horizons)[None, :])


def hours(b: Batch) -> float:
    return float(sum(b["t"][s, int(b.n[s]) - 1] - b["t"][s, 0] for s in range(b.S)) / 3600.0)
