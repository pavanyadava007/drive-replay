"""Shared fixtures: synthetic drives with exact signals, a small vehicle config, a predictor config."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from tools.mpred import data as D
from tools.mpred import models as M
from tools.mpred import predictors as PR

ROOT = Path(__file__).resolve().parents[3]
RADIUS = 0.3
VCFG = {"wheelbase_m": 2.588, "wheel_radius_m": RADIUS,
        "kinematic": {"steering_ratio": 15.0, "steer_offset_rad": 0.0},
        "dynamic": {"steering_ratio": 15.0, "steer_offset_rad": 0.0, "lf_m": 1.165, "mass_kg": 1650.0,
                    "yaw_inertia_kgm2": 2735.0, "cf_npr": 80000.0, "cr_npr": 120000.0, "min_speed_mps": 1.0},
        "steer_offset_by_vehicle": {"n008": {"kinematic": 0.0, "dynamic": 0.0}, "n015": {"kinematic": 0.0, "dynamic": 0.0}}}


def drive(n: int = 400, dt: float = 0.02, v0: float = 8.0, accel: float = 0.0, yaw_rate=0.0, x0=100.0, y0=-50.0,
          yaw0=0.4) -> np.ndarray:
    """Exact signals of a drive (columns of tools/mpred/extract.py). yaw_rate: constant or callable of t."""
    t = np.arange(n) * dt
    v = np.maximum(v0 + accel * t, 0.0)
    r = np.array([yaw_rate(tt) for tt in t]) if callable(yaw_rate) else np.full(n, float(yaw_rate))
    # integrate finely so the pose is consistent with the signals
    sub = 20
    x, y, yaw = x0, y0, yaw0
    rows = []
    for i in range(n):
        if i:
            for j in range(sub):
                tt = t[i - 1] + (j + 0.5) * dt / sub
                vv = max(v0 + accel * tt, 0.0)
                rr = yaw_rate(tt) if callable(yaw_rate) else yaw_rate
                yaw_mid = yaw + 0.5 * rr * dt / sub
                x += vv * math.cos(yaw_mid) * dt / sub
                y += vv * math.sin(yaw_mid) * dt / sub
                yaw += rr * dt / sub
        steer = 15.0 * math.atan(r[i] * 2.588 / max(v[i], 0.1))
        rpm = v[i] / (2 * math.pi / 60.0 * RADIUS)
        rows.append([t[i], x, y, yaw, v[i], 1, steer, r[i], rpm, accel, v[i] * r[i], accel])
    return np.array(rows)


@pytest.fixture()
def vcfg() -> dict:
    return json.loads(json.dumps(VCFG))


def random_spec(name: str, features: str, uses_state: bool, residual: bool, hidden: int | None, rng) -> PR.LearnedSpec:
    nin = (18 if features == "phys" else 33) + (3 if (uses_state and features == "raw") else 0)
    sizes = [nin] + ([hidden] if hidden else []) + [28]
    layers = [(rng.normal(0, 0.05, (a, b)), rng.normal(0, 0.01, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
    xm, xs = rng.normal(0, 1, nin), rng.uniform(0.5, 2.0, nin)
    if features == "phys":
        xm[0], xs[0] = 0.0, 1.0
    ym, ys = rng.normal(0, 0.01, 28), rng.uniform(0.01, 0.1, 28)
    return PR.LearnedSpec(name, features, uses_state, residual, xm, xs, ym, ys, layers)


@pytest.fixture()
def conf() -> dict:
    rng = np.random.default_rng(3)
    specs = {"ridge": random_spec("ridge", "phys", False, False, None, rng),
             "hybrid_ridge": random_spec("hybrid_ridge", "phys", True, True, None, rng),
             "mlp": random_spec("mlp", "raw", False, False, 8, rng),
             "hybrid_mlp": random_spec("hybrid_mlp", "raw", True, True, 8, rng)}
    kf = {"kf_cv": {"order": 2, "q": 1.0, "r_pos": 0.01}, "kf_ca": {"order": 3, "q": 3.0, "r_pos": 0.01}}
    ekf = M.EkfParams(s_jerk=2.0, s_rdot=0.1, r_scale=3.0, r_xy=0.001, r_psi=1e-4)
    return {"ctra": {"accel": "slope25"}, "kf": kf, "ekf": ekf.as_dict(), "ekf_params": ekf,
            "kf_params": {k: M.KfParams(**v) for k, v in kf.items()}, "specs": specs}


def find_binary() -> Path | None:
    for c in (os.environ.get("MPRED_EVAL"), "stack/mpred/mpred_eval", ROOT / "bazel-bin" / "stack" / "mpred" / "mpred_eval"):
        if c and Path(c).exists():
            return Path(c).resolve()
    return None


def batch_of(*tables: np.ndarray, vcfg: dict = VCFG) -> D.Batch:
    return D.from_arrays(list(tables), ["n008"] * len(tables), vcfg)
