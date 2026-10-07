"""Shared fixtures: a small synthetic scene set (circle and straight drives) and the vdyn_eval binary."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def find_binary() -> Path | None:
    for c in (os.environ.get("VDYN_EVAL"), "stack/vdyn/vdyn_eval", ROOT / "bazel-bin" / "stack" / "vdyn" / "vdyn_eval"):
        if c and Path(c).exists():
            return Path(c).resolve()
    return None


def write_scene(path: Path, v: float, steer_sw: float, ratio: float = 15.0, L: float = 2.588, r_w: float = 0.3) -> float:
    r = v * math.tan(steer_sw / ratio) / L
    lines = ["t_s,x,y,yaw,v_pose,steer_sw,r_imu,rpm_rear,drive"]
    n = 1000
    for i in range(n):
        t = 0.02 * i
        x, y = (v / r * math.sin(r * t), v / r * (1 - math.cos(r * t))) if r else (v * t, 0.0)
        lines.append(f"{t:.2f},{x:.4f},{y:.4f},{r * t:.6f},{v:.4f},{steer_sw:.6f},{r:.6f},{v * 60 / (2 * math.pi * r_w):.3f},1")
    path.write_text("\n".join(lines) + "\n")
    return 0.02 * (n - 1)


@pytest.fixture()
def scene_set(tmp_path: Path) -> dict:
    d = tmp_path / "scenes"
    d.mkdir()
    kept = {}
    for i, (v, sw) in enumerate([(8.0, 1.0), (5.0, -1.5), (10.0, 0.0), (6.0, 0.5)]):
        name = f"scene-90{i:02d}"
        dur = write_scene(d / f"{name}.csv", v, sw)
        kept[name] = {"vehicle": "n008" if i % 2 == 0 else "n015", "duration_s": dur}
    (d / "manifest.json").write_text(json.dumps({"kept": kept}))
    cfg = tmp_path / "zoe.json"
    cfg.write_text(json.dumps({
        "wheelbase_m": 2.588, "wheel_radius_m": 0.3,
        "kinematic": {"steering_ratio": 15.0, "steer_offset_rad": 0.0},
        "dynamic": {"steering_ratio": 15.0, "steer_offset_rad": 0.0, "lf_m": 1.165, "mass_kg": 1650.0,
                    "yaw_inertia_kgm2": 2735.0, "cf_npr": 80000.0, "cr_npr": 120000.0, "min_speed_mps": 1.0},
        "steer_offset_by_vehicle": {"n008": {"kinematic": 0.0, "dynamic": 0.0}, "n015": {"kinematic": 0.0, "dynamic": 0.0}}}))
    binary = find_binary()
    if binary is None:
        pytest.skip("vdyn_eval not built (bazel build //stack/vdyn:vdyn_eval)")
    return {"dir": d, "config": cfg, "binary": binary, "names": sorted(kept)}
