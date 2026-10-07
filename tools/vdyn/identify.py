"""Identify Renault Zoe single-track parameters from the training scenes and write configs/vdyn/renault_zoe.json.

    python tools/vdyn/identify.py --scenes ~/workspace/drive-replay-data/can_scenes [--eval bazel-bin/stack/vdyn/vdyn_eval]

Only training scenes are used for fitting; validation scenes only choose between a few assumed values of the
centre-of-gravity position (a prior is kept unless another value is clearly better); test scenes are never read.
What is identified, and how:

  wheel radius       least squares, pose speed against mean rear wheel rpm (rear wheels are not driven)
  kinematic model    steering ratio (shared) and steering-wheel offset (one per car): nonlinear least squares
                     (Gauss-Newton) of r_imu = v tan((sw - off) / i) / L on quasi-steady samples
  linear model       steering ratio, offset per car and understeer gradient K: r_imu = v (sw - off) / (i L (1 + K v^2));
                     K by a fine grid search with the other two by linear least squares at each K
  cornering stiffness  with K fixed, the rear stiffness Cr is chosen by grid search to minimise the yaw-rate
                     error of the simulated dynamic model (the C++ vdyn_eval binary) on the training scenes;
                     Cf follows from K. The CG position lf is chosen on the validation scenes from 3 assumed values.

Assumed, not identified (cited or stated in docs/VEHICLE_DYNAMICS.md): wheelbase 2.588 m, mass 1650 kg,
yaw inertia m lf lr.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WHEELBASE_M = 2.588      # Renault Zoe, https://en.wikipedia.org/wiki/Renault_Zoe (2,588 mm)
KERB_KG = 1468.0         # same source, kerb weight
MASS_KG = 1650.0         # assumption: kerb + sensor rig + driver and operator
LF_FRACTIONS = (0.40, 0.45, 0.50)  # CG to front axle as a fraction of L (55-60 % front axle load is typical for FWD)
LF_PRIOR = 0.45          # kept unless another candidate is more than LF_MARGIN better on the validation scenes
LF_MARGIN = 0.01
CR_GRID = [40e3, 60e3, 80e3, 100e3, 120e3, 150e3, 180e3, 220e3, 270e3, 330e3, 400e3]
MIN_SPEED = 3.0          # quasi-steady sample filter
MAX_STEER_RATE = 0.3     # rad/s of steering-wheel angle (0.2 s smoothing)
MAX_YAW_ACCEL = 0.1      # rad/s^2
MAX_YAW_RATE = 0.6       # rad/s


def load(scene_dir: Path, names: list[str], vehicle_of: dict[str, str]) -> dict[str, np.ndarray]:
    cols = ["t", "x", "y", "yaw", "v_pose", "sw", "r", "rpm", "drive"]
    tabs = [np.loadtxt(scene_dir / f"{n}.csv", delimiter=",", skiprows=1) for n in names]
    out = {c: np.concatenate([t[:, i] for t in tabs]) for i, c in enumerate(cols)}
    out["vehicle"] = np.concatenate([np.full(len(t), vehicle_of[n]) for n, t in zip(names, tabs, strict=True)])

    def rate(col: int) -> np.ndarray:  # per scene, 0.2 s central differences; edges never count as steady
        parts = []
        for t in tabs:
            v, tt = t[:, col], t[:, 0]
            d = np.full_like(v, np.inf)
            d[5:-5] = (v[10:] - v[:-10]) / (tt[10:] - tt[:-10])
            parts.append(d)
        return np.concatenate(parts)

    out["sw_rate"] = rate(5)
    out["r_rate"] = rate(6)
    return out


def quasi_steady(d: dict, v: np.ndarray) -> np.ndarray:
    return ((d["drive"] == 1) & (v >= MIN_SPEED) & (np.abs(d["sw_rate"]) < MAX_STEER_RATE)
            & (np.abs(d["r_rate"]) < MAX_YAW_ACCEL) & (np.abs(d["r"]) < MAX_YAW_RATE))


def onehot(groups: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = sorted(set(groups.tolist()))
    return names, np.column_stack([(groups == g).astype(float) for g in names])


def fit_kinematic(sw, v, r, groups) -> tuple[float, dict[str, float], float]:
    """r = v tan(a (sw - off_g)) / L, Gauss-Newton on (a, off_g) from the small-angle linear solution."""
    names, G = onehot(groups)
    X = np.column_stack([v * sw / WHEELBASE_M, -(v / WHEELBASE_M)[:, None] * G])
    c = np.linalg.lstsq(X, r, rcond=None)[0]
    p = np.concatenate([[c[0]], c[1:] / c[0]])
    for _ in range(30):
        a, offs = p[0], p[1:]
        off = G @ offs
        arg = a * (sw - off)
        sec2 = 1.0 / np.cos(arg) ** 2
        res = r - v * np.tan(arg) / WHEELBASE_M
        J = np.column_stack([v * sec2 * (sw - off) / WHEELBASE_M, -(v * sec2 * a / WHEELBASE_M)[:, None] * G])
        step = np.linalg.lstsq(J, res, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-12:
            break
    a, offs = p[0], p[1:]
    rmse = float(np.sqrt(np.mean((r - v * np.tan(a * (sw - G @ offs)) / WHEELBASE_M) ** 2)))
    return 1.0 / a, dict(zip(names, offs.tolist(), strict=True)), rmse


def fit_linear(sw, v, r, groups) -> tuple[float, dict[str, float], float, float]:
    """r = v (sw - off_g) / (i L (1 + K v^2)): K by grid search, (1/i, off_g / i) by linear least squares."""
    names, G = onehot(groups)

    def solve(K: float) -> tuple[float, np.ndarray]:
        g = v / (WHEELBASE_M * (1.0 + K * v * v))
        X = np.column_stack([g * sw, -g[:, None] * G])
        c = np.linalg.lstsq(X, r, rcond=None)[0]
        return float(np.sum((X @ c - r) ** 2)), c

    grid = np.arange(-0.002, 0.012, 1e-4)
    K = min(grid, key=lambda k: solve(k)[0])
    fine = np.arange(K - 1e-4, K + 1e-4, 1e-6)
    K = float(min(fine, key=lambda k: solve(k)[0]))
    sse, c = solve(K)
    return 1.0 / c[0], dict(zip(names, (c[1:] / c[0]).tolist(), strict=True)), K, math.sqrt(sse / len(r))


def stiffness_from(K: float, lf: float, cr: float) -> float:
    """Cf such that m (lr/Cf - lf/Cr) / L^2 = K."""
    lr = WHEELBASE_M - lf
    denom = K * WHEELBASE_M**2 / MASS_KG + lf / cr
    return lr / denom if denom > 0 else math.nan


def run_eval(binary: str, config: Path, scene_dir: Path, names: list[str], vehicle_of: dict[str, str], jobs: int) -> dict:
    """Pooled yaw-rate RMSE of the three yaw predictors over the scenes (C++ vdyn_eval --yaw-only)."""
    chunks = []
    for veh in sorted({vehicle_of[n] for n in names}):
        mine = [n for n in names if vehicle_of[n] == veh]
        chunks += [(veh, mine[i::jobs]) for i in range(jobs) if mine[i::jobs]]

    def one(job: tuple[str, list[str]]) -> list[dict]:
        veh, chunk = job
        cmd = [binary, "--config", str(config), "--yaw-only", "--vehicle", veh]
        for n in chunk:
            cmd += ["--scene", str(scene_dir / f"{n}.csv")]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return [json.loads(line) for line in out.splitlines()]

    with ThreadPoolExecutor(jobs) as ex:
        rows = [r for part in ex.map(one, chunks) for r in part]
    tot = {}
    for m in ("kinematic", "steady_state", "dynamic"):
        n = sum(r["yaw"][m]["all"][0] for r in rows if "yaw" in r)  # scenes that never move have no samples
        ss = sum(r["yaw"][m]["all"][2] for r in rows if "yaw" in r)
        tot[m] = math.sqrt(ss / n)
    return tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=ROOT / "configs" / "vdyn" / "split.json")
    ap.add_argument("--eval", default=str(ROOT / "bazel-bin" / "stack" / "vdyn" / "vdyn_eval"))
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "vdyn" / "renault_zoe.json")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    split = json.loads(args.split.read_text())
    manifest = json.loads((args.scenes / "manifest.json").read_text())["kept"]
    train, val = split["train"], split["val"]
    vehicle_of = {n: manifest[n]["vehicle"] for n in train + val}
    d = load(args.scenes, train, vehicle_of)

    # 1. wheel radius
    m = (d["drive"] == 1) & (d["v_pose"] > MIN_SPEED)
    k = float(np.sum(d["v_pose"][m] * d["rpm"][m]) / np.sum(d["rpm"][m] ** 2))
    wheel_radius = k * 60.0 / (2 * math.pi)
    v = d["rpm"] * k
    speed_rmse = float(np.sqrt(np.mean((v[m] - d["v_pose"][m]) ** 2)))

    # 2./3. steady-state fits on quasi-steady samples (speed from the wheels, as the models get it)
    q = quasi_steady(d, v)
    one_group = np.full(int(q.sum()), "all")
    ik, offk, rmse_k = fit_kinematic(d["sw"][q], v[q], d["r"][q], d["vehicle"][q])
    il, offl, K, rmse_l = fit_linear(d["sw"][q], v[q], d["r"][q], d["vehicle"][q])
    _, offk_shared, _ = fit_kinematic(d["sw"][q], v[q], d["r"][q], one_group)
    _, offl_shared, K_shared, rmse_shared = fit_linear(d["sw"][q], v[q], d["r"][q], one_group)
    per_vehicle = {}
    for veh in sorted(set(d["vehicle"].tolist())):  # diagnostics: everything fitted per car
        qv = q & (d["vehicle"] == veh)
        i_, o_, K_, e_ = fit_linear(d["sw"][qv], v[qv], d["r"][qv], np.full(int(qv.sum()), veh))
        per_vehicle[veh] = {"scenes": sum(1 for n in train if vehicle_of[n] == veh), "samples": int(qv.sum()),
                            "steering_ratio": round(i_, 3), "steer_offset_rad": round(o_[veh], 5),
                            "understeer_K_s2pm2": round(K_, 6), "rmse_rps": round(e_, 5)}

    base = {
        "wheelbase_m": WHEELBASE_M, "wheel_radius_m": round(wheel_radius, 5),
        "kinematic": {"steering_ratio": round(ik, 4), "steer_offset_rad": round(offk_shared["all"], 6)},
        "dynamic": {"steering_ratio": round(il, 4), "steer_offset_rad": round(offl_shared["all"], 6), "lf_m": 0.0,
                    "mass_kg": MASS_KG, "yaw_inertia_kgm2": 0.0, "cf_npr": 0.0, "cr_npr": 0.0, "min_speed_mps": 1.0},
        "steer_offset_by_vehicle": {veh: {"kinematic": round(offk[veh], 6), "dynamic": round(offl[veh], 6)} for veh in offk},
        "fcw_baseline": {"yaw_rate_decay_s": 1.0, "max_curvature": 0.2},
    }

    # 4. cornering stiffness (train) and CG position (val), through the C++ model
    search = []
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "base.json"
        for frac in LF_FRACTIONS:
            lf = frac * WHEELBASE_M
            lr = WHEELBASE_M - lf
            iz = MASS_KG * lf * lr
            best = None
            for cr in CR_GRID:
                cf = stiffness_from(K, lf, cr)
                if not cf > 0:
                    continue
                trial = json.loads(json.dumps(base))
                trial["dynamic"].update({"lf_m": lf, "yaw_inertia_kgm2": iz, "cf_npr": cf, "cr_npr": cr})
                cfg.write_text(json.dumps(trial))
                e = run_eval(args.eval, cfg, args.scenes, train, vehicle_of, args.jobs)["dynamic"]
                search.append({"lf_fraction": frac, "cr_npr": cr, "cf_npr": round(cf, 1), "train_dynamic_rmse_rps": round(e, 6)})
                if best is None or e < best[0]:
                    best = (e, cr, cf, trial)
            cfg.with_name("v.json").write_text(json.dumps(best[3]))
            ev = run_eval(args.eval, cfg.with_name("v.json"), args.scenes, val, vehicle_of, args.jobs)
            search.append({"lf_fraction": frac, "chosen_cr_npr": best[1],
                           "val_rmse_rps": {k2: round(x, 6) for k2, x in ev.items()}})
            results[frac] = (ev["dynamic"], frac, best)
    best_val = min(r[0] for r in results.values())
    chosen = results[LF_PRIOR] if results[LF_PRIOR][0] <= best_val * (1 + LF_MARGIN) else min(results.values())
    _, frac, (train_e, cr, cf, trial) = chosen
    lf = frac * WHEELBASE_M
    trial["dynamic"].update({"lf_m": round(lf, 4), "yaw_inertia_kgm2": round(MASS_KG * lf * (WHEELBASE_M - lf), 1),
                             "cf_npr": round(cf, 1), "cr_npr": cr})
    out = {
        "_comment": "Renault Zoe (nuScenes n008 Boston, n015 Singapore). Generated by tools/vdyn/identify.py from the "
                    "training scenes of configs/vdyn/split.json; do not edit by hand. Read by stack/vdyn (C++).",
        **trial,
        "identification": {
            "train_scenes": len(train), "train_samples": int(len(d["t"])), "quasi_steady_samples": int(q.sum()),
            "filters": {"min_speed_mps": MIN_SPEED, "max_steer_rate_rps": MAX_STEER_RATE, "max_yaw_accel_rps2": MAX_YAW_ACCEL,
                        "max_yaw_rate_rps": MAX_YAW_RATE, "drive_gear_only": True},
            "speed_rmse_mps": round(speed_rmse, 4),
            "kinematic_fit_rmse_rps": round(rmse_k, 6), "linear_fit_rmse_rps": round(rmse_l, 6),
            "linear_fit_rmse_one_shared_offset_rps": round(rmse_shared, 6), "understeer_K_one_shared_offset": round(K_shared, 7),
            "understeer_K_s2pm2": round(K, 7), "understeer_gradient_rad_per_mps2": round(K * WHEELBASE_M, 6),
            "understeer_gradient_deg_per_g": round(math.degrees(K * WHEELBASE_M * 9.81), 3),
            "cg_lf_fraction_chosen_on_val": frac,
            "cg_rule": f"prior {LF_PRIOR} L unless another candidate is more than {LF_MARGIN:.0%} better on val",
            "train_dynamic_yaw_rmse_rps": round(train_e, 6),
            "per_vehicle_linear_fit": per_vehicle, "stiffness_search": search,
            "assumed": {"wheelbase_m": f"{WHEELBASE_M} (Renault Zoe, Wikipedia)",
                        "mass_kg": f"{MASS_KG} (kerb {KERB_KG} + rig + 2 people)",
                        "yaw_inertia": "m lf lr", "lf_candidates_fraction_of_L": list(LF_FRACTIONS)},
        },
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k2: out[k2] for k2 in ("wheel_radius_m", "kinematic", "dynamic")}, indent=1))
    print(json.dumps({k2: v2 for k2, v2 in out["identification"].items() if k2 != "stiffness_search"}, indent=1))


if __name__ == "__main__":
    main()
