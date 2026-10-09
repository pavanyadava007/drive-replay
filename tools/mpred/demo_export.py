"""Data for the interactive AR-HUD demo (site/hud.html): the real fitted predictors on a few test scenes.

The demo shows no number that this module did not compute with the benchmark code path:

  - scenes, start points and truth: tools/mpred/data.py (test split, stride 5, the same start points as
    docs/MOTION_PREDICTION.md);
  - predictions: tools/mpred/predictors.predict_all with configs/mpred/predictors.json (the exported weights the
    report and the C++ library use), on the whole test split, so the filters see exactly the benchmark input;
  - noise: tools/mpred/experiment.perturb with the seed of experiment.noise_sweep for the same level, applied to the
    whole test split, so the draws are identical to the report's noise sweep;
  - errors: tools/mpred/metrics.errors.

One extra row that the report does not have: "no compensation", the overlay drawn from the newest pose itself (the
pose at the start sample, with the injected noise at that level) without any prediction. Its errors use the same
metric code.

Stored resolution: longitudinal / lateral error 1e-5 m, heading error 1e-7 rad (the page derives the position
error |(lon, lat)| and the 20 m overlay offset lat + 20 sin(head) from them, the formulas of metrics.py). Scene
means are stored at full precision from the unrounded errors and are checked against the cached benchmark results
(runs/mpred/results.pkl) when that file exists.

Scene choice: objective rules on the test split, applied in order; a rule whose best scene is already taken uses
its next best (see RULES).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.mpred import data as D
from tools.mpred import experiment as X
from tools.mpred import metrics as E
from tools.mpred import predictors as PR

GUI_METHODS = ("cv", "ctrv", "dynamic", "ekf", "mlp", "hybrid_ridge", "hybrid_mlp")
ROWS = ("stale", *GUI_METHODS)
SHORT = {"stale": "no compensation", "cv": "CV", "ctrv": "CTRV", "dynamic": "dynamic single track", "ekf": "EKF-CTRA",
         "mlp": "MLP", "hybrid_ridge": "EKF + ridge", "hybrid_mlp": "EKF + MLP"}
LEVELS = (0.0, 1.0, 2.0)
N_H = 6  # 50, 100, 150, 200, 300, 500 ms (the 1 s context horizon is left out)
HMS = [int(round(h * 1000)) for h in D.HORIZONS[:N_H]]
H200 = HMS.index(200)
Q_POS, Q_HEAD = 1e5, 1e7  # stored as integers: 1e-5 m, 1e-7 rad
OV_D = 20.0

RULES = (
    ("turn_in", "the start point with the largest yaw-rate change over the next 0.5 s at >= 6 m/s wheel speed "
                "(the rule of the report's scene trace)"),
    ("turning", "most start points labelled 'yaw-rate change (turn in / out)'"),
    ("curve", "most start points labelled 'steady curve'"),
    ("braking", "most start points labelled 'braking'"),
    ("accelerating", "most start points labelled 'accelerating'"),
    ("fast", "highest mean wheel speed over the start points"),
    ("low_speed", "most start points labelled 'low speed (< 5 m/s)'"),
    ("ctrv_hard", "largest scene mean |20 m overlay offset| of CTRV at 200 ms"),
    ("least_gain", "largest scene mean |20 m overlay offset| of EKF + MLP minus CTRV at 200 ms (the scene where the "
                   "hybrid helps least)"),
    ("noise", "largest growth of the EKF + MLP scene mean |20 m overlay offset| at 200 ms from noise level 0 to 1"),
)


@dataclass
class LevelResult:
    level: float
    preds: dict[str, np.ndarray]  # method -> (N, 7, 4) on all test start points
    err: dict[str, dict[str, np.ndarray]]  # method -> metric -> (N, 7)


def level_batch(b: D.Batch, vcfg: dict, level: float):
    """The perturbed batch of experiment.noise_sweep for this level (same seed, same draws)."""
    rng = np.random.default_rng(X.SEED + int(level * 100))
    nb, extra, _ = X.perturb(b, vcfg, {k: v * level for k, v in X.BASE_SIGMA.items()}, rng)
    return nb, extra


def stale(b: D.Batch, si: np.ndarray, ki: np.ndarray, n_h: int = len(D.HORIZONS)) -> np.ndarray:
    """No compensation: the newest pose (and wheel speed) held for every horizon. (N, n_h, 4)."""
    p = np.stack([b["x"][si, ki], b["y"][si, ki], b["yaw"][si, ki], b["v"][si, ki]], 1)
    return np.repeat(p[:, None, :], n_h, axis=1)


def run_level(b: D.Batch, si: np.ndarray, ki: np.ndarray, tru: np.ndarray, conf: dict, vcfg: dict,
              level: float) -> LevelResult:
    nb, extra = level_batch(b, vcfg, level)
    preds = PR.predict_all(nb, si, ki, conf, vcfg, GUI_METHODS, extra_std=extra)
    preds["stale"] = stale(nb, si, ki)
    return LevelResult(level, preds, {m: E.errors(p, tru) for m, p in preds.items()})


def scene_mean(e: np.ndarray, sel: np.ndarray) -> float:
    return float(np.abs(e[sel]).mean())


def select_scenes(sp: X.Split, lv0: LevelResult, lv1: LevelResult) -> list[dict]:
    b, si, ki = sp.b, sp.si, sp.ki
    lab = X.slice_labels(sp)
    S = b.S
    v = b["v"][si, ki]
    fut_r = D.interp_pose(b, si, (b["t"][si, ki] + 0.5)[:, None], keys=("r",))[:, 0, 0]
    has = np.bincount(si, minlength=S) > 0

    def per_scene(values: np.ndarray) -> np.ndarray:
        n = np.bincount(si, minlength=S)
        return np.where(n > 0, np.bincount(si, weights=values, minlength=S) / np.maximum(n, 1), np.nan)

    def count(slice_name: str) -> np.ndarray:
        c = np.bincount(si, weights=(lab == X.SLICES.index(slice_name)).astype(float), minlength=S)
        return np.where(has, c, np.nan)

    ov = lambda lv, m: np.abs(lv.err[m]["ov20"][:, H200])
    turn_score = np.where(v >= 6.0, np.abs(fut_r - b["r"][si, ki]), -1.0)
    best_turn = np.full(S, -np.inf)
    np.maximum.at(best_turn, si, turn_score)
    best_turn[~has] = np.nan
    scores = {
        "turn_in": best_turn,
        "turning": count("yaw-rate change (turn in / out)"),
        "curve": count("steady curve"),
        "braking": count("braking"),
        "accelerating": count("accelerating"),
        "fast": per_scene(v),
        "low_speed": count("low speed (< 5 m/s)"),
        "ctrv_hard": per_scene(ov(lv0, "ctrv")),
        "least_gain": per_scene(ov(lv0, "hybrid_mlp")) - per_scene(ov(lv0, "ctrv")),
        "noise": per_scene(ov(lv1, "hybrid_mlp")) - per_scene(ov(lv0, "hybrid_mlp")),
    }
    scores = {k: np.where(np.isfinite(v_), v_, -np.inf) for k, v_ in scores.items()}
    taken: list[int] = []
    out = []
    for key, rule in RULES:
        order = np.argsort(-scores[key], kind="stable")
        s = int(next(o for o in order if o not in taken and np.isfinite(scores[key][o])))
        taken.append(s)
        sel = si == s
        out.append({"key": key, "rule": rule, "scene_index": s, "score": float(scores[key][s]),
                    "label": scene_label(key, b, si, ki, s, sel, turn_score, lab)})
    return out


def scene_label(key: str, b: D.Batch, si, ki, s: int, sel: np.ndarray, turn_score: np.ndarray, lab: np.ndarray) -> str:
    v = b["v"][si[sel], ki[sel]]
    r = b["r"][si[sel], ki[sel]]
    speed = f"{np.median(v):.0f} m/s"
    if key == "turn_in":
        i = np.flatnonzero(sel)[int(np.argmax(turn_score[sel]))]
        r0 = b["r"][si[i], ki[i]]
        r1 = D.interp_pose(b, si[i:i + 1], (b["t"][si[i], ki[i]] + 0.5) * np.ones((1, 1)), keys=("r",))[0, 0, 0]
        side = "left" if (r1 if abs(r1) > abs(r0) else r0) > 0 else "right"  # yaw counter-clockwise: r > 0 turns left
        return f"Sharp {side} turn, {b['v'][si[i], ki[i]]:.0f} m/s"
    if key == "curve":
        m = lab[sel] == X.SLICES.index("steady curve")
        side = "left" if np.mean(r[m]) > 0 else "right"
        return f"Steady {side} curve, {np.median(v[m]):.0f} m/s"
    names = {"turning": "Turning in and out", "braking": "Braking", "accelerating": "Accelerating",
             "fast": "Fastest scene", "low_speed": "Low speed", "ctrv_hard": "Hardest for CTRV",
             "least_gain": "Least EKF + MLP gain", "noise": "Most noise-sensitive (EKF + MLP)"}
    return f"{names[key]}, {speed}"


def q(a: np.ndarray, scale: float) -> list[int]:
    return [int(x) for x in np.rint(np.asarray(a, dtype=float) * scale).ravel()]


def export_scene(sp: X.Split, pick: dict, levels: list[LevelResult]) -> dict:
    b, si, ki = sp.b, sp.si, sp.ki
    s = pick["scene_index"]
    sel = np.flatnonzero(si == s)
    n = int(b.n[s])
    t = b["t"][s, :n]
    x0, y0 = float(b["x"][s, 0]), float(b["y"][s, 0])
    doc = {
        "scene": b.names[s], "vehicle": b.vehicle[s], "location": b.location[s], "label": pick["label"],
        "rule": pick["rule"], "key": pick["key"],
        "track": {"t_ms": q(t - t[0], 1e3), "x_mm": q(b["x"][s, :n] - x0, 1e3), "y_mm": q(b["y"][s, :n] - y0, 1e3),
                  "yaw_1e5": q(b["yaw"][s, :n], 1e5), "v_cms": q(b["v"][s, :n], 1e2), "r_1e4": q(b["r"][s, :n], 1e4)},
        "t0": float(t[0]),
        "starts": {"k": [int(k) for k in ki[sel]], "t_ms": q(b["t"][s, ki[sel]] - t[0], 1e3)},
        "levels": {},
    }
    for lv in levels:
        L = {"rows": {}, "means": {}}
        for m in ROWS:
            e = lv.err[m]
            L["rows"][m] = {"lon": q(e["lon"][sel, :N_H], Q_POS), "lat": q(e["lat"][sel, :N_H], Q_POS),
                            "head": q(e["head"][sel, :N_H], Q_HEAD)}
            L["means"][m] = {k: [scene_mean(e[k][:, h], sel) for h in range(N_H)] for k in ("pos", "head", "ov20")}
        doc["levels"][f"{lv.level:g}"] = L
    return doc


def check_against_benchmark(results_pkl: Path, sp: X.Split, levels: list[LevelResult]) -> dict:
    """Per-scene mean |error| of every exported method, level and horizon against the cached benchmark sums."""
    import pickle

    R = pickle.loads(results_pkl.read_bytes())
    if not (np.array_equal(R["test"]["si"], sp.si) and np.array_equal(R["test"]["ki"], sp.ki)):
        raise RuntimeError("cached benchmark has different start points")
    worst, n = 0.0, 0
    for lv in levels:
        cached = R["noise"]["all"][str(float(lv.level))]
        for m in GUI_METHODS:
            for metric in ("pos", "head", "ov20"):
                for h in range(N_H):
                    sums = E.scene_sums(lv.err[m][metric][:, h], sp.si, sp.b.S)
                    ref = cached[m][metric][h]
                    ok = ref[:, 0] > 0
                    if not np.array_equal(sums[:, 0], ref[:, 0]):
                        raise RuntimeError(f"start-point counts differ for {m}")
                    d = np.abs(sums[ok, 1] / sums[ok, 0] - ref[ok, 1] / ref[ok, 0])
                    worst = max(worst, float(d.max()))
                    n += int(ok.sum())
    return {"compared_scene_means": n, "max_abs_diff": worst}


def write_all(out_dir: Path, scenes_dir: Path = D.DEFAULT_SCENES, results_pkl: Path | None = None, log=print) -> dict:
    vcfg = D.vehicle_config()
    conf = PR.load_config()
    sp = X.make_split("test", D.STRIDE, scenes_dir, vcfg)
    levels = []
    for lev in LEVELS:
        levels.append(run_level(sp.b, sp.si, sp.ki, sp.tru, conf, vcfg, lev))
        log(f"noise level {lev:g}: {sp.n} start points")
    check = None
    if results_pkl is not None and results_pkl.exists():
        check = check_against_benchmark(results_pkl, sp, levels)
        log(f"check against {results_pkl}: {check}")
        if check["max_abs_diff"] > 1e-9:
            raise RuntimeError(f"exported errors differ from the benchmark: {check}")
    picks = select_scenes(sp, levels[0], levels[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"_comment": "Generated by scripts/export_mpred_demo.py from the test split of the nuScenes CAN bus "
                         "expansion (CC BY-NC-SA 4.0); do not edit by hand.",
             "horizons_ms": HMS, "levels": [f"{lv:g}" for lv in LEVELS], "rows": list(ROWS), "short": SHORT,
             "labels": {m: PR.LABEL.get(m, "no compensation: the newest pose, no prediction") for m in ROWS},
             "noise_sigma_level1": X.BASE_SIGMA, "overlay_distance_m": OV_D,
             "resolution": {"lon_lat_m": 1 / Q_POS, "head_rad": 1 / Q_HEAD},
             "stride_samples": D.STRIDE, "predictor_meta": json.loads(PR.CONFIG.read_text()).get("meta", {}),
             "benchmark_check": check, "scenes": []}
    for p in picks:
        doc = export_scene(sp, p, levels)
        name = f"{doc['scene']}.json"
        (out_dir / name).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        index["scenes"].append({"file": name, "scene": doc["scene"], "label": doc["label"], "rule": doc["rule"],
                                "key": doc["key"], "vehicle": doc["vehicle"], "location": doc["location"],
                                "start_points": len(doc["starts"]["k"]),
                                "duration_s": doc["track"]["t_ms"][-1] / 1000.0})
        log(f"{doc['scene']:>11}  {p['key']:<13} {doc['label']}")
    (out_dir / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    return index
