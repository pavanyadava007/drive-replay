"""Regenerates docs/MOTION_PREDICTION.md, its figures (docs/figures/mpred_*.png), configs/mpred/predictors.json and
site/data/mpred.json: the ego motion prediction benchmark for AR-HUD latency compensation.

    python scripts/reproduce_mpred.py [--scenes ~/workspace/drive-replay-data/can_scenes_v2] [--report-only]

Needs: bazel on PATH, the scene tables of tools/mpred/extract.py (nuScenes CAN bus expansion, not in git) and the
v0.3.0 vehicle parameters configs/vdyn/renault_zoe.json. Runs everything in order, seeded: tuning on the validation
split, fitting on the training split, evaluation on the test split, robustness, latency and runtime studies, the
C++ build, Python vs C++ parity and the C++ benchmark. --report-only re-renders the document from the cached
results of the last full run (runs/mpred/results.pkl), for layout work only; the committed document always comes
from a full run. Every number in the document comes from this script. Never edit docs/MOTION_PREDICTION.md by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.mpred import data as D  # noqa: E402
from tools.mpred import experiment as X  # noqa: E402
from tools.mpred import metrics as E  # noqa: E402
from tools.mpred import models as M  # noqa: E402
from tools.mpred import predictors as PR  # noqa: E402
from tools.vdyn import pipeline as P  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SITE = ROOT / "site" / "data"
OUT = ROOT / "runs" / "mpred"
HMS = [int(round(h * 1000)) for h in D.HORIZONS]
H100, H200, H500 = HMS.index(100), HMS.index(200), HMS.index(500)
PLOT = ["cv", "ctrv", "ctra", "dynamic", "ekf", "hybrid_ridge", "hybrid_mlp"]
# fixed categorical slots (reference palette order), one per method, the same in every figure
COLOR = {"cv": "#2a78d6", "ctrv": "#eb6834", "ctra": "#1baf7a", "dynamic": "#eda100", "ekf": "#e87ba4",
         "hybrid_ridge": "#008300", "hybrid_mlp": "#4a3aa7"}
MARK = {"cv": "o", "ctrv": "s", "ctra": "^", "dynamic": "D", "ekf": "v", "hybrid_ridge": "P", "hybrid_mlp": "X"}
SHORT = {"cv": "CV", "ctrv": "CTRV", "ctra": "CTRA", "kinematic": "kinematic ST", "dynamic": "dynamic ST",
         "kf_cv": "KF-CV", "kf_ca": "KF-CA", "ekf": "EKF-CTRA", "ridge": "ridge", "mlp": "MLP",
         "hybrid_ridge": "EKF + ridge", "hybrid_mlp": "EKF + MLP", "hybrid_mlp_nopose": "EKF + MLP, no pose history"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def sh(*cmd: str, **kw) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True, **kw).stdout.strip()


def cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- the full run ---------------------------------------------------------------------------------------------------

def run_all(scenes_dir: Path) -> dict:
    R: dict = {"started": time.strftime("%Y-%m-%d"), "host": cpu(), "cpus": os.cpu_count()}
    R["commit"] = sh("git", "rev-parse", "--short=12", "HEAD")
    R["dirty"] = bool(sh("git", "status", "--porcelain", "--untracked-files=no"))
    vcfg = D.vehicle_config()
    log("loading scenes")
    t0 = time.monotonic()
    train = X.make_split("train", X.TRAIN_STRIDE, scenes_dir, vcfg)
    val = X.make_split("val", D.STRIDE, scenes_dir, vcfg)
    test = X.make_split("test", D.STRIDE, scenes_dir, vcfg)
    R["load_s"] = time.monotonic() - t0
    R["data"] = {sp.name: {"scenes": sp.b.S, "hours": D.hours(sp.b), "samples": int(sp.b.n.sum()), "start_points": sp.n,
                           "scenes_with_starts": int(len(np.unique(sp.si))), "stride": X.TRAIN_STRIDE if sp.name == "train" else D.STRIDE}
                 for sp in (train, val, test)}
    log(f"data {R['data']}")

    log("tuning (val) and fitting (train)")
    t0 = time.monotonic()
    cfgd, rec = X.tune(train, val, vcfg, log)
    R["tune_total_s"] = time.monotonic() - t0
    R["tuning"] = rec
    PR.save_config(PR.CONFIG, cfgd["ctra_accel"], cfgd["kf"], cfgd["ekf"], cfgd["learned"],
                   {"seed": X.SEED, "train_start_points": train.n, "train_stride": X.TRAIN_STRIDE,
                    "val_start_points": val.n, "objective": rec["objective"]})
    conf = PR.load_config()  # from here on: exactly the exported weights
    conf["specs"]["hybrid_mlp_nopose"] = cfgd["ablation"]
    del train

    log("test predictions")
    t0 = time.monotonic()
    preds = PR.predict_all(test.b, test.si, test.ki, conf, vcfg, PR.METHODS + ("hybrid_mlp_nopose",))
    R["predict_test_s"] = time.monotonic() - t0
    err = {m: E.errors(p, test.tru) for m, p in preds.items()}
    R["test"] = {"si": test.si, "ki": test.ki, "S": test.b.S, "vehicle": test.b.vehicle, "location": test.b.location,
                 "names": test.b.names, "slices": X.slice_labels(test)}
    R["err"] = {m: {k: v.astype(np.float32) for k, v in e.items()} for m, e in err.items()}
    R["traj_example"] = trajectory_example(test, preds)

    log("noise sweep")
    rob = [m for m in PR.METHODS]
    R["noise"] = X.noise_sweep(test, conf, vcfg, rob, log)
    log("dropouts")
    R["dropout"] = X.dropout_sweep(test, conf, vcfg, rob)
    log("asymmetric latency")
    R["latency"] = X.latency_experiment(test, conf)
    log("python runtime")
    R["py_runtime"] = X.python_latency(test, conf, vcfg)

    log("C++ build, parity, benchmark")
    sh("bazel", "build", "-c", "opt", "//stack/mpred:mpred_eval", "//stack/mpred:mpred_bench")
    OUT.mkdir(parents=True, exist_ok=True)
    lst = OUT / "test_scenes.txt"
    lst.write_text("".join(f"{scenes_dir / n}.csv {v}\n" for n, v in zip(test.b.names, test.b.vehicle, strict=True)))
    eval_bin, bench_bin = OUT / "mpred_eval", OUT / "mpred_bench"
    for src, dst in ((ROOT / "bazel-bin/stack/mpred/mpred_eval", eval_bin), (ROOT / "bazel-bin/stack/mpred/mpred_bench", bench_bin)):
        dst.unlink(missing_ok=True)  # bazel outputs are read-only, so a copy from an earlier run cannot be overwritten
        shutil.copy2(src, dst)
    predf = OUT / "cpp_preds.f64"
    t0 = time.monotonic()
    info = json.loads(sh(str(eval_bin), "--predictors", str(PR.CONFIG), "--vehicle-config", str(D.VEHICLE_CONFIG),
                         "--scene-list", str(lst), "--out", str(predf), "--stride", str(D.STRIDE),
                         "--history", str(D.HISTORY), "--min-speed", str(D.MIN_SPEED)))
    R["cpp_eval_s"] = time.monotonic() - t0
    R["parity"] = parity(predf, info, test, preds)
    predf.unlink()
    log(f"parity {R['parity']}")
    bench_json = OUT / "bench.json"
    core = str(min(3, (os.cpu_count() or 1) - 1))
    sh("taskset", "-c", core, str(bench_bin), "--predictors", str(PR.CONFIG), "--vehicle-config", str(D.VEHICLE_CONFIG),
       "--scene-list", str(lst), "--repeat", "3", "--out", str(bench_json))
    R["bench"] = json.loads(bench_json.read_text())
    R["bench"]["core"] = core
    R["compiler"] = sh("gcc", "--version").splitlines()[0]
    return R


def parity(predf: Path, info: dict, test: X.Split, preds: dict) -> dict:
    nm = info["methods"]
    rows = np.fromfile(predf, dtype=np.float64).reshape(-1, 3 + 4 * len(D.HORIZONS))
    key_c = rows[:, 0].astype(np.int64) * 1_000_000 + rows[:, 1].astype(np.int64)
    key_p = test.si * 1_000_000 + test.ki
    out = {}
    for mi, m in enumerate(nm):
        sel = rows[:, 2] == mi
        kc = key_c[sel]
        order = np.argsort(kc)
        pos = np.searchsorted(kc[order], key_p)
        ok = (pos < len(kc)) & (kc[order][np.minimum(pos, len(kc) - 1)] == key_p)
        if not ok.all():
            raise RuntimeError(f"parity: {(~ok).sum()} Python start points missing in the C++ output for {m}")
        c = rows[sel][order][pos, 3:].reshape(-1, len(D.HORIZONS), 4)
        p = preds[m]
        out[m] = {"n": int(len(key_p)), "pos_m": float(np.hypot(c[..., 0] - p[..., 0], c[..., 1] - p[..., 1]).max()),
                  "yaw_rad": float(np.abs(M.wrap(c[..., 2] - p[..., 2])).max()), "v_mps": float(np.abs(c[..., 3] - p[..., 3]).max())}
    return out


def trajectory_example(test: X.Split, preds: dict) -> dict:
    """The test scene that contains the start point with the largest yaw-rate change over the next 0.5 s at
    >= 6 m/s: its start points (time, IMU yaw rate) for the trace figure."""
    b, si, ki = test.b, test.si, test.ki
    fut = D.interp_pose(b, si, (b["t"][si, ki] + 0.5)[:, None], keys=("r",))[:, 0, 0]
    score = np.where(b["v"][si, ki] >= 6.0, np.abs(fut - b["r"][si, ki]), -1)
    i = int(np.argmax(score))
    s = int(si[i])
    rows = np.flatnonzero(si == s)
    return {"scene": b.names[s], "t": float(b["t"][s, ki[i]]), "rows": rows, "time": b["t"][s, ki[rows]],
            "r": b["r"][s, ki[rows]], "v": b["v"][s, ki[rows]]}


# ---- statistics -----------------------------------------------------------------------------------------------------

class Stats:
    def __init__(self, R: dict):
        self.R = R
        self.si = R["test"]["si"]
        self.S = R["test"]["S"]
        self.W = P.bootstrap_weights(self.S)

    def sums(self, m: str, metric: str, h: int, mask=None) -> np.ndarray:
        e = self.R["err"][m][metric][:, h].astype(np.float64)
        si = self.si
        if mask is not None:
            e, si = e[mask], si[mask]
        return E.scene_sums(e, si, self.S)

    def ci(self, m, metric, h, mask=None, kind="mean_abs") -> dict:
        return P.ci(self.sums(m, metric, h, mask), self.W, kind)

    def paired(self, a, b, metric, h, mask=None) -> dict:
        return P.paired(self.sums(a, metric, h, mask), self.sums(b, metric, h, mask), self.W)

    def p95(self, m, metric, h) -> float:
        return E.p95(self.R["err"][m][metric][:, h])

    def mean(self, m, metric, h, mask=None) -> float:
        e = np.abs(self.R["err"][m][metric][:, h].astype(np.float64))
        return float(e[mask].mean() if mask is not None else e.mean())


def sum_ci(sums: np.ndarray, W: np.ndarray) -> dict:
    """CI from stored (S, 3) per-scene sums."""
    return P.ci(sums, W, "mean_abs")


# ---- formatting ------------------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def civ(c: dict, nd: int = 3, scale: float = 1.0) -> str:
    if c["value"] is None:
        return "-"
    return f"{c['value'] * scale:.{nd}f} [{c['lo'] * scale:.{nd}f}, {c['hi'] * scale:.{nd}f}]"


def pct(p: dict) -> str:
    star = "*" if p["significant"] else ""
    return f"{p['rel'] * 100:+.1f} %{star}"


def pair_text(p: dict, nd: int = 4, scale: float = 1.0, unit: str = " m") -> str:
    sig = "yes" if p["significant"] else "no"
    if max(abs(p["lo"]), abs(p["hi"])) * scale < 0.5 * 10 ** -nd:  # too small for fixed point: scientific notation
        return (f"{p['diff'] * scale:+.1e}{unit} [{p['lo'] * scale:+.1e}, {p['hi'] * scale:+.1e}] "
                f"({p['rel'] * 100:+.2f} %), better in {p['a_better_scenes']}/{p['scenes']} scenes, CI excludes 0: {sig}")
    return (f"{p['diff'] * scale:+.{nd}f}{unit} [{p['lo'] * scale:+.{nd}f}, {p['hi'] * scale:+.{nd}f}] "
            f"({p['rel'] * 100:+.1f} %), better in {p['a_better_scenes']}/{p['scenes']} scenes, CI excludes 0: {sig}")


def us(x: float) -> str:
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


DEG = 180.0 / np.pi


# ---- figures --------------------------------------------------------------------------------------------------------

def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def figures(R: dict, st: Stats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": 150, "bbox_inches": "tight", "facecolor": SURFACE}

    # 1. error vs horizon: two charts side by side (never a second y axis)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title, ylab in ((axes[0], "pos", "Position error", "mean |error| (m, log scale)"),
                                    (axes[1], "ov20", "Overlay offset 20 m ahead", "mean |lateral offset| (m, log scale)")):
        for m in PLOT:
            y = [st.mean(m, metric, h) for h in range(len(HMS))]
            ax.plot(HMS, y, color=COLOR[m], marker=MARK[m], markersize=6, linewidth=2, label=SHORT[m])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(HMS)
        ax.set_xticklabels([str(h) for h in HMS])
        style(ax, title, "prediction horizon (ms, log scale)", ylab)
    axes[1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Test split, 192 scenes: mean error per horizon", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.savefig(FIG / "mpred_error_vs_horizon.png", **kw)
    plt.close(fig)

    # 2. distribution of the 20 m overlay offset at 200 ms (empirical CDF)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in PLOT:
        e = np.sort(np.abs(R["err"][m]["ov20"][:, H200]))
        q = np.arange(1, len(e) + 1) / len(e)
        ax.plot(np.maximum(e, 1e-4), q, color=COLOR[m], linewidth=2, label=SHORT[m])
    ax.set_xscale("log")
    ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(1.2e-4, 0.955, "p95", color=MUTED, fontsize=8, va="bottom")
    style(ax, "20 m overlay offset at 200 ms: share of start points below x", "|lateral overlay offset| (m, log scale)", "cumulative share")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(FIG / "mpred_overlay_cdf.png", **kw)
    plt.close(fig)

    # 3. noise sensitivity: two charts (overlay offset, position) vs noise level
    W = st.W
    lev = R["noise"]["levels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "ov20", "20 m overlay offset at 200 ms"), (axes[1], "pos", "Position error at 200 ms")):
        for m in PLOT:
            y = [sum_ci(R["noise"]["all"][str(lv)][m][metric][H200], W)["value"] for lv in lev]
            ax.plot(lev, y, color=COLOR[m], marker=MARK[m], markersize=6, linewidth=2, label=SHORT[m])
        ax.set_yscale("log")
        style(ax, title, "injected noise level (x base sigma, see text)", "mean |error| (m, log scale)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(FIG / "mpred_noise.png", **kw)
    plt.close(fig)

    # 4. C++ latency per prediction: p50 (filled) and p99 (open), log scale
    b = R["bench"]["methods"]
    order = ["cv", "ctrv", "ctra", "kinematic", "dynamic", "ekf", "ridge", "hybrid_ridge", "mlp", "hybrid_mlp",
             "history_push_and_ekf_step"]
    names = [SHORT.get(m, "history push + EKF step") for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    yy = np.arange(len(order))
    ax.scatter([b[m]["p50_ns"] for m in order], yy, color="#2a78d6", s=40, label="p50", zorder=3)
    ax.scatter([b[m]["p99_ns"] for m in order], yy, facecolors="none", edgecolors="#eb6834", s=40, linewidths=2,
               label="p99", zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    style(ax, "C++ latency per call, one core (all 7 horizons per prediction)", "ns per call (log scale)", "")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(FIG / "mpred_runtime.png", **kw)
    plt.close(fig)

    # 5. one scene over time: yaw rate (top) and the signed 20 m overlay offset at 200 ms (bottom), shared time axis
    ex = R["traj_example"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    a1.plot(ex["time"], np.degrees(ex["r"]), color=INK, linewidth=2)
    style(a1, f"{ex['scene']} (test): IMU yaw rate", "", "deg/s")
    for m in ("ctrv", "dynamic", "ekf", "hybrid_mlp"):
        a2.plot(ex["time"], R["err"][m]["ov20"][ex["rows"], H200] * 100, color=COLOR[m], marker=MARK[m], markersize=4,
                linewidth=1.5, label=SHORT[m])
    a2.axhline(0, color=MUTED, linewidth=0.8)
    style(a2, "signed 20 m overlay offset of the 200 ms prediction", "scene time of the prediction (s)", "cm (left positive)")
    a2.legend(fontsize=8, frameon=False, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "mpred_scene_trace.png", **kw)
    plt.close(fig)

    # 6. EKF tuning sensitivity: one parameter at a time around the chosen point (small multiples)
    grid = R["tuning"]["ekf"]["grid"]
    best = R["tuning"]["ekf"]["best"]
    keys = list(X.EKF_GRID)
    fig, axes = plt.subplots(1, len(keys), figsize=(13, 2.8), sharey=True)
    for ax, k in zip(axes, keys, strict=True):
        vals = X.EKF_GRID[k]
        ys = []
        for v in vals:
            g = [r for r in grid if all(r[q] == best[q] for q in keys if q != k) and r[k] == v]
            ys.append(g[0]["J"] if g else np.nan)
        xs = np.arange(len(vals))
        ax.plot(xs, ys, color="#2a78d6", marker="o", linewidth=2, markersize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in vals], fontsize=7)
        style(ax, k, "", "val objective (m)" if k == keys[0] else "")
    fig.suptitle("EKF: validation objective when one parameter moves away from the chosen point", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "mpred_ekf_sensitivity.png", **kw)
    plt.close(fig)


# ---- the document ---------------------------------------------------------------------------------------------------

def render(R: dict) -> str:
    st = Stats(R)
    W = st.W
    d = R["data"]
    tun = R["tuning"]
    lines: list[str] = []
    add = lines.append
    tot_h = sum(v["hours"] for v in d.values())
    tot_scenes = sum(v["scenes"] for v in d.values())
    tot_samples = sum(v["samples"] for v in d.values())
    meth = list(PR.METHODS)

    add("# Ego motion prediction for AR-HUD latency compensation")
    add("")
    add(f"Generated by `scripts/reproduce_mpred.py` on {R['started']} at commit `{R['commit']}`"
        f"{' (with uncommitted changes)' if R['dirty'] else ''}. Do not edit by hand. Host: {R['host']}, {R['cpus']} logical CPUs; "
        f"C++ built with `bazel build -c opt` ({R['compiler']}). Data: nuScenes CAN bus expansion (CC BY-NC-SA 4.0), "
        "two Renault Zoe cars (n008 Boston, n015 Singapore).")
    add("")
    add("## The question")
    add("")
    add("An augmented-reality head-up display draws contact-analog overlays (a navigation arrow on the lane, a marker on "
        "the car ahead) at a fixed place in the world. Between the newest sensor sample and the moment the frame is "
        "lit there is a latency (sensor transport, fusion, rendering, display scan-out), so the overlay must be drawn "
        "for where the car will be at display time, not where it was. This module compares ways to predict the ego pose "
        "(x, y, yaw) and speed 50, 100, 150, 200, 300 and 500 ms ahead (1 s for context) from what a function on the "
        "car has at the prediction instant: the recent localization poses, the wheel speed, the IMU yaw rate and "
        "acceleration and the steering angle.")
    add("")
    add("Errors are measured against the recorded pose at the display time. Besides the position error the table "
        "columns that matter for a HUD are the **heading error** and the **overlay offset**: an overlay placed D metres "
        "ahead along the predicted heading lands beside its target by exactly `lateral error + D sin(heading error)` "
        "(lateral in the true vehicle frame; `tools/mpred/metrics.py`). At D = 20 m a heading error of 0.1 deg alone "
        "moves the overlay by 3.5 cm, so heading dominates at look-ahead distances and position dominates near the car. "
        "The angular error seen from the reference point is atan(|offset| / D). The reference point is the middle of "
        "the rear axle (the nuScenes ego origin); the driver's eye is about a metre further forward, which changes D "
        "slightly, not the comparison.")
    add("")
    add("## Data, split and causality")
    add("")
    add(table(["split", "scenes", "hours", "pose samples", "start points", "start-point stride"],
              [[k, str(v["scenes"]), f"{v['hours']:.2f}", f"{v['samples']:,}", f"{v['start_points']:,}",
                f"every {v['stride']} samples"] for k, v in d.items()]))
    add("")
    add(f"- All {tot_scenes} scenes of the v0.3.0 CAN set ({tot_h:.2f} h, {tot_samples:,} pose samples at about 49 Hz), "
        "with the same fixed split (`configs/vdyn/split.json`: whole logs, stratified by car, seed 20261007). Fitting "
        "uses train only, every choice (filter noise, ridge strength, network size, acceleration source) is made on "
        "validation, and every number below is on test unless it says otherwise.")
    add("- `tools/mpred/extract.py` writes a second table set (`can_scenes_v2`, outside git) with the same scenes and "
        "pose rows as v0.3.0 (the extractor checks this), but every CAN and IMU signal is the latest message received at or before the pose time "
        "(sample and hold). The v0.3.0 tables interpolate linearly, which reads up to 10 ms into the future: fine for "
        "identification, a leak for a prediction benchmark. The v0.3.0 tables and results are unchanged.")
    add(f"- A start point is a pose sample with at least {D.HISTORY} samples (about 1 s) of history, the drive gear "
        f"from the start until {D.FUTURE_S} s later, and at least {D.MIN_SPEED} m/s wheel speed. Predictions read "
        "samples up to the start sample only; a unit test perturbs every later sample and checks that no prediction "
        "changes (`tools/mpred/tests/test_causality.py`). The truth at t + h is the recorded pose, interpolated "
        "linearly between the two neighbouring pose samples.")
    add("- Statistics: 95 % percentile bootstrap intervals over scenes (2,000 resamples, scenes are the unit; samples "
        "within a scene are strongly correlated), as in v0.3.0. Paired comparisons use the same resamples for both "
        "methods. p95 values are pooled over start points, without an interval.")
    add("")

    # methods
    add("## Methods")
    add("")
    tu_ekf = tun["ekf"]["best"]
    rows = []
    params = method_params(R)
    for m in meth:
        rows.append([f"`{m}`", PR.LABEL[m], PR.FAMILY[m], params[m]["inputs"], params[m]["how"]])
    add(table(["key", "method", "family", "inputs at the start sample", "parameters"], rows))
    add("")
    add("All physics predictors share one kernel: (x, y, yaw, v) with acceleration and yaw rate held, classical RK4 "
        "with 10 ms steps, the speed clamped at zero (the clamp came from a failing C++ unit test: an RK4 step that brakes "
        "through zero left -0.01 m/s). The filters run causally over every sample of a scene from its first sample; "
        "predictions start from the filtered state. The learned models predict all 7 horizons x (dx, dy, dyaw, dv) at "
        "once (28 outputs): `ridge` and `mlp` directly in the frame of the newest pose, the hybrids as a correction "
        "of the EKF-CTRA forecast. The dynamic and kinematic single-track models use the v0.3.0 parameters "
        "(`configs/vdyn/renault_zoe.json`) unchanged. Not done: a UKF (the measurement model is linear, so a UKF "
        "would differ from the EKF only in the 20 ms prediction step) and recurrent networks.")
    add("")

    # headline
    add("## 1. Accuracy on the test split")
    add("")
    for metric, title, nd, scale in (("pos", "Position error (m)", 4, 1.0), ("head", "Heading error (deg)", 3, DEG),
                                     ("ov20", "Overlay offset 20 m ahead (m)", 4, 1.0)):
        add(f"**{title}**, mean [95 % CI] and pooled p95:")
        add("")
        hdr = ["method"] + [f"{HMS[h]} ms" for h in (H100, H200, H500)] + [f"p95 {HMS[h]} ms" for h in (H100, H200, H500)]
        rows = []
        for m in meth:
            rows.append([PR.LABEL[m]] + [civ(st.ci(m, metric, h), nd, scale) for h in (H100, H200, H500)] +
                        [f"{st.p95(m, metric, h) * scale:.{nd}f}" for h in (H100, H200, H500)])
        add(table(hdr, rows))
        add("")
    add("All horizons, mean |error| (position m / heading deg / 20 m overlay offset m):")
    add("")
    rows = []
    for m in meth:
        rows.append([PR.LABEL[m]] + [f"{st.mean(m, 'pos', h):.3f} / {st.mean(m, 'head', h) * DEG:.3f} / {st.mean(m, 'ov20', h):.3f}"
                                     for h in range(len(HMS))])
    add(table(["method"] + [f"{h} ms" for h in HMS], rows))
    add("")
    add("![error vs horizon](figures/mpred_error_vs_horizon.png)")
    add("")
    add("Lateral, longitudinal and speed error at 200 and 500 ms (mean |error|), overlay offset at 10 and 50 m:")
    add("")
    rows = []
    for m in meth:
        rows.append([PR.LABEL[m]] + [f"{st.mean(m, k, h):.3f}" for k in ("lat", "lon", "speed") for h in (H200, H500)] +
                    [f"{st.mean(m, 'ov10', H200):.3f}", f"{st.mean(m, 'ov50', H200):.3f}"])
    add(table(["method", "lateral 200", "lateral 500", "longitudinal 200", "longitudinal 500", "speed 200 (m/s)",
               "speed 500 (m/s)", "offset 10 m, 200", "offset 50 m, 200"], rows))
    add("")
    # winners
    add("Best method per metric and horizon (lowest mean), and its paired difference to the runner-up:")
    add("")
    rows = []
    R["winners"] = {}
    for metric, nm, scale, unit, nd in (("pos", "position", 1.0, " m", 4), ("head", "heading", DEG, " deg", 4),
                                        ("ov20", "overlay 20 m", 1.0, " m", 4)):
        for h in (H100, H200, H500):
            ranked = sorted(meth, key=lambda m: st.mean(m, metric, h))
            p = st.paired(ranked[0], ranked[1], metric, h)
            R["winners"][f"{metric}@{HMS[h]}"] = {"best": ranked[0], "second": ranked[1], "pair": p}
            rows.append([nm, f"{HMS[h]} ms", SHORT[ranked[0]], SHORT[ranked[1]], pair_text(p, nd, scale, unit)])
    add(table(["metric", "horizon", "best", "runner-up", "best minus runner-up"], rows))
    add("")
    add("Paired comparisons, relative change of the mean |error| (A vs B; negative = A better; * = 95 % CI of the "
        "difference excludes 0):")
    add("")
    comps = [("hybrid_mlp", "cv"), ("hybrid_mlp", "ctrv"), ("hybrid_mlp", "ekf"), ("hybrid_mlp", "hybrid_ridge"),
             ("hybrid_ridge", "ekf"), ("ekf", "ctrv"), ("ekf", "ctra"), ("ekf", "cv"), ("ctra", "ctrv"), ("ctrv", "cv"),
             ("dynamic", "ctrv"), ("kinematic", "ctrv"), ("ridge", "ctra"), ("mlp", "ridge"), ("kf_ca", "kf_cv"),
             ("kf_ca", "ctrv")]
    rows = []
    R["pairs"] = {}
    for a, b_ in comps:
        cells = []
        for metric in ("pos", "head", "ov20"):
            for h in (H100, H200, H500):
                p = st.paired(a, b_, metric, h)
                R["pairs"][f"{a}|{b_}|{metric}|{HMS[h]}"] = p
                cells.append(pct(p))
        rows.append([f"{SHORT[a]} vs {SHORT[b_]}"] + cells)
    add(table(["A vs B"] + [f"{n} {HMS[h]}" for n in ("pos", "head", "ov20") for h in (H100, H200, H500)], rows))
    add("")
    for a, b_ in (("hybrid_mlp", "ctrv"), ("ekf", "ctrv"), ("hybrid_mlp", "ekf")):
        for metric, scale, unit in (("ov20", 1.0, " m"), ("head", DEG, " deg")):
            p = st.paired(a, b_, metric, H200)
            add(f"- {SHORT[a]} vs {SHORT[b_]}, {'20 m overlay offset' if metric == 'ov20' else 'heading error'} at 200 ms: "
                f"{pair_text(p, 4, scale, unit)}")
    add("")
    same = [st.mean(m, "ov20", H200) for m in ("ctrv", "ctra", "ekf")]
    ekf_head = st.paired("ekf", "ctrv", "head", H200)
    if abs(ekf_head["rel"]) < 0.01:
        add(f"The tuned EKF changes the heading forecast of CTRV by {ekf_head['rel'] * 100:+.2f} % at 200 ms: with the chosen "
            f"(large) yaw-acceleration noise s_rdot = {tun['ekf']['best']['s_rdot']} rad/s^2 its yaw-rate state follows the "
            "IMU yaw rate almost unchanged, and the CTRA forecast holds that yaw rate as CTRV does. Its gain over "
            "CTRV is in position (speed fused from the pose and the wheel speed); the learned residuals add the heading gain.")
        add("")
    add("![overlay offset distribution](figures/mpred_overlay_cdf.png)")
    add("")
    if (max(same) - min(same)) / min(same) < 0.01:
        add("CTRV, CTRA and EKF-CTRA have mean 20 m offsets within 1 % of each other at 200 ms "
            f"({', '.join(f'{v:.4f}' for v in same)} m), so their curves lie on top of each other.")
        add("")
    add("![scene trace](figures/mpred_scene_trace.png)")
    add("")
    ex = R["traj_example"]
    add(f"The scene is the one with the largest yaw-rate change in the next 0.5 s at 6 m/s or more over all test start "
        f"points ({ex['scene']}, at t = {ex['t']:.1f} s); every start point of the scene is shown.")
    add("")
    abl = st.paired("hybrid_mlp_nopose", "hybrid_mlp", "ov20", H200)
    abl5 = st.paired("hybrid_mlp_nopose", "hybrid_mlp", "pos", H500)
    add("**Ablation, pose history.** The MLP inputs include the three past poses in the frame of the newest pose. Whether the "
        "nuScenes pose is filtered causally on the car is not documented, so the hybrid MLP was also trained without "
        f"them (validation objective {tun['ablation_nopose']['J']:.4f} vs {tun['mlp']['hybrid_mlp']['candidates'][[c['hidden'] for c in tun['mlp']['hybrid_mlp']['candidates']].index(tun['mlp']['hybrid_mlp']['hidden'])]['J']:.4f} with). "
        f"On test, without vs with: 20 m overlay offset at 200 ms {pair_text(abl, 4)}; position at 500 ms {pair_text(abl5, 4)}.")
    add("")

    # slices
    add("## 2. Robustness by manoeuvre and by car")
    add("")
    add("Each start point gets one label from the recorded motion of the next 0.5 s (`experiment.slice_labels`, first "
        "match wins): low speed (< 5 m/s); yaw-rate change (|r(t + 0.5 s) - r(t)| > 3 deg/s: turning in or out); "
        "steady curve (|r| >= 4 deg/s); braking / accelerating (recorded speed change over 0.5 s <= -0.5 / >= +0.5 m/s); "
        "straight and steady (|r| < 1.5 deg/s, |speed change| < 0.25 m/s); other. The labels use the future only to "
        "group the results, never as an input.")
    add("")
    lab = R["test"]["slices"]
    sm = ["cv", "ctrv", "ctra", "dynamic", "kf_ca", "ekf", "hybrid_ridge", "hybrid_mlp"]
    R["slices"] = {}
    for metric, title, scale, nd in (("ov20", "20 m overlay offset at 200 ms (m)", 1.0, 3), ("head", "heading error at 500 ms (deg)", DEG, 3)):
        h = H200 if metric == "ov20" else H500
        rows = []
        for i, name in enumerate(X.SLICES):
            mask = lab == i
            if mask.sum() == 0:
                continue
            vals = {m: st.mean(m, metric, h, mask) for m in sm}
            best = min(vals, key=vals.get)
            R["slices"].setdefault(name, {})[metric] = {m: v for m, v in vals.items()}
            R["slices"][name]["n"] = int(mask.sum())
            hb = st.ci("hybrid_mlp", metric, h, mask)
            rows.append([name, f"{int(mask.sum()):,} ({mask.mean() * 100:.0f} %)"] +
                        [f"{vals[m] * scale:.{nd}f}" for m in sm] + [SHORT[best], civ(hb, nd, scale)])
        add(f"**{title}**, mean |error| per slice:")
        add("")
        add(table(["slice", "start points"] + [SHORT[m] for m in sm] + ["best", "EKF + MLP, 95 % CI"], rows))
        add("")
    p_tr = st.paired("hybrid_mlp", "ctrv", "ov20", H200, lab == 1)
    p_tr2 = st.paired("ekf", "ctrv", "ov20", H200, lab == 1)
    add(f"- Yaw-rate change slice, 20 m overlay offset at 200 ms: EKF + MLP vs CTRV {pair_text(p_tr, 4)}; "
        f"EKF-CTRA vs CTRV {pair_text(p_tr2, 4)}.")
    add("")
    add("By car (test split, mean [95 % CI]):")
    add("")
    veh = np.array(R["test"]["vehicle"])[R["test"]["si"]]
    loc = dict(zip(R["test"]["vehicle"], R["test"]["location"], strict=True))
    rows = []
    for v in sorted(set(R["test"]["vehicle"])):
        mask = veh == v
        nsc = len(set(np.array(R["test"]["si"])[mask]))
        for m in ("ctrv", "ekf", "hybrid_mlp"):
            rows.append([v, loc[v], str(nsc), f"{int(mask.sum()):,}", SHORT[m], civ(st.ci(m, "pos", H200, mask), 4),
                         civ(st.ci(m, "ov20", H200, mask), 4), civ(st.ci(m, "head", H500, mask), 3, DEG)])
    add(table(["car", "city", "scenes", "start points", "method", "position 200 ms (m)", "overlay 20 m, 200 ms (m)",
               "heading 500 ms (deg)"], rows))
    add("")

    # noise
    add("## 3. Noise sensitivity and lost samples")
    add("")
    bs = X.BASE_SIGMA
    add("White Gaussian noise is added to the inputs of every pose sample (the truth stays clean). Level 1 is: pose "
        f"position {bs['pose_xy']} m, pose heading {bs['pose_yaw'] * DEG:.2f} deg, wheel speed {bs['v']} m/s, yaw rate "
        f"{bs['r'] * DEG:.2f} deg/s, accelerations {bs['a']} m/s^2, steering wheel {bs['steer_sw'] * DEG:.2f} deg; the "
        "sweep scales all of them together. The filters are told the injected noise (it is added in quadrature to their "
        "measurement noise, no re-tuning); the learned models were trained on the recorded signals and get no such "
        "adaptation. Same start points as section 1.")
    add("")
    lev = R["noise"]["levels"]
    for metric, title in (("ov20", "20 m overlay offset at 200 ms (m)"), ("pos", "position error at 200 ms (m)")):
        rows = []
        for m in meth:
            rows.append([PR.LABEL[m]] + [f"{sum_ci(R['noise']['all'][str(lv)][m][metric][H200], W)['value']:.4f}" for lv in lev])
        add(f"**{title}**, mean |error| by noise level:")
        add("")
        add(table(["method"] + [f"level {lv:g}" for lv in lev], rows))
        add("")
    add("![noise sensitivity](figures/mpred_noise.png)")
    add("")
    add("One input group at a time at level 2 (20 m overlay offset at 200 ms, m):")
    add("")
    groups = list(X.NOISE_GROUPS)
    rows = []
    for m in meth:
        rows.append([PR.LABEL[m]] + [f"{sum_ci(R['noise']['groups'][g][m]['ov20'][H200], W)['value']:.4f}" for g in groups])
    add(table(["method"] + groups, rows))
    add("")
    add("Lost odometry samples: each CAN/IMU sample (wheel speed, yaw rate, accelerations, steering) is lost with "
        "probability p and the last received value is held; the pose always arrives. The EKF skips the lost updates. "
        "20 m overlay offset at 200 ms (m):")
    add("")
    rows = []
    for m in meth:
        rows.append([PR.LABEL[m]] + [f"{sum_ci(R['dropout']['res'][str(p)][m]['ov20'][H200], W)['value']:.4f}" for p in R["dropout"]["p"]])
    add(table(["method"] + [f"p = {p:g}" for p in R["dropout"]["p"]], rows))
    add("")

    # latency
    lt = R["latency"]
    mt = lt["_meta"]
    add("## 4. Late samples: pose 100 ms late, odometry 20 ms late")
    add("")
    add("With the same latency on every input, predicting to display time is predicting over latency + display delay, "
        "which section 1 already covers (for example 100 ms of input latency plus 100 ms to the display is the 200 ms "
        "column). The more realistic case is unequal latency: a localization pose that arrives "
        f"{mt['pose_latency_s'] * 1000:.0f} ms late while wheel speed and IMU arrive {mt['odo_latency_s'] * 1000:.0f} ms late. "
        "Now is t_k + 100 ms (k = newest pose); the overlay is drawn for now + h. *stale*: predict from pose k with the "
        "signals of sample k over 100 ms + h. *bridged*: dead-reckon from pose k with the odometry that has already "
        f"arrived (on average {mt['bridge_samples_mean']:.1f} samples), then predict the rest. *EKF bridged*: the filter "
        "state at k re-propagated with the odometry updates only (how a filter handles an out-of-sequence pose), then the "
        "CTRA forecast. *EKF, fresh pose*: the reference without latency (pose at or before now). Learned models are "
        "left out: they were trained for the fixed horizons from a fresh pose. Mean |error|:")
    add("")
    lm = ["ctrv_stale", "ctrv_bridged", "ctra_stale", "ctra_bridged", "ekf_stale", "ekf_bridged", "ekf_fresh_reference"]
    lab_l = {"ctrv_stale": "CTRV, stale", "ctrv_bridged": "CTRV, bridged", "ctra_stale": "CTRA, stale",
             "ctra_bridged": "CTRA, bridged", "ekf_stale": "EKF, stale", "ekf_bridged": "EKF bridged (odometry only)",
             "ekf_fresh_reference": "EKF, fresh pose (no latency)"}
    hz = [int(round(h * 1000)) for h in mt["horizons"]]
    rows = []
    for m in lm:
        rows.append([lab_l[m]] + [f"{sum_ci(lt[m]['pos'][i], W)['value']:.4f} / {sum_ci(lt[m]['ov20'][i], W)['value']:.4f}"
                                  for i in range(len(hz))])
    add(table(["method (position m / overlay 20 m m)"] + [f"now + {h} ms" for h in hz], rows))
    add("")
    i100 = hz.index(100)
    for a, b_ in (("ekf_bridged", "ekf_stale"), ("ctrv_bridged", "ctrv_stale"), ("ekf_bridged", "ekf_fresh_reference")):
        for metric in ("pos", "ov20"):
            p = P.paired(lt[a][metric][i100], lt[b_][metric][i100], W)
            R["pairs"][f"{a}|{b_}|{metric}|lat100"] = p
            add(f"- {lab_l[a]} vs {lab_l[b_]}, {'position' if metric == 'pos' else '20 m overlay offset'} at now + 100 ms: "
                f"{pair_text(p, 4)}")
    add("")

    # parameterization effort
    add("## 5. Parameterization effort")
    add("")
    rows = []
    for m in meth:
        q = params[m]
        rows.append([PR.LABEL[m], str(q["tuned"]), q["fitted"], q["fixed"], q["data"], q["time"]])
    add(table(["method", "tuned on val", "fitted on train", "fixed by hand / inherited", "data needed", "tune + fit time"], rows))
    add("")
    eg = [g["J"] for g in tun["ekf"]["grid"]]
    add(f"- Tuning objective (validation): {tun['objective']}.")
    add("- CTRA acceleration source (validation objective, m): " + ", ".join(f"{k} {v:.4f}" for k, v in tun["ctra"].items()) +
        f"; chosen `{min(tun['ctra'], key=tun['ctra'].get)}`.")
    for name in ("kf_cv", "kf_ca"):
        g = [r["J"] for r in tun["kf"][name]["grid"]]
        add(f"- {SHORT[name]}: {len(g)} grid points, objective {min(g):.4f} to {max(g):.4f} (chosen q {tun['kf'][name]['best']['q']}, "
            f"r {tun['kf'][name]['best']['r_pos']} m).")
    add(f"- EKF: {tun['ekf']['runs']} grid points in {tun['ekf']['s']:.0f} s; objective {min(eg):.4f} (chosen) to {max(eg):.4f} "
        f"(median {np.median(eg):.4f}, worst {(max(eg) / min(eg) - 1) * 100:.0f} % above the best). Chosen: " +
        ", ".join(f"{k} {tu_ekf[k]}" for k in X.EKF_GRID) + ". The figure moves one parameter at a time." +
        (" On the edge of its grid range: " + ", ".join(edge) + " (the optimum may lie beyond it; see the "
         "figure)." if (edge := [k for k in X.EKF_GRID if isinstance(X.EKF_GRID[k][0], float)
                                                 and tu_ekf[k] in (X.EKF_GRID[k][0], X.EKF_GRID[k][-1])]) else ""))
    add("- Ridge strength (validation objective by lambda): " + "; ".join(
        f"{SHORT[n]}: " + ", ".join(f"{k} {v:.4f}" for k, v in r["J_by_lambda"].items()) for n, r in tun["ridge"].items()) + ".")
    for n, r in tun["mlp"].items():
        add(f"- {SHORT[n]}: " + "; ".join(f"hidden {c['hidden']} objective {c['J']:.4f}, {c['train_s']:.0f} s, best epoch "
                                         f"{c['best_epoch']} of {c['epochs_run']}" for c in r["candidates"]) +
            f"; chosen {r['hidden']} ({r['params']:,} weights). Training: Adam, lr {X.MLP_TRAIN['lr']}, batch "
            f"{X.MLP_TRAIN['batch']}, weight decay {X.MLP_TRAIN['weight_decay']}, early stopping on validation MSE "
            f"(patience {X.MLP_TRAIN['patience']}), float32, seed {X.SEED}.")
    add(f"- Learned models are fitted on {tun['train_samples']:,} training start points (every {X.TRAIN_STRIDE}nd pose sample "
        f"of {d['train']['scenes']} scenes, {d['train']['hours']:.2f} h). Whole tuning + fitting run: {R['tune_total_s']:.0f} s.")
    add("")
    add("![EKF sensitivity](figures/mpred_ekf_sensitivity.png)")
    add("")

    # runtime
    b = R["bench"]
    bm = b["methods"]
    pr = R["py_runtime"]
    add("## 6. Runtime, real-time capability, C++ parity")
    add("")
    add(f"C++: `stack/mpred` (C++17), `mpred_bench` streams all {b['scenes']} test scenes {b['repeat']} times through the "
        f"history buffer and the EKF ({b['samples_streamed']:,} samples) and from the 50th sample of each scene calls every "
        f"predictor once per sample, pinned to one core (`taskset -c {b['core']}`). Each call is timed on its own; the "
        f"clock overhead (two consecutive reads, p50 {b['clock_overhead']['p50_ns']:.0f} ns) is included, not subtracted. "
        "Python: numpy, one start point per call (the real-time case), and vectorised over all test start points. Host CPU, "
        "not an automotive ECU.")
    add("")
    order = ["cv", "ctrv", "ctra", "kinematic", "dynamic", "ekf", "ridge", "mlp", "hybrid_ridge", "hybrid_mlp"]
    rows = []
    for m in order:
        n_par = params[m]["n_params"]
        rows.append([PR.LABEL[m], f"{bm[m]['p50_ns']:.0f}", f"{bm[m]['p99_ns']:.0f}", f"{bm[m]['p999_ns']:.0f}",
                     f"{bm[m]['max_ns'] / 1000:.1f}", us(pr[m]["p50_us"]), us(pr[m]["p99_us"]),
                     f"{pr['vectorised_us_per_prediction'][m]:.2f}", str(n_par)])
    m = "history_push_and_ekf_step"
    rows.append(["history push + EKF step (per sample)", f"{bm[m]['p50_ns']:.0f}", f"{bm[m]['p99_ns']:.0f}",
                 f"{bm[m]['p999_ns']:.0f}", f"{bm[m]['max_ns'] / 1000:.1f}", us(pr["ekf_step"]["p50_us"]),
                 us(pr["ekf_step"]["p99_us"]), f"{pr['vectorised_us_per_prediction']['ekf_filter_per_sample']:.2f}", "-"])
    add(table(["method", "C++ p50 ns", "C++ p99 ns", "C++ p99.9 ns", "C++ max us", "Python p50 us", "Python p99 us",
               "Python vectorised us/pred", "parameters"], rows))
    add("")
    cyc = bm["history_push_and_ekf_step"]["p99_ns"] + bm["hybrid_mlp"]["p99_ns"]
    cyc2 = bm["history_push_and_ekf_step"]["p99_ns"] + bm["ekf"]["p99_ns"]
    R["cycle_p99_ns"] = {"hybrid_mlp": cyc, "ekf": cyc2}
    add(f"- Real time: one display frame needs one history push + EKF step and one prediction. With the slowest "
        f"predictor (EKF + MLP) the sum of the two p99 values is {cyc / 1000:.1f} us, {cyc / 1e6 * 100:.2f} % of a 1 ms budget "
        f"and {cyc / 1e7 * 100:.3f} % of the 10 ms frame period of a 100 Hz display loop; with EKF-CTRA alone "
        f"{cyc2 / 1000:.1f} us. The worst single call over all runs was {max(v['max_ns'] for v in bm.values()) / 1000:.0f} us. "
        "This is a shared, non-real-time Linux host, so the maximum includes scheduling and cache effects and is not a "
        "worst-case execution time; a WCET claim needs the target hardware and a real-time OS.")
    add(f"- Memory: `History` {b['sizeof']['History']} bytes, `Ekf` {b['sizeof']['Ekf']} bytes, one `Prediction` "
        f"{b['sizeof']['Prediction']} bytes; learned weights in double precision: " +
        ", ".join(f"{SHORT[k]} {v:,} ({v * 8 / 1024:.1f} KiB)" for k, v in sorted(b["learned_params"].items())) +
        ". No allocation after construction on the prediction path.")
    add("")
    add("![runtime](figures/mpred_runtime.png)")
    add("")
    par = R["parity"]
    add(f"**Python vs C++ parity** (test split, every start point, all 7 horizons; the C++ side streams sample by sample "
        f"through the ring buffer and the filter, the Python side works on whole arrays; C++ run {R['cpp_eval_s']:.1f} s):")
    add("")
    rows = [[PR.LABEL[m], f"{v['n']:,}", f"{v['pos_m']:.1e}", f"{v['yaw_rad']:.1e}", f"{v['v_mps']:.1e}"] for m, v in par.items()]
    add(table(["method", "predictions", "max position diff (m)", "max yaw diff (rad)", "max speed diff (m/s)"], rows))
    add("")

    # limits
    add("## Assumptions and limits")
    add("")
    add("- Open-loop replay of recorded drives: the predictions never act on anything. The reference is the car's own "
        "localization pose (`pose`, about 49 Hz), not survey-grade ground truth; how it is filtered on the car is not "
        "documented, hence the pose-history ablation.")
    add("- One vehicle type (Renault Zoe, two cars), urban Boston and Singapore, mostly below 15 m/s and far from the "
        "tyre limit, as in v0.3.0. Nothing here says how the methods do on a highway, in snow or near the limit.")
    add("- The latencies are emulated on recorded timestamps (CAN reception times, 10-20 ms message periods); no real "
        "HUD pipeline, no display, no eye-point model; the overlay offset is computed in the ground plane from the rear axle.")
    add("- The v0.3.0 vehicle parameters (mass, CG and yaw inertia assumed) are reused unchanged.")
    add("- The injected noise is white and independent per sample; real sensor errors are biased and correlated.")
    add("- Timings are on a shared x86 host; an automotive ECU is slower and needs its own measurement. The C++ code is "
        "single threaded and allocation-free on the prediction path, but not certified, not MISRA-checked and not "
        "fixed-point.")
    add("- The learned models were fitted on 604 scenes of two cars; they learn this data's calibration (wheel radius "
        "vs pose speed, IMU vs pose heading) as much as motion. A different car needs new training data.")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("python tools/mpred/extract.py --can-zip can_bus.zip --out ~/workspace/drive-replay-data/can_scenes_v2")
    add("pip install -r tools/mpred/requirements.txt")
    add("python scripts/reproduce_mpred.py     # about " + f"{(R.get('total_s', 0) / 60):.0f} min on the host above")
    add("bazel test //stack/mpred:all //tools/mpred:all")
    add("```")
    add("")
    return "\n".join(lines)


def method_params(R: dict) -> dict:
    """Inputs, parameter counts and effort per method, from the tuning records."""
    tun = R["tuning"]
    mlp = tun["mlp"]
    rid = tun["ridge"]
    ekf_t = f"{tun['ekf']['s']:.0f} s grid ({tun['ekf']['runs']} runs)"
    ekf_fixed = "6 noise values (q_xy, q_psi, q_v, r_v, r_r, r_a bases)"
    v03 = "v0.3.0 identification"
    out = {
        "cv": {"inputs": "pose, wheel speed", "how": "none", "tuned": 0, "fitted": "-", "fixed": "wheel radius (v0.3.0)",
               "data": "none", "time": "-", "n_params": 0},
        "ctrv": {"inputs": "pose, wheel speed, IMU yaw rate", "how": "none", "tuned": 0, "fitted": "-",
                 "fixed": "wheel radius (v0.3.0)", "data": "none", "time": "-", "n_params": 0},
        "ctra": {"inputs": "+ acceleration", "how": "acceleration source (val)", "tuned": 1, "fitted": "-",
                 "fixed": "wheel radius", "data": "val (choice of 5)", "time": f"{tun['ctra_s']:.1f} s", "n_params": 1},
        "kinematic": {"inputs": "pose, wheel speed, steering", "how": v03, "tuned": 0,
                      "fitted": "steering ratio + 2 offsets (v0.3.0, train)", "fixed": "wheelbase", "data": "v0.3.0 train",
                      "time": "-", "n_params": 4},
        "dynamic": {"inputs": "pose, wheel speed, steering, IMU yaw rate", "how": v03, "tuned": 0,
                    "fitted": "ratio, offsets, Cr, K -> Cf (v0.3.0, train)", "fixed": "mass, CG, inertia (assumed)",
                    "data": "v0.3.0 train", "time": "-", "n_params": 9},
        "kf_cv": {"inputs": "pose position", "how": "q, r (val grid)", "tuned": 2, "fitted": "-", "fixed": "initial covariance",
                  "data": "val", "time": f"{tun['kf']['kf_cv']['s']:.1f} s", "n_params": 2},
        "kf_ca": {"inputs": "pose position", "how": "q, r (val grid)", "tuned": 2, "fitted": "-", "fixed": "initial covariance",
                  "data": "val", "time": f"{tun['kf']['kf_ca']['s']:.1f} s", "n_params": 2},
        "ekf": {"inputs": "pose, wheel speed, IMU yaw rate, accel", "how": "6 values (val grid)", "tuned": 6, "fitted": "-",
                "fixed": ekf_fixed, "data": "val", "time": ekf_t, "n_params": 12},
        "ridge": {"inputs": "signals at lags 0/5/25 (18 features)", "how": "weights (train), lambda (val)", "tuned": 1,
                  "fitted": f"{rid['ridge']['params']:,} weights", "fixed": "feature set", "data": "train + val",
                  "time": f"{rid['ridge']['s']:.1f} s", "n_params": rid["ridge"]["params"]},
        "mlp": {"inputs": "33 raw history values", "how": "weights (train), size (val)", "tuned": 1,
                "fitted": f"{mlp['mlp']['params']:,} weights", "fixed": "5 training settings", "data": "train + val",
                "time": f"{sum(c['train_s'] for c in mlp['mlp']['candidates']):.0f} s", "n_params": mlp["mlp"]["params"]},
        "hybrid_ridge": {"inputs": "EKF state + 18 features", "how": "EKF (val) + weights (train), lambda (val)", "tuned": 7,
                         "fitted": f"{rid['hybrid_ridge']['params']:,} weights", "fixed": ekf_fixed + ", feature set",
                         "data": "train + val", "time": f"EKF + {rid['hybrid_ridge']['s']:.1f} s",
                         "n_params": 12 + rid["hybrid_ridge"]["params"]},
        "hybrid_mlp": {"inputs": "EKF state + 33 raw history values", "how": "EKF (val) + weights (train), size (val)",
                       "tuned": 7, "fitted": f"{mlp['hybrid_mlp']['params']:,} weights",
                       "fixed": ekf_fixed + ", 5 training settings", "data": "train + val",
                       "time": f"EKF + {sum(c['train_s'] for c in mlp['hybrid_mlp']['candidates']):.0f} s",
                       "n_params": 12 + mlp["hybrid_mlp"]["params"]},
    }
    return out


def site_json(R: dict) -> dict:
    st = Stats(R)
    W = st.W
    out = {"data": R["data"], "horizons_ms": HMS, "labels": {m: PR.LABEL[m] for m in PR.METHODS}, "acc": {},
           "noise": {}, "bench": {m: {k: R["bench"]["methods"][m][k] for k in ("p50_ns", "p99_ns")} for m in R["bench"]["methods"]},
           "parity": R["parity"], "commit": R["commit"], "host": R["host"]}
    for m in PR.METHODS:
        out["acc"][m] = {metric: [st.ci(m, metric, h) for h in range(len(HMS))] for metric in ("pos", "head", "ov20")}
        out["noise"][m] = [sum_ci(R["noise"]["all"][str(lv)][m]["ov20"][H200], W)["value"] for lv in R["noise"]["levels"]]
    out["noise_levels"] = R["noise"]["levels"]
    out["winners"] = {k: {"best": v["best"], "second": v["second"]} for k, v in R["winners"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=Path, default=D.DEFAULT_SCENES)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    cache = OUT / "results.pkl"
    if args.report_only:
        R = pickle.loads(cache.read_bytes())
    else:
        t0 = time.monotonic()
        R = run_all(args.scenes)
        R["total_s"] = time.monotonic() - t0
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(R))
    log("figures and report")
    st = Stats(R)
    figures(R, st)
    doc = render(R)
    (DOCS / "MOTION_PREDICTION.md").write_text(doc)
    SITE.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list | tuple):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return o

    (SITE / "mpred.json").write_text(json.dumps(clean(site_json(R))) + "\n")
    for f in ("mpred_error_vs_horizon.png", "mpred_overlay_cdf.png", "mpred_noise.png", "mpred_runtime.png",
              "mpred_scene_trace.png"):
        shutil.copy2(FIG / f, SITE / f)
    log(f"wrote {DOCS / 'MOTION_PREDICTION.md'}")


if __name__ == "__main__":
    main()
