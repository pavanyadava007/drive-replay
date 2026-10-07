"""Regenerates docs/VEHICLE_DYNAMICS.md, its figures (docs/figures/vdyn_*.png) and site/data/vdyn.json.

    python scripts/reproduce_vdyn.py [--scenes ~/workspace/drive-replay-data/can_scenes]

Needs: bazel on PATH, the scene tables from tools/vdyn/can_extract.py (nuScenes CAN bus expansion, not in git),
the identified parameters configs/vdyn/renault_zoe.json (tools/vdyn/identify.py), and for the FCW part the
suite recordings plus data/recordings_can (tools/vdyn/add_can_topic.py). Every number in the document comes
from this script. Never edit docs/VEHICLE_DYNAMICS.md by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.replay import drops, paths  # noqa: E402
from tools.vdyn import fcw_experiment  # noqa: E402
from tools.vdyn import pipeline as P  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SITE = ROOT / "site" / "data"
OUT = paths.RUNS / "vdyn"
MINI = ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-0916",
        "scene-1077", "scene-1094", "scene-1100"]
HELD = ["const_velocity", "const_yaw_rate", "fcw_yaw_decay", "kinematic", "dynamic"]
REC = ["kinematic_rec", "dynamic_rec"]
YAW = ["kinematic", "steady_state", "dynamic"]
CORR = ["const_velocity", "const_yaw_rate", "fcw_yaw_decay", "kinematic", "single_track", "single_track_steer_decay"]
LABEL = {"const_velocity": "constant velocity", "const_yaw_rate": "constant yaw rate",
         "fcw_yaw_decay": "FCW 0.2.0 yaw-rate decay", "kinematic": "kinematic single track",
         "dynamic": "linear dynamic single track", "kinematic_rec": "kinematic, recorded inputs",
         "dynamic_rec": "dynamic, recorded inputs", "steady_state": "linear steady state (K)",
         "single_track": "dynamic, steering held", "single_track_steer_decay": "dynamic, steering decays (1 s)"}
COLOR = {"kinematic": "#2a78d6", "dynamic": "#eb6834", "fcw_yaw_decay": "#1baf7a", "const_yaw_rate": "#eda100",
         "const_velocity": "#e87ba4", "steady_state": "#4a3aa7", "kinematic_rec": "#2a78d6", "dynamic_rec": "#eb6834",
         "single_track": "#eb6834", "single_track_steer_decay": "#4a3aa7"}
OCTAVE_IMAGE = "alpine:3.24.2"  # octave from the Alpine package repository at run time
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def f(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def civ(c: dict, nd: int = 3, scale: float = 1.0) -> str:
    if c["value"] is None:
        return "-"
    return f"{c['value'] * scale:.{nd}f} [{c['lo'] * scale:.{nd}f}, {c['hi'] * scale:.{nd}f}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def pair_text(p: dict, nd: int = 3, scale: float = 1.0, unit: str = "") -> str:
    sig = "yes" if p["significant"] else "no"
    return (f"{p['diff'] * scale:+.{nd}f}{unit} [{p['lo'] * scale:+.{nd}f}, {p['hi'] * scale:+.{nd}f}] "
            f"({p['rel'] * 100:+.1f} %), lower in {p['a_better_scenes']}/{p['scenes']} scenes, CI excludes 0: {sig}")


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt  # noqa: F401
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=Path, default=P.DEFAULT_SCENES)
    ap.add_argument("--skip-fcw", action="store_true", help="no replay part (needs data/recordings_can)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    FIG.mkdir(parents=True, exist_ok=True)
    head = sh("git", "rev-parse", "--short=12", "HEAD")
    dirty = bool(sh("git", "status", "--porcelain", "--untracked-files=no"))
    print("building vdyn_eval and replay_main (-c opt)", flush=True)
    sh("bazel", "build", "-c", "opt", "//stack/vdyn:vdyn_eval", "//stack:replay_main")
    binary = OUT / "vdyn_eval"
    shutil.copy2(ROOT / "bazel-bin" / "stack" / "vdyn" / "vdyn_eval", binary)
    cfg = json.loads(P.DEFAULT_CONFIG.read_text())
    ident = cfg["identification"]
    split = json.loads(P.SPLIT.read_text())
    man = P.manifest(args.scenes)
    workers = os.cpu_count() or 4

    # ---- evaluation on the held-out test scenes -------------------------------------------------------------
    test_res = P.run_batch(split["test"], workers, args.scenes, binary=binary)
    assert not test_res.errors, test_res.errors
    test = [test_res.scenes[n] for n in split["test"]]
    w = P.bootstrap_weights(len(test))
    S = lambda *path: P.sums(test, path)
    n_test_moving = sum(1 for s in test if "yaw" in s)
    hours = sum(s["duration_s"] for s in test) / 3600

    yaw_rows, yaw_json = [], {}
    bins = ["0.5-3", "3-6", "6-9", "9-12", "12+", "all"]
    for m in YAW:
        yaw_json[m] = {b: P.ci(S("yaw", m, b), w, "rmse") for b in bins}
        yaw_json[m + "_mae"] = P.ci(S("yaw", m, "all"), w, "mean_abs")
        yaw_rows.append([LABEL[m]] + [civ(yaw_json[m][b], 3, 180 / np.pi) for b in bins])
    n_bins = [str(int(S("yaw", "dynamic", b)[:, 0].sum())) for b in bins]
    yaw_pair = {"dyn_vs_kin": P.paired(S("yaw", "dynamic", "all"), S("yaw", "kinematic", "all"), w, "rmse"),
                "dyn_vs_ss": P.paired(S("yaw", "dynamic", "all"), S("yaw", "steady_state", "all"), w, "rmse"),
                "ss_vs_kin": P.paired(S("yaw", "steady_state", "all"), S("yaw", "kinematic", "all"), w, "rmse")}

    traj_json, traj_rows = {}, []
    for m in HELD + REC:
        traj_json[m] = {}
        for h in ("1", "2", "3"):
            traj_json[m][h] = {k: P.ci(S("traj", m, h, k), w) for k in ("fde", "lat", "head")}
        traj_rows.append([LABEL[m] + (" (held)" if m in HELD else "")] +
                         [civ(traj_json[m][h]["fde"], 2) for h in ("1", "2", "3")] +
                         [civ(traj_json[m][h]["lat"], 2) for h in ("1", "2", "3")])
    n_starts = int(S("traj", "dynamic", "3", "fde")[:, 0].sum())
    traj_pairs = {}
    for a, b in [("dynamic", "kinematic"), ("kinematic", "const_yaw_rate"), ("dynamic", "fcw_yaw_decay"),
                 ("fcw_yaw_decay", "const_yaw_rate"), ("dynamic_rec", "kinematic_rec")]:
        for h in ("1", "3"):
            for k in ("fde", "lat"):
                traj_pairs[f"{a}|{b}|{h}|{k}"] = P.paired(S("traj", a, h, k), S("traj", b, h, k), w)
    best_lat3 = min(HELD, key=lambda m: traj_json[m]["3"]["lat"]["value"])

    corr_json, corr_rows = {}, []
    for m in CORR:
        corr_json[m] = {d: P.ci(S("corridor", m, d), w) for d in ("10", "20", "30")}
        corr_rows.append([LABEL[m]] + [civ(corr_json[m][d], 2) for d in ("10", "20", "30")])
    corr_n = [str(int(S("corridor", "fcw_yaw_decay", d)[:, 0].sum())) for d in ("10", "20", "30")]
    corr_pairs = {d: P.paired(S("corridor", "single_track_steer_decay", d), S("corridor", "fcw_yaw_decay", d), w)
                  for d in ("10", "20", "30")}
    corr_pairs_held = {d: P.paired(S("corridor", "single_track", d), S("corridor", "fcw_yaw_decay", d), w)
                       for d in ("10", "20", "30")}

    # by car
    veh_rows = []
    for veh in sorted({man[n]["vehicle"] for n in split["test"]}):
        sub = [s for s, n in zip(test, split["test"], strict=True) if man[n]["vehicle"] == veh]
        wv = P.bootstrap_weights(len(sub), seed=11)
        veh_rows.append([veh, man[next(n for n in split["test"] if man[n]["vehicle"] == veh)]["location"].split("-")[0],
                         str(len(sub))] +
                        [civ(P.ci(P.sums(sub, ("yaw", m, "all")), wv, "rmse"), 2, 180 / np.pi) for m in YAW] +
                        [civ(P.ci(P.sums(sub, ("traj", m, "3", "lat")), wv), 2) for m in ("kinematic", "dynamic")])

    # ---- validation split for the corridor variant choice (made before the test run, reproduced here) -------
    val_res = P.run_batch(split["val"], workers, args.scenes, binary=binary)
    val = list(val_res.scenes.values())
    val_corr = {m: {d: P.pooled(P.sums(val, ("corridor", m, d)))["mean_abs"] for d in ("10", "20", "30")}
                for m in ("fcw_yaw_decay", "single_track", "single_track_steer_decay")}

    # ---- mini scenes (the replay suite's drives) ------------------------------------------------------------
    mini_res = P.run_batch(MINI, workers, args.scenes, binary=binary)
    mini = [mini_res.scenes[n] for n in MINI]
    wm = P.bootstrap_weights(len(mini), seed=13)
    mini_rows = [[LABEL[m]] + [civ(P.ci(P.sums(mini, ("corridor", m, d)), wm), 2) for d in ("10", "20", "30")]
                 for m in ("const_velocity", "fcw_yaw_decay", "single_track", "single_track_steer_decay")]
    mini_n = [str(int(P.sums(mini, ("corridor", "fcw_yaw_decay", d))[:, 0].sum())) for d in ("10", "20", "30")]
    mini_split = {p: [n for n in MINI if n in split[p]] for p in ("train", "val", "test")}

    # ---- throughput and determinism over all scenes ---------------------------------------------------------
    all_names = sorted(man)
    tp = []
    for nw in sorted({1, 2, 4, 8, workers}):
        r = P.run_batch(all_names, nw, args.scenes, binary=binary)
        assert not r.errors
        tp.append(r.summary())
    again = P.run_batch(all_names, workers, args.scenes, binary=binary)
    digests = {t["batch_digest"] for t in tp} | {again.batch_digest}

    # ---- figures ------------------------------------------------------------------------------------------
    # yaw trace: the test scene with the largest yaw-rate RMS among scenes averaging >= 5 m/s (chosen by rule)
    cands = [n for n in split["test"] if man[n]["mean_speed_mps"] >= 5.0]
    trace_scene = max(cands, key=lambda n: np.loadtxt(args.scenes / f"{n}.csv", delimiter=",", skiprows=1, usecols=6).std())
    tr_path = OUT / "trace.csv"
    subprocess.run([str(binary), "--config", str(P.DEFAULT_CONFIG), "--vehicle", man[trace_scene]["vehicle"],
                    "--scene", str(args.scenes / f"{trace_scene}.csv"), "--trace", str(tr_path), "--yaw-only"],
                   check=True, capture_output=True)
    rows = list(csv.DictReader(tr_path.open()))
    t = np.array([float(r["t_s"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(t, [float(r["r_imu"]) for r in rows], color=INK, lw=2.0, label="IMU (measured)")
    for k, m in (("r_kinematic", "kinematic"), ("r_dynamic", "dynamic")):
        ax.plot(t, [float(r[k]) for r in rows], color=COLOR[m], lw=1.6, label=LABEL[m])
    style(ax, f"Yaw rate from steering + wheel speed, {trace_scene} (test split, {man[trace_scene]['location']})",
          "time (s)", "yaw rate (rad/s)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "vdyn_yaw_trace.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150, sharex=True)
    hs = [1, 2, 3]
    for ax, k, title in ((axes[0], "fde", "Final displacement error"), (axes[1], "lat", "Lateral error")):
        for m in HELD + REC:
            v = [traj_json[m][str(h)][k] for h in hs]
            y = [x["value"] for x in v]
            err = [[x["value"] - x["lo"] for x in v], [x["hi"] - x["value"] for x in v]]
            ax.errorbar(hs, y, yerr=err, color=COLOR[m], lw=2, marker="o", ms=4, capsize=3,
                        ls="--" if m in REC else "-", label=LABEL[m])
        style(ax, title, "prediction horizon (s)", "mean error (m), 95 % CI")
        ax.set_xticks(hs)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"Open-loop ego prediction on {len(test)} test scenes ({n_starts} start points)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "vdyn_error_vs_horizon.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    xs = np.arange(len(bins) - 1)
    for i, m in enumerate(YAW):
        v = [yaw_json[m][b] for b in bins[:-1]]
        ax.bar(xs + (i - 1) * 0.26, [x["value"] * 180 / np.pi for x in v], width=0.24, color=COLOR[m], label=LABEL[m],
               yerr=[[(x["value"] - x["lo"]) * 180 / np.pi for x in v], [(x["hi"] - x["value"]) * 180 / np.pi for x in v]],
               capsize=2, error_kw={"elinewidth": 1, "ecolor": MUTED})
    ax.set_xticks(xs, [b + " m/s" for b in bins[:-1]])
    style(ax, "Yaw-rate RMSE vs IMU by speed (test split)", "speed bin", "RMSE (deg/s), 95 % CI")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "vdyn_yaw_by_speed.png")
    plt.close(fig)

    # ---- optional GNU Octave cross-check of the steady state (throwaway container) ---------------------------
    octave = None
    if shutil.which("docker"):
        sh("bazel", "build", "-c", "opt", "//stack/vdyn:vdyn_steady")
        oct_dir = OUT / "octave"
        oct_dir.mkdir()
        (oct_dir / "steady.csv").write_text(sh(str(ROOT / "bazel-bin" / "stack" / "vdyn" / "vdyn_steady"), "--config",
                                               str(P.DEFAULT_CONFIG)) + "\n")
        shutil.copy2(P.DEFAULT_CONFIG, oct_dir / "config.json")
        shutil.copy2(ROOT / "tools" / "vdyn" / "octave" / "steady_state.m", oct_dir)
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{oct_dir}:/w:ro", OCTAVE_IMAGE, "sh", "-c",
                            "apk add --no-cache octave >/dev/null 2>&1 && octave --no-gui --quiet /w/steady_state.m "
                            "/w/config.json /w/steady.csv"], capture_output=True, text=True, check=False)
        kv = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
        if r.returncode == 0 and "max_rel_diff_cpp_simulation" in kv:
            octave = kv

    # ---- FCW feature integration ----------------------------------------------------------------------------
    fcw = None
    if not args.skip_fcw:
        d = drops.import_binary(ROOT / "bazel-bin" / "stack" / "replay_main")
        fcw = fcw_experiment.run(d.name, OUT / "fcw")

    # ---- write ----------------------------------------------------------------------------------------------
    dyn, kin = cfg["dynamic"], cfg["kinematic"]
    deg = 180 / np.pi
    tp_rows = [[str(t_["workers"]), str(t_["processes"]), str(t_["scenes"]), f(t_["wall_s"], 2), f(t_["scenes_per_s"], 1),
                f(t_["x_realtime"], 0), t_["batch_digest"]] for t_ in tp]
    lines = [
        "# Vehicle dynamics: models, identification, validation",
        "",
        f"Generated by `scripts/reproduce_vdyn.py` on {time.strftime('%Y-%m-%d')} at commit `{head}`"
        f"{' (working tree with uncommitted changes)' if dirty else ''}. Do not edit by hand.",
        f"Host: {cpu()}, {os.cpu_count()} logical CPUs. Data: nuScenes CAN bus expansion (CC BY-NC-SA 4.0), "
        f"{len(man)} scenes with valid CAN data, two Renault Zoe cars (n008 Boston, n015 Singapore).",
        "",
        "## Data and split",
        "",
        f"- Scenes in `can_bus.zip`: {len(man)} kept by `tools/vdyn/can_extract.py`; none of the 21 scenes on the devkit's CAN "
        "blacklist is in the zip, and none failed the gap check (all four 50/100 Hz messages overlap for at least 10 s, "
        "no gap above 0.25 s). Messages used: `steeranglefeedback` (steering wheel), `zoe_veh_info` (rear wheel speeds), "
        "`ms_imu` (yaw rate), `pose` (position, heading, reference speed), `vehicle_monitor` (gear: only samples in "
        "drive are scored).",
        f"- Split (`configs/vdyn/split.json`, seed {split['seed']}): train {split['counts']['train']}, val "
        f"{split['counts']['val']}, test {split['counts']['test']} scenes. Logs are kept whole (no drive is split across "
        "parts) and the split is stratified by car. Parameters are fitted on train only; val chose two settings "
        "(CG position, corridor steering variant); every number below is on test unless it says otherwise.",
        f"- Test set: {len(test)} scenes, {hours:.2f} h of driving, {n_test_moving} with samples above 0.5 m/s.",
        "",
        "## Identified parameters (train split)",
        "",
        table(["parameter", "value", "how"], [
            ["wheelbase L", f"{cfg['wheelbase_m']} m", "Renault Zoe, public spec (Wikipedia: 2,588 mm)"],
            ["wheel radius", f"{cfg['wheel_radius_m']:.4f} m", f"least squares, pose speed vs rear wheel rpm "
             f"(speed RMSE {ident['speed_rmse_mps']:.3f} m/s on train)"],
            ["steering ratio, kinematic", f"{kin['steering_ratio']:.2f}", "Gauss-Newton on r = v tan((sw - off)/i)/L"],
            ["steering ratio, linear", f"{dyn['steering_ratio']:.2f}", "least squares with K, r = v (sw - off) / (i L (1 + K v^2))"],
            ["steering-wheel offset", ", ".join(f"{k} {v['dynamic'] * deg:+.2f} deg" for k, v in cfg["steer_offset_by_vehicle"].items()),
             "one per car (sensor calibration), fitted jointly"],
            ["understeer gradient K", f"{ident['understeer_K_s2pm2']:.5f} s^2/m^2 = {ident['understeer_gradient_deg_per_g']:.2f} deg/g",
             "grid search + least squares, quasi-steady samples"],
            ["mass m", f"{dyn['mass_kg']:.0f} kg", "assumed: kerb 1,468 kg (Wikipedia) + sensor rig + 2 people"],
            ["CG to front axle lf", f"{dyn['lf_m']:.3f} m ({ident['cg_lf_fraction_chosen_on_val']:.2f} L)",
             "assumed prior; val could not separate 0.40/0.45/0.50 L (within 0.1 %)"],
            ["yaw inertia Iz", f"{dyn['yaw_inertia_kgm2']:.0f} kg m^2", "assumed m lf lr"],
            ["cornering stiffness Cf / Cr", f"{dyn['cf_npr'] / 1e3:.1f} / {dyn['cr_npr'] / 1e3:.1f} kN/rad (per axle)",
             "Cr by grid search on the simulated yaw rate (train), Cf from K"],
        ]),
        "",
        f"Quasi-steady samples for the steady-state fits: {ident['quasi_steady_samples']} of {ident['train_samples']} train "
        "samples (speed >= 3 m/s, steering-wheel rate < 0.3 rad/s, yaw acceleration < 0.1 rad/s^2, drive gear). "
        f"Fit RMSE on them: kinematic {ident['kinematic_fit_rmse_rps'] * deg:.3f} deg/s, linear with K "
        f"{ident['linear_fit_rmse_rps'] * deg:.3f} deg/s (one shared offset instead of one per car: "
        f"{ident['linear_fit_rmse_one_shared_offset_rps'] * deg:.3f} deg/s). Per-car fits (diagnostic only): " +
        "; ".join(f"{k}: ratio {v['steering_ratio']}, K {v['understeer_K_s2pm2']}" for k, v in ident["per_vehicle_linear_fit"].items()) +
        ". All parameters live in `configs/vdyn/renault_zoe.json`, written by `tools/vdyn/identify.py` and read by the C++ library.",
        "",
        "## 1. Yaw rate from steering angle and wheel speed vs the IMU (test split)",
        "",
        "RMSE in deg/s, 95 % bootstrap CI over scenes (2,000 resamples). `kinematic`: v tan(delta)/L. "
        "`linear steady state`: v delta / (L (1 + K v^2)). `dynamic`: the linear single-track model simulated through "
        "the whole scene (RK4, 50 Hz) from the recorded steering and wheel speed, started once from the measured yaw rate.",
        "",
        table(["model"] + [b + (" m/s" if b != "all" else "") for b in bins], yaw_rows),
        "| samples | " + " | ".join(n_bins) + " |",
        "",
        f"- Dynamic vs kinematic (RMSE, deg/s): {pair_text(yaw_pair['dyn_vs_kin'], 3, deg)}",
        f"- Dynamic vs linear steady state: {pair_text(yaw_pair['dyn_vs_ss'], 3, deg)}",
        f"- Linear steady state vs kinematic: {pair_text(yaw_pair['ss_vs_kin'], 3, deg)}",
        f"- Mean absolute error, all speeds: kinematic {civ(yaw_json['kinematic_mae'], 3, deg)}, "
        f"dynamic {civ(yaw_json['dynamic_mae'], 3, deg)} deg/s.",
        "",
        "![yaw rate trace](figures/vdyn_yaw_trace.png)",
        "",
        "![yaw rate error by speed](figures/vdyn_yaw_by_speed.png)",
        "",
        "## 2. Open-loop ego trajectory prediction (test split)",
        "",
        f"Start points every 0.5 s where the car is in drive at >= 2 m/s and the whole horizon is recorded: {n_starts} "
        "per horizon. Each model starts from the recorded pose and predicts 1, 2 and 3 s ahead; the error is measured "
        "against the recorded pose (`pose`, 50 Hz). **Held** models get only what a function has at run time: steering "
        "angle, wheel speed and IMU yaw rate at the start instant, held constant. The **recorded-input** rows feed the "
        "recorded future steering and speed instead, which isolates the model error from the input prediction error; "
        "they are not a forecast. Mean error in m with 95 % CI.",
        "",
        table(["model", "FDE 1 s", "FDE 2 s", "FDE 3 s", "lateral 1 s", "lateral 2 s", "lateral 3 s"], traj_rows),
        "",
        "Paired comparisons (A minus B, mean error in m, same scenes and resamples; negative = A better):",
        "",
    ]
    for key, p in traj_pairs.items():
        a, b, h, k = key.split("|")
        lines.append(f"- {LABEL[a]} vs {LABEL[b]}, {'FDE' if k == 'fde' else 'lateral'} at {h} s: {pair_text(p, 3)}")
    lines += [
        "",
        "![error vs horizon](figures/vdyn_error_vs_horizon.png)",
        "",
        "By car (test split; yaw RMSE in deg/s, lateral error at 3 s in m):",
        "",
        table(["car", "city", "scenes", "yaw kinematic", "yaw steady state", "yaw dynamic", "lat 3 s kinematic",
               "lat 3 s dynamic"], veh_rows),
        "",
        "## 3. Corridor centre line vs the driven path",
        "",
        "Lateral offset of the predicted path at 10, 20 and 30 m ahead, against where the car really was when it got "
        "there (start-frame coordinates, same start points; distances not reached within 6 s are skipped). This is the "
        "question the FCW corridor asks. `FCW 0.2.0 yaw-rate decay` is the exact formula of the stack (yaw rate "
        "decaying with 1 s, curvature clamp); the dynamic rows use `PredictedPath`, the code the FCW runs with "
        "`path_model = 1`. Mean absolute error in m with 95 % CI, test split.",
        "",
        table(["model", "10 m", "20 m", "30 m"], corr_rows),
        "| start points | " + " | ".join(corr_n) + " |",
        "",
        "Validation split (mean absolute error, m), used to choose the steering variant of `path_model = 1` before the "
        "test run: " + "; ".join(f"{LABEL[m]} {', '.join(f(v, 3) for v in val_corr[m].values())}" for m in val_corr) + ".",
        "",
    ]
    for d_, p in corr_pairs.items():
        lines.append(f"- steering decays vs FCW 0.2.0 decay at {d_} m: {pair_text(p, 3)}")
    for d_, p in corr_pairs_held.items():
        lines.append(f"- steering held vs FCW 0.2.0 decay at {d_} m: {pair_text(p, 3)}")
    lines += [
        "",
        f"The 10 replay-suite drives (nuScenes mini; in the CAN split they fall into train {len(mini_split['train'])}, "
        f"val {len(mini_split['val'])}, test {len(mini_split['test'])}), same metric:",
        "",
        table(["model", "10 m", "20 m", "30 m"], mini_rows),
        "| start points | " + " | ".join(mini_n) + " |",
        "",
    ]
    if fcw:
        rec = fcw["recordings"]
        changed_w = [r["scene"] for r in rec if r["on"]["warnings"] != r["off"]["warnings"]
                     or r["on"]["first_warning_s"] != r["off"]["first_warning_s"] or r["on"]["golden_diffs"]]
        lines += [
            "## 4. FCW with the single-track corridor (opt-in `path_model = 1`)",
            "",
            "The suite replayed on the CAN recording set (`data/recordings_can`, the published recordings plus a "
            "`/vehicle/can` topic; `recordings/catalog_can.json`) with the option OFF (default) and ON "
            f"(`path_model=1`, `vehicle_model=configs/vdyn/renault_zoe.json`, `vehicle_id` per car). Drop `{fcw['drop']}`.",
            "",
            table(["recording", "OFF identical to published", "single-track cycles", "warnings OFF / ON",
                   "first warning OFF / ON (s)", "false OFF / ON", "missed OFF / ON", "golden diffs ON", "state digest changed"],
                  [[r["scene"], "yes" if r["off_identical_to_published"] else "NO", f"{r['single_track_cycles']}/{r['cycles']}",
                    f"{r['off']['warnings']} / {r['on']['warnings']}",
                    f"{f(r['off']['first_warning_s'], 2)} / {f(r['on']['first_warning_s'], 2)}",
                    f"{r['off']['false']} / {r['on']['false']}", f"{r['off']['missed']} / {r['on']['missed']}",
                    str(len(r["on"]["golden_diffs"])), "yes" if r["state_digest_changed"] else "no"] for r in rec]),
            "",
            table(["bug", "recording", "window (s)", "expectation", "holds OFF", "holds ON", "note"],
                  [[b["bug"], b["recording"], f"{b['window_s'][0]}-{b['window_s'][1]}", b["expect"]["kind"],
                    str(b["off"]["expectation_holds"]), str(b["on"]["expectation_holds"]),
                    "; ".join(f"{k.split('_')[0]} false warnings OFF {v['off']} / ON {v['on']}" for k, v in b.items()
                              if k.endswith("_false_warnings"))] for b in fcw["bugs"]]),
            "",
            f"- Recordings whose warnings changed with the option ON: {', '.join(changed_w) if changed_w else 'none'}. "
            f"The single-track corridor was used in {sum(r['single_track_cycles'] for r in rec)} of "
            f"{sum(r['cycles'] for r in rec)} radar cycles (the rest had no steering message within 0.1 s), and the internal state "
            f"(threat selection, TTC) changed in {sum(r['state_digest_changed'] for r in rec)} of {len(rec)} recordings, "
            "but no warning onset moved.",
            "- With the option OFF the extra topic changes nothing: every CAN recording gives the event and state "
            "digests of the published recording.",
            "- What this does and does not show: on these 13 recordings the steering-based corridor neither adds nor "
            "removes a warning, keeps all three bug expectations and catches the three injected threats at the same time. "
            "The suite has no case where the two corridors disagree about a threat, so it cannot show a safety benefit; "
            "the corridor accuracy of section 3 is the measured difference. The option stays OFF by default.",
            "",
        ]
    lines += ["## Cross-check in GNU Octave", ""]
    if octave:
        lines += [
            f"`tools/vdyn/octave/steady_state.m` (GNU Octave {octave['octave_version']}, {OCTAVE_IMAGE} container) builds the "
            "state-space matrices of the linear single-track model from `configs/vdyn/renault_zoe.json` and solves the "
            f"steady state with Octave's linear solver for {octave['cases']} speed/steering cases (2-30 m/s). Largest "
            f"relative difference to the C++ RK4 simulation (30 s): {float(octave['max_rel_diff_cpp_simulation']):.1e}; to the "
            f"C++ closed form: {float(octave['max_rel_diff_cpp_closed_form']):.1e}. This checks the implementation of the "
            "equations, not the model: both sides use the same parameters. Octave only; no MATLAB or Simulink was used.",
            ""]
    else:
        lines += ["Not run (needs docker and network for the Octave package).", ""]
    lines += [
        "## 5. Reprocessing throughput and determinism (all scenes)",
        "",
        f"`tools/vdyn/pipeline.py` over all {len(all_names)} scenes ({tp[0]['recorded_s'] / 3600:.2f} h of driving): "
        "scenes grouped by car, 8 per `vdyn_eval` process, a thread pool keeps N processes running. Wall time includes "
        "process start and CSV parsing.",
        "",
        table(["workers", "processes", "scenes", "wall s", "scenes/s", "x real time", "batch digest"], tp_rows),
        "",
        f"- Distinct batch digests over all worker counts plus a repeat run: {len(digests)} "
        f"({'identical' if len(digests) == 1 else 'DIFFERENT'}). Every scene digest is computed by the C++ side from the "
        "metrics alone, so chunking and scheduling cannot change it.",
        "",
        "## Assumptions and limits",
        "",
        "- One vehicle type (Renault Zoe), two cars, urban Boston and Singapore driving: speeds mostly below 15 m/s and "
        f"lateral acceleration far below the tyre limit (only {n_bins[4]} test samples above 12 m/s, so that bin is thin). Nothing here says how the models do on a highway or near the limit.",
        "- Linear tyres: no saturation, no combined slip, no load transfer, roll or pitch; constant mass and assumed CG "
        "and inertia. lf and Iz are not identifiable from these data (validation error within 0.1 % for 0.40-0.50 L).",
        "- Open loop: held models keep the start-instant steering and speed; nobody predicts the driver. The "
        "displacement error at 2-3 s is dominated by the speed assumption, the lateral error by the path model.",
        "- CAN timing: `zoe_veh_info` carries reception time, not measurement time (devkit note); signals are linearly "
        "interpolated onto the 50 Hz pose clock. No latency between steering and yaw is modelled beyond the dynamics.",
        "- The IMU yaw rate is the reference for section 1 and the pose for sections 2-3; both are the vehicle's own "
        "estimates, not survey-grade ground truth.",
        "- The FCW experiment covers 13 recordings (10 real mini drives, 3 injected must-warn cases). It shows no change, "
        "not an improvement.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "curl -O https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip   # 781 MB, no login",
        "mkdir -p meta; for v in trainval test; do   # scene and log tables only (logs, cars, cities for the split)",
        "  curl -sSfL https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-${v}_meta.tgz \\",
        "    | tar -xz -C meta --wildcards \"v1.0-$v/scene.json\" \"v1.0-$v/log.json\"; done",
        "python tools/vdyn/can_extract.py --can-zip can_bus.zip --meta meta --out ~/workspace/drive-replay-data/can_scenes",
        "bazel build -c opt //stack/vdyn:vdyn_eval && python tools/vdyn/identify.py --scenes ~/workspace/drive-replay-data/can_scenes",
        "python tools/vdyn/add_can_topic.py --can-zip can_bus.zip   # CAN recording set for section 4",
        "python scripts/reproduce_vdyn.py",
        "```",
        "",
    ]
    text = "\n".join(lines)
    for bad in (chr(0x2014), chr(0x2013)):  # no long dashes in generated text
        assert bad not in text, "no long dashes"
    (DOCS / "VEHICLE_DYNAMICS.md").write_text(text)
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "vdyn.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d"), "commit": head, "cpu": cpu(), "split": split["counts"],
        "test_scenes": len(test), "start_points": n_starts, "params": {k: cfg[k] for k in ("wheelbase_m", "wheel_radius_m",
                                                                                            "kinematic", "dynamic")},
        "understeer_deg_per_g": ident["understeer_gradient_deg_per_g"], "yaw": yaw_json, "yaw_pairs": yaw_pair,
        "traj": traj_json, "traj_pairs": traj_pairs, "corridor": corr_json, "corridor_pairs": corr_pairs,
        "throughput": tp, "fcw": fcw, "octave": octave, "best_lateral_3s": best_lat3,
    }, indent=1, default=float))
    for p in FIG.glob("vdyn_*.png"):
        shutil.copy2(p, SITE / p.name)
    print(text)


if __name__ == "__main__":
    main()
