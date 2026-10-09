"""The benchmark experiments: tuning on validation, fitting on train, and the robustness, latency and runtime
studies on test. scripts/reproduce_mpred.py calls these and turns the results into docs/MOTION_PREDICTION.md.

Seeds: every random draw (noise, dropouts, MLP initialisation and batches, bootstrap) uses a fixed seed.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace

import numpy as np

from tools.mpred import data as D
from tools.mpred import learn as L
from tools.mpred import metrics as E
from tools.mpred import models as M
from tools.mpred import predictors as PR

SEED = 20261009
ACCEL_CANDIDATES = ("ax_imu", "ax_can", "slope5", "slope10", "slope25")
KF_Q = {2: (0.3, 1.0, 3.0, 10.0, 30.0), 3: (0.3, 1.0, 3.0, 10.0, 30.0)}
KF_R = (0.001, 0.005, 0.02, 0.05)
EKF_GRID = {"r_xy": (0.001, 0.003), "r_psi": (1e-4, 3e-4, 1e-3), "s_jerk": (3.0, 9.0, 27.0), "s_rdot": (0.15, 0.45, 1.35),
            "r_scale": (3.0, 10.0, 30.0), "accel": ("ax_can", "ax_imu")}
RIDGE_LAMBDA = (1e-5, 1e-3, 1e-1, 1e1, 1e3)
MLP_HIDDEN = ((32, 32), (64, 64))
MLP_TRAIN = {"epochs": 120, "batch": 512, "lr": 1e-3, "weight_decay": 1e-5, "patience": 6}
TRAIN_STRIDE = 2

# Injected noise at level 1 (std, white, per pose sample); the sweep multiplies all of them by the level.
BASE_SIGMA = {"pose_xy": 0.05, "pose_yaw": 0.003, "v": 0.1, "r": 0.005, "a": 0.2, "steer_sw": 0.01}
NOISE_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)
NOISE_GROUPS = {"pose": ("pose_xy", "pose_yaw"), "wheel speed": ("v",), "yaw rate": ("r",), "acceleration": ("a",),
                "steering": ("steer_sw",)}
DROP_P = (0.0, 0.1, 0.3, 0.5)
POSE_LATENCY_S, ODO_LATENCY_S = 0.10, 0.02


@dataclass
class Split:
    name: str
    b: D.Batch
    si: np.ndarray
    ki: np.ndarray
    tru: np.ndarray

    @property
    def n(self) -> int:
        return len(self.si)


def make_split(part: str, stride: int = D.STRIDE, scenes_dir=D.DEFAULT_SCENES, cfg: dict | None = None) -> Split:
    b = D.load(D.split_names(part), scenes_dir, cfg)
    si, ki = D.start_points(b, stride=stride)
    return Split(part, b, si, ki, D.truth(b, si, ki))


def anchor_of(sp: Split) -> np.ndarray:
    return np.stack([sp.b["x"][sp.si, sp.ki], sp.b["y"][sp.si, sp.ki], sp.b["yaw"][sp.si, sp.ki]], 1)


def J(pred: np.ndarray, sp: Split) -> float:
    return E.objective(E.errors(pred, sp.tru))


# ---- tuning and fitting -----------------------------------------------------------------------------------------

def tune(train: Split, val: Split, vcfg: dict, log=print) -> tuple[dict, dict]:
    """Returns (config for predictors.save_config, records of every tuning step for the report)."""
    rec: dict = {"objective": "mean over 100/200/300/500 ms of (mean position error + mean |20 m overlay offset|), "
                              "validation split, metres"}
    t0 = time.monotonic()
    rec["ctra"] = {s: J(M.predict_ctra(val.b, val.si, val.ki, M.accel_signal(val.b, s)), val) for s in ACCEL_CANDIDATES}
    ctra_accel = min(rec["ctra"], key=rec["ctra"].get)
    rec["ctra_s"] = time.monotonic() - t0
    log(f"CTRA acceleration: {ctra_accel}")

    kf, rec["kf"] = {}, {}
    for name, order in (("kf_cv", 2), ("kf_ca", 3)):
        t0 = time.monotonic()
        grid = {}
        for q, r in itertools.product(KF_Q[order], KF_R):
            p = M.KfParams(order, q, r)
            grid[(q, r)] = J(M.predict_kf(M.run_kf(val.b, p), val.si, val.ki, D.HORIZONS), val)
        q, r = min(grid, key=grid.get)
        kf[name] = {"order": order, "q": q, "r_pos": r}
        rec["kf"][name] = {"grid": [{"q": a, "r_pos": b_, "J": v} for (a, b_), v in grid.items()], "best": kf[name],
                           "s": time.monotonic() - t0}
        log(f"{name}: q {q} r {r} J {grid[(q, r)]:.4f}")

    t0 = time.monotonic()
    keys = list(EKF_GRID)
    egrid = []
    for vals in itertools.product(*(EKF_GRID[k] for k in keys)):
        p = M.EkfParams(**dict(zip(keys, vals, strict=True)))
        st = M.run_ekf(val.b, p)
        egrid.append({**dict(zip(keys, vals, strict=True)), "J": J(M.predict_from_state(st, val.si, val.ki), val)})
    best = min(egrid, key=lambda g: g["J"])
    ekf = M.EkfParams(**{k: best[k] for k in keys})
    rec["ekf"] = {"grid": egrid, "best": ekf.as_dict(), "J": best["J"], "s": time.monotonic() - t0, "runs": len(egrid)}
    log(f"EKF: {best}")

    # learned models: features on train (stride 2) and val, EKF states with the tuned filter
    t0 = time.monotonic()
    feats = {}
    for sp in (train, val):
        acc = M.accel_signal(sp.b, ctra_accel)
        st_all = M.run_ekf(sp.b, ekf)
        st = st_all[sp.si, sp.ki]
        base = M.rollout(st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5])
        feats[sp.name] = {
            "phys": L.phys_features(sp.b, sp.si, sp.ki, acc), "phys_s": L.phys_features(sp.b, sp.si, sp.ki, acc, st),
            "raw": L.raw_features(sp.b, sp.si, sp.ki, acc), "raw_s": L.raw_features(sp.b, sp.si, sp.ki, acc, st),
            "y_direct": L.to_anchor(anchor_of(sp), sp.tru, sp.b["v"][sp.si, sp.ki]),
            "y_res": L.residual_target(base, sp.tru, st[:, 2]), "st": st_all}
    rec["features_s"] = time.monotonic() - t0
    rec["train_samples"] = train.n
    tr, va = feats["train"], feats["val"]

    def evaluate(spec: PR.LearnedSpec) -> float:
        return J(PR.predict_learned(spec, val.b, val.si, val.ki, M.accel_signal(val.b, ctra_accel), va["st"]), val)

    learned, rec["ridge"], rec["mlp"] = {}, {}, {}
    for name, fkey, residual in (("ridge", "phys", False), ("hybrid_ridge", "phys_s", True)):
        t0 = time.monotonic()
        ykey = "y_res" if residual else "y_direct"
        cands = {}
        for lam in RIDGE_LAMBDA:
            m = L.Ridge.fit(tr[fkey], tr[ykey], lam)
            spec = PR.LearnedSpec.from_ridge(name, m, "phys", fkey.endswith("_s"), residual)
            cands[lam] = (evaluate(spec), spec)
        lam = min(cands, key=lambda k: cands[k][0])
        learned[name] = cands[lam][1]
        rec["ridge"][name] = {"lambda": lam, "J_by_lambda": {str(k): v[0] for k, v in cands.items()},
                              "s": time.monotonic() - t0, "params": learned[name].n_params()}
        log(f"{name}: lambda {lam} J {cands[lam][0]:.4f}")
    for name, fkey, residual in (("mlp", "raw", False), ("hybrid_mlp", "raw_s", True)):
        ykey = "y_res" if residual else "y_direct"
        cands = {}
        for hid in MLP_HIDDEN:
            t0 = time.monotonic()
            m = L.train_mlp(tr[fkey], tr[ykey], va[fkey], va[ykey], hidden=hid, seed=SEED, **MLP_TRAIN)
            spec = PR.LearnedSpec.from_mlp(name, m, "raw", fkey.endswith("_s"), residual)
            cands[hid] = (evaluate(spec), spec, time.monotonic() - t0, m.history)
            log(f"{name} {hid}: J {cands[hid][0]:.4f} ({cands[hid][2]:.0f} s, best epoch {m.history[-1]['best_epoch']})")
        hid = min(cands, key=lambda k: cands[k][0])
        learned[name] = cands[hid][1]
        rec["mlp"][name] = {"hidden": list(hid), "params": learned[name].n_params(),
                            "candidates": [{"hidden": list(h), "J": c[0], "train_s": c[2], "best_epoch": c[3][-1]["best_epoch"],
                                            "epochs_run": len(c[3]) - 1} for h, c in cands.items()]}
    # ablation: the hybrid MLP without the pose-history inputs (Python only)
    t0 = time.monotonic()
    hid = tuple(rec["mlp"]["hybrid_mlp"]["hidden"])
    drop = lambda f: np.concatenate([f[:, :24], f[:, 33:]], 1)
    m = L.train_mlp(drop(tr["raw_s"]), tr["y_res"], drop(va["raw_s"]), va["y_res"], hidden=hid, seed=SEED, **MLP_TRAIN)
    abl = PR.LearnedSpec.from_mlp("hybrid_mlp_nopose", m, "raw", True, True, drop_pose=True)
    rec["ablation_nopose"] = {"J": evaluate(abl), "train_s": time.monotonic() - t0}
    config = {"ctra_accel": ctra_accel, "kf": kf, "ekf": ekf, "learned": learned, "ablation": abl}
    return config, rec


# ---- perturbations -------------------------------------------------------------------------------------------------

def perturb(b: D.Batch, vcfg: dict, sigma: dict, rng: np.random.Generator, drop_p: float = 0.0):
    """Copy of b with white Gaussian noise (stds in `sigma`, keys of BASE_SIGMA) on the inputs, and odometry samples
    (wheel speed, yaw rate, accelerations, steering) lost with probability drop_p (the last value is held).
    Returns (batch, extra_std for the filters, odometry availability mask)."""
    sig = dict(b.sig)
    S, T = b["t"].shape

    def noisy(key, std):
        return sig[key] + rng.normal(0.0, std, (S, T)) if std > 0 else sig[key]

    sig["x"] = noisy("x", sigma.get("pose_xy", 0.0))
    sig["y"] = noisy("y", sigma.get("pose_xy", 0.0))
    sig["yaw"] = noisy("yaw", sigma.get("pose_yaw", 0.0))
    sig["r"] = noisy("r", sigma.get("r", 0.0))
    sig["ax_imu"] = noisy("ax_imu", sigma.get("a", 0.0))
    sig["ax_can"] = noisy("ax_can", sigma.get("a", 0.0))
    sig["steer"] = noisy("steer", sigma.get("steer_sw", 0.0))
    # wheel speed noise on the converted speed (m/s), so the std is in m/s
    v_noise = rng.normal(0.0, sigma["v"], (S, T)) if sigma.get("v", 0.0) > 0 else 0.0
    avail = None
    if drop_p > 0:
        lost = rng.random((S, T)) < drop_p
        lost[:, 0] = False
        avail = (~lost).astype(float)
        idx = np.maximum.accumulate(np.where(lost, 0, np.arange(T)[None, :]), axis=1)
        rows = np.arange(S)[:, None]
        for k in ("r", "ax_imu", "ax_can", "steer", "rpm"):
            sig[k] = sig[k][rows, idx]
        if not np.isscalar(v_noise):
            v_noise = v_noise[rows, idx]
    nb = replace(b, sig=sig)
    D.derive(nb.sig, nb.vehicle, vcfg)
    nb.sig["v"] = nb.sig["v"] + v_noise
    extra = {"pose_xy": sigma.get("pose_xy", 0.0), "pose_yaw": sigma.get("pose_yaw", 0.0), "v": sigma.get("v", 0.0),
             "r": sigma.get("r", 0.0), "a": sigma.get("a", 0.0)}
    return nb, extra, avail


KEY_METRICS = ("pos", "lat", "head", "ov20")


def summarize(err: dict, si: np.ndarray, n_scenes: int) -> dict:
    """Per-scene sums (n, sum |e|, sum e^2) for the key metrics at every horizon: {metric: (H, S, 3)}."""
    return {k: np.stack([E.scene_sums(err[k][:, h], si, n_scenes) for h in range(err[k].shape[1])]) for k in KEY_METRICS}


def noise_sweep(test: Split, conf: dict, vcfg: dict, methods, log=print) -> dict:
    out = {"levels": list(NOISE_LEVELS), "base_sigma": BASE_SIGMA, "all": {}, "groups": {}}
    for lev in NOISE_LEVELS:
        rng = np.random.default_rng(SEED + int(lev * 100))
        nb, extra, _ = perturb(test.b, vcfg, {k: v * lev for k, v in BASE_SIGMA.items()}, rng)
        preds = PR.predict_all(nb, test.si, test.ki, conf, vcfg, methods, extra_std=extra)
        out["all"][str(lev)] = {m: summarize(E.errors(p, test.tru), test.si, test.b.S) for m, p in preds.items()}
        log(f"noise level {lev}")
    for g, keys in NOISE_GROUPS.items():
        rng = np.random.default_rng(SEED + 7)
        nb, extra, _ = perturb(test.b, vcfg, {k: 2.0 * BASE_SIGMA[k] for k in keys}, rng)
        preds = PR.predict_all(nb, test.si, test.ki, conf, vcfg, methods, extra_std=extra)
        out["groups"][g] = {m: summarize(E.errors(p, test.tru), test.si, test.b.S) for m, p in preds.items()}
    return out


def dropout_sweep(test: Split, conf: dict, vcfg: dict, methods) -> dict:
    out = {"p": list(DROP_P), "res": {}}
    for p in DROP_P:
        rng = np.random.default_rng(SEED + int(p * 1000))
        nb, extra, avail = perturb(test.b, vcfg, {}, rng, drop_p=p)
        preds = PR.predict_all(nb, test.si, test.ki, conf, vcfg, methods, odo_avail=avail)
        out["res"][str(p)] = {m: summarize(E.errors(pr, test.tru), test.si, test.b.S) for m, pr in preds.items()}
    return out


# ---- asymmetric latency ------------------------------------------------------------------------------------------

def latency_experiment(test: Split, conf: dict, horizons=D.HORIZONS[:6]) -> dict:
    """The pose arrives POSE_LATENCY_S late, wheel speed / IMU ODO_LATENCY_S late. 'now' = t_k + pose latency,
    k = newest pose; the overlay is drawn for now + h. stale: predict from pose k over pose latency + h with the
    signals of sample k. bridged: dead-reckon from pose k with the odometry received until now - odo latency
    (sample j), then predict the rest. EKF bridged: the filter state at k re-propagated with the odometry
    updates of k+1..j (no pose), then the CTRA forecast."""
    b, si, k = test.b, test.si, test.ki
    t = b["t"]
    tk = t[si, k]
    j = np.array([np.searchsorted(t[s], tt + POSE_LATENCY_S - ODO_LATENCY_S + 1e-9, side="right") - 1
                  for s, tt in zip(si, tk, strict=True)])
    j = np.maximum(j, k)
    tru = D.truth(b, si, k, horizons, offset_s=POSE_LATENCY_S)
    acc = M.accel_signal(b, conf["ctra"]["accel"])
    x, y, p = b["x"][si, k], b["y"][si, k], b["yaw"][si, k]
    xb, yb, pb = M.dead_reckon(b, si, k, j, x, y, p)
    ekf_p = conf["ekf_params"]
    Xk, Pk = M.run_ekf_with_cov(b, ekf_p, k, si)
    Xj = M.ekf_odometry_only(b, ekf_p, Xk, Pk, si, k, j)
    out = {}
    preds = {m: np.empty(tru.shape) for m in ("ctrv_stale", "ctrv_bridged", "ctra_stale", "ctra_bridged", "ekf_stale",
                                              "ekf_bridged", "ekf_fresh_reference")}
    rem_base = tk + POSE_LATENCY_S - t[si, j]
    for hi, h in enumerate(horizons):
        H = POSE_LATENCY_S + h
        hs = np.full(len(si), H)
        v, r, a = b["v"][si, k], b["r"][si, k], acc[si, k]
        preds["ctrv_stale"][:, hi] = M.rollout_to(x, y, p, v, 0.0, r, hs, H)
        preds["ctra_stale"][:, hi] = M.rollout_to(x, y, p, v, a, r, hs, H)
        rem = rem_base + h
        vj, rj, aj = b["v"][si, j], b["r"][si, j], acc[si, j]
        preds["ctrv_bridged"][:, hi] = M.rollout_to(xb, yb, pb, vj, 0.0, rj, rem, H)
        preds["ctra_bridged"][:, hi] = M.rollout_to(xb, yb, pb, vj, aj, rj, rem, H)
        preds["ekf_stale"][:, hi] = M.rollout_to(*Xk.T, hs, H)
        preds["ekf_bridged"][:, hi] = M.rollout_to(*Xj.T, rem, H)
    # reference without latency: the same display time predicted from a fresh pose (sample at or before t_k + 100 ms)
    kf = np.array([np.searchsorted(t[s], tt + POSE_LATENCY_S + 1e-9, side="right") - 1 for s, tt in zip(si, tk, strict=True)])
    Xf, _ = M.run_ekf_with_cov(b, ekf_p, kf, si)
    for hi, h in enumerate(horizons):
        remf = tk + POSE_LATENCY_S + h - t[si, kf]
        preds["ekf_fresh_reference"][:, hi] = M.rollout_to(*Xf.T, remf, h + 0.02)
    for m, pr in preds.items():
        out[m] = summarize(E.errors(pr, tru), si, b.S)
    out["_meta"] = {"pose_latency_s": POSE_LATENCY_S, "odo_latency_s": ODO_LATENCY_S, "horizons": list(horizons),
                    "bridge_samples_mean": float((j - k).mean())}
    return out


# ---- slices -------------------------------------------------------------------------------------------------------

SLICES = ("low speed (< 5 m/s)", "yaw-rate change (turn in / out)", "steady curve", "braking", "accelerating",
          "straight, steady", "other")


def slice_labels(sp: Split) -> np.ndarray:
    """One manoeuvre label per start point, from the recorded motion over the next 0.5 s (first match wins):
    low speed: wheel speed < 5 m/s; yaw-rate change: |r(t+0.5) - r(t)| > 3 deg/s; steady curve: |r| >= 4 deg/s;
    braking / accelerating: recorded speed change over 0.5 s <= -0.5 / >= +0.5 m/s (|a| >= 1 m/s^2); straight,
    steady: |r| < 1.5 deg/s and |speed change| < 0.25 m/s."""
    b, si, ki = sp.b, sp.si, sp.ki
    fut = D.interp_pose(b, si, (b["t"][si, ki] + 0.5)[:, None], keys=("r", "v_pose"))[:, 0]
    r0, r1 = b["r"][si, ki], fut[:, 0]
    dv = fut[:, 1] - b["v_pose"][si, ki]
    rmean = 0.5 * (r0 + r1)
    deg = np.radians
    lab = np.full(len(si), len(SLICES) - 1)
    conds = [b["v"][si, ki] < 5.0, np.abs(r1 - r0) > deg(3.0), np.abs(rmean) >= deg(4.0), dv <= -0.5, dv >= 0.5,
             (np.abs(rmean) < deg(1.5)) & (np.abs(dv) < 0.25)]
    for i in range(len(conds) - 1, -1, -1):
        lab[conds[i]] = i
    return lab


# ---- Python runtime ----------------------------------------------------------------------------------------------

def python_latency(test: Split, conf: dict, vcfg: dict, n: int = 1000) -> dict:
    """Wall time of one prediction (batch of 1, numpy) per method, and of one EKF step; plus the vectorised cost per
    prediction over all test start points."""
    rng = np.random.default_rng(SEED)
    pick = rng.choice(test.n, size=min(n, test.n), replace=False)
    b = test.b
    acc = M.accel_signal(b, conf["ctra"]["accel"])
    st = M.run_ekf(b, conf["ekf_params"])
    out = {}

    def time_calls(fn):
        ts = []
        for i in pick:
            a = time.perf_counter_ns()
            fn(test.si[i:i + 1], test.ki[i:i + 1])
            ts.append(time.perf_counter_ns() - a)
        ts = np.array(ts) / 1000.0
        return {"p50_us": float(np.percentile(ts, 50)), "p99_us": float(np.percentile(ts, 99)), "n": len(ts)}

    out["cv"] = time_calls(lambda s, k: M.predict_cv(b, s, k))
    out["ctrv"] = time_calls(lambda s, k: M.predict_ctrv(b, s, k))
    out["ctra"] = time_calls(lambda s, k: M.predict_ctra(b, s, k, acc))
    out["kinematic"] = time_calls(lambda s, k: M.predict_kinematic(b, s, k, vcfg["wheelbase_m"]))
    out["dynamic"] = time_calls(lambda s, k: M.predict_dynamic(b, s, k, vcfg))
    out["ekf"] = time_calls(lambda s, k: M.predict_from_state(st, s, k))
    for m in PR.LEARNED:
        out[m] = time_calls(lambda s, k, m=m: PR.predict_learned(conf["specs"][m], b, s, k, acc, st))
    # one EKF step (predict + 6 scalar updates) for a single filter
    p = conf["ekf_params"]
    var = [s_ * s_ for s_ in p.meas_std()]
    qd = np.array([p.q_xy, p.q_xy, p.q_psi, p.q_v, p.s_jerk ** 2, p.s_rdot ** 2])
    X, P, I6 = st[0, 60][None, :].copy(), np.eye(6)[None] * 1e-3, np.eye(6)
    z = np.array([b[key][0, 61] for key in ("x", "y", "yaw", "v", "r", p.accel)])
    ts = []
    for _ in range(len(pick)):
        a = time.perf_counter_ns()
        Xn, Pn = M.ekf_predict(X, P, np.array([0.02]), qd, I6)
        for jj, zi in enumerate(M.EKF_UPDATE_ORDER):
            Xn, Pn = M.ekf_update(Xn, Pn, zi, z[jj:jj + 1], var[jj])
        ts.append(time.perf_counter_ns() - a)
    ts = np.array(ts) / 1000.0
    out["ekf_step"] = {"p50_us": float(np.percentile(ts, 50)), "p99_us": float(np.percentile(ts, 99)), "n": len(ts)}
    # vectorised: all test start points at once
    vec = {}
    for m in PR.CPP:
        a = time.perf_counter_ns()
        PR.predict_all(b, test.si, test.ki, conf, vcfg, (m,), ekf_states=st)
        vec[m] = (time.perf_counter_ns() - a) / 1000.0 / test.n
    a = time.perf_counter_ns()
    M.run_ekf(b, conf["ekf_params"])
    vec["ekf_filter_per_sample"] = (time.perf_counter_ns() - a) / 1000.0 / float(sum(b.n))
    out["vectorised_us_per_prediction"] = vec
    return out
