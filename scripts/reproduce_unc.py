"""Regenerates docs/UNCERTAINTY.md and its figures (docs/figures/unc_*.png): uncertainty estimation for the learned
ego motion predictor (deep ensemble, MC dropout, heteroscedastic MLP vs an ego-state failure monitor) and an
uncertainty-gated fallback from EKF + MLP to the EKF alone, on clean data, rare manoeuvres and sensor noise.

    python scripts/reproduce_unc.py [--scenes ~/workspace/drive-replay-data/can_scenes_v2] [--report-only]

Needs torch (CPU is enough, see tools/mpred/requirements.txt), the scene tables of tools/mpred/extract.py and the
v0.4.0 predictor config configs/mpred/predictors.json (tuned EKF, CTRA acceleration signal). Fits on the training
split, picks every threshold and scale on the validation split, reports on the test split. --report-only re-renders
from runs/unc/results.pkl. Every number in the document comes from this script. Never edit docs/UNCERTAINTY.md by hand.
"""
from __future__ import annotations

import argparse
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tools.mpred import data as D  # noqa: E402
from tools.mpred import experiment as X  # noqa: E402
from tools.mpred import learn as L  # noqa: E402
from tools.mpred import metrics as E  # noqa: E402
from tools.mpred import models as M  # noqa: E402
from tools.mpred import predictors as PR  # noqa: E402
from tools.mpred import uncertainty as U  # noqa: E402
from tools.vdyn import pipeline as P  # noqa: E402

DOCS, FIG, OUT = ROOT / "docs", ROOT / "docs" / "figures", ROOT / "runs" / "unc"
SEED = 20261011
NOISE = (2.0, 4.0)
BUDGET = 0.05  # fallback budget on the clean validation split
H = U.H_IDX
EGO_COLS = (1, 2, 3, 4, 9, 10, 13, 14, 17)  # v, a, r, d, r-r5, r-r25, d-d5, d-d25, vw-vw25 of learn.phys_features
SCORES = {"ens_total": "ensemble, total variance", "ens_epi": "ensemble, epistemic only", "ens_ale": "ensemble, aleatoric only",
          "het": "single heteroscedastic MLP", "mcd_total": "MC dropout, total variance", "mcd_epi": "MC dropout, epistemic only",
          "ego": "ego-state monitor (logistic regression)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"ens_total": "#4a3aa7", "ens_epi": "#2a78d6", "mcd_total": "#eb6834", "het": "#1baf7a", "ego": "#eda100",
         "random": "#9a9893", "oracle": "#52514e"}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def prep(sp: X.Split, conf: dict, b: D.Batch | None = None, extra: dict | None = None) -> dict:
    b = b or sp.b
    acc = M.accel_signal(b, conf["ctra"]["accel"])
    st_all = M.run_ekf(b, conf["ekf_params"], extra_std=extra)
    st = st_all[sp.si, sp.ki]
    base = M.rollout(st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5])
    det = PR.predict_learned(conf["specs"]["hybrid_mlp"], b, sp.si, sp.ki, acc, st_all)
    phys = L.phys_features(b, sp.si, sp.ki, acc)
    return {"f": L.raw_features(b, sp.si, sp.ki, acc, st), "y": L.residual_target(base, sp.tru, st[:, 2]), "base": base,
            "yaw": st[:, 2], "ekf": M.predict_from_state(st_all, sp.si, sp.ki), "det": det,
            "ego": np.abs(phys[:, EGO_COLS])}


def ov(pred: np.ndarray, tru: np.ndarray) -> np.ndarray:
    return E.errors(pred, tru)["ov20"][..., H]


def evaluate(models: dict, pp: dict, sp: X.Split) -> dict:
    """Signed 200 ms / 20 m overlay errors and predictive variances of every model on one condition."""
    out = {"si": sp.si, "ekf": ov(pp["ekf"], sp.tru), "det": ov(pp["det"], sp.tru), "ego_x": pp["ego"]}
    for key, passes in (("ens", 1), ("mcd", U.MC_PASSES)):
        mus, vas = U.sample_outputs(models[key], pp["f"], passes=passes)
        e_k = np.stack([ov(L.apply_residual(pp["base"], m, pp["yaw"]), sp.tru) for m in mus])
        out[key] = ov(L.apply_residual(pp["base"], mus.mean(0), pp["yaw"]), sp.tru)
        out[key + "_ale"] = U.overlay_var_aleatoric(vas).mean(0)
        out[key + "_epi"] = e_k.var(0)
        if key == "ens":
            out["het"] = e_k[0]
            out["het_ale"] = U.overlay_var_aleatoric(vas[:1])[0]
    return out


def scores(r: dict, ego_model) -> dict:
    return {"ens_total": r["ens_ale"] + r["ens_epi"], "ens_epi": r["ens_epi"], "ens_ale": r["ens_ale"], "het": r["het_ale"],
            "mcd_total": r["mcd_ale"] + r["mcd_epi"], "mcd_epi": r["mcd_epi"], "ego": U.predict_logistic(ego_model, r["ego_x"])}


def run_all(scenes: Path) -> dict:
    torch.set_num_threads(16)
    t_all = time.monotonic()
    vcfg, conf = D.vehicle_config(), PR.load_config()
    log("loading splits")
    tr = X.make_split("train", stride=X.TRAIN_STRIDE, scenes_dir=scenes, cfg=vcfg)
    va = X.make_split("val", scenes_dir=scenes, cfg=vcfg)
    te = X.make_split("test", scenes_dir=scenes, cfg=vcfg)
    ptr, pva, pte = prep(tr, conf), prep(va, conf), prep(te, conf)
    R: dict = {"n": {"train": tr.n, "val": va.n, "test": te.n, "test_scenes": te.b.S, "test_hours": D.hours(te.b)}}

    log("training the deep ensemble")
    t0 = time.monotonic()
    ens = U.fit_ensemble(ptr["f"], ptr["y"], pva["f"], pva["y"], [SEED + i for i in range(U.N_MEMBERS)], log=log)
    R["train_s_ens"] = time.monotonic() - t0
    log("training the MC dropout network")
    t0 = time.monotonic()
    mcd = U.fit_ensemble(ptr["f"], ptr["y"], pva["f"], pva["y"], [SEED + 100], dropout=U.DROPOUT_P, log=log)
    R["train_s_mcd"] = time.monotonic() - t0
    R["history"] = {"ens": ens.history, "mcd": mcd.history}
    models = {"ens": ens, "mcd": mcd}

    # ---- validation: failure threshold, variance scales, ego monitor, gate thresholds ----
    rv = evaluate(models, pva, va)
    tau = float(np.percentile(np.abs(rv["ens"]), 95))
    fail_v = np.abs(rv["ens"]) > tau
    ego_model = U.fit_logistic(rv["ego_x"], fail_v.astype(float))
    sv = scores(rv, ego_model)
    scale = {k: U.var_scale(rv[k], np.sqrt(rv[k + "_ale"] + rv.get(k + "_epi", 0.0))) for k in ("ens", "mcd")}
    scale["het"] = U.var_scale(rv["het"], np.sqrt(rv["het_ale"]))
    gate = {k: float(np.percentile(s, 100 * (1 - BUDGET))) for k, s in sv.items()}
    R["val"] = {"tau_fail_m": tau, "scale": scale, "gate": gate, "fail_rate": float(fail_v.mean()),
                "ego_auroc_val": U.auroc(sv["ego"], fail_v)}
    log(f"failure threshold {tau * 100:.2f} cm; scales {scale}")

    # ---- test conditions ----
    vb = X.make_split("test", scenes_dir=scenes, cfg=vcfg)  # fresh copy for the perturbations
    conds = {"clean": evaluate(models, pte, te)}
    for lev in NOISE:
        rng = np.random.default_rng(X.SEED + int(lev * 100))  # same draws as the v0.4.0 noise sweep
        nb, extra, _ = X.perturb(vb.b, vcfg, {k: v * lev for k, v in X.BASE_SIGMA.items()}, rng)
        conds[f"noise x{lev:g}"] = evaluate(models, prep(te, conf, nb, extra), te)
        log(f"noise level {lev:g} done")
    labels = X.slice_labels(te)
    W = P.bootstrap_weights(te.b.S, b=1000, seed=SEED)
    R["conds"] = {}
    for name, r in conds.items():
        R["conds"][name] = summarize(r, scores(r, ego_model), tau, gate, scale, W, te.b.S)
        log(f"{name}: ens AUROC {R['conds'][name]['auroc']['ens_total']:.3f}, ego {R['conds'][name]['auroc']['ego']:.3f}")
    rc = conds["clean"]
    sc = scores(rc, ego_model)
    # shift detection: can the score tell a noisy frame from a clean one (same frames, both conditions)?
    R["ood"] = {}
    for name, r in conds.items():
        if name == "clean":
            continue
        sn = scores(r, ego_model)
        lab = np.r_[np.zeros(len(rc["si"]), bool), np.ones(len(r["si"]), bool)]
        R["ood"][name] = {k: U.auroc(np.r_[sc[k], sn[k]], lab) for k in SCORES}
    R["slices"] = {}
    for i, nm in enumerate(X.SLICES):
        m = labels == i
        f = np.abs(rc["ens"][m]) > tau
        R["slices"][nm] = {"n": int(m.sum()), "fail_rate": float(f.mean()),
                           "auroc": {k: U.auroc(sc[k][m], f) for k in ("ens_total", "ens_epi", "mcd_total", "ego")},
                           "mean": {k: float(np.abs(rc[k][m]).mean()) for k in ("ekf", "det", "ens")},
                           "gated": float(np.abs(np.where(sc["ens_total"][m] > gate["ens_total"], rc["ekf"][m], rc["ens"][m])).mean()),
                           "fallback": float((sc["ens_total"][m] > gate["ens_total"]).mean())}
    R["curves"] = {n: curves(r, scores(r, ego_model), scale) for n, r in conds.items() if n in ("clean", f"noise x{NOISE[-1]:g}")}
    R["runtime"] = runtime(models, pte["f"])
    R["env"] = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
                "seed": SEED, "total_s": time.monotonic() - t_all}
    return R


def summarize(r: dict, s: dict, tau: float, gate: dict, scale: dict, W: np.ndarray, S: int) -> dict:
    fail = np.abs(r["ens"]) > tau
    si = r["si"]
    out = {"n": len(si), "fail_rate": float(fail.mean()), "tau": tau}
    out["acc"] = {k: {"mean": float(np.abs(r[k]).mean()), "p95": E.p95(r[k])} for k in ("ekf", "det", "het", "ens", "mcd")}
    cal = {}
    for k in ("het", "ens", "mcd"):
        sig = np.sqrt(r[k + "_ale"] + (r[k + "_epi"] if k != "het" else 0.0))
        cal[k] = {"nll": U.gaussian_nll(r[k], sig), "cal": U.calibration_error(r[k], sig),
                  "nll_s": U.gaussian_nll(r[k], scale[k] * sig), "cal_s": U.calibration_error(r[k], scale[k] * sig),
                  "sigma_cm": float(np.mean(scale[k] * sig) * 100), "epi_share": float(np.mean(r[k + "_epi"] / sig ** 2)) if k != "het" else 0.0}
    out["cal"] = cal
    out["auroc"] = {k: U.auroc(v, fail) for k, v in s.items()}
    out["rc90"] = {k: float(U.risk_coverage(v, r["ens"])[0][40]) for k, v in s.items()}  # grid[40] = 0.90 coverage
    out["aurc"] = {k: U.risk_coverage(v, r["ens"])[1] for k, v in s.items()}
    # bootstrap over scenes: AUROC difference ensemble - ego monitor
    diff = np.array([U.auroc(s["ens_total"], fail, W[b, si]) - U.auroc(s["ego"], fail, W[b, si]) for b in range(len(W))])
    out["auroc_diff"] = {"value": out["auroc"]["ens_total"] - out["auroc"]["ego"], "lo": float(np.nanpercentile(diff, 2.5)),
                         "hi": float(np.nanpercentile(diff, 97.5))}
    # gated fallback to the EKF at the validation thresholds
    g = {"always learned (EKF + MLP ensemble)": (np.zeros(len(si), bool), r["ens"]), "always EKF": (np.ones(len(si), bool), r["ekf"])}
    for k in ("ens_total", "ens_epi", "mcd_total", "ego"):
        fb = s[k] > gate[k]
        g[f"gate on {SCORES[k]}"] = (fb, np.where(fb, r["ekf"], r["ens"]))
    out["gate"] = {}
    base_sums = E.scene_sums(r["ens"], si, S)
    for name, (fb, e) in g.items():
        sums = E.scene_sums(e, si, S)
        out["gate"][name] = {"fallback": float(fb.mean()), "mean": float(np.abs(e).mean()), "p95": E.p95(e),
                             "fail": float((np.abs(e) > tau).mean()), "vs_learned": P.paired(sums, base_sums, W)}
    return out


def curves(r: dict, s: dict, scale: dict) -> dict:
    grid = U.RC_GRID
    rc = {k: U.risk_coverage(s[k], r["ens"], grid)[0] for k in ("ens_total", "ens_epi", "mcd_total", "het", "ego")}
    rc["oracle"] = U.risk_coverage(np.abs(r["ens"]), r["ens"], grid)[0]
    rc["random"] = np.full(len(grid), np.abs(r["ens"]).mean())
    rel = {}
    for k in ("het", "ens", "mcd"):
        sig = np.sqrt(r[k + "_ale"] + (r[k + "_epi"] if k != "het" else 0.0))
        rel[k] = (U.coverage(r[k], sig), U.coverage(r[k], scale[k] * sig))
    return {"grid": grid, "rc": rc, "rel": rel}


def runtime(models: dict, f: np.ndarray, n: int = 500) -> dict:
    """Wall time of one prediction (batch of 1, PyTorch CPU, 1 thread): single net, ensemble, MC dropout."""
    torch.set_num_threads(1)
    out = {}
    for name, key, k, passes in (("single heteroscedastic MLP", "ens", 1, 1), (f"deep ensemble, {U.N_MEMBERS} members", "ens", None, 1),
                                 (f"MC dropout, {U.MC_PASSES} passes", "mcd", None, U.MC_PASSES)):
        fm = models[key]
        sub = U.Fitted(fm.xs, fm.ys, fm.nets[:k] if k else fm.nets, fm.dropout, [])
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            U.sample_outputs(sub, f[i:i + 1], passes=passes)
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts[20:]) * 1e6
        out[name] = {"p50_us": float(np.median(ts)), "p99_us": float(np.percentile(ts, 99))}
    torch.set_num_threads(16)
    return out


# ---- report -----------------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def cm(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}"


def ci_cm(p: dict) -> str:
    v = p["value"] if "value" in p else p["diff"]
    return f"{v * 100:+.2f} [{p['lo'] * 100:+.2f}, {p['hi'] * 100:+.2f}]"


def figures(R: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": 150, "bbox_inches": "tight"}

    def style(ax, title, xl, yl):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xl, fontsize=9, color=MUTED)
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cond in zip(axes, R["curves"], strict=True):
        c = R["curves"][cond]
        for k, lab in (("ens_total", "ensemble (total)"), ("mcd_total", "MC dropout"), ("het", "single het. MLP"),
                       ("ego", "ego-state monitor"), ("random", "random"), ("oracle", "oracle")):
            ax.plot(c["grid"] * 100, c["rc"][k] * 100, color=COLOR[k], lw=1.6 if k not in ("random", "oracle") else 1.0,
                    ls="--" if k in ("random", "oracle") else "-", label=lab)
        style(ax, f"Risk-coverage, test, {cond}", "coverage: frames kept (%)", "mean |20 m overlay offset| at 200 ms (cm)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(FIG / "unc_risk_coverage.png", **kw)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cond in zip(axes, R["curves"], strict=True):
        c = R["curves"][cond]
        ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=":")
        for k, col, lab in (("ens", COLOR["ens_total"], "ensemble"), ("mcd", COLOR["mcd_total"], "MC dropout"), ("het", COLOR["het"], "single het.")):
            raw, scaled = c["rel"][k]
            ax.plot(U.LEVELS * 100, raw * 100, color=col, lw=1.0, ls="--")
            ax.plot(U.LEVELS * 100, scaled * 100, color=col, lw=1.6, label=f"{lab} (solid: val-scaled, dashed: raw)")
        style(ax, f"Interval coverage, test, {cond}", "nominal coverage (%)", "observed coverage (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "unc_reliability.png", **kw)
    plt.close(fig)


def render(R: dict) -> str:
    out: list[str] = []
    add = out.append
    v, n = R["val"], R["n"]
    add("# Uncertainty estimation and a gated fallback for the learned motion predictor\n")
    add("Generated by `scripts/reproduce_unc.py`. Do not edit by hand.\n")
    add("## Question\n")
    add("The EKF + MLP residual hybrid is the most accurate predictor in [MOTION_PREDICTION.md](MOTION_PREDICTION.md) on clean "
        "data, and the least robust one under sensor noise. Production dual stacks pair a learned driving model with a classic "
        "stack and hand over when the learned part does not know what to do. This study asks the same question at small "
        "scale: **which uncertainty signal flags, frame by frame, that the learned prediction is about to fail, and does a "
        "fallback to the EKF alone on that signal help on rare manoeuvres and under sensor noise?**\n")
    add("## Setup\n")
    add(f"- Data: nuScenes CAN bus expansion, split by log (configs/vdyn/split.json). Train {n['train']:,} start points "
        f"(stride 2), validation {n['val']:,}, test {n['test']:,} in {n['test_scenes']} scenes ({n['test_hours']:.2f} h).")
    add("- Learned model: same inputs and residual targets as `hybrid_mlp` (36 inputs: signal history, past poses, EKF state; "
        "28 outputs). Heteroscedastic heads (mean + log variance per output), Gaussian NLL, Adam, early stopping on the "
        "validation NLL. PyTorch on CPU.")
    add(f"- Methods: deep ensemble of {U.N_MEMBERS} members (different seeds); MC dropout (p = {U.DROPOUT_P}, {U.MC_PASSES} "
        "passes); a single heteroscedastic MLP (ensemble member 0, aleatoric only); and a cheap baseline that uses no model "
        "internals, a logistic regression on the ego state (|speed|, |acceleration|, |yaw rate|, |steering| and their recent "
        "changes) trained on validation failure labels.")
    add("- Monitored quantity: the 20 m overlay offset at the 200 ms horizon (cm). Aleatoric variance is propagated from the "
        "residual outputs as var(dy) + 20^2 var(dyaw) (small angle, diagonal); epistemic variance is the spread of the member "
        "(or pass) offsets, computed exactly.")
    add(f"- Failure: |offset| of the ensemble mean above **{cm(v['tau_fail_m'])} cm**, the 95th percentile on the clean "
        "validation split (so 5 % of clean validation frames are failures by construction). The threshold stays fixed under shift.")
    add(f"- Every scale and threshold comes from the clean validation split: variance scales (NLL-optimal) ens "
        f"{v['scale']['ens']:.2f}, MC dropout {v['scale']['mcd']:.2f}, single {v['scale']['het']:.2f}; gate thresholds at a "
        f"{BUDGET * 100:.0f} % fallback budget on clean validation data.")
    add("- Shift: the v0.4.0 noise sweep inputs at 2x and 4x the base noise (pose, wheel speed, yaw rate, acceleration, "
        "steering), same random draws; the EKF is told the extra noise, the networks are not (they were trained on clean data).")
    add("- Statistics: 95 % percentile bootstrap intervals over test scenes (1,000 resamples).\n")

    cl = R["conds"]["clean"]
    nk = f"noise x{NOISE[-1]:g}"
    nz = R["conds"][nk]
    add("## Results in short\n")
    add(f"- Accuracy: the ensemble mean is at {cm(cl['acc']['ens']['mean'])} cm mean |offset| on clean test data vs "
        f"{cm(cl['acc']['det']['mean'])} cm for the deterministic v0.4.0 `hybrid_mlp` and {cm(cl['acc']['ekf']['mean'])} cm "
        f"for the EKF alone. Under {nk}: ensemble {cm(nz['acc']['ens']['mean'])} cm, EKF {cm(nz['acc']['ekf']['mean'])} cm.")
    add(f"- Failure detection, clean test (AUROC): ensemble total {cl['auroc']['ens_total']:.3f}, MC dropout "
        f"{cl['auroc']['mcd_total']:.3f}, single heteroscedastic {cl['auroc']['het']:.3f}, ego-state monitor "
        f"{cl['auroc']['ego']:.3f}. Ensemble minus ego monitor: {cl['auroc_diff']['value']:+.3f} "
        f"[{cl['auroc_diff']['lo']:+.3f}, {cl['auroc_diff']['hi']:+.3f}].")
    add(f"- Shift detection (AUROC, clean vs {nk} frames): ensemble total {R['ood'][nk]['ens_total']:.3f}, epistemic only "
        f"{R['ood'][nk]['ens_epi']:.3f}, MC dropout {R['ood'][nk]['mcd_total']:.3f}, single heteroscedastic {R['ood'][nk]['het']:.3f}, "
        f"ego monitor {R['ood'][nk]['ego']:.3f}. The ensemble's epistemic share of the variance rises from "
        f"{cl['cal']['ens']['epi_share'] * 100:.0f} % (clean) to {nz['cal']['ens']['epi_share'] * 100:.0f} %.")
    add(f"- Calibration under {nk}: calibration error ensemble {nz['cal']['ens']['cal_s']:.3f}, MC dropout "
        f"{nz['cal']['mcd']['cal_s']:.3f}, single {nz['cal']['het']['cal_s']:.3f} (clean: {cl['cal']['ens']['cal_s']:.3f} / "
        f"{cl['cal']['mcd']['cal_s']:.3f} / {cl['cal']['het']['cal_s']:.3f}).")
    add(f"- Under {nk} (failure rate {nz['fail_rate'] * 100:.1f} % vs {cl['fail_rate'] * 100:.1f} % clean): ensemble "
        f"{nz['auroc']['ens_total']:.3f}, epistemic only {nz['auroc']['ens_epi']:.3f}, MC dropout {nz['auroc']['mcd_total']:.3f}, "
        f"ego monitor {nz['auroc']['ego']:.3f}; difference ensemble - ego {nz['auroc_diff']['value']:+.3f} "
        f"[{nz['auroc_diff']['lo']:+.3f}, {nz['auroc_diff']['hi']:+.3f}].")
    gk = "gate on " + SCORES["ens_total"]
    add(f"- Gated fallback on the ensemble's total variance: clean {cl['gate'][gk]['fallback'] * 100:.1f} % of frames fall back, "
        f"mean |offset| {ci_cm(cl['gate'][gk]['vs_learned'])} cm vs always learned; under {nk} "
        f"{nz['gate'][gk]['fallback'] * 100:.1f} % fall back, {ci_cm(nz['gate'][gk]['vs_learned'])} cm vs always learned "
        f"(always EKF: {ci_cm(nz['gate']['always EKF']['vs_learned'])} cm).\n")

    add("## Accuracy (200 ms, 20 m overlay offset, cm)\n")
    rows = []
    for name, c in R["conds"].items():
        rows.append([name] + [f"{cm(c['acc'][k]['mean'])} / {cm(c['acc'][k]['p95'])}" for k in ("ekf", "det", "het", "ens", "mcd")])
    add(table(["condition", "EKF alone", "EKF + MLP v0.4.0 (MSE)", "single het.", "ensemble mean", "MC dropout mean"], rows))
    add("\nmean |offset| / 95th percentile. NLL training costs some mean accuracy against the MSE-trained v0.4.0 model; the "
        "ensemble mean recovers part of it.\n")

    add("## Calibration (Gaussian predictive interval of the offset)\n")
    rows = []
    for name, c in R["conds"].items():
        for k, lab in (("het", "single het."), ("ens", "ensemble"), ("mcd", "MC dropout")):
            q = c["cal"][k]
            rows.append([name, lab, f"{q['nll']:.3f}", f"{q['nll_s']:.3f}", f"{q['cal']:.3f}", f"{q['cal_s']:.3f}",
                         f"{q['sigma_cm']:.2f}", "-" if k == "het" else f"{q['epi_share'] * 100:.0f} %"])
    add(table(["condition", "model", "NLL raw", "NLL scaled", "calib. error raw", "calib. error scaled", "mean sigma (cm, scaled)",
               "epistemic share"], rows))
    add("\nCalibration error = mean |observed - nominal coverage| of the central intervals at 5 ... 95 % (0 = perfect). "
        "Scaled = sigma times the NLL-optimal factor from the clean validation split. NLL in log-metres.\n")
    add("![interval coverage](figures/unc_reliability.png)\n")

    add("## Failure detection (AUROC) and selective prediction\n")
    rows = []
    for k, lab in SCORES.items():
        rows.append([lab] + [f"{c['auroc'][k]:.3f}" for c in R["conds"].values()] +
                    [f"{cm(c['rc90'][k])}" for c in R["conds"].values()])
    conds = list(R["conds"])
    add(table(["score"] + [f"AUROC {c}" for c in conds] + [f"mean at 90 % cov. {c}" for c in conds], rows))
    add("\nFailure rates: " + ", ".join(f"{c} {R['conds'][c]['fail_rate'] * 100:.1f} %" for c in conds) +
        f". The ego-state monitor reaches AUROC {v['ego_auroc_val']:.3f} on the validation split it was trained on. "
        "'mean at 90 % cov.' = mean |offset| (cm) of the 90 % of frames with the lowest score.\n")
    add("![risk-coverage](figures/unc_risk_coverage.png)\n")

    add("## Shift detection (frame level)\n")
    rows = [[lab] + [f"{R['ood'][c][k]:.3f}" for c in R["ood"]] for k, lab in SCORES.items()]
    add(table(["score"] + [f"AUROC clean vs {c}" for c in R["ood"]], rows))
    add("\nThe same test frames scored once clean and once with input noise; AUROC of the score for 'this frame is noisy'. "
        "Per-frame failure ranking inside a shifted condition is weak for every score (table above), because most frames fail; "
        "what the variance does carry is the condition itself.\n")
    add("## Rare manoeuvres (clean test)\n")
    rows = []
    for nm, s in R["slices"].items():
        rows.append([nm, f"{s['n']:,}", f"{s['fail_rate'] * 100:.1f} %", f"{s['auroc']['ens_total']:.3f}", f"{s['auroc']['mcd_total']:.3f}",
                     f"{s['auroc']['ego']:.3f}", cm(s["mean"]["ens"]), cm(s["mean"]["ekf"]), cm(s["gated"]), f"{s['fallback'] * 100:.1f} %"])
    add(table(["slice", "frames", "failure rate", "AUROC ens.", "AUROC MCD", "AUROC ego", "ens. mean (cm)", "EKF (cm)",
               "gated (cm)", "fallback"], rows))
    add("\nSlices as in MOTION_PREDICTION.md (labels from the recorded motion over the next 0.5 s; first match wins).\n")

    add("## Gated fallback to the EKF\n")
    for name, c in R["conds"].items():
        add(f"**{name}**\n")
        rows = [[g, f"{q['fallback'] * 100:.1f} %", cm(q["mean"]), cm(q["p95"]), f"{q['fail'] * 100:.1f} %", ci_cm(q["vs_learned"])]
                for g, q in c["gate"].items()]
        add(table(["policy", "fallback", "mean (cm)", "p95 (cm)", "failure rate", "mean vs always learned (cm, 95 % CI)"], rows))
        add("")
    add("## Cost\n")
    rows = [[k, f"{q['p50_us']:.0f}", f"{q['p99_us']:.0f}"] for k, q in R["runtime"].items()]
    add(table(["model", "p50 (us)", "p99 (us)"], rows))
    add(f"\nOne prediction, batch of 1, PyTorch {R['env']['torch']} on 1 CPU thread (Python overhead included; not the C++ path "
        f"of stack/mpred). Training: ensemble {R['train_s_ens'] / 60:.1f} min, MC dropout {R['train_s_mcd'] / 60:.1f} min on "
        f"{R['env']['threads']} threads.\n")
    add("## Limits\n")
    add("- Open-loop ego motion prediction on 192 test scenes of one vehicle (Renault Zoe, Boston and Singapore), not an "
        "end-to-end driving model and not closed-loop planning.")
    add("- The noise shift is synthetic (white Gaussian on the inputs) and therefore easy to detect (even the ego-state "
        "monitor separates it); real sensor faults are structured and subtler. The shift-detection AUROCs are an upper bound.")
    add("- The aleatoric propagation to the overlay offset is a small-angle, diagonal approximation.")
    add("- The failure threshold is the 95th percentile of the clean validation error, a choice; other thresholds change the AUROCs.")
    add(f"- Seeds fixed ({R['env']['seed']}); one training run per model, no seed-to-seed spread reported.\n")
    add(f"Environment: Python {R['env']['python']}, PyTorch {R['env']['torch']}, numpy {R['env']['numpy']}; full run "
        f"{R['env']['total_s'] / 60:.1f} min.")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, default=D.DEFAULT_SCENES)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pkl = OUT / "results.pkl"
    if a.report_only:
        R = pickle.loads(pkl.read_bytes())
    else:
        R = run_all(a.scenes)
        pkl.write_bytes(pickle.dumps(R))
    figures(R)
    (DOCS / "UNCERTAINTY.md").write_text(render(R))
    log("wrote docs/UNCERTAINTY.md")


if __name__ == "__main__":
    main()
