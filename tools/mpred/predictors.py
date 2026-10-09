"""The predictor set: method names, the JSON config read by stack/mpred, and one call that runs every method.

configs/mpred/predictors.json holds what was tuned or fitted (written by scripts/reproduce_mpred.py): the CTRA
acceleration source, the EKF parameters and the learned models (standardisation + layers). The Python numbers in
docs/MOTION_PREDICTION.md are computed from exactly these exported weights (`LearnedSpec`), and the C++ library
reads the same file, so the parity check compares the same model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.mpred import learn as L
from tools.mpred import models as M
from tools.mpred.data import HORIZONS, ROOT, Batch

CONFIG = ROOT / "configs" / "mpred" / "predictors.json"

METHODS = ("cv", "ctrv", "ctra", "kinematic", "dynamic", "kf_cv", "kf_ca", "ekf", "ridge", "mlp", "hybrid_ridge",
           "hybrid_mlp")
LEARNED = ("ridge", "mlp", "hybrid_ridge", "hybrid_mlp")
CPP = ("cv", "ctrv", "ctra", "kinematic", "dynamic", "ekf", *LEARNED)
LABEL = {
    "cv": "CV (constant velocity)",
    "ctrv": "CTRV (constant turn rate and velocity)",
    "ctra": "CTRA (constant turn rate and acceleration)",
    "kinematic": "kinematic single track, inputs held",
    "dynamic": "linear dynamic single track, inputs held",
    "kf_cv": "KF-CV (linear Kalman, pose position only)",
    "kf_ca": "KF-CA (linear Kalman, pose position only)",
    "ekf": "EKF-CTRA (pose + wheel speed + IMU yaw rate + accel)",
    "ridge": "ridge regression on physical features",
    "mlp": "MLP on the raw history window",
    "hybrid_ridge": "hybrid: EKF-CTRA + ridge residual",
    "hybrid_mlp": "hybrid: EKF-CTRA + MLP residual",
    "hybrid_mlp_nopose": "hybrid MLP without the pose-history inputs (ablation)",
}
FAMILY = {"cv": "kinematic model", "ctrv": "kinematic model", "ctra": "kinematic model", "kinematic": "kinematic model",
          "dynamic": "dynamic model", "kf_cv": "Kalman filter", "kf_ca": "Kalman filter", "ekf": "Kalman filter",
          "ridge": "regression", "mlp": "data-driven", "hybrid_ridge": "hybrid", "hybrid_mlp": "hybrid",
          "hybrid_mlp_nopose": "hybrid"}


@dataclass
class LearnedSpec:
    """A learned predictor as exported: features, standardisation, layers (tanh between), output scaling."""
    name: str
    features: str  # "phys" or "raw"
    uses_state: bool
    residual: bool
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]
    drop_pose: bool = False  # ablation only (not exportable to C++)

    def forward(self, f: np.ndarray) -> np.ndarray:
        h = (f - self.x_mean) / self.x_std
        for i, (w, b_) in enumerate(self.layers):
            h = h @ w + b_
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h * self.y_std + self.y_mean

    def n_params(self) -> int:
        return int(sum(w.size + b_.size for w, b_ in self.layers))

    def to_json(self) -> dict:
        return {"features": self.features, "uses_state": self.uses_state, "residual": self.residual,
                "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(), "y_mean": self.y_mean.tolist(),
                "y_std": self.y_std.tolist(),
                "layers": [{"in": int(w.shape[0]), "out": int(w.shape[1]), "w": w.ravel().tolist(), "b": b_.tolist()}
                           for w, b_ in self.layers]}

    @staticmethod
    def from_json(name: str, d: dict) -> LearnedSpec:
        layers = [(np.array(lay["w"], dtype=float).reshape(lay["in"], lay["out"]), np.array(lay["b"], dtype=float))
                  for lay in d["layers"]]
        return LearnedSpec(name, d["features"], d["uses_state"], d["residual"], np.array(d["x_mean"]), np.array(d["x_std"]),
                           np.array(d["y_mean"]), np.array(d["y_std"]), layers)

    @staticmethod
    def from_ridge(name: str, m: L.Ridge, features: str, uses_state: bool, residual: bool) -> LearnedSpec:
        n_out = m.w.shape[1]
        return LearnedSpec(name, features, uses_state, residual, m.xs.mean.astype(float), m.xs.std.astype(float),
                           np.zeros(n_out), np.ones(n_out), [(m.w.astype(float), np.zeros(n_out))])

    @staticmethod
    def from_mlp(name: str, m: L.MLP, features: str, uses_state: bool, residual: bool, drop_pose: bool = False) -> LearnedSpec:
        return LearnedSpec(name, features, uses_state, residual, m.xs.mean.astype(float), m.xs.std.astype(float),
                           m.ys.mean.astype(float), m.ys.std.astype(float), m.layers, drop_pose)


def features(spec_features: str, b: Batch, si, ki, accel: np.ndarray, state: np.ndarray | None, drop_pose: bool = False):
    if spec_features == "phys":
        return L.phys_features(b, si, ki, accel, state)
    f = L.raw_features(b, si, ki, accel, state)
    if drop_pose:  # the 9 pose-history columns sit after the 24 signal columns
        f = np.concatenate([f[:, :24], f[:, 33:]], 1)
    return f


def predict_learned(spec: LearnedSpec, b: Batch, si, ki, accel: np.ndarray, ekf_states: np.ndarray | None) -> np.ndarray:
    st = ekf_states[si, ki] if ekf_states is not None else None
    f = features(spec.features, b, si, ki, accel, st if spec.uses_state else None, spec.drop_pose)
    y = spec.forward(f)
    if spec.residual:
        base = M.rollout(st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4], st[:, 5])
        return L.apply_residual(base, y, st[:, 2])
    anchor = np.stack([b["x"][si, ki], b["y"][si, ki], b["yaw"][si, ki]], 1)
    return L.from_anchor(anchor, y, b["v"][si, ki])


def save_config(path: Path, ctra_accel: str, kf: dict, ekf: M.EkfParams, learned: dict[str, LearnedSpec], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_comment": "Ego motion predictors for AR-HUD latency compensation. Generated by scripts/reproduce_mpred.py "
                       "(tuned on the validation split, learned models fitted on the training split of "
                       "configs/vdyn/split.json); do not edit by hand. Read by stack/mpred (C++) and tools/mpred.",
           "horizons_s": list(HORIZONS), "ctra": {"accel": ctra_accel}, "kf": kf, "ekf": ekf.as_dict(),
           "learned": {k: v.to_json() for k, v in sorted(learned.items())}, "meta": meta}
    path.write_text(json.dumps(doc, indent=None, separators=(",", ":")) + "\n")


def load_config(path: Path | None = None) -> dict:
    d = json.loads((path or CONFIG).read_text())
    d["ekf_params"] = M.EkfParams(**d["ekf"])
    d["kf_params"] = {k: M.KfParams(**v) for k, v in d["kf"].items()}
    d["specs"] = {k: LearnedSpec.from_json(k, v) for k, v in d["learned"].items()}
    return d


def predict_all(b: Batch, si, ki, conf: dict, vehicle_cfg: dict, methods=METHODS, extra_std: dict | None = None,
                odo_avail: np.ndarray | None = None, ekf_states: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every requested method at the start points (si, ki) of batch b. extra_std / odo_avail describe injected
    noise and lost odometry samples so the filters can account for them (noise-aware R, skipped updates)."""
    out = {}
    acc = M.accel_signal(b, conf["ctra"]["accel"])
    need_ekf = any(m in methods for m in ("ekf", "hybrid_ridge", "hybrid_mlp", "hybrid_mlp_nopose")) or any(
        conf["specs"][m].uses_state for m in methods if m in conf["specs"])
    if need_ekf and ekf_states is None:
        ekf_states = M.run_ekf(b, conf["ekf_params"], odo_avail=odo_avail, extra_std=extra_std)
    for m in methods:
        if m == "cv":
            out[m] = M.predict_cv(b, si, ki)
        elif m == "ctrv":
            out[m] = M.predict_ctrv(b, si, ki)
        elif m == "ctra":
            out[m] = M.predict_ctra(b, si, ki, acc)
        elif m == "kinematic":
            out[m] = M.predict_kinematic(b, si, ki, vehicle_cfg["wheelbase_m"])
        elif m == "dynamic":
            out[m] = M.predict_dynamic(b, si, ki, vehicle_cfg)
        elif m in ("kf_cv", "kf_ca"):
            ev = (extra_std or {}).get("pose_xy", 0.0) ** 2
            st = M.run_kf(b, conf["kf_params"][m], extra_var=ev)
            out[m] = M.predict_kf(st, si, ki, HORIZONS)
        elif m == "ekf":
            out[m] = M.predict_from_state(ekf_states, si, ki)
        else:
            out[m] = predict_learned(conf["specs"][m], b, si, ki, acc, ekf_states)
    return out
