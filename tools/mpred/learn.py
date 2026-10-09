"""Regression and data-driven predictors: features, ridge regression, a small MLP (numpy only), hybrids.

Targets, per horizon h (7 horizons x 4 = 28 outputs), in the frame of an anchor pose (xa, ya, yaw_a) at the
start sample:
  direct   the recorded displacement (dx, dy) in the anchor frame, the heading change and the speed change
           against the wheel speed at the start; anchor = the measured pose
  residual (hybrids) what the recorded pose differs from a physics prediction: (dx, dy) of truth minus base in
           the anchor frame, heading and speed differences; anchor = the EKF pose, base = EKF-CTRA

Features read samples k, k-5, ..., k-25 only (no future), see `phys_features` and `raw_features`;
stack/mpred/mpred.cc computes the same vectors from a ring buffer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tools.mpred.data import Batch
from tools.mpred.models import wrap

RAW_LAGS = (0, 5, 10, 15, 20, 25)
POSE_LAGS = (5, 10, 25)
PHYS_NAMES = ("1", "v", "a", "r", "d", "v*r", "v*d", "v*a", "v*v*d", "r-r5", "r-r25", "v*(r-r5)", "v*(r-r25)",
              "d-d5", "d-d25", "v*(d-d5)", "v*(d-d25)", "vw-vw25")


def phys_features(b: Batch, si, ki, accel: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """18 physically motivated features. v, a, r come from the filter state (x, y, yaw, v, a, r) when given,
    else from the measurements (wheel speed, the chosen acceleration signal, IMU yaw rate). d = road-wheel
    angle from the steering wheel (kinematic map)."""
    vw, rr, dd = b["v"], b["r"], b["d_kin"]
    if state is None:
        v, a, r = vw[si, ki], accel[si, ki], rr[si, ki]
    else:
        v, a, r = state[:, 3], state[:, 4], state[:, 5]
    d = dd[si, ki]
    r5, r25 = rr[si, ki] - rr[si, ki - 5], rr[si, ki] - rr[si, ki - 25]
    d5, d25 = d - dd[si, ki - 5], d - dd[si, ki - 25]
    cols = [np.ones_like(v), v, a, r, d, v * r, v * d, v * a, v * v * d, r5, r25, v * r5, v * r25, d5, d25, v * d5, v * d25,
            vw[si, ki] - vw[si, ki - 25]]
    return np.stack(cols, 1)


def raw_features(b: Batch, si, ki, accel: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """History window: wheel speed, yaw rate, road-wheel angle and acceleration at lags 0..25 samples (24), the
    past poses in the current pose frame at lags 5, 10, 25 (9); plus the filter's v, a, r when given (3)."""
    cols = [b[key][si, ki - lag] for key in ("v", "r", "d_kin") for lag in RAW_LAGS]
    cols += [accel[si, ki - lag] for lag in RAW_LAGS]
    x0, y0, p0 = b["x"][si, ki], b["y"][si, ki], b["yaw"][si, ki]
    c, s = np.cos(p0), np.sin(p0)
    for lag in POSE_LAGS:
        dx, dy = b["x"][si, ki - lag] - x0, b["y"][si, ki - lag] - y0
        cols += [c * dx + s * dy, -s * dx + c * dy, wrap(b["yaw"][si, ki - lag] - p0)]
    if state is not None:
        cols += [state[:, 3], state[:, 4], state[:, 5]]
    return np.stack(cols, 1)


# ---- targets ----------------------------------------------------------------------------------------------------

def to_anchor(anchor: np.ndarray, pose: np.ndarray, base_v: np.ndarray) -> np.ndarray:
    """(N, H, 4) global x, y, yaw, v -> (N, H*4) anchor-frame dx, dy, dyaw, dv. anchor (N, 3), base_v (N,) or (N, H)."""
    c, s = np.cos(anchor[:, 2])[:, None], np.sin(anchor[:, 2])[:, None]
    dx, dy = pose[..., 0] - anchor[:, 0, None], pose[..., 1] - anchor[:, 1, None]
    bv = base_v if base_v.ndim == 2 else base_v[:, None]
    out = np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(pose[..., 2] - anchor[:, 2, None]), pose[..., 3] - bv], -1)
    return out.reshape(len(anchor), -1)


def from_anchor(anchor: np.ndarray, y: np.ndarray, base_v: np.ndarray) -> np.ndarray:
    y = y.reshape(len(anchor), -1, 4)
    c, s = np.cos(anchor[:, 2])[:, None], np.sin(anchor[:, 2])[:, None]
    bv = base_v if base_v.ndim == 2 else base_v[:, None]
    return np.stack([anchor[:, 0, None] + c * y[..., 0] - s * y[..., 1], anchor[:, 1, None] + s * y[..., 0] + c * y[..., 1],
                     anchor[:, 2, None] + y[..., 2], bv + y[..., 3]], -1)


def residual_target(base: np.ndarray, tru: np.ndarray, anchor_yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(anchor_yaw)[:, None], np.sin(anchor_yaw)[:, None]
    dx, dy = tru[..., 0] - base[..., 0], tru[..., 1] - base[..., 1]
    out = np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(tru[..., 2] - base[..., 2]), tru[..., 3] - base[..., 3]], -1)
    return out.reshape(len(base), -1)


def apply_residual(base: np.ndarray, res: np.ndarray, anchor_yaw: np.ndarray) -> np.ndarray:
    res = res.reshape(base.shape)
    c, s = np.cos(anchor_yaw)[:, None], np.sin(anchor_yaw)[:, None]
    return np.stack([base[..., 0] + c * res[..., 0] - s * res[..., 1], base[..., 1] + s * res[..., 0] + c * res[..., 1],
                     base[..., 2] + res[..., 2], base[..., 3] + res[..., 3]], -1)


# ---- standardisation and ridge --------------------------------------------------------------------------------

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, keep_first: bool = False) -> Scaler:
        m, s = x.mean(0), x.std(0)
        s = np.where(s > 1e-12, s, 1.0)
        if keep_first:  # the bias column stays 1
            m, s = m.copy(), s.copy()
            m[0], s[0] = 0.0, 1.0
        return Scaler(m, s)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class Ridge:
    xs: Scaler
    w: np.ndarray  # (features, outputs) on standardised features
    lam: float

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, lam: float) -> Ridge:
        xs = Scaler.fit(x, keep_first=True)
        z = xs(x)
        reg = lam * np.eye(z.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(z.T @ z + reg, z.T @ y)
        return Ridge(xs, w, lam)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.xs(x) @ self.w

    def n_params(self) -> int:
        return int(self.w.size)


# ---- MLP --------------------------------------------------------------------------------------------------------

@dataclass
class MLP:
    xs: Scaler
    ys: Scaler
    layers: list[tuple[np.ndarray, np.ndarray]]  # (W, b) per layer; tanh on all but the last
    history: list[dict]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = self.xs(x)
        for i, (w, b_) in enumerate(self.layers):
            h = h @ w + b_
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h * self.ys.std + self.ys.mean

    def n_params(self) -> int:
        return int(sum(w.size + b_.size for w, b_ in self.layers))


def train_mlp(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, hidden: tuple[int, ...] = (64, 64),
              epochs: int = 60, batch: int = 512, lr: float = 1e-3, weight_decay: float = 1e-5, patience: int = 6,
              seed: int = 20261009) -> MLP:
    """Adam on the standardised MSE; keeps the epoch with the lowest validation MSE (early stopping)."""
    rng = np.random.default_rng(seed)
    xs, ys = Scaler.fit(x), Scaler.fit(y)
    X, Y = xs(x).astype(np.float32), ys(y).astype(np.float32)
    XV, YV = xs(xv).astype(np.float32), ys(yv).astype(np.float32)
    sizes = [X.shape[1], *hidden, Y.shape[1]]
    params = []
    for a, b_ in zip(sizes[:-1], sizes[1:], strict=True):
        lim = np.sqrt(6.0 / (a + b_))
        params += [rng.uniform(-lim, lim, (a, b_)).astype(np.float32), np.zeros(b_, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    nl = len(sizes) - 1

    def forward(xb):
        acts = [xb]
        h = xb
        for i in range(nl):
            h = h @ params[2 * i] + params[2 * i + 1]
            if i < nl - 1:
                h = np.tanh(h)
            acts.append(h)
        return acts

    def val_mse():
        return float(np.mean((forward(XV)[-1] - YV) ** 2))

    best, best_ep, best_params, hist = val_mse(), 0, [p.copy() for p in params], []
    for ep in range(1, epochs + 1):
        t0 = time.monotonic()
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            acts = forward(X[idx])
            g = 2.0 * (acts[-1] - Y[idx]) / (len(idx) * Y.shape[1])
            tot += float(np.sum((acts[-1] - Y[idx]) ** 2))
            grads = [None] * len(params)
            for li in range(nl - 1, -1, -1):
                grads[2 * li] = acts[li].T @ g + weight_decay * params[2 * li]
                grads[2 * li + 1] = g.sum(0)
                if li > 0:
                    g = (g @ params[2 * li].T) * (1.0 - acts[li] ** 2)
            step += 1
            for j, gp in enumerate(grads):
                m[j] = b1 * m[j] + (1 - b1) * gp
                v[j] = b2 * v[j] + (1 - b2) * gp * gp
                params[j] -= lr * (m[j] / (1 - b1 ** step)) / (np.sqrt(v[j] / (1 - b2 ** step)) + eps)
        vm = val_mse()
        hist.append({"epoch": ep, "train_mse": tot / (len(X) * Y.shape[1]), "val_mse": vm, "s": time.monotonic() - t0})
        if vm < best:
            best, best_ep, best_params = vm, ep, [p.copy() for p in params]
        elif ep - best_ep >= patience:
            break
    layers = [(best_params[2 * i].astype(np.float64), best_params[2 * i + 1].astype(np.float64)) for i in range(nl)]
    hist.append({"best_epoch": best_ep, "best_val_mse": best})
    return MLP(xs, ys, layers, hist)
