"""Uncertainty estimation for the learned motion predictor (EKF + MLP residual), and an uncertainty-gated fallback
to the physics filter.

Setting: the EKF + MLP hybrid of v0.4.0 is the most accurate predictor on clean data, but it is the least robust one
under sensor noise. Mercedes-style dual stacks run a learned planner next to a classic stack and hand over when the
learned part "does not know what to do". Here the learned part is the residual MLP and the classic part is the
EKF-CTRA rollout. The question: which uncertainty signal tells, per frame, that the learned prediction is about to
be wrong, and does gating on it help on rare manoeuvres and under sensor noise?

Models (PyTorch, CPU), all on the same inputs and targets as `hybrid_mlp` (raw history + EKF state -> 28 residuals):
  het      one heteroscedastic MLP: mean + log variance per output, Gaussian NLL (aleatoric only)
  ens      deep ensemble of N_MEMBERS het MLPs with different seeds (Lakshminarayanan et al. 2017); predictive
           variance = mean aleatoric + variance of the member means (epistemic)
  mcd      the same network with dropout after each hidden layer, MC_PASSES stochastic passes at test time
           (Gal and Ghahramani 2016)
Monitored quantity: the 20 m overlay offset at the 200 ms horizon (lateral error + 20 m * sin(heading error)), the
headline metric of docs/MOTION_PREDICTION.md. Its aleatoric variance is propagated from the residual outputs as
var(dy) + 20^2 var(dyaw) at 200 ms (small-angle, diagonal covariance); its epistemic variance is the spread of the
offsets of the members / passes, computed exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tools.mpred import learn as L

N_MEMBERS = 5
MC_PASSES = 30
DROPOUT_P = 0.1
HIDDEN = (64, 64)
TRAIN = {"epochs": 60, "batch": 512, "lr": 1e-3, "weight_decay": 1e-5, "patience": 6}
LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0
D_OVERLAY = 20.0
H_IDX = 3  # 200 ms in data.HORIZONS


class HetNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden=HIDDEN, dropout: float = 0.0):
        super().__init__()
        layers, a = [], n_in
        for h in hidden:
            layers += [nn.Linear(a, h), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            a = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(a, 2 * n_out)
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.head(self.body(x))
        return o[:, :self.n_out], o[:, self.n_out:].clamp(LOGVAR_MIN, LOGVAR_MAX)


def nll_loss(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()


@dataclass
class Fitted:
    xs: L.Scaler
    ys: L.Scaler
    nets: list[HetNet]
    dropout: float
    history: list[dict]


def train_het(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, seed: int, dropout: float = 0.0,
              xs: L.Scaler | None = None, ys: L.Scaler | None = None, **kw) -> tuple[HetNet, dict]:
    """Adam on the Gaussian NLL of the standardised targets; keeps the epoch with the lowest validation NLL."""
    cfg = {**TRAIN, **kw}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, Y = torch.from_numpy(xs(x).astype(np.float32)), torch.from_numpy(ys(y).astype(np.float32))
    XV, YV = torch.from_numpy(xs(xv).astype(np.float32)), torch.from_numpy(ys(yv).astype(np.float32))
    net = HetNet(X.shape[1], Y.shape[1], dropout=dropout)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    def val_nll() -> float:
        net.eval()
        with torch.no_grad():
            mu, lv = net(XV)
            return float(nll_loss(mu, lv, YV))

    best, best_ep, best_state, bad = val_nll(), 0, {k: v.clone() for k, v in net.state_dict().items()}, 0
    for ep in range(1, cfg["epochs"] + 1):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(X)))
        for i in range(0, len(X), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            mu, lv = net(X[idx])
            loss = nll_loss(mu, lv, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_nll()
        if v < best - 1e-5:
            best, best_ep, best_state, bad = v, ep, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"best_epoch": best_ep, "epochs_run": ep, "val_nll": best}


def fit_ensemble(x, y, xv, yv, seeds, dropout: float = 0.0, log=print) -> Fitted:
    xs, ys = L.Scaler.fit(x), L.Scaler.fit(y)
    nets, hist = [], []
    for s in seeds:
        net, h = train_het(x, y, xv, yv, seed=s, dropout=dropout, xs=xs, ys=ys)
        nets.append(net)
        hist.append({"seed": s, **h})
        log(f"  member seed {s}: best epoch {h['best_epoch']} of {h['epochs_run']}, val NLL {h['val_nll']:.4f}")
    return Fitted(xs, ys, nets, dropout, hist)


def sample_outputs(f: Fitted, x: np.ndarray, passes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Means and variances in target units: (K, N, outputs) each, K = members x passes. With dropout the
    networks run in train mode (MC dropout); without it passes must be 1."""
    X = torch.from_numpy(f.xs(x).astype(np.float32))
    mus, vas = [], []
    with torch.no_grad():
        for net in f.nets:
            net.train(f.dropout > 0)
            for _ in range(passes):
                mu, lv = net(X)
                mus.append(mu.numpy() * f.ys.std + f.ys.mean)
                vas.append(np.exp(lv.numpy()) * f.ys.std ** 2)
            net.eval()
    return np.stack(mus), np.stack(vas)


def overlay_var_aleatoric(var_res: np.ndarray, h: int = H_IDX, d: float = D_OVERLAY) -> np.ndarray:
    """(K, N, 28) residual variances -> (K, N) variance of the overlay offset at horizon h (small-angle)."""
    v = var_res.reshape(*var_res.shape[:2], -1, 4)[..., h, :]
    return v[..., 1] + d * d * v[..., 2]


# ---- metrics (numpy) -------------------------------------------------------------------------------------------

def auroc(score: np.ndarray, label: np.ndarray, w: np.ndarray | None = None) -> float:
    """Area under the ROC curve of `score` for the positive class `label` (bool), with optional sample weights
    (bootstrap counts). Ties count one half."""
    w = np.ones(len(score)) if w is None else w
    order = np.argsort(score, kind="mergesort")
    s, lab, ww = score[order], label[order], w[order]
    wp, wn = np.where(lab, ww, 0.0), np.where(lab, 0.0, ww)
    P, N = wp.sum(), wn.sum()
    if P == 0 or N == 0:
        return float("nan")
    # group ties: negatives strictly below + half of the tied negatives
    uniq, start = np.unique(s, return_index=True)
    neg_c = np.concatenate([[0.0], np.cumsum(wn)])
    end = np.append(start[1:], len(s))
    below = neg_c[start]
    tied = neg_c[end] - neg_c[start]
    pos_g = np.add.reduceat(wp, start)
    return float(np.sum(pos_g * (below + 0.5 * tied)) / (P * N))


LEVELS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def coverage(e: np.ndarray, sigma: np.ndarray, levels=LEVELS) -> np.ndarray:
    """Observed fraction of |e| inside the central Gaussian interval of each nominal level."""
    z = np.abs(e) / sigma
    q = np.array([math.sqrt(2.0) * _erfinv(p) for p in levels])
    return (z[:, None] <= q[None, :]).mean(0)


def _erfinv(p: float) -> float:
    lo, hi = 0.0, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if math.erf(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def calibration_error(e: np.ndarray, sigma: np.ndarray) -> float:
    """Mean |observed - nominal coverage| over the levels 5 % ... 95 % (regression ECE, Kuleshov et al. 2018)."""
    return float(np.mean(np.abs(coverage(e, sigma) - LEVELS)))


def gaussian_nll(e: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * (e / sigma) ** 2))


def var_scale(e: np.ndarray, sigma: np.ndarray) -> float:
    """NLL-optimal factor s for sigma -> s * sigma (post-hoc recalibration on the validation split)."""
    return float(np.sqrt(np.mean((e / sigma) ** 2)))


RC_GRID = np.linspace(0.5, 1.0, 51)


def risk_coverage(score: np.ndarray, err: np.ndarray, grid=RC_GRID) -> tuple[np.ndarray, float]:
    """Mean |err| of the kept samples when only the fraction `grid` with the lowest score is kept; and the area
    under the full risk-coverage curve (AURC, mean over every coverage step)."""
    order = np.argsort(score, kind="mergesort")
    cm = np.cumsum(np.abs(err)[order]) / np.arange(1, len(err) + 1)
    idx = np.clip((grid * len(err)).astype(int) - 1, 0, len(err) - 1)
    return cm[idx], float(cm.mean())


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 50, lam: float = 1e-3) -> tuple[L.Scaler, np.ndarray]:
    """L2-regularised logistic regression by Newton steps (the ego-state failure monitor)."""
    xs = L.Scaler.fit(x)
    Z = np.c_[np.ones(len(x)), xs(x)]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * np.r_[0.0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return xs, w


def predict_logistic(model: tuple[L.Scaler, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs, w = model
    return 1.0 / (1.0 + np.exp(-(np.c_[np.ones(len(x)), xs(x)] @ w)))
