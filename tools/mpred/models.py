"""Physics-based predictors and filters (Python prototypes; the C++ versions are in stack/mpred).

Every predictor returns the predicted global pose and speed (x, y, yaw, v) at each horizon: an array
(N start points, H horizons, 4). Frame and signs as in stack/vdyn: x forward, y left, yaw counter-clockwise,
reference point = middle of the rear axle (the nuScenes ego origin).

Prediction kernel (shared by CV, CTRV, CTRA and the filters): state (x, y, yaw, v) with acceleration a and yaw
rate r held, classical RK4 with a fixed 10 ms step; the speed never goes negative (a car that brakes to a stop
stays there). CV: a = r = 0. CTRV: a = 0. CTRA: both held. stack/mpred/mpred.cc implements the same arithmetic.

Filters run causally over every sample of every scene (all scenes at once, vectorised over the scene axis):
  KF-CV, KF-CA   linear Kalman filters per axis in the map frame, measuring the pose position only
  EKF-CTRA       state (x, y, yaw, v, a, r), CTRA process model (one RK4 step per pose sample, Jacobian of the
                 Euler step), measurements pose x, y, yaw, wheel speed, IMU yaw rate and a longitudinal
                 acceleration, applied as sequential scalar updates (the measurement model is linear and R is
                 diagonal, so this equals the joint update and needs no matrix inverse)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tools.mpred.data import DT_INT, H_STEPS, Batch

TWO_PI = 2.0 * np.pi


def wrap(a: np.ndarray) -> np.ndarray:
    return a - TWO_PI * np.rint(a / TWO_PI)


# ---- prediction kernel ------------------------------------------------------------------------------------------

def _f(psi, v, a, r):
    vp = np.maximum(v, 0.0)
    dv = np.where((v > 0.0) | (a > 0.0), a, 0.0)
    return vp * np.cos(psi), vp * np.sin(psi), r, dv


def rk4_step(x, y, psi, v, a, r, dt):
    h2 = 0.5 * dt
    k1 = _f(psi, v, a, r)
    k2 = _f(psi + h2 * k1[2], v + h2 * k1[3], a, r)
    k3 = _f(psi + h2 * k2[2], v + h2 * k2[3], a, r)
    k4 = _f(psi + dt * k3[2], v + dt * k3[3], a, r)
    s = dt / 6.0
    vn = v + s * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    # a step that brakes through zero would otherwise leave a small negative speed
    return (x + s * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y + s * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            psi + s * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]), np.maximum(vn, 0.0))


def rollout(x, y, psi, v, a, r, steps=H_STEPS, dt=DT_INT) -> np.ndarray:
    """CTRA kernel from the given state; output (N, len(steps), 4) after each number of fixed steps."""
    x, y, psi, v = (np.asarray(q, dtype=float).copy() for q in (x, y, psi, v))
    a = np.broadcast_to(np.asarray(a, dtype=float), x.shape)
    r = np.broadcast_to(np.asarray(r, dtype=float), x.shape)
    out = np.empty(x.shape + (len(steps), 4))
    want = {s: j for j, s in enumerate(steps)}
    for i in range(1, max(steps) + 1):
        x, y, psi, v = rk4_step(x, y, psi, v, a, r, dt)
        if i in want:
            out[:, want[i]] = np.stack([x, y, psi, v], -1)
    return out


def rollout_to(x, y, psi, v, a, r, horizon: np.ndarray, nominal_s: float) -> np.ndarray:
    """One arbitrary horizon per start point: ceil(nominal/10 ms) equal steps of horizon/n. Returns (N, 4)."""
    n = max(1, int(np.ceil(nominal_s / DT_INT - 1e-9)))
    return rollout(x, y, psi, v, a, r, steps=(n,), dt=np.asarray(horizon) / n)[:, 0]


# ---- signals ----------------------------------------------------------------------------------------------------

def accel_signal(b: Batch, source: str) -> np.ndarray:
    """Longitudinal acceleration per sample: a recorded signal, or 'slopeW' = least-squares slope of the wheel
    speed over the last W samples (causal; 0 for the first W - 1 samples)."""
    if not source.startswith("slope"):
        return b[source]
    w = int(source[5:])
    t, v = b["t"], b["v"]
    out = np.zeros_like(v)
    T = t.shape[1]
    if T < w:
        return out
    tw = np.stack([t[:, w - 1 - j: T - j] for j in range(w)], 0)  # (W, S, T-W+1), j = 0 newest
    vw = np.stack([v[:, w - 1 - j: T - j] for j in range(w)], 0)
    tm, vm = tw.mean(0), vw.mean(0)
    num = ((tw - tm) * (vw - vm)).sum(0)
    den = ((tw - tm) ** 2).sum(0)
    out[:, w - 1:] = num / np.where(den > 0, den, 1.0)
    return out


def anchor(b: Batch, si, ki):
    return b["x"][si, ki], b["y"][si, ki], b["yaw"][si, ki]


def predict_cv(b: Batch, si, ki, steps=H_STEPS):
    x, y, p = anchor(b, si, ki)
    return rollout(x, y, p, b["v"][si, ki], 0.0, 0.0, steps)


def predict_ctrv(b: Batch, si, ki, steps=H_STEPS):
    x, y, p = anchor(b, si, ki)
    return rollout(x, y, p, b["v"][si, ki], 0.0, b["r"][si, ki], steps)


def predict_ctra(b: Batch, si, ki, accel: np.ndarray, steps=H_STEPS):
    x, y, p = anchor(b, si, ki)
    return rollout(x, y, p, b["v"][si, ki], accel[si, ki], b["r"][si, ki], steps)


def predict_kinematic(b: Batch, si, ki, wheelbase: float, steps=H_STEPS):
    """Kinematic single track with steering and speed held: yaw rate v tan(delta) / L (v0.3.0 parameters)."""
    x, y, p = anchor(b, si, ki)
    v = b["v"][si, ki]
    return rollout(x, y, p, v, 0.0, v * np.tan(b["d_kin"][si, ki]) / wheelbase, steps)


# ---- linear dynamic single track (port of stack/vdyn LinearSingleTrack, steering and speed held) ----------------

def predict_dynamic(b: Batch, si, ki, cfg: dict, steps=H_STEPS, dt=DT_INT):
    d = cfg["dynamic"]
    L, lf, m, Iz, cf, cr = cfg["wheelbase_m"], d["lf_m"], d["mass_kg"], d["yaw_inertia_kgm2"], d["cf_npr"], d["cr_npr"]
    lr = L - lf
    x, y, psi = (q.copy() for q in anchor(b, si, ki))
    v = b["v"][si, ki]
    delta = b["d_dyn"][si, ki]
    r = b["r"][si, ki].copy()
    slow = v < d["min_speed_mps"]
    if np.any(slow):  # as stack/vdyn: below min_speed the slip angles are undefined and the model moves kinematically
        out = np.empty(x.shape + (len(steps), 4))
        out[slow] = rollout(x[slow], y[slow], psi[slow], v[slow], 0.0, v[slow] * np.tan(delta[slow]) / L, steps, dt)
        sel = np.flatnonzero(~slow)
        if len(sel):
            out[~slow] = predict_dynamic(b, si[sel], ki[sel], cfg, steps, dt)
        return out
    vy = (((lr * cr - lf * cf) / (m * v) - v) * r + cf / m * delta) * m * v / (cf + cr)
    lam = (cf + cr) / (m * v) + (lf * lf * cf + lr * lr * cr) / (Iz * v)
    nsub = np.maximum(1, np.ceil(lam * dt / 1.0)).astype(int)
    h = dt / nsub

    def deriv(psi_, vy_, r_):
        dvy = -(cf + cr) / (m * v) * vy_ + ((lr * cr - lf * cf) / (m * v) - v) * r_ + cf / m * delta
        dr = (lr * cr - lf * cf) / (Iz * v) * vy_ - (lf * lf * cf + lr * lr * cr) / (Iz * v) * r_ + lf * cf / Iz * delta
        vy_rear = vy_ - lr * r_
        return (v * np.cos(psi_) - vy_rear * np.sin(psi_), v * np.sin(psi_) + vy_rear * np.cos(psi_), r_, dvy, dr)

    out = np.empty(x.shape + (len(steps), 4))
    want = {s: j for j, s in enumerate(steps)}
    for i in range(1, max(steps) + 1):
        for j in range(int(nsub.max())):
            act = j < nsub
            st = (x, y, psi, vy, r)
            k1 = deriv(psi, vy, r)
            s2 = [q + 0.5 * h * k for q, k in zip(st, k1, strict=True)]
            k2 = deriv(s2[2], s2[3], s2[4])
            s3 = [q + 0.5 * h * k for q, k in zip(st, k2, strict=True)]
            k3 = deriv(s3[2], s3[3], s3[4])
            s4 = [q + h * k for q, k in zip(st, k3, strict=True)]
            k4 = deriv(s4[2], s4[3], s4[4])
            new = [q + h / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4) for q, a1, a2, a3, a4 in zip(st, k1, k2, k3, k4, strict=True)]
            x, y, psi, vy, r = (np.where(act, nq, q) for nq, q in zip(new, st, strict=True))
        if i in want:
            out[:, want[i]] = np.stack([x, y, psi, v], -1)
    return out


# ---- linear Kalman filters (per axis, map frame, pose position measured) ----------------------------------------

@dataclass(frozen=True)
class KfParams:
    order: int = 2  # 2: constant velocity (x, v), 3: constant acceleration (x, v, a)
    q: float = 1.0  # white-noise PSD of the highest derivative: acceleration (CV) or jerk (CA)
    r_pos: float = 0.02  # m, measurement std of the pose position


def _kf_q(order: int, q: float, dt: np.ndarray) -> np.ndarray:
    d = dt[:, None, None]
    if order == 2:
        return q * np.concatenate([np.concatenate([d ** 3 / 3, d ** 2 / 2], 2), np.concatenate([d ** 2 / 2, d], 2)], 1)
    rows = [[d ** 5 / 20, d ** 4 / 8, d ** 3 / 6], [d ** 4 / 8, d ** 3 / 3, d ** 2 / 2], [d ** 3 / 6, d ** 2 / 2, d]]
    return q * np.concatenate([np.concatenate(r_, 2) for r_ in rows], 1)


def run_kf(b: Batch, p: KfParams, extra_var: float = 0.0) -> np.ndarray:
    """Filtered (S, T, 2 axes, order) states of the per-axis linear KF."""
    n = p.order
    t = b["t"]
    S, T = t.shape
    z = np.concatenate([b["x"], b["y"]], 0)  # rows: axis x of every scene, then axis y
    v0 = np.concatenate([b["v"][:, 0] * np.cos(b["yaw"][:, 0]), b["v"][:, 0] * np.sin(b["yaw"][:, 0])])
    X = np.zeros((2 * S, n))
    X[:, 0], X[:, 1] = z[:, 0], v0
    P = np.zeros((2 * S, n, n))
    P[:, 0, 0] = p.r_pos ** 2 + extra_var
    P[:, 1, 1] = 1.0
    if n == 3:
        P[:, 2, 2] = 4.0
    R = p.r_pos ** 2 + extra_var
    out = np.empty((2 * S, T, n))
    out[:, 0] = X
    tt = np.concatenate([t, t], 0)
    for k in range(1, T):
        dt = tt[:, k] - tt[:, k - 1]
        F = np.broadcast_to(np.eye(n), (2 * S, n, n)).copy()
        F[:, 0, 1] = dt
        if n == 3:
            F[:, 0, 2] = 0.5 * dt * dt
            F[:, 1, 2] = dt
        X = np.einsum("sij,sj->si", F, X)
        P = F @ P @ F.transpose(0, 2, 1) + _kf_q(n, p.q, dt)
        Sv = P[:, 0, 0] + R
        K = P[:, :, 0] / Sv[:, None]
        X = X + K * (z[:, k] - X[:, 0])[:, None]
        P = P - K[:, :, None] * P[:, None, 0, :]
        out[:, k] = X
    return np.stack([out[:S], out[S:]], 2)  # (S, T, 2, n)


def predict_kf(states: np.ndarray, si, ki, horizons) -> np.ndarray:
    st = states[si, ki]  # (N, 2, n)
    n = st.shape[-1]
    h = np.asarray(horizons)[None, :, None]
    pos = st[:, None, :, 0] + st[:, None, :, 1] * h
    vel = st[:, None, :, 1] + 0.0 * h
    if n == 3:
        pos = pos + 0.5 * st[:, None, :, 2] * h * h
        vel = vel + st[:, None, :, 2] * h
    yaw = np.arctan2(vel[..., 1], vel[..., 0])
    return np.stack([pos[..., 0], pos[..., 1], yaw, np.hypot(vel[..., 0], vel[..., 1])], -1)


# ---- EKF on the CTRA state --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EkfParams:
    s_jerk: float = 2.0  # m/s^3, std of the white-noise jerk driving a (tuned)
    s_rdot: float = 0.1  # rad/s^2, std of the white-noise yaw acceleration driving r (tuned)
    r_scale: float = 1.0  # multiplies the odometry measurement stds r_v, r_r, r_a (tuned)
    accel: str = "ax_can"  # acceleration measurement (tuned: ax_can or ax_imu)
    q_xy: float = 1e-3  # m^2/s, small process noise on the position (fixed)
    q_psi: float = 1e-5  # rad^2/s (fixed)
    q_v: float = 1e-2  # (m/s)^2/s (fixed)
    r_xy: float = 0.02  # m, pose position (fixed: the pose is the reference and very smooth)
    r_psi: float = 0.002  # rad, pose heading (fixed)
    r_v: float = 0.05  # m/s, wheel speed (fixed, scaled by r_scale)
    r_r: float = 0.005  # rad/s, IMU yaw rate (fixed, scaled by r_scale)
    r_a: float = 0.3  # m/s^2, acceleration (fixed, scaled by r_scale)

    def as_dict(self) -> dict:
        return asdict(self)

    def meas_std(self) -> tuple[float, ...]:
        """Measurement stds in update order: x, y, yaw, v, r, a."""
        k = self.r_scale
        return (self.r_xy, self.r_xy, self.r_psi, self.r_v * k, self.r_r * k, self.r_a * k)


EKF_UPDATE_ORDER = (0, 1, 2, 3, 5, 4)  # state indices of the measurements x, y, yaw, v, r, a


def run_ekf(b: Batch, p: EkfParams, pose_avail: np.ndarray | None = None, odo_avail: np.ndarray | None = None,
            extra_std: dict | None = None, upto: int | None = None) -> np.ndarray:
    """Filtered states (S, T, 6) = x, y, yaw, v, a, r after the updates of each sample.

    pose_avail / odo_avail (S, T bool) skip updates of samples that did not arrive. extra_std adds known injected
    noise (keys pose_xy, pose_yaw, v, r, a) in quadrature to R."""
    t = b["t"]
    S, T = t.shape
    T = T if upto is None else upto
    acc = b[p.accel]
    Z = [b["x"], b["y"], b["yaw"], b["v"], b["r"], acc]
    std = list(p.meas_std())
    if extra_std:
        add = [extra_std.get("pose_xy", 0), extra_std.get("pose_xy", 0), extra_std.get("pose_yaw", 0),
               extra_std.get("v", 0), extra_std.get("r", 0), extra_std.get("a", 0)]
        std = [float(np.hypot(s_, a_)) for s_, a_ in zip(std, add, strict=True)]
    var = [s_ * s_ for s_ in std]
    avail = [pose_avail, pose_avail, pose_avail, odo_avail, odo_avail, odo_avail]
    qd = np.array([p.q_xy, p.q_xy, p.q_psi, p.q_v, p.s_jerk ** 2, p.s_rdot ** 2])
    X = np.stack([b["x"][:, 0], b["y"][:, 0], b["yaw"][:, 0], b["v"][:, 0], acc[:, 0], b["r"][:, 0]], 1)
    P = np.zeros((S, 6, 6))
    P[:, range(6), range(6)] = [var[0], var[1], var[2], var[3], var[5], var[4]]
    out = np.empty((S, T, 6))
    out[:, 0] = X
    I6 = np.eye(6)
    for k in range(1, T):
        dt = t[:, k] - t[:, k - 1]
        X, P = ekf_predict(X, P, dt, qd, I6)
        for j, zi in enumerate(EKF_UPDATE_ORDER):
            X, P = ekf_update(X, P, zi, Z[j][:, k], var[j], None if avail[j] is None else avail[j][:, k])
        out[:, k] = X
    return out


def ekf_update(X, P, zi: int, z, var: float, avail=None):
    """Scalar measurement of state component zi (the yaw innovation is wrapped). avail = 0 skips the update."""
    innov = z - X[:, zi]
    if zi == 2:
        innov = wrap(innov)
    Sv = P[:, zi, zi] + var
    K = P[:, :, zi] / Sv[:, None]
    if avail is not None:
        K = K * avail[:, None]
    return X + K * innov[:, None], P - K[:, :, None] * P[:, None, zi, :]


def ekf_predict(X, P, dt, qd, I6):
    x, y, psi, v, a, r = X.T
    vp = np.maximum(v, 0.0)
    F = np.broadcast_to(I6, P.shape).copy()
    F[:, 0, 2] = -dt * vp * np.sin(psi)
    F[:, 0, 3] = dt * np.cos(psi)
    F[:, 1, 2] = dt * vp * np.cos(psi)
    F[:, 1, 3] = dt * np.sin(psi)
    F[:, 2, 5] = dt
    F[:, 3, 4] = dt
    nx, ny, npsi, nv = rk4_step(x, y, psi, v, a, r, dt)
    Xn = np.stack([nx, ny, npsi, nv, a, r], 1)
    Pn = F @ P @ F.transpose(0, 2, 1)
    Pn[:, range(6), range(6)] += qd[None, :] * dt[:, None]
    return Xn, Pn


def predict_from_state(states: np.ndarray, si, ki, steps=H_STEPS, use_accel: bool = True) -> np.ndarray:
    st = states[si, ki]
    return rollout(st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4] if use_accel else 0.0, st[:, 5], steps)


# ---- asymmetric latency: odometry bridge --------------------------------------------------------------------------

def dead_reckon(b: Batch, si, k_from, k_to, x, y, psi, v_key: str = "v", r_key: str = "r"):
    """Integrate the recorded odometry (wheel speed, IMU yaw rate) from sample k_from to k_to (k_to >= k_from):
    one RK4 step per pose interval with the mean of the two samples held. Returns x, y, yaw at t[k_to]."""
    x, y, psi = x.copy(), y.copy(), psi.copy()
    for j in range(int((k_to - k_from).max())):
        k = k_from + j
        act = k < k_to
        kn = np.minimum(k + 1, k_to)
        dt = np.where(act, b["t"][si, kn] - b["t"][si, k], 0.0)
        v = 0.5 * (b[v_key][si, k] + b[v_key][si, kn])
        r = 0.5 * (b[r_key][si, k] + b[r_key][si, kn])
        x, y, psi, _ = rk4_step(x, y, psi, v, 0.0, r, dt)
    return x, y, psi


def ekf_odometry_only(b: Batch, p: EkfParams, X: np.ndarray, P: np.ndarray, si, k_from, k_to):
    """Re-propagate an EKF state from sample k_from to k_to with the odometry updates only (no pose): how a filter
    handles a pose that arrives late while wheel speed, yaw rate and acceleration are already in."""
    std = p.meas_std()
    var = [s_ * s_ for s_ in std]
    qd = np.array([p.q_xy, p.q_xy, p.q_psi, p.q_v, p.s_jerk ** 2, p.s_rdot ** 2])
    I6 = np.eye(6)
    X, P = X.copy(), P.copy()
    acc = b[p.accel]
    for j in range(int((k_to - k_from).max())):
        k = k_from + j + 1
        act = k <= k_to
        kk = np.minimum(k, k_to)
        dt = np.where(act, b["t"][si, kk] - b["t"][si, kk - 1], 0.0)
        Xn, Pn = ekf_predict(X, P, dt, qd, I6)
        for jj, (zi, z) in enumerate(((3, b["v"]), (5, b["r"]), (4, acc))):
            Xn, Pn = ekf_update(Xn, Pn, zi, z[si, kk], var[3 + jj])
        X = np.where(act[:, None], Xn, X)
        P = np.where(act[:, None, None], Pn, P)
    return X


def run_ekf_with_cov(b: Batch, p: EkfParams, ks: np.ndarray, si: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like run_ekf, but also returns the covariance at the requested (scene, sample) pairs."""
    t = b["t"]
    S, T = t.shape
    acc = b[p.accel]
    Z = [b["x"], b["y"], b["yaw"], b["v"], b["r"], acc]
    var = [s_ * s_ for s_ in p.meas_std()]
    qd = np.array([p.q_xy, p.q_xy, p.q_psi, p.q_v, p.s_jerk ** 2, p.s_rdot ** 2])
    X = np.stack([b["x"][:, 0], b["y"][:, 0], b["yaw"][:, 0], b["v"][:, 0], acc[:, 0], b["r"][:, 0]], 1)
    P = np.zeros((S, 6, 6))
    P[:, range(6), range(6)] = [var[0], var[1], var[2], var[3], var[5], var[4]]
    Xo = np.empty((len(ks), 6))
    Po = np.empty((len(ks), 6, 6))
    I6 = np.eye(6)
    order = {}
    for i, (s, k) in enumerate(zip(si, ks, strict=True)):
        order.setdefault(int(k), []).append((i, int(s)))
    for k in range(T):
        if k > 0:
            dt = t[:, k] - t[:, k - 1]
            X, P = ekf_predict(X, P, dt, qd, I6)
            for j, zi in enumerate(EKF_UPDATE_ORDER):
                X, P = ekf_update(X, P, zi, Z[j][:, k], var[j])
        for i, s in order.get(k, []):
            Xo[i], Po[i] = X[s], P[s]
    return Xo, Po
