"""Error metrics for ego motion prediction, with the AR-HUD overlay offset.

For a predicted pose (x^, y^, yaw^) and the recorded pose (x, y, yaw) at the display time:

  position error   |p^ - p|
  longitudinal /   the position error resolved in the TRUE vehicle frame at the display time
  lateral error
  heading error    yaw^ - yaw, wrapped to (-pi, pi]
  overlay offset   a contact-analog overlay drawn D metres ahead along the predicted heading lands at
  at distance D    p^ + D (cos yaw^, sin yaw^) instead of p + D (cos yaw, sin yaw). Its lateral misplacement
                   in the true frame is exactly  lateral error + D sin(heading error).  At 20 m a heading error
                   of 0.1 deg alone moves the overlay by 3.5 cm.
  angular error    atan(|overlay offset| / D): how far the overlay appears beside its target, seen from the
                   reference point (the rear axle; the driver's eye is roughly a metre further forward)

Per-scene sums (n, sum |e|, sum e^2) feed the bootstrap over scenes of tools/vdyn/pipeline.py.
"""
from __future__ import annotations

import numpy as np

from tools.mpred.models import wrap

LOOKAHEAD_M = (10.0, 20.0, 50.0)
METRICS = ("pos", "lon", "lat", "head", "speed", "ov10", "ov20", "ov50")


def errors(pred: np.ndarray, tru: np.ndarray) -> dict[str, np.ndarray]:
    """pred, tru: (N, H, 4) = x, y, yaw, v. Returns signed errors (N, H) per metric (pos is non-negative)."""
    dx = pred[..., 0] - tru[..., 0]
    dy = pred[..., 1] - tru[..., 1]
    c, s = np.cos(tru[..., 2]), np.sin(tru[..., 2])
    lat = -s * dx + c * dy
    head = wrap(pred[..., 2] - tru[..., 2])
    out = {"pos": np.hypot(dx, dy), "lon": c * dx + s * dy, "lat": lat, "head": head, "speed": pred[..., 3] - tru[..., 3]}
    for d in LOOKAHEAD_M:
        out[f"ov{int(d)}"] = lat + d * np.sin(head)
    return out


def overlay_angle_deg(offset: np.ndarray, d: float) -> np.ndarray:
    return np.degrees(np.arctan2(np.abs(offset), d))


def scene_sums(e: np.ndarray, si: np.ndarray, n_scenes: int) -> np.ndarray:
    """(n_scenes, 3) = [count, sum |e|, sum e^2] of one error vector."""
    out = np.zeros((n_scenes, 3))
    out[:, 0] = np.bincount(si, minlength=n_scenes)
    out[:, 1] = np.bincount(si, weights=np.abs(e), minlength=n_scenes)
    out[:, 2] = np.bincount(si, weights=e * e, minlength=n_scenes)
    return out


def p95(e: np.ndarray) -> float:
    return float(np.percentile(np.abs(e), 95)) if len(e) else float("nan")


def objective(err: dict[str, np.ndarray], hidx=(1, 3, 4, 5)) -> float:
    """The single tuning objective (validation split): mean position error + mean |20 m overlay offset|, averaged
    over the 100, 200, 300 and 500 ms horizons. Both terms are in metres."""
    return float(np.mean([np.abs(err["pos"][:, h]).mean() + np.abs(err["ov20"][:, h]).mean() for h in hidx]))
