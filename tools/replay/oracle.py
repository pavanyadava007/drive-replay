"""Ground-truth oracle: judges the stack's warnings against the annotated objects in the recording.

The oracle has hindsight the stack does not: it knows where the car actually drove next (the recorded ego
trajectory) and where every annotated object really was. It answers two questions per recording:

  false warning  a warning onset with no annotated object in the driven path whose true time-to-collision
                 was below ttc_justify_s
  missed threat  a key frame where an annotated road user, seen by the radar (>= 1 radar point), was in the
                 driven path with a true TTC below ttc_must_warn_s, and no warning was active within
                 +-match_window_s

Objects without radar points are counted as not observable, never as misses: a radar-only stack cannot see them.
"""
from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass

from tools.replay.recording import RecordingView

ROAD_USERS = ("vehicle.", "human.pedestrian")


@dataclass(frozen=True)
class OracleParams:
    ego_half_width_m: float = 0.95
    path_margin_m: float = 0.3
    front_bumper_m: float = 3.7
    horizon_s: float = 4.0
    ttc_justify_s: float = 3.2
    ttc_must_warn_s: float = 1.5
    min_ego_speed_mps: float = 2.0
    match_window_s: float = 0.5
    min_closing_mps: float = 0.3


class Oracle:
    def __init__(self, rec: RecordingView, p: OracleParams | None = None):
        self.rec, self.p = rec, p or OracleParams()
        self.ego_t = [e["t_us"] for e in rec.ego]
        self.gt_t = [g["t_us"] for g in rec.gt]

    # --- geometry -------------------------------------------------------------------------------------
    def _ego_at(self, t_us: int) -> dict:
        i = min(max(bisect.bisect_left(self.ego_t, t_us), 0), len(self.ego_t) - 1)
        if i > 0 and abs(self.ego_t[i - 1] - t_us) < abs(self.ego_t[i] - t_us):
            i -= 1
        return self.rec.ego[i]

    def _path(self, t_us: int) -> list[tuple[float, float, float]]:
        """Driven path from t: (x, y, arc length) of the recorded ego positions, shifted to the front bumper."""
        i0 = bisect.bisect_left(self.ego_t, t_us)
        pts, s = [], 0.0
        for e in self.rec.ego[max(i0 - 1, 0):]:
            if e["t_us"] > t_us + self.p.horizon_s * 1e6:
                break
            if pts:
                s += math.hypot(e["x"] - pts[-1][0], e["y"] - pts[-1][1])
            pts.append((e["x"], e["y"], s))
        return pts

    def _objects_at(self, t_us: int) -> list[dict]:
        """Annotated objects linearly interpolated between the bracketing key frames."""
        j = bisect.bisect_left(self.gt_t, t_us)
        if j <= 0 or j >= len(self.gt_t):
            g = self.rec.gt[min(max(j, 0), len(self.gt_t) - 1)]
            dt = (t_us - g["t_us"]) * 1e-6
            return [{**o, "x": o["x"] + o["vx"] * dt, "y": o["y"] + o["vy"] * dt} for o in g["objects"]]
        a, b = self.rec.gt[j - 1], self.rec.gt[j]
        w = (t_us - a["t_us"]) / (b["t_us"] - a["t_us"])
        bmap = {o["id"]: o for o in b["objects"]}
        out = []
        for o in a["objects"]:
            q = bmap.get(o["id"])
            if q is None:
                continue
            out.append({**o, "x": o["x"] + w * (q["x"] - o["x"]), "y": o["y"] + w * (q["y"] - o["y"]),
                        "num_radar_pts": max(o["num_radar_pts"], q["num_radar_pts"])})
        return out

    def threats(self, t_us: int) -> list[dict]:
        """Objects in the driven path at t with their true gap, closing speed and TTC."""
        ego = self._ego_at(t_us)
        path = self._path(t_us)
        if len(path) < 2:
            return []
        out = []
        for o in self._objects_at(t_us):
            best = None
            for (x0, y0, s0), (x1, y1, _) in itertools.pairwise(path):
                dx, dy = x1 - x0, y1 - y0
                seg = math.hypot(dx, dy)
                if seg < 1e-6:
                    continue
                u = ((o["x"] - x0) * dx + (o["y"] - y0) * dy) / (seg * seg)
                if not 0.0 <= u <= 1.0:
                    continue
                lat = abs((o["x"] - x0) * dy - (o["y"] - y0) * dx) / seg
                if best is None or lat < best[0]:
                    best = (lat, s0 + u * seg, dx / seg, dy / seg)
            if best is None:
                continue
            lat, s, tx, ty = best
            if lat > self.p.ego_half_width_m + o["width"] / 2 + self.p.path_margin_m:
                continue
            gap = s - self.p.front_bumper_m - o["length"] / 2
            if gap <= 0:
                continue
            closing = ego["speed_mps"] - (o["vx"] * tx + o["vy"] * ty)
            ttc = gap / closing if closing > self.p.min_closing_mps else math.inf
            out.append({"id": o["id"], "category": o["category"], "gap_m": round(gap, 2),
                        "closing_mps": round(closing, 2), "ttc_s": round(ttc, 2) if ttc != math.inf else None,
                        "radar_pts": o["num_radar_pts"], "lateral_m": round(lat, 2)})
        return sorted(out, key=lambda r: r["ttc_s"] if r["ttc_s"] is not None else 1e9)

    # --- verdicts -------------------------------------------------------------------------------------
    def judge(self, events: list[dict]) -> dict:
        p = self.p
        onsets = [e for e in events if e["kind"] == "fcw_warning_on"]
        warnings = []
        for e in onsets:
            cands = [th for th in self.threats(e["t_us"]) if th["ttc_s"] is not None and th["ttc_s"] < p.ttc_justify_s]
            warnings.append({"t_us": e["t_us"], "t_s": round((e["t_us"] - self.rec.start_us) * 1e-6, 2),
                             "track": e["track"], "stack_ttc_s": e["ttc"],
                             "verdict": "justified" if cands else "false", "gt": cands[:1]})
        active = self._active_intervals(events)
        missed, unobservable = [], 0
        for g in self.rec.gt:
            t = g["t_us"]
            if self._ego_at(t)["speed_mps"] < p.min_ego_speed_mps:
                continue
            for th in self.threats(t):
                if th["ttc_s"] is None or th["ttc_s"] >= p.ttc_must_warn_s or not th["category"].startswith(ROAD_USERS):
                    continue
                if th["radar_pts"] < 1:
                    unobservable += 1
                    continue
                w = p.match_window_s * 1e6
                if not any(a - w <= t <= b + w for a, b in active):
                    missed.append({"t_us": t, "t_s": round((t - self.rec.start_us) * 1e-6, 2), **th})
        n_false = sum(w["verdict"] == "false" for w in warnings)
        return {"warnings": warnings, "false_warnings": n_false, "missed_threats": len(missed), "missed": missed,
                "not_observable": unobservable, "requirements_met": n_false == 0 and not missed}

    def _active_intervals(self, events: list[dict]) -> list[tuple[int, int]]:
        out, start = [], None
        for e in events:
            if e["kind"] == "fcw_warning_on":
                start = e["t_us"]
            elif e["kind"] == "fcw_warning_off" and start is not None:
                out.append((start, e["t_us"]))
                start = None
        if start is not None:
            out.append((start, self.rec.end_us))
        return out
