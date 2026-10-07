"""Small REST job service around the reprocessing pipeline (Python standard library only).

    python -m tools.vdyn.service --port 8080 [--scenes-dir DIR] [--binary PATH] [--config PATH]

Endpoints (JSON in, JSON out):
  GET  /healthz          liveness plus what the service can see (binary, config, number of scenes)
  GET  /scenes?split=X   scene names of a split (train, val, test) or of all scenes
  POST /jobs             {"scenes": [...] | "split": "test", "model": "dynamic", "params": {"dynamic.cr_npr": 1.2e5},
                          "workers": 4} -> 202 {"id": ..., "status": "queued"}
  GET  /jobs             all jobs, newest first (id, status, scenes)
  GET  /jobs/{id}        status, progress and, once done, pooled metrics of the model, throughput and digests

A job is one batch run of tools/vdyn/pipeline.py: the C++ vdyn_eval over the scenes, in parallel. `params`
override identified parameters for this job only (the same keys as `vdyn_eval --set`). This is a
cloud-ready containerised service (see Dockerfile.vdyn), run and tested locally; it has not been deployed
anywhere. There is no authentication: run it on a trusted network only.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tools.vdyn import pipeline as P

MODELS = {
    # model -> (yaw predictor or None, trajectory key or None, corridor key or None)
    "kinematic": ("kinematic", "kinematic", "kinematic"),
    "dynamic": ("dynamic", "dynamic", "single_track_steer_decay"),
    "steady_state": ("steady_state", None, None),
    "const_yaw_rate": (None, "const_yaw_rate", "const_yaw_rate"),
    "const_velocity": (None, "const_velocity", "const_velocity"),
    "fcw_yaw_decay": (None, "fcw_yaw_decay", "fcw_yaw_decay"),
    "kinematic_rec": (None, "kinematic_rec", None),
    "dynamic_rec": (None, "dynamic_rec", None),
}
PARAM_KEYS = {"wheel_radius_m", "wheelbase_m", "kinematic.steering_ratio", "kinematic.steer_offset_rad",
              "dynamic.steering_ratio", "dynamic.steer_offset_rad", "dynamic.lf_m", "dynamic.mass_kg",
              "dynamic.yaw_inertia_kgm2", "dynamic.cf_npr", "dynamic.cr_npr", "dynamic.min_speed_mps",
              "fcw_baseline.yaw_rate_decay_s", "fcw_baseline.max_curvature"}
MAX_SCENES = 2000


class BadRequest(ValueError):
    pass


class JobStore:
    def __init__(self, scenes_dir: Path, binary: Path, config: Path, max_parallel_jobs: int = 2):
        self.scenes_dir, self.binary, self.config = scenes_dir, binary, config
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max_parallel_jobs)

    def known_scenes(self) -> list[str]:
        try:
            return sorted(P.manifest(self.scenes_dir))
        except OSError:
            return []

    def validate(self, body: dict) -> dict:
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        known = set(self.known_scenes())
        if "split" in body:
            if body["split"] not in ("train", "val", "test", "all"):
                raise BadRequest("split must be train, val, test or all")
            scenes = sorted(known) if body["split"] == "all" else P.split_scenes(body["split"])
        else:
            scenes = body.get("scenes")
            if not isinstance(scenes, list) or not scenes or not all(isinstance(s, str) for s in scenes):
                raise BadRequest("give 'scenes' (non-empty list of scene names) or 'split'")
        if len(scenes) > MAX_SCENES:
            raise BadRequest(f"at most {MAX_SCENES} scenes per job")
        unknown = [s for s in scenes if s not in known]
        if unknown:
            raise BadRequest(f"unknown scenes: {', '.join(unknown[:5])}")
        model = body.get("model", "dynamic")
        if model not in MODELS:
            raise BadRequest(f"model must be one of {', '.join(sorted(MODELS))}")
        params = body.get("params", {})
        if not isinstance(params, dict):
            raise BadRequest("params must be an object")
        for k, v in params.items():
            if k not in PARAM_KEYS:
                raise BadRequest(f"unknown parameter {k}")
            if isinstance(v, bool) or not isinstance(v, int | float):
                raise BadRequest(f"parameter {k} must be a number")
        workers = body.get("workers", os.cpu_count() or 4)
        if not isinstance(workers, int) or not 1 <= workers <= 64:
            raise BadRequest("workers must be an integer in 1..64")
        return {"scenes": sorted(set(scenes)), "model": model, "params": params, "workers": workers}

    def submit(self, body: dict) -> dict:
        spec = self.validate(body)
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "created": time.time(), "model": spec["model"],
               "params": spec["params"], "scenes": len(spec["scenes"]), "progress": {"done": 0, "total": len(spec["scenes"])}}
        with self.lock:
            self.jobs[job_id] = job
        self.pool.submit(self._run, job_id, spec)
        return {"id": job_id, "status": "queued"}

    def _run(self, job_id: str, spec: dict) -> None:
        job = self.jobs[job_id]
        job["status"] = "running"

        def progress(done: int, total: int) -> None:
            job["progress"] = {"done": done, "total": total}

        try:
            sets = [f"{k}={v!r}" for k, v in sorted(spec["params"].items())]
            res = P.run_batch(spec["scenes"], spec["workers"], self.scenes_dir, self.config, self.binary, sets=sets,
                              progress=progress)
            if res.errors:
                raise RuntimeError("; ".join(res.errors))
            job["metrics"] = metrics(list(res.scenes.values()), spec["model"])
            job["result"] = res.summary()
            job["scene_digests"] = {n: s["digest"] for n, s in sorted(res.scenes.items())}
            job["status"] = "done"
        except Exception as e:  # any failure is reported on the job; the service keeps running
            job["status"] = "failed"
            job["error"] = str(e)[-1000:]
        job["finished"] = time.time()

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)


def metrics(rows: list[dict], model: str) -> dict:
    yaw, traj, corr = MODELS[model]
    out: dict = {"model": model, "scenes": len(rows)}
    if yaw:
        p = P.pooled(P.sums(rows, ("yaw", yaw, "all")))
        out["yaw_rate"] = {"samples": p["n"], "rmse_rps": p["rmse"], "mae_rps": p["mean_abs"]}
    if traj:
        out["trajectory"] = {}
        for h in ("1", "2", "3"):
            f, lat = P.pooled(P.sums(rows, ("traj", traj, h, "fde"))), P.pooled(P.sums(rows, ("traj", traj, h, "lat")))
            out["trajectory"][f"{h}s"] = {"starts": f["n"], "fde_mean_m": f["mean_abs"], "lateral_mean_m": lat["mean_abs"]}
    if corr:
        out["corridor_mean_abs_m"] = {f"{d}m": P.pooled(P.sums(rows, ("corridor", corr, d)))["mean_abs"]
                                      for d in ("10", "20", "30")}
    return out


def make_handler(store: JobStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "vdyn-service/0.3"

        def log_message(self, fmt: str, *args) -> None:  # quiet by default
            if os.environ.get("VDYN_SERVICE_LOG"):
                super().log_message(fmt, *args)

        def send(self, code: int, body: dict) -> None:
            data = json.dumps(body, indent=1).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            if parts == ["healthz"]:
                self.send(200, {"status": "ok", "binary": store.binary.exists(), "config": store.config.exists(),
                                "scenes": len(store.known_scenes())})
            elif parts == ["scenes"]:
                split = parse_qs(url.query).get("split", ["all"])[0]
                if split == "all":
                    self.send(200, {"split": split, "scenes": store.known_scenes()})
                elif split in ("train", "val", "test"):
                    self.send(200, {"split": split, "scenes": P.split_scenes(split)})
                else:
                    self.send(400, {"error": "split must be train, val, test or all"})
            elif parts == ["jobs"]:
                jobs = sorted(store.jobs.values(), key=lambda j: j["created"], reverse=True)
                self.send(200, {"jobs": [{k: j[k] for k in ("id", "status", "model", "scenes")} for j in jobs]})
            elif len(parts) == 2 and parts[0] == "jobs":
                job = store.get(parts[1])
                self.send(200, job) if job else self.send(404, {"error": f"no job {parts[1]}"})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlparse(self.path).path.rstrip("/") != "/jobs":
                self.send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 1 << 20:
                    raise BadRequest("body too large")
                body = json.loads(self.rfile.read(n) or b"null")
                self.send(202, store.submit(body))
            except (BadRequest, json.JSONDecodeError) as e:
                self.send(400, {"error": str(e)})

    return Handler


def serve(host: str, port: int, store: JobStore) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(store))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scenes-dir", type=Path, default=P.DEFAULT_SCENES)
    ap.add_argument("--binary", type=Path, default=P.DEFAULT_BINARY)
    ap.add_argument("--config", type=Path, default=P.DEFAULT_CONFIG)
    args = ap.parse_args()
    httpd = serve(args.host, args.port, JobStore(args.scenes_dir, args.binary, args.config))
    print(f"vdyn service on http://{args.host}:{httpd.server_address[1]}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
