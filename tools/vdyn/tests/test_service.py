"""REST job service: a real server on an ephemeral port, driven by a small urllib test client."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from tools.vdyn import pipeline as P
from tools.vdyn import service


class Client:
    def __init__(self, base: str):
        self.base = base

    def call(self, method: str, path: str, body=None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, job_id: str, timeout_s: float = 30.0) -> dict:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            _, job = self.call("GET", f"/jobs/{job_id}")
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(0.05)
        raise TimeoutError(job_id)


@pytest.fixture()
def client(scene_set):
    store = service.JobStore(scene_set["dir"], scene_set["binary"], scene_set["config"])
    httpd = service.serve("127.0.0.1", 0, store)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield Client(f"http://127.0.0.1:{httpd.server_address[1]}")
    httpd.shutdown()
    httpd.server_close()


def test_healthz_and_scenes(client, scene_set):
    code, body = client.call("GET", "/healthz")
    assert code == 200 and body["status"] == "ok" and body["binary"] and body["scenes"] == 4
    code, body = client.call("GET", "/scenes")
    assert body["scenes"] == scene_set["names"]


def test_job_lifecycle_and_metrics_match_the_pipeline(client, scene_set):
    code, body = client.call("POST", "/jobs", {"scenes": scene_set["names"], "model": "kinematic", "workers": 2})
    assert code == 202 and body["status"] == "queued"
    job = client.wait(body["id"])
    assert job["status"] == "done", job.get("error")
    assert job["progress"] == {"done": 4, "total": 4}
    m = job["metrics"]
    assert m["scenes"] == 4 and m["yaw_rate"]["rmse_rps"] < 1e-5
    assert m["trajectory"]["3s"]["fde_mean_m"] < 1e-3
    direct = P.run_batch(scene_set["names"], 1, scene_set["dir"], scene_set["config"], scene_set["binary"])
    assert job["result"]["batch_digest"] == direct.batch_digest
    code, listing = client.call("GET", "/jobs")
    assert listing["jobs"][0]["id"] == body["id"]


def test_params_override_reaches_the_model(client, scene_set):
    ids = []
    for params in ({}, {"dynamic.cr_npr": 60000}):
        _, b = client.call("POST", "/jobs", {"scenes": scene_set["names"], "model": "dynamic", "params": params})
        ids.append(b["id"])
    a, b = (client.wait(i) for i in ids)
    assert a["status"] == b["status"] == "done"
    assert a["result"]["batch_digest"] != b["result"]["batch_digest"]


@pytest.mark.parametrize("body, msg", [
    ({"scenes": ["scene-0000"]}, "unknown scenes"),
    ({"scenes": []}, "non-empty"),
    ({"split": "nope"}, "split must be"),
    ({"scenes": ["scene-9000"], "model": "magic"}, "model must be"),
    ({"scenes": ["scene-9000"], "params": {"dynamic.mass": 1}}, "unknown parameter"),
    ({"scenes": ["scene-9000"], "params": {"dynamic.cr_npr": "big"}}, "must be a number"),
    ({"scenes": ["scene-9000"], "workers": 0}, "workers"),
])
def test_bad_requests_are_rejected(client, body, msg):
    code, resp = client.call("POST", "/jobs", body)
    assert code == 400 and msg in resp["error"]


def test_unknown_job_and_path(client):
    assert client.call("GET", "/jobs/doesnotexist")[0] == 404
    assert client.call("GET", "/nope")[0] == 404
    assert client.call("POST", "/elsewhere", {})[0] == 404
