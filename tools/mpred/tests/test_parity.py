"""The C++ predictors (stack/mpred, streamed sample by sample) against the Python prototypes on synthetic drives."""
import json
import subprocess

import numpy as np
import pytest

from tools.mpred import data as D
from tools.mpred import predictors as PR
from tools.mpred.tests.conftest import drive, find_binary

COLUMNS = ["t_s", "x", "y", "yaw", "v_pose", "drive", "steer_sw", "r_imu", "rpm_rear", "ax_imu", "ay_imu", "ax_can"]


def test_cpp_matches_python(tmp_path, conf, vcfg):
    binary = find_binary()
    if binary is None:
        pytest.skip("mpred_eval not built (bazel build //stack/mpred:mpred_eval)")
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    tabs = {"scene-9001": drive(n=320, yaw_rate=lambda t: 0.15 * np.sin(0.5 * t), accel=0.5),
            "scene-9002": drive(n=320, v0=12.0, yaw_rate=-0.05, accel=-0.8)}
    kept = {}
    for name, tab in tabs.items():
        np.savetxt(scenes / f"{name}.csv", tab, delimiter=",", comments="", header=",".join(COLUMNS),
                   fmt=["%.4f", "%.4f", "%.4f", "%.6f", "%.4f", "%d", "%.6f", "%.6f", "%.3f", "%.5f", "%.5f", "%.4f"])
        kept[name] = {"vehicle": "n008", "location": "synthetic"}
    (scenes / "manifest.json").write_text(json.dumps({"kept": kept}))
    vpath = tmp_path / "vehicle.json"
    vpath.write_text(json.dumps(vcfg))
    ppath = tmp_path / "predictors.json"
    PR.save_config(ppath, conf["ctra"]["accel"], conf["kf"], conf["ekf_params"], conf["specs"], {})
    lst = tmp_path / "list.txt"
    lst.write_text("".join(f"{scenes / n}.csv n008\n" for n in tabs))
    out = tmp_path / "preds.f64"
    info = json.loads(subprocess.run([str(binary), "--predictors", str(ppath), "--vehicle-config", str(vpath),
                                      "--scene-list", str(lst), "--out", str(out)], capture_output=True, text=True,
                                     check=True).stdout)
    rows = np.fromfile(out, dtype=np.float64).reshape(-1, 31)
    b = D.load(list(tabs), scenes, vcfg)
    si, ki = D.start_points(b)
    py = PR.predict_all(b, si, ki, PR.load_config(ppath), vcfg, PR.CPP)
    assert len(si) > 20
    for mi, m in enumerate(info["methods"]):
        sel = rows[rows[:, 2] == mi]
        lut = {(int(r[0]), int(r[1])): r[3:].reshape(7, 4) for r in sel}
        c = np.stack([lut[(int(s), int(k))] for s, k in zip(si, ki, strict=True)])
        assert np.abs(c - py[m]).max() < 1e-9, m
