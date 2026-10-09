"""Overlay-offset metric, target transforms, ridge, MLP and the exported model format."""
import json

import numpy as np

from tools.mpred import learn as L
from tools.mpred import metrics as E
from tools.mpred import predictors as PR


def test_overlay_offset_is_lateral_error_plus_heading_term():
    tru = np.zeros((1, 1, 4))
    tru[0, 0] = [10.0, 5.0, 0.3, 8.0]
    pred = tru.copy()
    pred[0, 0, 2] += np.radians(0.1)  # heading only
    e = E.errors(pred, tru)
    assert abs(e["ov20"][0, 0] - 20 * np.sin(np.radians(0.1))) < 1e-12
    assert abs(e["ov20"][0, 0] - 0.0349) < 1e-4
    pred = tru.copy()
    pred[0, 0, 0] += -np.sin(0.3) * 0.2  # 0.2 m to the left of the true pose
    pred[0, 0, 1] += np.cos(0.3) * 0.2
    e = E.errors(pred, tru)
    assert abs(e["lat"][0, 0] - 0.2) < 1e-12 and abs(e["lon"][0, 0]) < 1e-12
    assert all(abs(e[f"ov{d}"][0, 0] - 0.2) < 1e-12 for d in (10, 20, 50))
    assert abs(E.overlay_angle_deg(np.array([0.2]), 20.0)[0] - np.degrees(np.arctan(0.01))) < 1e-12


def test_scene_sums():
    s = E.scene_sums(np.array([1.0, -2.0, 3.0]), np.array([0, 0, 2]), 3)
    assert s.tolist() == [[2, 3, 5], [0, 0, 0], [1, 3, 9]]


def test_anchor_and_residual_transforms_round_trip():
    rng = np.random.default_rng(0)
    anchor = rng.normal(0, 1, (5, 3))
    pose = rng.normal(0, 1, (5, 7, 4))
    v0 = rng.normal(5, 1, 5)
    back = L.from_anchor(anchor, L.to_anchor(anchor, pose, v0), v0)
    assert np.allclose(back[..., :2], pose[..., :2]) and np.allclose(back[..., 3], pose[..., 3])
    base = pose + rng.normal(0, 0.1, pose.shape)
    res = L.residual_target(base, pose, anchor[:, 2])
    assert np.allclose(L.apply_residual(base, res, anchor[:, 2]), pose)


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(1)
    x = np.c_[np.ones(500), rng.normal(0, 2, (500, 4))]
    w = rng.normal(0, 1, (5, 3))
    m = L.Ridge.fit(x, x @ w, 1e-9)
    assert np.allclose(m(x), x @ w, atol=1e-6)


def test_mlp_learns_a_smooth_function():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (4000, 3))
    y = np.c_[np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2], x[:, 0] ** 2]
    m = L.train_mlp(x[:3000], y[:3000], x[3000:], y[3000:], hidden=(16, 16), epochs=30, patience=30, lr=3e-3)
    assert m.history[-1]["best_val_mse"] < 0.1 * m.history[0]["val_mse"] + 1e-3
    err = np.mean((m(x[3000:]) - y[3000:]) ** 2) / np.var(y[3000:])
    assert err < 0.1


def test_exported_spec_predicts_like_the_trained_model():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (300, 33))
    y = rng.normal(0, 1, (300, 28))
    m = L.train_mlp(x, y, x[:50], y[:50], hidden=(8,), epochs=2)
    spec = PR.LearnedSpec.from_mlp("mlp", m, "raw", False, False)
    again = PR.LearnedSpec.from_json("mlp", json.loads(json.dumps(spec.to_json())))
    assert np.array_equal(again.forward(x), m(x))
    r = L.Ridge.fit(np.c_[np.ones(300), x[:, :17]], y, 1.0)
    rs = PR.LearnedSpec.from_ridge("ridge", r, "phys", False, False)
    rs2 = PR.LearnedSpec.from_json("ridge", json.loads(json.dumps(rs.to_json())))
    xx = np.c_[np.ones(300), x[:, :17]]
    assert np.allclose(rs2.forward(xx), r(xx), atol=1e-12)
