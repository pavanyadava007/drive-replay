"""Batch pipeline: determinism across worker counts and chunking, and the scene-level statistics."""
import numpy as np

from tools.vdyn import pipeline as P


def test_batch_digest_is_independent_of_workers_and_chunking(scene_set):
    kw = {"scenes_dir": scene_set["dir"], "config": scene_set["config"], "binary": scene_set["binary"]}
    a = P.run_batch(scene_set["names"], 1, chunk=1, **kw)
    b = P.run_batch(scene_set["names"], 4, chunk=3, **kw)
    assert not a.errors and not b.errors
    assert a.batch_digest == b.batch_digest
    assert len(a.scenes) == 4
    assert a.recorded_s > 70


def test_kinematic_scenes_are_reproduced_by_the_kinematic_model(scene_set):
    res = P.run_batch(scene_set["names"], 2, scene_set["dir"], scene_set["config"], scene_set["binary"])
    rows = list(res.scenes.values())
    assert P.pooled(P.sums(rows, ("yaw", "kinematic", "all")))["rmse"] < 1e-5
    assert P.pooled(P.sums(rows, ("traj", "kinematic", "3", "fde")))["mean_abs"] < 1e-3
    assert P.pooled(P.sums(rows, ("traj", "const_velocity", "3", "fde")))["mean_abs"] > 0.5


def test_parameter_override_changes_the_digest(scene_set):
    kw = {"scenes_dir": scene_set["dir"], "config": scene_set["config"], "binary": scene_set["binary"]}
    a = P.run_batch(scene_set["names"], 2, **kw)
    b = P.run_batch(scene_set["names"], 2, sets=["dynamic.cr_npr=90000"], **kw)
    assert a.batch_digest != b.batch_digest


def test_bootstrap_ci_and_paired_difference():
    rng = np.random.default_rng(0)
    n = rng.integers(50, 100, size=40).astype(float)
    a = np.column_stack([n, n * 1.0 + rng.normal(0, 2, 40), n * 1.5])
    b = np.column_stack([n, n * 2.0, n * 4.5])
    w = P.bootstrap_weights(40, b=500)
    assert w.sum(axis=1).tolist() == [40.0] * 500
    c = P.ci(a, w)
    assert c["lo"] <= c["value"] <= c["hi"]
    p = P.paired(a, b, w)
    assert p["diff"] < 0 and p["hi"] < 0 and p["significant"]
    assert p["a_better_scenes"] == 40
    assert P.pooled(np.zeros((3, 3)))["n"] == 0
