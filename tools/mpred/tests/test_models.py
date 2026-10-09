"""Physics predictors and filters on drives whose answer is known exactly."""
import numpy as np

from tools.mpred import data as D
from tools.mpred import models as M
from tools.mpred.tests.conftest import batch_of, drive


def test_ctrv_follows_the_exact_circle():
    v, r = 12.0, 0.25
    out = M.rollout(np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, v), 0.0, np.full(1, r))
    for j, h in enumerate(D.HORIZONS):
        assert abs(out[0, j, 0] - v / r * np.sin(r * h)) < 1e-9
        assert abs(out[0, j, 1] - v / r * (1 - np.cos(r * h))) < 1e-9
        assert abs(out[0, j, 2] - r * h) < 1e-12


def test_ctra_accelerates_and_never_reverses():
    acc = M.rollout(np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, 5.0), 2.0, np.zeros(1))
    assert abs(acc[0, -1, 0] - 6.0) < 1e-12 and abs(acc[0, -1, 3] - 7.0) < 1e-12
    brk = M.rollout(np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, 2.0), -6.0, np.zeros(1))
    assert np.all(brk[0, :, 3] >= 0.0)
    assert abs(brk[0, -1, 0] - 1.0 / 3.0) < 2e-3
    assert np.all(np.diff(brk[0, :, 0]) >= 0)


def test_predictors_reproduce_a_constant_turn():
    b = batch_of(drive(yaw_rate=0.15))
    si, ki = D.start_points(b)
    assert len(si) > 10
    tru = D.truth(b, si, ki)
    for pred in (M.predict_ctrv(b, si, ki), M.predict_kinematic(b, si, ki, 2.588)):
        assert np.abs(pred[..., :2] - tru[..., :2]).max() < 1e-3
        assert np.abs(pred[..., 2] - tru[..., 2]).max() < 1e-6
    cv = M.predict_cv(b, si, ki)
    assert np.abs(cv[..., :2] - tru[..., :2]).max() > 0.5


def test_slope_acceleration_is_exact_on_a_ramp_and_causal():
    b = batch_of(drive(accel=1.2))
    a = M.accel_signal(b, "slope25")
    assert np.allclose(a[0, 24:int(b.n[0])], 1.2, atol=1e-6)
    assert np.all(a[0, :24] == 0.0)
    si, ki = D.start_points(b)
    pred = M.predict_ctra(b, si, ki, a)
    tru = D.truth(b, si, ki)
    assert np.abs(pred[..., 0] - tru[..., 0]).max() < 1e-3


def test_dynamic_model_is_mirror_symmetric_and_straight_at_zero_steer(vcfg):
    left, right, straight = drive(yaw_rate=0.1), drive(yaw_rate=-0.1, yaw0=0.4), drive(yaw_rate=0.0)
    for tab in (left, right):
        tab[:, 1:3] -= tab[0, 1:3]
    b = batch_of(left, right, straight, vcfg=vcfg)
    si, ki = np.array([0, 1, 2]), np.array([100, 100, 100])
    p = M.predict_dynamic(b, si, ki, vcfg)
    a0 = np.stack([b["x"][si, ki], b["y"][si, ki], b["yaw"][si, ki]], 1)
    rel = []
    for i in range(2):
        c, s = np.cos(a0[i, 2]), np.sin(a0[i, 2])
        dx, dy = p[i, :, 0] - a0[i, 0], p[i, :, 1] - a0[i, 1]
        rel.append((c * dx + s * dy, -s * dx + c * dy))
    assert np.allclose(rel[0][0], rel[1][0], atol=1e-9)
    assert np.allclose(rel[0][1], -rel[1][1], atol=1e-9)
    assert rel[0][1][-1] > 0.1
    assert np.allclose(p[2, :, 2], b["yaw"][2, 100], atol=1e-12)


def test_dynamic_model_below_min_speed_moves_kinematically(vcfg):
    b = batch_of(drive(v0=0.6, yaw_rate=0.05), vcfg=vcfg)
    si, ki = np.array([0]), np.array([80])
    p = M.predict_dynamic(b, si, ki, vcfg)
    v = b["v"][0, 80]
    k = M.rollout(b["x"][0, 80:81], b["y"][0, 80:81], b["yaw"][0, 80:81], np.array([v]), 0.0,
                  np.array([v * np.tan(b["d_dyn"][0, 80]) / vcfg["wheelbase_m"]]))
    assert np.allclose(p, k)


def test_kalman_filters_track_a_straight_line():
    b = batch_of(drive(yaw_rate=0.0, accel=0.0))
    si, ki = D.start_points(b)
    tru = D.truth(b, si, ki)
    for order in (2, 3):
        st = M.run_kf(b, M.KfParams(order, 1.0, 0.01))
        pred = M.predict_kf(st, si, ki, D.HORIZONS)
        assert np.abs(pred[..., :2] - tru[..., :2]).max() < 0.02
        assert np.abs(M.wrap(pred[..., 2] - tru[..., 2])).max() < 1e-3


def test_ekf_tracks_a_turn_and_skips_lost_samples():
    tab = drive(yaw_rate=0.12, accel=0.3)
    b = batch_of(tab)
    p = M.EkfParams(r_xy=0.001, r_psi=1e-4, r_scale=3.0)
    st = M.run_ekf(b, p)
    k = 300
    assert abs(st[0, k, 3] - b["v"][0, k]) < 0.05
    assert abs(st[0, k, 5] - 0.12) < 2e-3
    si, ki = D.start_points(b)
    pred = M.predict_from_state(st, si, ki)
    tru = D.truth(b, si, ki)
    assert np.abs(pred[:, :6, :2] - tru[:, :6, :2]).max() < 0.02
    # a wrong wheel speed flagged as lost must not move the filter
    bad = b.with_signals(v=np.where(np.arange(b.T)[None, :] >= 200, 50.0, b["v"]))
    avail = (np.arange(b.T)[None, :] < 200).astype(float)
    st2 = M.run_ekf(bad, p, odo_avail=avail)
    assert abs(st2[0, 250, 3] - b["v"][0, 250]) < 0.2


def test_odometry_bridge_matches_the_drive():
    b = batch_of(drive(yaw_rate=0.1))
    si, k = np.array([0, 0]), np.array([100, 150])
    j = k + 4
    x, y, p = M.dead_reckon(b, si, k, j, b["x"][si, k], b["y"][si, k], b["yaw"][si, k])
    assert np.allclose(x, b["x"][si, j], atol=1e-4) and np.allclose(y, b["y"][si, j], atol=1e-4)
    assert np.allclose(p, b["yaw"][si, j], atol=1e-6)
