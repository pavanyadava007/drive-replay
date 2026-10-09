"""No future leakage: a prediction made at sample k must not change when anything after k changes."""
import numpy as np

from tools.mpred import data as D
from tools.mpred import predictors as PR
from tools.mpred.tests.conftest import batch_of, drive


def test_no_prediction_reads_samples_after_its_start(conf, vcfg):
    tab = drive(n=300, yaw_rate=lambda t: 0.2 * np.sin(0.7 * t), accel=0.4)
    k = 150
    other = tab.copy()
    rng = np.random.default_rng(9)
    other[k + 1:, 1:4] += rng.normal(0, 0.5, other[k + 1:, 1:4].shape)  # pose
    other[k + 1:, 6:12] += rng.normal(0, 0.5, other[k + 1:, 6:12].shape)  # steering, IMU, wheels, accelerations
    si, ki = np.array([0]), np.array([k])
    a = PR.predict_all(batch_of(tab, vcfg=vcfg), si, ki, conf, vcfg)
    b = PR.predict_all(batch_of(other, vcfg=vcfg), si, ki, conf, vcfg)
    assert set(a) == set(PR.METHODS)
    for m in PR.METHODS:
        assert np.array_equal(a[m], b[m]), m
        assert np.all(np.isfinite(a[m])), m
    # and a change at sample k does change them (the test can fail)
    third = tab.copy()
    third[k, 7] += 0.05  # yaw rate at k
    c = PR.predict_all(batch_of(third, vcfg=vcfg), si, ki, conf, vcfg, ("ctrv", "ekf"))
    assert not np.array_equal(a["ctrv"], c["ctrv"]) and not np.array_equal(a["ekf"], c["ekf"])


def test_start_points_need_history_drive_and_speed():
    tab = drive(n=300)
    tab[200:, 5] = 0  # out of drive from sample 200
    b = batch_of(tab)
    si, ki = D.start_points(b)
    assert ki.min() >= D.HISTORY
    t = b["t"][0]
    assert np.all(t[ki] + D.FUTURE_S < t[199] + 1e-9)
    slow = drive(n=300, v0=0.5)
    assert len(D.start_points(batch_of(slow))[0]) == 0
