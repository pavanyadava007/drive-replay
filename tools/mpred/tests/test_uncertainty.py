"""Metrics of the uncertainty study (AUROC, interval calibration, NLL scale, risk-coverage) and one tiny training run."""
import numpy as np
import pytest

U = pytest.importorskip("tools.mpred.uncertainty", exc_type=ImportError)  # needs torch


def test_auroc_matches_pairwise_count_with_ties_and_weights():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=300), 1)
    y = rng.random(300) < 0.3
    pos, neg = s[y], s[~y]
    ref = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert abs(U.auroc(s, y) - ref) < 1e-12
    w = rng.integers(0, 3, 300).astype(float)
    wp, wn = w[y], w[~y]
    pair = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    assert abs(U.auroc(s, y, w) - (wp[:, None] * wn[None, :] * pair).sum() / (wp.sum() * wn.sum())) < 1e-12
    assert U.auroc(-s, y) == pytest.approx(1 - U.auroc(s, y))


def test_calibration_and_scale_on_gaussian_errors():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 2.0, 50_000)
    assert U.calibration_error(e, np.full(e.size, 2.0)) < 0.01
    assert U.calibration_error(e, np.full(e.size, 1.0)) > 0.15
    assert U.var_scale(e, np.full(e.size, 1.0)) == pytest.approx(2.0, rel=0.02)
    assert U.gaussian_nll(e, np.full(e.size, 2.0)) < U.gaussian_nll(e, np.full(e.size, 1.0))


def test_risk_coverage_oracle_is_lowest():
    rng = np.random.default_rng(2)
    err = np.abs(rng.normal(size=1000))
    curve_oracle, aurc_oracle = U.risk_coverage(err, err)
    _, aurc_rand = U.risk_coverage(rng.random(1000), err)
    assert curve_oracle[-1] == pytest.approx(err.mean())
    assert aurc_oracle < aurc_rand


def test_heteroscedastic_ensemble_learns_input_dependent_noise():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (4000, 1))
    y = np.c_[x[:, 0] + rng.normal(size=4000) * (0.05 + 0.5 * (x[:, 0] > 0))]
    f = U.fit_ensemble(x[:3000], y[:3000], x[3000:], y[3000:], seeds=[1, 2], log=lambda m: None)
    mus, vas = U.sample_outputs(f, np.array([[-0.5], [0.5]]))
    assert mus.shape == (2, 2, 1)
    assert vas[:, 1, 0].mean() > 10 * vas[:, 0, 0].mean()  # the noisy half gets the larger variance
