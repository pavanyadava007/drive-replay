"""The AR-HUD demo data (tools/mpred/demo_export.py) must equal direct predictor calls of the benchmark code path."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mpred import data as D
from tools.mpred import demo_export as DE
from tools.mpred import experiment as X
from tools.mpred import metrics as E
from tools.mpred import predictors as PR
from tools.mpred.tests.conftest import ROOT, batch_of, drive


def synthetic_split(vcfg) -> X.Split:
    b = batch_of(drive(n=320, yaw_rate=lambda t: 0.2 * np.sin(0.6 * t), accel=0.3),
                 drive(n=320, v0=11.0, yaw_rate=-0.04, accel=-0.6), vcfg=vcfg)
    si, ki = D.start_points(b)
    return X.Split("test", b, si, ki, D.truth(b, si, ki))


def decode(row: dict, n_starts: int) -> dict:
    lon = np.array(row["lon"]).reshape(n_starts, DE.N_H) / DE.Q_POS
    lat = np.array(row["lat"]).reshape(n_starts, DE.N_H) / DE.Q_POS
    head = np.array(row["head"]).reshape(n_starts, DE.N_H) / DE.Q_HEAD
    # the page's formulas (site/hud.js), as in tools/mpred/metrics.py
    return {"lon": lon, "lat": lat, "head": head, "pos": np.hypot(lon, lat), "ov20": lat + DE.OV_D * np.sin(head)}


def test_export_equals_direct_predictor_calls(conf, vcfg):
    sp = synthetic_split(vcfg)
    levels = [DE.run_level(sp.b, sp.si, sp.ki, sp.tru, conf, vcfg, lev) for lev in (0.0, 1.0)]
    # level 0: the clean batch through predict_all; level 1: the seeded perturbation of the noise sweep
    direct0 = PR.predict_all(sp.b, sp.si, sp.ki, conf, vcfg, DE.GUI_METHODS)
    rng = np.random.default_rng(X.SEED + 100)
    nb, extra, _ = X.perturb(sp.b, vcfg, dict(X.BASE_SIGMA), rng)
    direct1 = PR.predict_all(nb, sp.si, sp.ki, conf, vcfg, DE.GUI_METHODS, extra_std=extra)
    for m in DE.GUI_METHODS:
        assert np.array_equal(levels[0].preds[m], direct0[m]), m
        assert np.array_equal(levels[1].preds[m], direct1[m]), m
        assert not np.allclose(direct0[m], direct1[m]), m  # the noise reaches every method
    stale = levels[0].preds["stale"]
    assert np.array_equal(stale[:, 3, :3], np.stack([sp.b["x"][sp.si, sp.ki], sp.b["y"][sp.si, sp.ki],
                                                     sp.b["yaw"][sp.si, sp.ki]], 1))

    pick = {"scene_index": 0, "rule": "test", "key": "turn_in", "label": "test"}
    doc = json.loads(json.dumps(DE.export_scene(sp, pick, levels)))
    sel = sp.si == 0
    assert doc["starts"]["k"] == [int(k) for k in sp.ki[sel]]
    for lv, direct in ((levels[0], direct0), (levels[1], direct1)):
        L = doc["levels"][f"{lv.level:g}"]
        for m in DE.GUI_METHODS:
            ref = E.errors(direct[m], sp.tru)
            got = decode(L["rows"][m], int(sel.sum()))
            for k in ("lon", "lat", "pos", "ov20"):
                assert np.allclose(got[k], ref[k][sel, :DE.N_H], atol=2e-5, rtol=0), (m, k)
            assert np.allclose(got["head"], ref["head"][sel, :DE.N_H], atol=1e-7, rtol=0), m
            for k in ("pos", "head", "ov20"):
                means = np.abs(ref[k][sel, :DE.N_H]).mean(0)
                assert np.allclose(L["means"][m][k], means, atol=0, rtol=1e-12), (m, k)


HUD = ROOT / "site" / "data" / "hud"


@pytest.mark.skipif(not (HUD / "index.json").exists() or not (D.DEFAULT_SCENES / "manifest.json").exists(),
                    reason="needs the exported demo data and the nuScenes CAN tables (outside git)")
def test_exported_files_match_the_real_predictors():
    index = json.loads((HUD / "index.json").read_text())
    entry = index["scenes"][0]
    doc = json.loads((HUD / entry["file"]).read_text())
    vcfg = D.vehicle_config()
    conf = PR.load_config()
    # level 0 needs only this scene: the filters run per scene, so a one-scene batch gives the same states
    b = D.load([doc["scene"]], D.DEFAULT_SCENES, vcfg)
    ks = np.array(doc["starts"]["k"])
    pick = np.array([0, len(ks) // 3, len(ks) // 2, len(ks) - 1])
    si, ki = np.zeros(len(pick), dtype=np.int64), ks[pick]
    assert np.array_equal(D.start_points(b)[1], ks)
    tru = D.truth(b, si, ki)
    preds = PR.predict_all(b, si, ki, conf, vcfg, DE.GUI_METHODS)
    L = doc["levels"]["0"]
    for m in DE.GUI_METHODS:
        ref = E.errors(preds[m], tru)
        got = decode(L["rows"][m], len(ks))
        for k in ("lat", "pos", "ov20"):
            assert np.allclose(got[k][pick], ref[k][:, :DE.N_H], atol=2e-5, rtol=0), (m, k)
        assert np.allclose(got["head"][pick], ref["head"][:, :DE.N_H], atol=1e-7, rtol=0), m
    # the recorded track as stored
    t = np.array(doc["track"]["t_ms"]) / 1e3
    assert np.allclose(t, b["t"][0, :b.n[0]] - b["t"][0, 0], atol=6e-4)
    assert Path(HUD / entry["file"]).stat().st_size < 1_000_000
