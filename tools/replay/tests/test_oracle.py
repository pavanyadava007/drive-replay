"""Oracle geometry and verdicts on a synthetic straight drive at 10 m/s."""
from tools.replay.oracle import Oracle
from tools.replay.recording import RecordingView

T0 = 1_000_000_000


def drive(objects, seconds=6.0):
    ego = [{"t_us": T0 + int(k * 5e4), "x": 0.5 * k, "y": 0.0, "yaw": 0.0, "speed_mps": 10.0, "yaw_rate_rps": 0.0}
           for k in range(int(seconds * 20) + 1)]
    gt = [{"t_us": T0 + int(k * 5e5), "objects": objects} for k in range(int(seconds * 2) + 1)]
    return RecordingView({"t_us": T0}, ego, gt)


def car(x, y, radar=3, vx=0.0):
    return {"id": 1, "category": "vehicle.car", "x": x, "y": y, "yaw": 0.0, "length": 4.5, "width": 1.9,
            "vx": vx, "vy": 0.0, "num_radar_pts": radar}


def test_stopped_car_in_path_has_true_ttc():
    o = Oracle(drive([car(40.0, 0.2)]))
    th = o.threats(T0)
    assert len(th) == 1
    # gap = 40 - front bumper 3.7 - half length 2.25 = 34.05 m at 10 m/s closing
    assert abs(th[0]["gap_m"] - 34.05) < 0.01
    assert abs(th[0]["ttc_s"] - 3.4) < 0.01


def test_object_beside_the_path_is_not_a_threat():
    assert Oracle(drive([car(30.0, 3.5)])).threats(T0) == []


def test_lead_car_at_same_speed_has_no_ttc():
    th = Oracle(drive([car(30.0, 0.0, vx=10.0)])).threats(T0)
    assert th[0]["ttc_s"] is None


def test_warning_verdicts():
    o = Oracle(drive([car(30.0, 0.0)]))
    ev = [{"t_us": T0 + 500_000, "kind": "fcw_warning_on", "track": 3, "ttc": 2.0}]
    assert o.judge(ev)["warnings"][0]["verdict"] == "justified"
    empty = Oracle(drive([car(30.0, 4.0)]))
    assert empty.judge(ev)["warnings"][0]["verdict"] == "false"


def test_missed_threat_only_when_radar_sees_it():
    # car at 25 m: gap 19 m at 10 m/s -> TTC < 1.5 s from about 0.4 s on
    seen = Oracle(drive([car(25.0, 0.0, radar=2)])).judge([])
    assert seen["missed_threats"] > 0 and not seen["requirements_met"]
    unseen = Oracle(drive([car(25.0, 0.0, radar=0)])).judge([])
    assert unseen["missed_threats"] == 0 and unseen["not_observable"] > 0


def test_active_warning_covers_the_threat():
    o = Oracle(drive([car(25.0, 0.0)]))
    ev = [{"t_us": T0, "kind": "fcw_warning_on", "track": 1, "ttc": 2.0}]
    assert o.judge(ev)["missed_threats"] == 0
