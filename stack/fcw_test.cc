#include "stack/fcw.h"

#include <gtest/gtest.h>

namespace dr {
namespace {

Track target(int id, double x, double y, double vx) {
  Track t;
  t.id = id; t.s = {x, y, vx, 0}; t.confirmed = true;
  return t;
}

EgoState ego(double v, double yaw_rate = 0) {
  EgoState e;
  e.speed = v; e.yaw_rate = yaw_rate;
  return e;
}

// Runs the debounce: the first warn_confirm_cycles - 1 steps must be silent; returns the events of the last one.
std::vector<FcwEvent> settle(Fcw& f, std::vector<Track>& tr, const EgoState& e, const Params& p, int64_t t0 = 0) {
  for (int k = 0; k < p.warn_confirm_cycles - 1; ++k) EXPECT_TRUE(f.step(t0 + k, tr, e).empty());
  return f.step(t0 + p.warn_confirm_cycles - 1, tr, e);
}

TEST(Fcw, WarnsAfterConfirmCyclesAndReleasesWithHysteresis) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(1, p.front_bumper_m + 10.0, 0.2, -6.0)};  // TTC 1.67 s
  auto ev = settle(f, tr, ego(8), p);
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].kind, "fcw_warning_on");
  EXPECT_EQ(ev[0].track_id, 1);
  EXPECT_NEAR(ev[0].ttc, 10.0 / 6.0, 1e-9);
  tr[0].s[0] = p.front_bumper_m + 30.0;  // TTC 5 s
  for (int k = 0; k < p.release_cycles - 1; ++k) EXPECT_TRUE(f.step(2 + k, tr, ego(8)).empty());
  ev = f.step(9, tr, ego(8));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].kind, "fcw_warning_off");
}

TEST(Fcw, BrakeRequestBelowBrakeTtc) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(3, p.front_bumper_m + 5.0, 0.0, -8.0)};  // TTC 0.625 s
  const auto ev = settle(f, tr, ego(10), p);
  ASSERT_EQ(ev.size(), 2u);
  EXPECT_EQ(ev[0].kind, "fcw_warning_on");
  EXPECT_EQ(ev[1].kind, "brake_request_on");
}

TEST(Fcw, SilentBelowMinimumEgoSpeedOrOutsideCorridorOrUnconfirmed) {
  Params p;
  Fcw f(p);
  std::vector<Track> slow{target(1, 8, 0, -3)};
  std::vector<Track> side{target(2, 8, 3.0, -3)};
  std::vector<Track> tent{target(3, 8, 0, -3)};
  tent[0].confirmed = false;
  for (int k = 0; k < p.warn_confirm_cycles + 3; ++k) {
    EXPECT_TRUE(f.step(k, slow, ego(1.0)).empty());
    EXPECT_TRUE(f.step(k, side, ego(8)).empty());
    EXPECT_TRUE(f.step(k, tent, ego(8)).empty());
  }
}

TEST(Fcw, Bug0001IgnoresOncomingTraffic) {
  Params p;
  Fcw f(p);
  // ego 8.5 m/s, target closes at 16 m/s: it drives towards us at 7.5 m/s over ground
  std::vector<Track> tr{target(1, p.front_bumper_m + 32.0, 0.4, -16.0)};
  for (int k = 0; k < p.warn_confirm_cycles + 3; ++k) EXPECT_TRUE(f.step(k, tr, ego(8.5)).empty());
  // the same geometry with a stopped car (closing = ego speed) must still warn
  Fcw g(p);
  std::vector<Track> stopped{target(2, p.front_bumper_m + 15.0, 0.4, -8.5)};
  EXPECT_EQ(settle(g, stopped, ego(8.5), p).size(), 1u);
}

TEST(Fcw, Bug0002CorridorFollowsTheTurn) {
  Params p;
  // turning left at 0.13 rad/s and 5.6 m/s: with the yaw rate decaying (tau 1 s) the predicted path is
  // about 1.3 m to the left at x = 15.3 m (a constant arc would say 2.7 m)
  std::vector<Track> right{target(1, 15.3, -0.86, -5.6)};
  std::vector<Track> on_arc{target(2, 15.3, 1.3, -5.6)};
  Fcw a(p), b(p);
  for (int k = 0; k < p.warn_confirm_cycles + 3; ++k) EXPECT_TRUE(a.step(k, right, ego(5.6, 0.131)).empty());
  EXPECT_EQ(settle(b, on_arc, ego(5.6, 0.131), p).size(), 1u);
}

TEST(Fcw, Bug0003CorridorStraightensAtTheExitOfATurn) {
  Params p;
  // scene-0916 at 9.41 s: leaving a right turn at 4.44 m/s, yaw rate -0.361 rad/s; a parked car 7.9 m ahead,
  // 3.2 m to the right. A constant-curvature arc puts the path at -2.5 m there; with decay it is at -1.5 m.
  std::vector<Track> parked{target(1, 7.91, -3.22, -2.18)};
  Fcw f(p);
  for (int k = 0; k < p.warn_confirm_cycles + 3; ++k) EXPECT_TRUE(f.step(k, parked, ego(4.44, -0.361)).empty());
  Params arc = p;
  arc.yaw_rate_decay_s = 0.0;  // the be3c4e4 behaviour
  Fcw g(arc);
  EXPECT_EQ(settle(g, parked, ego(4.44, -0.361), arc).size(), 1u);
}

TEST(Fcw, OffEventNamesTheWarnedTrack) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(7, p.front_bumper_m + 8.0, 0.0, -5.0)};
  ASSERT_EQ(settle(f, tr, ego(8), p).size(), 1u);
  std::vector<Track> other{target(9, p.front_bumper_m + 40.0, 0.0, -5.0)};  // TTC 8 s, not a threat
  std::vector<FcwEvent> ev;
  for (int k = 0; k < p.release_cycles; ++k) ev = f.step(2 + k, other, ego(8));
  ASSERT_EQ(ev.size(), 1u);
  EXPECT_EQ(ev[0].kind, "fcw_warning_off");
  EXPECT_EQ(ev[0].track_id, 7);
}

TEST(Fcw, PicksMostCriticalTarget) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(1, 20, 0, -6), target(2, 9, 0.5, -6)};
  f.step(0, tr, ego(8));
  EXPECT_EQ(f.threat().track_id, 2);
}

}  // namespace
}  // namespace dr
