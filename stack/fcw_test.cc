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

TEST(Fcw, WarnsAfterConfirmCyclesAndReleasesWithHysteresis) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(1, p.front_bumper_m + 10.0, 0.2, -6.0)};  // TTC 1.67 s
  EXPECT_TRUE(f.step(0, tr, ego(8)).empty());
  auto ev = f.step(1, tr, ego(8));
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
  f.step(0, tr, ego(10));
  const auto ev = f.step(1, tr, ego(10));
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
  for (int k = 0; k < 5; ++k) {
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
  for (int k = 0; k < 5; ++k) EXPECT_TRUE(f.step(k, tr, ego(8.5)).empty());
  // the same geometry with a stopped car (closing = ego speed) must still warn
  Fcw g(p);
  std::vector<Track> stopped{target(2, p.front_bumper_m + 15.0, 0.4, -8.5)};
  g.step(0, stopped, ego(8.5));
  EXPECT_EQ(g.step(1, stopped, ego(8.5)).size(), 1u);
}

TEST(Fcw, Bug0002CorridorFollowsTheTurn) {
  Params p;
  // turning left at 0.13 rad/s and 5.6 m/s: at x = 15 m the path is about 2.6 m to the left
  std::vector<Track> right{target(1, 15.3, -0.86, -5.6)};
  std::vector<Track> on_arc{target(2, 15.3, 2.7, -5.6)};
  Fcw a(p), b(p);
  for (int k = 0; k < 4; ++k) EXPECT_TRUE(a.step(k, right, ego(5.6, 0.131)).empty());
  b.step(0, on_arc, ego(5.6, 0.131));
  EXPECT_EQ(b.step(1, on_arc, ego(5.6, 0.131)).size(), 1u);
}

TEST(Fcw, OffEventNamesTheWarnedTrack) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(7, p.front_bumper_m + 8.0, 0.0, -5.0)};
  f.step(0, tr, ego(8));
  f.step(1, tr, ego(8));
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
