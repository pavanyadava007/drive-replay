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

TEST(Fcw, PicksMostCriticalTarget) {
  Params p;
  Fcw f(p);
  std::vector<Track> tr{target(1, 20, 0, -6), target(2, 9, 0.5, -6)};
  f.step(0, tr, ego(8));
  EXPECT_EQ(f.threat().track_id, 2);
}

}  // namespace
}  // namespace dr
