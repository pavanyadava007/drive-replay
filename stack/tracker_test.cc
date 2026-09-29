#include "stack/tracker.h"

#include <cmath>

#include <gtest/gtest.h>

namespace dr {
namespace {

Detection det(double x, double y, double vx, double vy = 0) {
  Detection d;
  d.x = x; d.y = y; d.vx = vx; d.vy = vy; d.n_points = 1;
  return d;
}

TEST(Tracker, ConfirmsAfterConsecutiveHitsAndConvergesOnVelocity) {
  Params p;
  Tracker tr(p);
  const double dt = 0.075;
  double x = 40.0;
  for (int k = 0; k < 40; ++k) {
    tr.step({det(x, 0.3, -5.0)}, k == 0 ? 0.0 : dt, 0.0);
    x -= 5.0 * dt;
    if (k == p.confirm_hits - 2) {
      EXPECT_FALSE(tr.tracks()[0].confirmed);
    }
  }
  ASSERT_EQ(tr.tracks().size(), 1u);
  const Track& t = tr.tracks()[0];
  EXPECT_TRUE(t.confirmed);
  EXPECT_EQ(t.id, 1);
  EXPECT_NEAR(t.s[2], -5.0, 0.05);
  EXPECT_NEAR(t.s[0], x + 5.0 * dt, 0.2);
}

TEST(Tracker, DeletesAfterMisses) {
  Params p;
  Tracker tr(p);
  tr.step({det(20, 0, 0)}, 0.0, 0.0);
  for (int k = 0; k < p.delete_misses; ++k) tr.step({}, 0.075, 0.0);
  EXPECT_TRUE(tr.tracks().empty());
}

TEST(Tracker, KeepsIdentitiesOfTwoCrossingFreeTargets) {
  Params p;
  Tracker tr(p);
  for (int k = 0; k < 20; ++k) tr.step({det(30 - 0.3 * k, -2.0, -4.0), det(25, 2.0, 0.0)}, k ? 0.075 : 0.0, 0.0);
  ASSERT_EQ(tr.tracks().size(), 2u);
  EXPECT_LT(tr.tracks()[0].s[1], 0.0);  // track 1 stays the one on the right
  EXPECT_GT(tr.tracks()[1].s[1], 0.0);
}

TEST(Tracker, EgoRotationMovesStaticTargetSideways) {
  Params p;
  Tracker tr(p);
  tr.step({det(20, 0, 0)}, 0.0, 0.0);
  tr.step({}, 0.1, 0.5);  // ego turns left 0.05 rad: a static point ahead drifts to the right
  EXPECT_LT(tr.tracks()[0].s[1], -0.9);
}

}  // namespace
}  // namespace dr
