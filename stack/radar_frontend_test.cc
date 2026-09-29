#include "stack/radar_frontend.h"

#include <cmath>

#include <gtest/gtest.h>

namespace dr {
namespace {

RadarPoint pt(double x, double y, double vx, int ambig = 3) {
  RadarPoint r;
  r.x = x; r.y = y; r.vx = vx; r.ambig_state = ambig;
  return r;
}

TEST(RadarFrontend, AppliesMountingOffsetAndFiltersInvalidReturns) {
  Params p;
  Extrinsic ext;
  ext.tx = 3.412; ext.valid = true;
  RadarScan s;
  s.points = {pt(10, 0, -5), pt(10, 0, -5, /*ambig=*/1), pt(200, 0, 0)};
  s.points[0].invalid_state = 0;
  const auto v = to_vehicle_frame(s, ext, p);
  ASSERT_EQ(v.size(), 1u);  // ambiguous and out-of-range points dropped
  EXPECT_NEAR(v[0].x, 13.412, 1e-9);
}

TEST(RadarFrontend, RotatesPositionAndVelocity) {
  Params p;
  Extrinsic ext;
  ext.yaw = M_PI / 2; ext.valid = true;
  RadarScan s;
  s.points = {pt(10, 0, -5)};
  const auto v = to_vehicle_frame(s, ext, p);
  EXPECT_NEAR(v[0].x, 0.0, 1e-9);
  EXPECT_NEAR(v[0].y, 10.0, 1e-9);
  EXPECT_NEAR(v[0].vy, -5.0, 1e-9);
}

TEST(RadarFrontend, ClustersByDistanceAndVelocity) {
  Params p;
  const std::vector<RadarPoint> pts = {pt(10, 0, -5), pt(10.8, 0.4, -5.2), pt(10.5, 0.2, 3.0), pt(30, 0, 0)};
  const auto d = cluster(pts, p);
  ASSERT_EQ(d.size(), 3u);  // the third point is close but moves the other way
  EXPECT_EQ(d[0].n_points, 2);
  EXPECT_NEAR(d[0].x, 10.4, 1e-9);
}

TEST(RadarFrontend, ClusterOrderIsStable) {
  Params p;
  const std::vector<RadarPoint> pts = {pt(30, 0, 0), pt(10, 0, -5), pt(10.5, 0, -5), pt(30.5, 0, 0)};
  const auto d = cluster(pts, p);
  ASSERT_EQ(d.size(), 2u);
  EXPECT_NEAR(d[0].x, 30.25, 1e-9);
  EXPECT_NEAR(d[1].x, 10.25, 1e-9);
}

}  // namespace
}  // namespace dr
