#include "stack/vdyn/evaluate.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace dr::vdyn {
namespace {

constexpr double kPi = 3.14159265358979323846;

VehicleConfig config() {
  VehicleConfig c;
  c.wheel_radius_m = 0.3;
  c.kinematic_steering = SteeringMap{15.0, 0.0};
  c.dynamic_steering = SteeringMap{15.0, 0.0};
  return c;
}

// A synthetic scene driven exactly by the kinematic model: constant speed and steering, on a circle.
SceneTable circle_scene(const VehicleConfig& c, double v, double steer_sw) {
  SceneTable s;
  s.name = "synthetic";
  const double r = kinematic_yaw_rate(c.dynamic.wheelbase_m, c.kinematic_steering.road_wheel(steer_sw), v);
  for (int i = 0; i <= 1000; ++i) {
    const double t = 0.02 * i;
    s.t.push_back(t);
    s.x.push_back(v / r * std::sin(r * t));
    s.y.push_back(v / r * (1.0 - std::cos(r * t)));
    s.yaw.push_back(r * t);
    s.v_pose.push_back(v);
    s.steer_sw.push_back(steer_sw);
    s.r_imu.push_back(r);
    s.rpm_rear.push_back(v * 60.0 / (2.0 * kPi * c.wheel_radius_m));
    s.drive.push_back(1);
  }
  return s;
}

double rms(const nlohmann::json& acc) { return std::sqrt(acc[2].get<double>() / acc[0].get<double>()); }

TEST(Evaluate, KinematicSceneIsReproducedByTheKinematicModel) {
  const VehicleConfig c = config();
  const SceneTable s = circle_scene(c, 8.0, 1.0);
  const nlohmann::json j = evaluate_scene(s, c, EvalOptions{});
  EXPECT_LT(rms(j["yaw"]["kinematic"]["all"]), 1e-9);
  EXPECT_LT(rms(j["traj"]["kinematic"]["3"]["fde"]), 1e-4);
  EXPECT_LT(rms(j["traj"]["const_yaw_rate"]["3"]["fde"]), 1e-4);
  EXPECT_LT(rms(j["traj"]["kinematic_rec"]["3"]["fde"]), 1e-4);
  EXPECT_GT(rms(j["traj"]["const_velocity"]["3"]["fde"]), 1.0);   // a straight line misses the circle
  EXPECT_GT(rms(j["traj"]["fcw_yaw_decay"]["3"]["lat"]), 0.5);    // so does a decaying yaw rate
  EXPECT_LT(rms(j["corridor"]["kinematic"]["20"]), 0.02);
  EXPECT_EQ(j["yaw"]["kinematic"]["all"][0].get<double>(), 1001.0);
  EXPECT_GT(j["starts"].get<int>(), 30);
}

TEST(Evaluate, DigestIsDeterministicAndSensitive) {
  const VehicleConfig c = config();
  const SceneTable s = circle_scene(c, 8.0, 1.0);
  const auto a = evaluate_scene(s, c, EvalOptions{});
  const auto b = evaluate_scene(s, c, EvalOptions{});
  EXPECT_EQ(a["digest"], b["digest"]);
  VehicleConfig c2 = c;
  c2.dynamic.cr_npr *= 1.1;
  EXPECT_NE(evaluate_scene(s, c2, EvalOptions{})["digest"], a["digest"]);
}

TEST(Evaluate, NotDrivingSamplesAreNotScored) {
  const VehicleConfig c = config();
  SceneTable s = circle_scene(c, 8.0, 1.0);
  for (size_t i = 0; i < 500; ++i) s.drive[i] = 0;
  const auto j = evaluate_scene(s, c, EvalOptions{});
  EXPECT_EQ(j["yaw"]["kinematic"]["all"][0].get<double>(), 501.0);
}

TEST(Evaluate, ReadsTheExtractorCsv) {
  const std::string path = ::testing::TempDir() + "/scene-9999.csv";
  {
    std::ofstream f(path);
    f << "t_s,x,y,yaw,v_pose,steer_sw,r_imu,rpm_rear,drive\n";
    for (int i = 0; i < 20; ++i) f << 0.02 * i << ",1,2,0.1,5,0.2,0.01,160,1\n";
  }
  const SceneTable s = read_scene_csv(path);
  EXPECT_EQ(s.name, "scene-9999");
  EXPECT_EQ(s.size(), 20u);
  EXPECT_DOUBLE_EQ(s.rpm_rear[3], 160.0);
  std::remove(path.c_str());
}

}  // namespace
}  // namespace dr::vdyn
