#include "stack/vdyn/vdyn.h"

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace dr::vdyn {
namespace {

VehicleParams test_params() {
  VehicleParams p;
  p.wheelbase_m = 2.588;
  p.lf_m = 1.165;
  p.mass_kg = 1650.0;
  p.yaw_inertia_kgm2 = 1650.0 * 1.165 * (2.588 - 1.165);
  p.cf_npr = 80000.0;
  p.cr_npr = 120000.0;
  return p;
}

State run(const Model& m, const Input& u, double measured_r, double seconds, double dt = 0.02, State s0 = {}) {
  State s = s0;
  m.init(s, u, measured_r);
  const int n = static_cast<int>(std::lround(seconds / dt));
  for (int i = 0; i < n; ++i) m.step(s, u, u, dt);
  return s;
}

bool finite(const State& s) {
  return std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.psi) && std::isfinite(s.vy) && std::isfinite(s.r);
}

TEST(Vdyn, ZeroSteerDrivesStraight) {
  const KinematicBicycle kin(2.588);
  const LinearSingleTrack dyn(test_params());
  for (const Model* m : std::vector<const Model*>{&kin, &dyn}) {
    const State s = run(*m, Input{0.0, 10.0}, 0.0, 3.0);
    EXPECT_NEAR(s.x, 30.0, 1e-9) << m->name();
    EXPECT_NEAR(s.y, 0.0, 1e-12) << m->name();
    EXPECT_NEAR(s.psi, 0.0, 1e-12) << m->name();
  }
}

TEST(Vdyn, SteeringSignIsMirrorSymmetric) {
  const KinematicBicycle kin(2.588);
  const LinearSingleTrack dyn(test_params());
  for (const Model* m : std::vector<const Model*>{&kin, &dyn}) {
    State a, b;
    m->init(a, Input{0.03, 8.0}, 0.0);
    m->init(b, Input{-0.03, 8.0}, 0.0);
    for (int i = 0; i < 200; ++i) {  // time-varying steering, mirrored
      const double d0 = 0.03 * std::sin(0.05 * i), d1 = 0.03 * std::sin(0.05 * (i + 1));
      m->step(a, Input{d0, 8.0}, Input{d1, 8.0}, 0.02);
      m->step(b, Input{-d0, 8.0}, Input{-d1, 8.0}, 0.02);
    }
    EXPECT_NEAR(a.x, b.x, 1e-9) << m->name();
    EXPECT_NEAR(a.y, -b.y, 1e-9) << m->name();
    EXPECT_NEAR(a.psi, -b.psi, 1e-12) << m->name();
    EXPECT_NEAR(a.r, -b.r, 1e-12) << m->name();
    EXPECT_GT(std::fabs(a.y), 0.1) << m->name();
  }
  // left steering turns left
  EXPECT_GT(run(dyn, Input{0.02, 10.0}, 0.0, 2.0).y, 0.0);
}

TEST(Vdyn, LinearModelSteadyStateYawRateMatchesUndersteerFormula) {
  const VehicleParams p = test_params();
  const LinearSingleTrack dyn(p);
  const double K = p.understeer_K();
  EXPECT_GT(K, 0.0);  // lr/Cf > lf/Cr: understeer with these parameters
  for (double v : {3.0, 5.0, 10.0, 15.0, 20.0}) {
    const double delta = 0.02;
    const State s = run(dyn, Input{delta, v}, 0.0, 15.0);
    const double expect = v * delta / (p.wheelbase_m * (1.0 + K * v * v));
    EXPECT_NEAR(s.r, expect, 1e-9 * std::max(1.0, std::fabs(expect))) << "v " << v;
    EXPECT_NEAR(s.r, linear_steady_yaw_rate(p.wheelbase_m, K, delta, v), 1e-9);
  }
}

TEST(Vdyn, DynamicConvergesToKinematicAtLowSpeed) {
  const VehicleParams p = test_params();
  const LinearSingleTrack dyn(p);
  const double delta = 0.05;
  double prev = 1e9;
  for (double v : {15.0, 10.0, 5.0, 2.0, 1.2}) {
    const double r_dyn = run(dyn, Input{delta, v}, 0.0, 20.0).r;
    const double r_kin = kinematic_yaw_rate(p.wheelbase_m, delta, v);
    const double rel = std::fabs(r_dyn - r_kin) / r_kin;
    EXPECT_LT(rel, prev) << "v " << v;  // the gap shrinks with speed
    prev = rel;
  }
  EXPECT_LT(prev, 0.01);  // within 1 % of kinematic at 1.2 m/s
}

TEST(Vdyn, StandstillStaysPutWithoutNaN) {
  const KinematicBicycle kin(2.588);
  const LinearSingleTrack dyn(test_params());
  const ConstantYawRate cyr;
  const ConstantVelocity cv;
  const YawRateDecay decay(1.0, 0.2);
  for (const Model* m : std::vector<const Model*>{&kin, &dyn, &cyr, &cv, &decay}) {
    const State s = run(*m, Input{0.3, 0.0}, 0.0, 5.0, 0.02, State{1.0, 2.0, 0.5, 0.0, 0.0});
    EXPECT_TRUE(finite(s)) << m->name();
    EXPECT_DOUBLE_EQ(s.x, 1.0) << m->name();
    EXPECT_DOUBLE_EQ(s.y, 2.0) << m->name();
    EXPECT_DOUBLE_EQ(s.psi, 0.5) << m->name();
  }
  // a stop from speed through the low-speed fallback, with steering, stays finite
  State s;
  dyn.init(s, Input{0.2, 3.0}, 0.0);
  for (int i = 0; i < 300; ++i) {
    const double v0 = std::max(0.0, 3.0 - 0.01 * i), v1 = std::max(0.0, 3.0 - 0.01 * (i + 1));
    dyn.step(s, Input{0.2, v0}, Input{0.2, v1}, 0.02);
    ASSERT_TRUE(finite(s)) << "step " << i;
  }
  EXPECT_DOUBLE_EQ(s.r, 0.0);
}

TEST(Vdyn, Rk4ArcMatchesTheCircle) {
  const ConstantYawRate cyr;
  const double v = 10.0, r = 0.3, t = 3.0;
  const State s = run(cyr, Input{0.0, v}, r, t);
  EXPECT_NEAR(s.x, v / r * std::sin(r * t), 1e-7);
  EXPECT_NEAR(s.y, v / r * (1.0 - std::cos(r * t)), 1e-7);
  EXPECT_NEAR(s.psi, r * t, 1e-12);
}

TEST(Vdyn, YawRateDecayHeadingAndClamp) {
  const YawRateDecay decay(1.0, 0.2);
  const double v = 10.0, r0 = 0.25, t = 2.0;
  const State s = run(decay, Input{0.0, v}, r0, t);
  EXPECT_NEAR(s.psi, r0 * 1.0 * (1.0 - std::exp(-t / 1.0)), 1e-7);
  // curvature clamp of the FCW: 2 rad/s at 5 m/s is clamped to 0.2 1/m * 5 m/s = 1 rad/s
  State c;
  decay.init(c, Input{0.0, 5.0}, 2.0);
  EXPECT_DOUBLE_EQ(c.r, 1.0);
}

TEST(Vdyn, DecayCorridorFormulaAgreesWithTheModelAtSmallAngles) {
  // decay_path_offset is the small-angle closed form the FCW uses; the integrated model must agree when the
  // heading change stays small.
  const YawRateDecay decay(1.0, 0.2);
  PredictedPath path;
  const double v = 12.0, r0 = 0.02;
  path.predict(decay, Input{0.0, v}, r0, 4.0, 0.02);
  for (double x : {10.0, 20.0, 30.0}) EXPECT_NEAR(path.offset(x), decay_path_offset(v, r0, x, 0.2, 1.0), 0.01) << x;
}

TEST(Vdyn, PredictedPathOffset) {
  const ConstantVelocity cv;
  const ConstantYawRate cyr;
  PredictedPath straight, arc;
  straight.predict(cv, Input{0.0, 10.0}, 0.0, 4.0, 0.05);
  EXPECT_DOUBLE_EQ(straight.offset(25.0), 0.0);
  EXPECT_DOUBLE_EQ(straight.offset(60.0), 0.0);  // extrapolated beyond the horizon
  const double v = 10.0, r = 0.2, R = v / r;
  arc.predict(cyr, Input{0.0, v}, r, 4.0, 0.01);
  const double x = 20.0;
  EXPECT_NEAR(arc.offset(x), R - std::sqrt(R * R - x * x), 0.02);
}

TEST(Vdyn, DecayingSteeringPathLiesBetweenStraightAndHeld) {
  const LinearSingleTrack dyn(test_params());
  PredictedPath held, decayed;
  const Input u{0.05, 8.0};
  const double r0 = linear_steady_yaw_rate(2.588, test_params().understeer_K(), 0.05, 8.0);
  held.predict(dyn, u, r0, 4.0, 0.05);
  decayed.predict(dyn, u, r0, 4.0, 0.05, 1.0);
  for (double x : {5.0, 15.0, 25.0}) {
    EXPECT_GT(decayed.offset(x), 0.0) << x;
    EXPECT_LT(decayed.offset(x), held.offset(x)) << x;
  }
}

TEST(Vdyn, LowSpeedSubstepsKeepRk4Stable) {
  const LinearSingleTrack dyn(test_params());
  const State s = run(dyn, Input{0.1, 1.05}, 0.0, 10.0, 0.1);
  EXPECT_TRUE(finite(s));
  EXPECT_NEAR(s.r, linear_steady_yaw_rate(2.588, test_params().understeer_K(), 0.1, 1.05), 1e-6);
}

TEST(Vdyn, ShippedConfigLoadsAndIsPhysical) {
  const VehicleConfig c = load_vehicle_config("configs/vdyn/renault_zoe.json");
  EXPECT_DOUBLE_EQ(c.dynamic.wheelbase_m, 2.588);
  EXPECT_GT(c.dynamic.lr(), 0.0);
  EXPECT_GT(c.kinematic_steering.ratio, 5.0);
  EXPECT_LT(c.kinematic_steering.ratio, 30.0);
  EXPECT_GT(c.wheel_radius_m, 0.2);
  EXPECT_LT(c.wheel_radius_m, 0.4);
  EXPECT_THROW(load_vehicle_config("does/not/exist.json"), std::runtime_error);
}

}  // namespace
}  // namespace dr::vdyn
