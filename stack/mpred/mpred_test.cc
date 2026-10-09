#include "stack/mpred/mpred.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace dr::mpred {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Samples of a car driving a circle (or a straight line for r = 0) at constant speed, with exact signals.
Sample circle_sample(int k, double v, double r, double dt = 0.02) {
  Sample s;
  s.t = k * dt;
  s.yaw = r * s.t;
  if (r == 0.0) {
    s.x = v * s.t;
  } else {
    s.x = v / r * std::sin(r * s.t);
    s.y = v / r * (1.0 - std::cos(r * s.t));
  }
  s.v = v;
  s.r = r;
  s.d_kin = std::atan(r * 2.588 / std::max(v, 0.1));
  s.d_dyn = s.d_kin;
  return s;
}

TEST(Mpred, ConstantVelocityIsAStraightLine) {
  const Prediction p = rollout(1.0, 2.0, 0.3, 10.0, 0.0, 0.0);
  for (size_t k = 0; k < kHorizons; ++k) {
    EXPECT_NEAR(p[k].x, 1.0 + 10.0 * kHorizonS[k] * std::cos(0.3), 1e-12);
    EXPECT_NEAR(p[k].y, 2.0 + 10.0 * kHorizonS[k] * std::sin(0.3), 1e-12);
    EXPECT_DOUBLE_EQ(p[k].yaw, 0.3);
    EXPECT_DOUBLE_EQ(p[k].v, 10.0);
  }
}

TEST(Mpred, CtrvFollowsTheExactCircle) {
  const double v = 12.0, r = 0.25;
  const Prediction p = rollout(0, 0, 0, v, 0.0, r);
  for (size_t k = 0; k < kHorizons; ++k) {
    const double t = kHorizonS[k];
    EXPECT_NEAR(p[k].x, v / r * std::sin(r * t), 1e-9);
    EXPECT_NEAR(p[k].y, v / r * (1 - std::cos(r * t)), 1e-9);
    EXPECT_NEAR(p[k].yaw, r * t, 1e-12);
  }
}

TEST(Mpred, CtraAcceleratesAndStopsAtZeroSpeed) {
  const Prediction acc = rollout(0, 0, 0, 5.0, 2.0, 0.0);
  EXPECT_NEAR(acc.back().x, 5.0 + 0.5 * 2.0, 1e-12);
  EXPECT_NEAR(acc.back().v, 7.0, 1e-12);
  // braking at -6 m/s^2 from 2 m/s: stops after 1/3 s at 1/3 m and stays there (no reversing)
  const Prediction brk = rollout(0, 0, 0, 2.0, -6.0, 0.0);
  EXPECT_NEAR(brk.back().x, 1.0 / 3.0, 2e-3);
  EXPECT_GE(brk.back().v, 0.0);
  EXPECT_LT(brk.back().v, 1e-9);
  for (size_t k = 1; k < kHorizons; ++k) EXPECT_GE(brk[k].x, brk[k - 1].x);
}

TEST(Mpred, WrapAngle) {
  EXPECT_NEAR(wrap_angle(3 * kPi / 2), -kPi / 2, 1e-15);
  EXPECT_NEAR(wrap_angle(-3 * kPi / 2), kPi / 2, 1e-15);
  EXPECT_DOUBLE_EQ(wrap_angle(0.1), 0.1);
}

TEST(Mpred, HistoryLagsAndSlope) {
  History h(AccelSpec::parse("slope25"));
  for (int k = 0; k < 60; ++k) {
    Sample s;
    s.t = 0.02 * k;
    s.v = 3.0 + 1.5 * s.t;  // exactly 1.5 m/s^2
    h.push(s);
    if (k < 24) {
      EXPECT_EQ(h.at(0).accel, 0.0);
    }
  }
  EXPECT_TRUE(h.full());
  EXPECT_NEAR(h.at(0).accel, 1.5, 1e-9);
  EXPECT_NEAR(h.at(25).t, 0.02 * 34, 1e-12);
  EXPECT_NEAR(h.at(0).t, 0.02 * 59, 1e-12);
  EXPECT_THROW(AccelSpec::parse("slope99"), std::runtime_error);
  EXPECT_THROW(AccelSpec::parse("gps"), std::runtime_error);
}

TEST(Mpred, EkfTracksACircleAndPredictsIt) {
  EkfParams p;
  p.r_xy = 0.001;
  p.r_psi = 1e-4;
  Ekf ekf(p);
  const double v = 8.0, r = 0.15;
  for (int k = 0; k < 300; ++k) ekf.step(circle_sample(k, v, r));
  EXPECT_NEAR(ekf.state()[3], v, 1e-3);
  EXPECT_NEAR(ekf.state()[4], 0.0, 1e-2);
  EXPECT_NEAR(ekf.state()[5], r, 1e-4);
  const Prediction pr = ekf.forecast();
  const double t0 = 299 * 0.02;
  for (size_t k = 0; k < kHorizons; ++k) {
    const double t = t0 + kHorizonS[k];
    EXPECT_NEAR(pr[k].x, v / r * std::sin(r * t), 5e-3);
    EXPECT_NEAR(pr[k].y, v / r * (1 - std::cos(r * t)), 5e-3);
  }
  // covariance stays symmetric and positive on the diagonal
  for (int i = 0; i < Ekf::N; ++i) {
    EXPECT_GT(ekf.cov()[static_cast<size_t>(i)][static_cast<size_t>(i)], 0.0);
    for (int j = 0; j < Ekf::N; ++j)
      EXPECT_NEAR(ekf.cov()[static_cast<size_t>(i)][static_cast<size_t>(j)], ekf.cov()[static_cast<size_t>(j)][static_cast<size_t>(i)], 1e-12);
  }
}

TEST(Mpred, EkfScalarUpdatesDoNotDependOnOrder) {
  // independent measurements with a diagonal R: sequential scalar updates equal the joint update in any order
  EkfParams p;
  Ekf a(p), b(p);
  const Sample s0 = circle_sample(0, 5.0, 0.1);
  a.init(s0);
  b.init(s0);
  a.predict(0.02);
  b.predict(0.02);
  a.update(0, 0.11, 4e-4);
  a.update(3, 5.2, 2.5e-3);
  a.update(5, 0.12, 2.5e-5);
  b.update(5, 0.12, 2.5e-5);
  b.update(3, 5.2, 2.5e-3);
  b.update(0, 0.11, 4e-4);
  for (size_t i = 0; i < Ekf::N; ++i) {
    EXPECT_NEAR(a.state()[i], b.state()[i], 1e-12);
    for (size_t j = 0; j < Ekf::N; ++j) EXPECT_NEAR(a.cov()[i][j], b.cov()[i][j], 1e-12);
  }
}

TEST(Mpred, EkfSkipsMissingMeasurements) {
  EkfParams p;
  Ekf a(p), b(p);
  for (int k = 0; k < 10; ++k) {
    a.step(circle_sample(k, 6.0, 0.0));
    b.step(circle_sample(k, 6.0, 0.0), true, k < 5);
  }
  Sample odd = circle_sample(10, 6.0, 0.0);
  odd.v = 100.0;  // a wrong wheel speed that must not be used when odometry is flagged missing
  b.step(odd, true, false);
  EXPECT_LT(std::fabs(b.state()[3] - 6.0), 0.5);
}

TEST(Mpred, PhysicalFeaturesOfASteadyStraightDrive) {
  History h(AccelSpec::parse("slope25"));
  for (int k = 0; k < 30; ++k) h.push(circle_sample(k, 10.0, 0.0));
  double f[kPhysFeatures];
  ASSERT_EQ(phys_features(h, nullptr, f), kPhysFeatures);
  EXPECT_DOUBLE_EQ(f[0], 1.0);
  EXPECT_DOUBLE_EQ(f[1], 10.0);
  EXPECT_NEAR(f[2], 0.0, 1e-9);
  for (int i = 3; i < kPhysFeatures; ++i) EXPECT_NEAR(f[i], 0.0, 1e-9) << i;
  double g[kRawFeatures + 3];
  Ekf::Vec st{0, 0, 0, 9.0, 0.5, 0.01};
  ASSERT_EQ(raw_features(h, &st, g), kRawFeatures + 3);
  EXPECT_NEAR(g[24], -10.0 * 5 * 0.02, 1e-9);  // pose 5 samples back: 1 m behind, on the axis
  EXPECT_NEAR(g[25], 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(g[33], 9.0);
}

std::string write_config(const std::string& learned) {
  const char* dir = std::getenv("TEST_TMPDIR");
  const std::string path = std::string(dir ? dir : "/tmp") + "/mpred_test_predictors.json";
  std::ofstream(path) << R"({"ctra": {"accel": "slope25"},
    "ekf": {"s_jerk": 2, "s_rdot": 0.1, "r_scale": 1, "accel": "ax_can", "q_xy": 0.001, "q_psi": 1e-05,
            "q_v": 0.01, "r_xy": 0.001, "r_psi": 0.0001, "r_v": 0.05, "r_r": 0.005, "r_a": 0.3},
    "learned": {)" << learned << "}}";
  return path;
}

std::string zeros(int n, double v = 0.0) {
  std::string s = "[";
  for (int i = 0; i < n; ++i) {
    char b[32];
    std::snprintf(b, sizeof b, "%s%g", i ? "," : "", v);
    s += b;
  }
  return s + "]";
}

TEST(Mpred, LearnedModelWithZeroWeightsReturnsItsBase) {
  // all-zero ridge weights: the direct model predicts no motion, the residual model the EKF forecast
  const int nf = kPhysFeatures, no = kHorizons * 4;
  const std::string m = std::string(R"({"features": "phys", "uses_state": true, "residual": RES, "x_mean": )") +
                        zeros(nf) + R"(, "x_std": )" + zeros(nf, 1) + R"(, "y_mean": )" + zeros(no) +
                        R"(, "y_std": )" + zeros(no, 1) + R"(, "layers": [{"in": 18, "out": 28, "w": )" +
                        zeros(nf * no) + R"(, "b": )" + zeros(no) + "}]}";
  std::string direct = m, residual = m;
  direct.replace(direct.find("RES"), 3, "false");
  residual.replace(residual.find("RES"), 3, "true");
  const PredictorSet ps = load_predictors(write_config("\"d\": " + direct + ", \"r\": " + residual));
  ASSERT_EQ(ps.learned.size(), 2u);
  EXPECT_EQ(ps.learned.at("d").n_params(), nf * no + no);
  History h(ps.ctra_accel);
  Ekf ekf(ps.ekf);
  for (int k = 0; k < 40; ++k) {
    h.push(circle_sample(k, 7.0, 0.1));
    ekf.step(circle_sample(k, 7.0, 0.1));
  }
  const Prediction d = predict_learned(ps.learned.at("d"), h, &ekf);
  const Prediction r = predict_learned(ps.learned.at("r"), h, &ekf);
  const Prediction base = ekf.forecast();
  for (size_t k = 0; k < kHorizons; ++k) {
    EXPECT_DOUBLE_EQ(d[k].x, h.at(0).x);
    EXPECT_DOUBLE_EQ(d[k].yaw, h.at(0).yaw);
    EXPECT_DOUBLE_EQ(r[k].x, base[k].x);
    EXPECT_DOUBLE_EQ(r[k].v, base[k].v);
  }
}

TEST(Mpred, MlpForwardPass) {
  // 1 input -> 1 tanh unit -> 28 outputs, every output = 2 * tanh(0.5 * x) + 1 after scaling
  LearnedModel m;
  m.x_mean = {0.0};
  m.x_std = {1.0};
  m.y_mean.assign(kHorizons * 4, 1.0);
  m.y_std.assign(kHorizons * 4, 2.0);
  m.layers.push_back(Layer{1, 1, {0.5}, {0.0}});
  m.layers.push_back(Layer{1, kHorizons * 4, std::vector<double>(kHorizons * 4, 1.0), std::vector<double>(kHorizons * 4, 0.0)});
  const double x = 0.8;
  double y[kHorizons * 4];
  m.eval(&x, y);
  for (double v : y) EXPECT_NEAR(v, 2.0 * std::tanh(0.4) + 1.0, 1e-15);
  EXPECT_EQ(m.n_params(), 2 + 2 * kHorizons * 4);
}

TEST(Mpred, BadConfigIsRejected) {
  EXPECT_THROW(load_predictors("/nonexistent/predictors.json"), std::runtime_error);
  const std::string bad = R"("x": {"features": "phys", "uses_state": false, "residual": false, "x_mean": [0],
      "x_std": [1], "y_mean": [0], "y_std": [1], "layers": [{"in": 1, "out": 1, "w": [1], "b": [0]}]})";
  EXPECT_THROW(load_predictors(write_config(bad)), std::runtime_error);
}

}  // namespace
}  // namespace dr::mpred
