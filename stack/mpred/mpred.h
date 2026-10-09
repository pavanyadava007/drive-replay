// Ego motion prediction for latency compensation of an augmented-reality head-up display (AR-HUD).
//
// A contact-analog overlay is rendered for the vehicle pose at display time, which lies 50-500 ms after the
// newest sensor sample. This library predicts the ego pose (x, y, yaw) and speed at fixed horizons from the
// samples received so far. It is the C++17 port of the Python prototypes in tools/mpred (same arithmetic; the
// parity on the test split is in docs/MOTION_PREDICTION.md).
//
// Frame and signs as stack/vdyn: x forward, y left, yaw counter-clockwise, reference point = middle of the rear
// axle. Everything is double precision, allocation-free after construction, single threaded.
//
//   rollout       CTRA kernel: (x, y, yaw, v) with acceleration a and yaw rate r held, classical RK4 with fixed
//                 10 ms steps, speed clamped at zero. CV is a = r = 0, CTRV is a = 0.
//   History       ring buffer of the newest kMaxLag + 1 samples; computes the configured acceleration signal
//   Ekf           extended Kalman filter on the CTRA state (x, y, yaw, v, a, r): RK4 mean, Jacobian of the Euler
//                 step, sequential scalar updates for pose x, y, yaw, wheel speed, IMU yaw rate, acceleration
//   LearnedModel  ridge regression or a small tanh MLP on history features, either direct (relative to the
//                 newest pose) or as a residual on top of the EKF-CTRA forecast (hybrid)
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

namespace dr::mpred {

inline constexpr int kHorizons = 7;
inline constexpr std::array<double, kHorizons> kHorizonS{0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00};
inline constexpr std::array<int, kHorizons> kHorizonSteps{5, 10, 15, 20, 30, 50, 100};
inline constexpr double kDtInt = 0.01;  // s, integration step of every rollout
inline constexpr int kMaxLag = 25;      // samples, the oldest history a feature reads

struct Pose {
  double x = 0, y = 0, yaw = 0, v = 0;
};
using Prediction = std::array<Pose, kHorizons>;

double wrap_angle(double a);

// One classical RK4 step of the CTRA kernel.
void rk4_step(double& x, double& y, double& yaw, double& v, double a, double r, double dt);
// The kernel over all horizons (100 steps of 10 ms, sampled at kHorizonSteps).
Prediction rollout(double x, double y, double yaw, double v, double a, double r);

// One input sample on the pose clock, in SI units: wheel speed in m/s, road-wheel angles in rad.
struct Sample {
  double t = 0;
  double x = 0, y = 0, yaw = 0;  // localization pose
  double v = 0;                  // wheel speed
  double r = 0;                  // IMU yaw rate
  double d_kin = 0, d_dyn = 0;   // road-wheel angle from the steering wheel (kinematic / dynamic map)
  double ax_imu = 0, ax_can = 0;
  double accel = 0;              // the configured acceleration signal, set by History::push
};

// "ax_imu", "ax_can" (recorded signals) or "slopeW": least-squares slope of the wheel speed over the newest W
// samples (0 until W samples have been seen).
struct AccelSpec {
  enum class Source { kImu, kCan, kSlope } source = Source::kSlope;
  int window = 25;
  static AccelSpec parse(const std::string& text);
};

class History {
 public:
  explicit History(AccelSpec a = {}) : accel_(a) {}
  void push(Sample s);
  // 0 = newest; lag must be below count() and at most kMaxLag
  const Sample& at(int lag) const { return buf_[static_cast<size_t>(((n_ - 1 - lag) % kCap + kCap) % kCap)]; }
  long count() const { return n_; }
  bool full() const { return n_ >= kCap; }

 private:
  static constexpr long kCap = kMaxLag + 1;
  AccelSpec accel_;
  std::array<Sample, kCap> buf_{};
  long n_ = 0;
};

// Predictions from the newest sample of a history (no filter). wheelbase for the kinematic single track.
Prediction predict_cv(const History& h);
Prediction predict_ctrv(const History& h);
Prediction predict_ctra(const History& h);
Prediction predict_kinematic(const History& h, double wheelbase_m);

struct EkfParams {
  double s_jerk = 2.0, s_rdot = 0.1, r_scale = 1.0;  // tuned on the validation split
  std::string accel = "ax_can";                     // acceleration measurement: ax_can or ax_imu
  double q_xy = 1e-3, q_psi = 1e-5, q_v = 1e-2;      // fixed process noise (per second)
  double r_xy = 0.02, r_psi = 0.002, r_v = 0.05, r_r = 0.005, r_a = 0.3;  // measurement stds
};

class Ekf {
 public:
  static constexpr int N = 6;  // x, y, yaw, v, a, r
  using Vec = std::array<double, N>;
  using Mat = std::array<Vec, N>;

  explicit Ekf(const EkfParams& p);
  void init(const Sample& s);
  // Predict to s.t, then apply the measurements of s (x, y, yaw, v, r, a). pose / odometry false skip them.
  void step(const Sample& s, bool pose = true, bool odometry = true);
  void predict(double dt);
  void update(int index, double z, double var, bool angle = false);
  Prediction forecast() const { return rollout(x_[0], x_[1], x_[2], x_[3], x_[4], x_[5]); }
  const Vec& state() const { return x_; }
  const Mat& cov() const { return P_; }
  bool initialized() const { return init_; }

 private:
  EkfParams p_;
  std::array<double, 6> var_{};  // measurement variances in update order x, y, yaw, v, r, a
  Vec x_{};
  Mat P_{};
  double t_ = 0;
  bool init_ = false;
};

struct Layer {
  int in = 0, out = 0;
  std::vector<double> w;  // row-major (in x out)
  std::vector<double> b;
};

// Ridge regression (one linear layer, no output scaling) or a tanh MLP; outputs kHorizons x (dx, dy, dyaw, dv).
struct LearnedModel {
  std::string name;
  bool raw_features = false;  // false: the 18 physical features, true: the raw history window
  bool uses_state = false;    // features read the EKF v, a, r
  bool residual = false;      // true: correction of the EKF-CTRA forecast; false: relative to the newest pose
  std::vector<double> x_mean, x_std, y_mean, y_std;
  std::vector<Layer> layers;
  void eval(const double* features, double* out) const;  // out: kHorizons * 4
  int n_params() const;
  int n_inputs() const { return layers.empty() ? 0 : layers.front().in; }
};

inline constexpr int kPhysFeatures = 18;
inline constexpr int kRawFeatures = 33;  // + 3 with the filter state
int phys_features(const History& h, const Ekf::Vec* state, double* out);
int raw_features(const History& h, const Ekf::Vec* state, double* out);

// Needs a full history; ekf is required for models with uses_state or residual.
Prediction predict_learned(const LearnedModel& m, const History& h, const Ekf* ekf);

// configs/mpred/predictors.json (written by scripts/reproduce_mpred.py).
struct PredictorSet {
  AccelSpec ctra_accel;
  EkfParams ekf;
  std::map<std::string, LearnedModel> learned;
};
PredictorSet load_predictors(const std::string& path);

}  // namespace dr::mpred
