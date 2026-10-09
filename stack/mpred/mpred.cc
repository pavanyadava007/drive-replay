#include "stack/mpred/mpred.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace dr::mpred {
namespace {

constexpr double kTwoPi = 6.28318530717958647692;

// CTRA derivative of (x, y, yaw, v); the speed never goes below zero.
inline void deriv(double yaw, double v, double a, double r, double d[4]) {
  const double vp = v > 0.0 ? v : 0.0;
  d[0] = vp * std::cos(yaw);
  d[1] = vp * std::sin(yaw);
  d[2] = r;
  d[3] = (v > 0.0 || a > 0.0) ? a : 0.0;
}

double num(const nlohmann::json& j, const char* key) {
  if (!j.contains(key)) throw std::runtime_error(std::string("predictor config: missing key ") + key);
  return j.at(key).get<double>();
}

std::vector<double> vec(const nlohmann::json& j, const char* key) {
  if (!j.contains(key)) throw std::runtime_error(std::string("predictor config: missing key ") + key);
  return j.at(key).get<std::vector<double>>();
}

}  // namespace

double wrap_angle(double a) { return a - kTwoPi * std::nearbyint(a / kTwoPi); }

void rk4_step(double& x, double& y, double& yaw, double& v, double a, double r, double dt) {
  const double h2 = 0.5 * dt;
  double k1[4], k2[4], k3[4], k4[4];
  deriv(yaw, v, a, r, k1);
  deriv(yaw + h2 * k1[2], v + h2 * k1[3], a, r, k2);
  deriv(yaw + h2 * k2[2], v + h2 * k2[3], a, r, k3);
  deriv(yaw + dt * k3[2], v + dt * k3[3], a, r, k4);
  const double s = dt / 6.0;
  x = x + s * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
  y = y + s * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
  yaw = yaw + s * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
  v = v + s * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]);
  if (v < 0.0) v = 0.0;  // a step that brakes through zero would otherwise leave a small negative speed
}

Prediction rollout(double x, double y, double yaw, double v, double a, double r) {
  Prediction out{};
  int next = 0;
  for (int i = 1; i <= kHorizonSteps.back(); ++i) {
    rk4_step(x, y, yaw, v, a, r, kDtInt);
    if (i == kHorizonSteps[static_cast<size_t>(next)]) out[static_cast<size_t>(next++)] = Pose{x, y, yaw, v};
  }
  return out;
}

// ---- history --------------------------------------------------------------------------------------------------

AccelSpec AccelSpec::parse(const std::string& text) {
  AccelSpec a;
  if (text == "ax_imu") a.source = Source::kImu;
  else if (text == "ax_can") a.source = Source::kCan;
  else if (text.rfind("slope", 0) == 0) {
    a.source = Source::kSlope;
    a.window = std::stoi(text.substr(5));
    if (a.window < 2 || a.window > kMaxLag + 1) throw std::runtime_error("accel slope window out of range: " + text);
  } else throw std::runtime_error("unknown acceleration source " + text);
  return a;
}

void History::push(Sample s) {
  buf_[static_cast<size_t>(n_ % kCap)] = s;
  ++n_;
  Sample& cur = buf_[static_cast<size_t>((n_ - 1) % kCap)];
  switch (accel_.source) {
    case AccelSpec::Source::kImu: cur.accel = cur.ax_imu; break;
    case AccelSpec::Source::kCan: cur.accel = cur.ax_can; break;
    case AccelSpec::Source::kSlope: {
      const int w = accel_.window;
      if (n_ < w) {
        cur.accel = 0.0;
        break;
      }
      // least-squares slope of v over t, sums from the newest sample to the oldest (as the Python prototype)
      double st = 0, sv = 0;
      for (int j = 0; j < w; ++j) st += at(j).t;
      for (int j = 0; j < w; ++j) sv += at(j).v;
      const double tm = st / w, vm = sv / w;
      double n = 0, d = 0;
      for (int j = 0; j < w; ++j) n += (at(j).t - tm) * (at(j).v - vm);
      for (int j = 0; j < w; ++j) d += (at(j).t - tm) * (at(j).t - tm);
      cur.accel = n / (d > 0 ? d : 1.0);
      break;
    }
  }
}

Prediction predict_cv(const History& h) {
  const Sample& s = h.at(0);
  return rollout(s.x, s.y, s.yaw, s.v, 0.0, 0.0);
}

Prediction predict_ctrv(const History& h) {
  const Sample& s = h.at(0);
  return rollout(s.x, s.y, s.yaw, s.v, 0.0, s.r);
}

Prediction predict_ctra(const History& h) {
  const Sample& s = h.at(0);
  return rollout(s.x, s.y, s.yaw, s.v, s.accel, s.r);
}

Prediction predict_kinematic(const History& h, double wheelbase_m) {
  const Sample& s = h.at(0);
  return rollout(s.x, s.y, s.yaw, s.v, 0.0, s.v * std::tan(s.d_kin) / wheelbase_m);
}

// ---- EKF ------------------------------------------------------------------------------------------------------

Ekf::Ekf(const EkfParams& p) : p_(p) {
  const double k = p.r_scale;
  const std::array<double, 6> std_{p.r_xy, p.r_xy, p.r_psi, p.r_v * k, p.r_r * k, p.r_a * k};
  for (int i = 0; i < 6; ++i) var_[static_cast<size_t>(i)] = std_[static_cast<size_t>(i)] * std_[static_cast<size_t>(i)];
  if (p.accel != "ax_can" && p.accel != "ax_imu") throw std::runtime_error("EKF acceleration must be ax_can or ax_imu");
}

void Ekf::init(const Sample& s) {
  x_ = Vec{s.x, s.y, s.yaw, s.v, p_.accel == "ax_can" ? s.ax_can : s.ax_imu, s.r};
  P_ = Mat{};
  // state order x, y, yaw, v, a, r; variances in update order x, y, yaw, v, r, a
  const std::array<double, 6> d{var_[0], var_[1], var_[2], var_[3], var_[5], var_[4]};
  for (int i = 0; i < N; ++i) P_[static_cast<size_t>(i)][static_cast<size_t>(i)] = d[static_cast<size_t>(i)];
  t_ = s.t;
  init_ = true;
}

void Ekf::predict(double dt) {
  const double psi = x_[2], v = x_[3];
  const double vp = v > 0.0 ? v : 0.0;
  Mat F{};
  for (int i = 0; i < N; ++i) F[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;
  F[0][2] = -dt * vp * std::sin(psi);
  F[0][3] = dt * std::cos(psi);
  F[1][2] = dt * vp * std::cos(psi);
  F[1][3] = dt * std::sin(psi);
  F[2][5] = dt;
  F[3][4] = dt;
  rk4_step(x_[0], x_[1], x_[2], x_[3], x_[4], x_[5], dt);
  Mat FP{}, out{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < N; ++k) acc += F[static_cast<size_t>(i)][static_cast<size_t>(k)] * P_[static_cast<size_t>(k)][static_cast<size_t>(j)];
      FP[static_cast<size_t>(i)][static_cast<size_t>(j)] = acc;
    }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < N; ++k) acc += FP[static_cast<size_t>(i)][static_cast<size_t>(k)] * F[static_cast<size_t>(j)][static_cast<size_t>(k)];
      out[static_cast<size_t>(i)][static_cast<size_t>(j)] = acc;
    }
  const std::array<double, 6> q{p_.q_xy, p_.q_xy, p_.q_psi, p_.q_v, p_.s_jerk * p_.s_jerk, p_.s_rdot * p_.s_rdot};
  for (int i = 0; i < N; ++i) out[static_cast<size_t>(i)][static_cast<size_t>(i)] += q[static_cast<size_t>(i)] * dt;
  P_ = out;
}

void Ekf::update(int index, double z, double var, bool angle) {
  const auto i = static_cast<size_t>(index);
  double innov = z - x_[i];
  if (angle) innov = wrap_angle(innov);
  const double s = P_[i][i] + var;
  Vec K{}, row = P_[i];
  for (size_t a = 0; a < N; ++a) K[a] = P_[a][i] / s;
  for (size_t a = 0; a < N; ++a) x_[a] = x_[a] + K[a] * innov;
  for (size_t a = 0; a < N; ++a)
    for (size_t b = 0; b < N; ++b) P_[a][b] = P_[a][b] - K[a] * row[b];
}

void Ekf::step(const Sample& s, bool pose, bool odometry) {
  if (!init_) {
    init(s);
    return;
  }
  predict(s.t - t_);
  t_ = s.t;
  if (pose) {
    update(0, s.x, var_[0]);
    update(1, s.y, var_[1]);
    update(2, s.yaw, var_[2], true);
  }
  if (odometry) {
    update(3, s.v, var_[3]);
    update(5, s.r, var_[4]);
    update(4, p_.accel == "ax_can" ? s.ax_can : s.ax_imu, var_[5]);
  }
}

// ---- learned models -------------------------------------------------------------------------------------------

void LearnedModel::eval(const double* features, double* out) const {
  // two scratch buffers sized by the widest layer; the models here are at most a few hundred wide
  double bufa[512], bufb[512];
  const int nin = n_inputs();
  for (int i = 0; i < nin; ++i) bufa[i] = (features[i] - x_mean[static_cast<size_t>(i)]) / x_std[static_cast<size_t>(i)];
  double* cur = bufa;
  double* nxt = bufb;
  for (size_t l = 0; l < layers.size(); ++l) {
    const Layer& L = layers[l];
    // row-major weights: walk them in memory order (input outer, output inner) so the inner loop vectorises
    for (int j = 0; j < L.out; ++j) nxt[j] = 0.0;
    for (int i = 0; i < L.in; ++i) {
      const double xi = cur[i];
      const double* wr = L.w.data() + static_cast<size_t>(i) * static_cast<size_t>(L.out);
      for (int j = 0; j < L.out; ++j) nxt[j] += xi * wr[j];
    }
    for (int j = 0; j < L.out; ++j) {
      const double acc = nxt[j] + L.b[static_cast<size_t>(j)];
      nxt[j] = (l + 1 < layers.size()) ? std::tanh(acc) : acc;
    }
    std::swap(cur, nxt);
  }
  const int nout = layers.back().out;
  for (int j = 0; j < nout; ++j) out[j] = cur[j] * y_std[static_cast<size_t>(j)] + y_mean[static_cast<size_t>(j)];
}

int LearnedModel::n_params() const {
  int n = 0;
  for (const Layer& L : layers) n += static_cast<int>(L.w.size() + L.b.size());
  return n;
}

int phys_features(const History& h, const Ekf::Vec* state, double* f) {
  const Sample &s0 = h.at(0), &s5 = h.at(5), &s25 = h.at(25);
  const double v = state ? (*state)[3] : s0.v;
  const double a = state ? (*state)[4] : s0.accel;
  const double r = state ? (*state)[5] : s0.r;
  const double d = s0.d_kin;
  const double r5 = s0.r - s5.r, r25 = s0.r - s25.r, d5 = d - s5.d_kin, d25 = d - s25.d_kin;
  const double vals[kPhysFeatures] = {1.0,    v,      a,      r,     d,      v * r,     v * d,     v * a,     v * v * d,
                                      r5,     r25,    v * r5, v * r25, d5,   d25,       v * d5,    v * d25,   s0.v - s25.v};
  for (int i = 0; i < kPhysFeatures; ++i) f[i] = vals[i];
  return kPhysFeatures;
}

int raw_features(const History& h, const Ekf::Vec* state, double* f) {
  static constexpr int kLags[6] = {0, 5, 10, 15, 20, 25};
  static constexpr int kPoseLags[3] = {5, 10, 25};
  int n = 0;
  for (int lag : kLags) f[n++] = h.at(lag).v;
  for (int lag : kLags) f[n++] = h.at(lag).r;
  for (int lag : kLags) f[n++] = h.at(lag).d_kin;
  for (int lag : kLags) f[n++] = h.at(lag).accel;
  const Sample& s0 = h.at(0);
  const double c = std::cos(s0.yaw), s = std::sin(s0.yaw);
  for (int lag : kPoseLags) {
    const Sample& p = h.at(lag);
    const double dx = p.x - s0.x, dy = p.y - s0.y;
    f[n++] = c * dx + s * dy;
    f[n++] = -s * dx + c * dy;
    f[n++] = wrap_angle(p.yaw - s0.yaw);
  }
  if (state) {
    f[n++] = (*state)[3];
    f[n++] = (*state)[4];
    f[n++] = (*state)[5];
  }
  return n;
}

Prediction predict_learned(const LearnedModel& m, const History& h, const Ekf* ekf) {
  if (!h.full()) throw std::runtime_error("predict_learned: history not full");
  if ((m.uses_state || m.residual) && (ekf == nullptr || !ekf->initialized()))
    throw std::runtime_error("predict_learned: model " + m.name + " needs the EKF");
  double f[64];
  const Ekf::Vec* st = m.uses_state ? &ekf->state() : nullptr;
  const int nf = m.raw_features ? raw_features(h, st, f) : phys_features(h, st, f);
  if (nf != m.n_inputs()) throw std::runtime_error("predict_learned: feature count mismatch for " + m.name);
  double y[kHorizons * 4];
  m.eval(f, y);
  Prediction out{};
  if (m.residual) {
    const Prediction base = ekf->forecast();
    const double c = std::cos(ekf->state()[2]), s = std::sin(ekf->state()[2]);
    for (size_t k = 0; k < kHorizons; ++k) {
      const double* r = y + 4 * k;
      out[k] = Pose{base[k].x + c * r[0] - s * r[1], base[k].y + s * r[0] + c * r[1], base[k].yaw + r[2], base[k].v + r[3]};
    }
  } else {
    const Sample& a = h.at(0);
    const double c = std::cos(a.yaw), s = std::sin(a.yaw);
    for (size_t k = 0; k < kHorizons; ++k) {
      const double* r = y + 4 * k;
      out[k] = Pose{a.x + c * r[0] - s * r[1], a.y + s * r[0] + c * r[1], a.yaw + r[2], a.v + r[3]};
    }
  }
  return out;
}

PredictorSet load_predictors(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open predictor config " + path);
  const nlohmann::json j = nlohmann::json::parse(in);
  PredictorSet ps;
  ps.ctra_accel = AccelSpec::parse(j.at("ctra").at("accel").get<std::string>());
  const auto& e = j.at("ekf");
  EkfParams& p = ps.ekf;
  p.s_jerk = num(e, "s_jerk");
  p.s_rdot = num(e, "s_rdot");
  p.r_scale = num(e, "r_scale");
  p.accel = e.at("accel").get<std::string>();
  p.q_xy = num(e, "q_xy");
  p.q_psi = num(e, "q_psi");
  p.q_v = num(e, "q_v");
  p.r_xy = num(e, "r_xy");
  p.r_psi = num(e, "r_psi");
  p.r_v = num(e, "r_v");
  p.r_r = num(e, "r_r");
  p.r_a = num(e, "r_a");
  for (auto it = j.at("learned").begin(); it != j.at("learned").end(); ++it) {
    const auto& m = it.value();
    LearnedModel lm;
    lm.name = it.key();
    lm.raw_features = m.at("features").get<std::string>() == "raw";
    lm.uses_state = m.at("uses_state").get<bool>();
    lm.residual = m.at("residual").get<bool>();
    lm.x_mean = vec(m, "x_mean");
    lm.x_std = vec(m, "x_std");
    lm.y_mean = vec(m, "y_mean");
    lm.y_std = vec(m, "y_std");
    for (const auto& L : m.at("layers")) {
      Layer layer;
      layer.in = L.at("in").get<int>();
      layer.out = L.at("out").get<int>();
      layer.w = L.at("w").get<std::vector<double>>();
      layer.b = L.at("b").get<std::vector<double>>();
      if (static_cast<int>(layer.w.size()) != layer.in * layer.out || static_cast<int>(layer.b.size()) != layer.out ||
          layer.in > 512 || layer.out > 512)
        throw std::runtime_error("predictor config: bad layer shape in " + lm.name);
      lm.layers.push_back(std::move(layer));
    }
    if (lm.layers.empty() || lm.layers.back().out != kHorizons * 4 ||
        static_cast<int>(lm.x_mean.size()) != lm.n_inputs() || static_cast<int>(lm.x_std.size()) != lm.n_inputs() ||
        static_cast<int>(lm.y_mean.size()) != kHorizons * 4 || static_cast<int>(lm.y_std.size()) != kHorizons * 4 ||
        lm.n_inputs() > 64)
      throw std::runtime_error("predictor config: inconsistent model " + lm.name);
    ps.learned[lm.name] = std::move(lm);
  }
  return ps;
}

}  // namespace dr::mpred
