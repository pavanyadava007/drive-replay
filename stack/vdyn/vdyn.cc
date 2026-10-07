#include "stack/vdyn/vdyn.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace dr::vdyn {
namespace {

Input lerp(const Input& a, const Input& b, double w) {
  return Input{a.delta + w * (b.delta - a.delta), a.v + w * (b.v - a.v)};
}

State add(const State& s, const State& d, double h) {
  return State{s.x + h * d.x, s.y + h * d.y, s.psi + h * d.psi, s.vy + h * d.vy, s.r + h * d.r};
}

// Classical RK4 for a derivative function f(state, input) -> derivative, input linear over the step.
template <typename F>
void rk4(State& s, const Input& u0, const Input& u1, double dt, F f) {
  const Input um = lerp(u0, u1, 0.5);
  State k1, k2, k3, k4;
  f(s, u0, k1);
  f(add(s, k1, 0.5 * dt), um, k2);
  f(add(s, k2, 0.5 * dt), um, k3);
  f(add(s, k3, dt), u1, k4);
  s.x += dt / 6.0 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x);
  s.y += dt / 6.0 * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
  s.psi += dt / 6.0 * (k1.psi + 2 * k2.psi + 2 * k3.psi + k4.psi);
  s.vy += dt / 6.0 * (k1.vy + 2 * k2.vy + 2 * k3.vy + k4.vy);
  s.r += dt / 6.0 * (k1.r + 2 * k2.r + 2 * k3.r + k4.r);
}

double num(const nlohmann::json& j, const char* key) {
  if (!j.contains(key)) throw std::runtime_error(std::string("vehicle config: missing key ") + key);
  return j.at(key).get<double>();
}

}  // namespace

double VehicleParams::understeer_K() const {
  const double L = wheelbase_m;
  return mass_kg * (lr() / cf_npr - lf_m / cr_npr) / (L * L);
}

VehicleConfig load_vehicle_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open vehicle config " + path);
  const nlohmann::json j = nlohmann::json::parse(in);
  VehicleConfig c;
  c.wheel_radius_m = num(j, "wheel_radius_m");
  const auto& k = j.at("kinematic");
  c.kinematic_steering = SteeringMap{num(k, "steering_ratio"), num(k, "steer_offset_rad")};
  const auto& d = j.at("dynamic");
  c.dynamic_steering = SteeringMap{num(d, "steering_ratio"), num(d, "steer_offset_rad")};
  c.dynamic.wheelbase_m = num(j, "wheelbase_m");
  c.dynamic.lf_m = num(d, "lf_m");
  c.dynamic.mass_kg = num(d, "mass_kg");
  c.dynamic.yaw_inertia_kgm2 = num(d, "yaw_inertia_kgm2");
  c.dynamic.cf_npr = num(d, "cf_npr");
  c.dynamic.cr_npr = num(d, "cr_npr");
  c.dynamic.min_speed_mps = num(d, "min_speed_mps");
  if (j.contains("fcw_baseline")) {
    c.fcw_decay_tau_s = num(j["fcw_baseline"], "yaw_rate_decay_s");
    c.fcw_max_curvature = num(j["fcw_baseline"], "max_curvature");
  }
  if (j.contains("steer_offset_by_vehicle"))
    for (auto it = j["steer_offset_by_vehicle"].begin(); it != j["steer_offset_by_vehicle"].end(); ++it)
      c.offsets_by_vehicle[it.key()] = VehicleConfig::Offsets{num(it.value(), "kinematic"), num(it.value(), "dynamic")};
  if (c.dynamic.lf_m <= 0 || c.dynamic.lr() <= 0 || c.dynamic.cf_npr <= 0 || c.dynamic.cr_npr <= 0 ||
      c.kinematic_steering.ratio <= 0 || c.dynamic_steering.ratio <= 0)
    throw std::runtime_error("vehicle config " + path + ": non-physical parameters");
  return c;
}

VehicleConfig VehicleConfig::for_vehicle(const std::string& id) const {
  VehicleConfig c = *this;
  const auto it = offsets_by_vehicle.find(id);
  if (it != offsets_by_vehicle.end()) {
    c.kinematic_steering.offset_rad = it->second.kinematic;
    c.dynamic_steering.offset_rad = it->second.dynamic;
  }
  return c;
}

double kinematic_yaw_rate(double wheelbase_m, double delta, double v) { return v * std::tan(delta) / wheelbase_m; }

double linear_steady_yaw_rate(double wheelbase_m, double K, double delta, double v) {
  return v * delta / (wheelbase_m * (1.0 + K * v * v));
}

void Model::init(State& s, const Input&, double measured_yaw_rate) const {
  s.r = measured_yaw_rate;
  s.vy = 0.0;
}

// --- kinematic ------------------------------------------------------------------------------------------
void KinematicBicycle::init(State& s, const Input& u, double) const {
  s.r = kinematic_yaw_rate(L_, u.delta, u.v);
  s.vy = 0.0;
}

void KinematicBicycle::step(State& s, const Input& u0, const Input& u1, double dt) const {
  const double L = L_;
  rk4(s, u0, u1, dt, [L](const State& st, const Input& u, State& d) {
    d.x = u.v * std::cos(st.psi);
    d.y = u.v * std::sin(st.psi);
    d.psi = kinematic_yaw_rate(L, u.delta, u.v);
    d.vy = 0.0;
    d.r = 0.0;
  });
  s.r = kinematic_yaw_rate(L, u1.delta, u1.v);
}

// --- linear dynamic single track --------------------------------------------------------------------------
void LinearSingleTrack::deriv(const State& s, const Input& u, State& d) const {
  const double m = p_.mass_kg, Iz = p_.yaw_inertia_kgm2, cf = p_.cf_npr, cr = p_.cr_npr;
  const double lf = p_.lf_m, lr = p_.lr(), v = u.v;
  d.vy = -(cf + cr) / (m * v) * s.vy + ((lr * cr - lf * cf) / (m * v) - v) * s.r + cf / m * u.delta;
  d.r = (lr * cr - lf * cf) / (Iz * v) * s.vy - (lf * lf * cf + lr * lr * cr) / (Iz * v) * s.r + lf * cf / Iz * u.delta;
  d.psi = s.r;
  const double vy_rear = s.vy - lr * s.r;  // lateral velocity of the rear-axle reference point
  d.x = v * std::cos(s.psi) - vy_rear * std::sin(s.psi);
  d.y = v * std::sin(s.psi) + vy_rear * std::cos(s.psi);
}

void LinearSingleTrack::init(State& s, const Input& u, double measured_yaw_rate) const {
  s.r = measured_yaw_rate;
  if (u.v < p_.min_speed_mps) {
    s.vy = p_.lr() * s.r;
    return;
  }
  // lateral velocity in equilibrium with the measured yaw rate and the current steering (dvy/dt = 0)
  const double m = p_.mass_kg, cf = p_.cf_npr, cr = p_.cr_npr, lf = p_.lf_m, lr = p_.lr(), v = u.v;
  s.vy = (((lr * cr - lf * cf) / (m * v) - v) * s.r + cf / m * u.delta) * m * v / (cf + cr);
}

void LinearSingleTrack::step(State& s, const Input& u0, const Input& u1, double dt) const {
  const double v_lo = std::min(u0.v, u1.v);
  if (v_lo < p_.min_speed_mps) {
    // Slip angles are undefined near standstill: move kinematically and keep the states consistent with it.
    KinematicBicycle(p_.wheelbase_m).step(s, u0, u1, dt);
    s.vy = p_.lr() * s.r;
    return;
  }
  // The lateral dynamics get stiff at low speed (eigenvalues ~ C / (m v)); substep so RK4 stays stable.
  const double lam = (p_.cf_npr + p_.cr_npr) / (p_.mass_kg * v_lo) +
                     (p_.lf_m * p_.lf_m * p_.cf_npr + p_.lr() * p_.lr() * p_.cr_npr) / (p_.yaw_inertia_kgm2 * v_lo);
  const int n = std::max(1, static_cast<int>(std::ceil(lam * dt / 1.0)));
  const double h = dt / n;
  for (int i = 0; i < n; ++i) {
    const Input a = lerp(u0, u1, static_cast<double>(i) / n), b = lerp(u0, u1, static_cast<double>(i + 1) / n);
    rk4(s, a, b, h, [this](const State& st, const Input& u, State& d) { deriv(st, u, d); });
  }
}

// --- baselines ----------------------------------------------------------------------------------------------
void ConstantYawRate::step(State& s, const Input& u0, const Input& u1, double dt) const {
  rk4(s, u0, u1, dt, [](const State& st, const Input& u, State& d) {
    d.x = u.v * std::cos(st.psi);
    d.y = u.v * std::sin(st.psi);
    d.psi = st.r;
    d.vy = 0.0;
    d.r = 0.0;
  });
}

void ConstantVelocity::init(State& s, const Input&, double) const {
  s.r = 0.0;
  s.vy = 0.0;
}

void ConstantVelocity::step(State& s, const Input& u0, const Input& u1, double dt) const {
  rk4(s, u0, u1, dt, [](const State& st, const Input& u, State& d) {
    d.x = u.v * std::cos(st.psi);
    d.y = u.v * std::sin(st.psi);
    d.psi = 0.0;
    d.vy = 0.0;
    d.r = 0.0;
  });
}

void YawRateDecay::init(State& s, const Input& u, double measured_yaw_rate) const {
  const double v = std::max(u.v, 0.5);
  s.r = std::clamp(measured_yaw_rate / v, -max_k_, max_k_) * v;
  s.vy = 0.0;
}

void YawRateDecay::step(State& s, const Input& u0, const Input& u1, double dt) const {
  const double tau = tau_;
  rk4(s, u0, u1, dt, [tau](const State& st, const Input& u, State& d) {
    d.x = u.v * std::cos(st.psi);
    d.y = u.v * std::sin(st.psi);
    d.psi = st.r;
    d.vy = 0.0;
    d.r = tau > 0.0 ? -st.r / tau : 0.0;
  });
}

// --- path prediction for the FCW corridor -------------------------------------------------------------------
void PredictedPath::predict(const Model& m, const Input& u, double measured_yaw_rate, double horizon_s, double dt,
                            double steer_decay_s) {
  pts_.clear();
  State s;
  m.init(s, u, measured_yaw_rate);
  pts_.push_back(s);
  const int n = static_cast<int>(std::lround(horizon_s / dt));
  const double f = steer_decay_s > 0.0 ? std::exp(-dt / steer_decay_s) : 1.0;
  Input a = u;
  for (int i = 0; i < n; ++i) {
    const Input b{a.delta * f, u.v};
    m.step(s, a, b, dt);
    pts_.push_back(s);
    a = b;
  }
}

double PredictedPath::offset(double x) const {
  if (pts_.empty() || x <= pts_.front().x) return pts_.empty() ? 0.0 : pts_.front().y;
  for (size_t i = 1; i < pts_.size(); ++i) {
    const State &a = pts_[i - 1], &b = pts_[i];
    if (b.x <= a.x) break;  // the path turned back past 90 degrees: extrapolate from here
    if (x <= b.x) return a.y + (x - a.x) / (b.x - a.x) * (b.y - a.y);
  }
  // beyond the predicted horizon: straight along the last monotone heading (clamped below 80 degrees)
  size_t last = 0;
  for (size_t i = 1; i < pts_.size() && pts_[i].x > pts_[i - 1].x; ++i) last = i;
  const State& e = pts_[last];
  const double psi = std::clamp(e.psi, -1.396, 1.396);
  return e.y + (x - e.x) * std::tan(psi);
}

double decay_path_offset(double speed, double yaw_rate, double x, double max_curvature, double tau) {
  // heading(t) = k v tau (1 - e^{-t/tau}); lateral offset by small-angle integration at constant speed
  const double v = std::max(speed, 0.5);
  const double k = std::clamp(yaw_rate / v, -max_curvature, max_curvature);
  const double tt = std::max(x, 0.0) / v;  // time to reach the longitudinal position x
  return tau > 0.0 ? k * v * v * tau * (tt - tau * (1.0 - std::exp(-tt / tau))) : 0.5 * k * x * x;
}

}  // namespace dr::vdyn
