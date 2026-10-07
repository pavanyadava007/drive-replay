#include "stack/vdyn/evaluate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>

namespace dr::vdyn {
namespace {

constexpr double kPi = 3.14159265358979323846;

double wrap(double a) {
  while (a > kPi) a -= 2 * kPi;
  while (a < -kPi) a += 2 * kPi;
  return a;
}

// n, sum |e|, sum e^2
struct Acc {
  double n = 0, sa = 0, ss = 0;
  void add(double e) { n += 1; sa += std::fabs(e); ss += e * e; }
  nlohmann::json json() const { return nlohmann::json::array({n, sa, ss}); }
};

std::string key(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}

std::string bin_label(const std::vector<double>& edges, double v) {
  if (v < edges.front()) return "";
  for (size_t i = 0; i + 1 < edges.size(); ++i)
    if (v < edges[i + 1]) return key(edges[i]) + "-" + key(edges[i + 1]);
  return key(edges.back()) + "+";
}

}  // namespace

std::string fnv1a_hex(const std::string& text) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : text) { h ^= c; h *= 1099511628211ULL; }
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h));
  return b;
}

SceneTable read_scene_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open scene table " + path);
  SceneTable s;
  const auto slash = path.find_last_of('/');
  s.name = path.substr(slash == std::string::npos ? 0 : slash + 1);
  if (s.name.size() > 4 && s.name.substr(s.name.size() - 4) == ".csv") s.name.resize(s.name.size() - 4);
  std::string line;
  std::getline(in, line);
  if (line != "t_s,x,y,yaw,v_pose,steer_sw,r_imu,rpm_rear,drive") throw std::runtime_error(path + ": unexpected header");
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    double v[9];
    if (std::sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                    &v[7], &v[8]) != 9)
      throw std::runtime_error(path + ": bad row: " + line);
    s.t.push_back(v[0]); s.x.push_back(v[1]); s.y.push_back(v[2]); s.yaw.push_back(v[3]); s.v_pose.push_back(v[4]);
    s.steer_sw.push_back(v[5]); s.r_imu.push_back(v[6]); s.rpm_rear.push_back(v[7]);
    s.drive.push_back(static_cast<int>(v[8]));
  }
  if (s.size() < 10) throw std::runtime_error(path + ": too few rows");
  return s;
}

nlohmann::json evaluate_scene(const SceneTable& s, const VehicleConfig& c, const EvalOptions& o, std::vector<TraceRow>* trace) {
  const size_t n = s.size();
  const double L = c.dynamic.wheelbase_m;
  std::vector<double> v_in(n), d_kin(n), d_dyn(n);
  for (size_t i = 0; i < n; ++i) {
    v_in[i] = s.rpm_rear[i] * 2.0 * kPi * c.wheel_radius_m / 60.0;  // speed from the rear (non-driven) wheels
    d_kin[i] = c.kinematic_steering.road_wheel(s.steer_sw[i]);
    d_dyn[i] = c.dynamic_steering.road_wheel(s.steer_sw[i]);
  }
  const KinematicBicycle kin(L);
  const LinearSingleTrack dyn(c.dynamic);
  const ConstantYawRate cyr;
  const ConstantVelocity cv;
  const YawRateDecay decay(c.fcw_decay_tau_s, c.fcw_max_curvature);
  const double K = c.dynamic.understeer_K();

  // ---- yaw rate over the whole scene -------------------------------------------------------------------
  std::map<std::string, std::map<std::string, Acc>> yaw;
  Acc speed;
  State sd;
  dyn.init(sd, Input{d_dyn[0], v_in[0]}, s.r_imu[0]);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) dyn.step(sd, Input{d_dyn[i - 1], v_in[i - 1]}, Input{d_dyn[i], v_in[i]}, s.t[i] - s.t[i - 1]);
    const double r_kin = kinematic_yaw_rate(L, d_kin[i], v_in[i]);
    const double r_ss = linear_steady_yaw_rate(L, K, d_dyn[i], v_in[i]);
    if (trace) trace->push_back(TraceRow{s.t[i], v_in[i], s.r_imu[i], r_kin, r_ss, sd.r});
    if (!s.drive[i] || s.v_pose[i] < o.min_yaw_speed) continue;
    speed.add(v_in[i] - s.v_pose[i]);
    const std::string b = bin_label(o.speed_edges, s.v_pose[i]);
    for (const auto& [name, r] : {std::pair<const char*, double>{"kinematic", r_kin}, {"steady_state", r_ss},
                                  {"dynamic", sd.r}}) {
      yaw[name][b].add(r - s.r_imu[i]);
      yaw[name]["all"].add(r - s.r_imu[i]);
    }
  }

  // ---- open-loop trajectories and corridor centre lines ------------------------------------------------
  struct Entry { const char* name; const Model* m; const std::vector<double>* delta; bool recorded; };
  const std::vector<Entry> models = {
      {"const_velocity", &cv, &d_dyn, false}, {"const_yaw_rate", &cyr, &d_dyn, false},
      {"fcw_yaw_decay", &decay, &d_dyn, false}, {"kinematic", &kin, &d_kin, false},
      {"dynamic", &dyn, &d_dyn, false}, {"kinematic_rec", &kin, &d_kin, true}, {"dynamic_rec", &dyn, &d_dyn, true}};
  std::map<std::string, std::map<std::string, std::map<std::string, Acc>>> traj;  // model -> horizon -> metric
  std::map<std::string, std::map<std::string, Acc>> corridor;                     // model -> look-ahead
  const double h_max = *std::max_element(o.horizons_s.begin(), o.horizons_s.end());
  int starts = 0;
  for (size_t i0 = 0; i0 < (o.yaw_only ? 0 : n); i0 += static_cast<size_t>(o.stride)) {
    // horizon indices: first sample at or after t0 + H
    std::vector<size_t> hidx;
    for (double H : o.horizons_s) {
      size_t j = i0;
      while (j < n && s.t[j] < s.t[i0] + H - 1e-6) ++j;
      hidx.push_back(j);
    }
    const size_t jmax = *std::max_element(hidx.begin(), hidx.end());
    if (jmax >= n || v_in[i0] < o.min_start_speed) continue;
    bool driving = true;
    for (size_t k = i0; k <= jmax; ++k) driving = driving && s.drive[k];
    if (!driving) continue;
    starts += 1;
    for (const Entry& e : models) {
      const std::vector<double>& d = *e.delta;
      State st{s.x[i0], s.y[i0], s.yaw[i0], 0.0, 0.0};
      const Input held{d[i0], v_in[i0]};
      e.m->init(st, held, s.r_imu[i0]);
      size_t k = i0;
      for (size_t h = 0; h < hidx.size(); ++h) {
        for (; k < hidx[h]; ++k) {
          const double dt = s.t[k + 1] - s.t[k];
          if (e.recorded) e.m->step(st, Input{d[k], v_in[k]}, Input{d[k + 1], v_in[k + 1]}, dt);
          else e.m->step(st, held, held, dt);
        }
        const size_t j = hidx[h];
        const double ex = st.x - s.x[j], ey = st.y - s.y[j];
        const double lat = -ex * std::sin(s.yaw[j]) + ey * std::cos(s.yaw[j]);
        auto& slot = traj[e.name][key(o.horizons_s[h])];
        slot["fde"].add(std::hypot(ex, ey));
        slot["lat"].add(lat);
        slot["head"].add(wrap(st.psi - s.yaw[j]));
      }
    }
    // corridor: the driven path in the start vehicle frame
    const double c0 = std::cos(s.yaw[i0]), s0 = std::sin(s.yaw[i0]);
    std::vector<std::pair<double, double>> driven;
    for (size_t k = i0; k < n && s.t[k] <= s.t[i0] + h_max + 3.0; ++k) {
      const double dx = s.x[k] - s.x[i0], dy = s.y[k] - s.y[i0];
      driven.emplace_back(c0 * dx + s0 * dy, -s0 * dx + c0 * dy);
    }
    std::map<std::string, PredictedPath> paths;
    const Input u{d_dyn[i0], std::max(v_in[i0], 0.5)}, uk{d_kin[i0], std::max(v_in[i0], 0.5)};
    paths["single_track"].predict(dyn, u, s.r_imu[i0], o.corridor_horizon_s, o.corridor_dt);
    paths["single_track_steer_decay"].predict(dyn, u, s.r_imu[i0], o.corridor_horizon_s, o.corridor_dt, c.fcw_decay_tau_s);
    paths["kinematic"].predict(kin, uk, s.r_imu[i0], o.corridor_horizon_s, o.corridor_dt);
    paths["const_yaw_rate"].predict(cyr, u, s.r_imu[i0], o.corridor_horizon_s, o.corridor_dt);
    for (double D : o.lookahead_m) {
      double y_true = 0;
      bool found = false;
      for (size_t k = 1; k < driven.size(); ++k) {
        if (driven[k].first <= driven[k - 1].first) break;
        if (driven[k].first >= D) {
          const auto &a = driven[k - 1], &b = driven[k];
          y_true = a.second + (D - a.first) / (b.first - a.first) * (b.second - a.second);
          found = true;
          break;
        }
      }
      if (!found) continue;
      const std::string kd = key(D);
      corridor["fcw_yaw_decay"][kd].add(
          decay_path_offset(v_in[i0], s.r_imu[i0], D, c.fcw_max_curvature, c.fcw_decay_tau_s) - y_true);
      corridor["const_velocity"][kd].add(0.0 - y_true);
      for (const auto& [name, p] : paths) corridor[name][kd].add(p.offset(D) - y_true);
    }
  }

  nlohmann::json out;
  out["scene"] = s.name;
  out["samples"] = n;
  out["duration_s"] = s.t.back() - s.t.front();
  out["starts"] = starts;
  out["speed_err"] = speed.json();
  for (const auto& [m, bins] : yaw)
    for (const auto& [b, a] : bins) out["yaw"][m][b] = a.json();
  for (const auto& [m, hs] : traj)
    for (const auto& [h, ms] : hs)
      for (const auto& [k, a] : ms) out["traj"][m][h][k] = a.json();
  for (const auto& [m, ds] : corridor)
    for (const auto& [d, a] : ds) out["corridor"][m][d] = a.json();
  out["digest"] = fnv1a_hex(out.dump());
  return out;
}

}  // namespace dr::vdyn
