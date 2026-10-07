// vdyn_eval: vehicle-dynamics validation of recorded scenes, one JSON line of metrics per scene.
//
//   vdyn_eval --config configs/vdyn/renault_zoe.json --scene a.csv [--scene b.csv ...] [--scene-list FILE]
//             [--out metrics.jsonl] [--trace yaw_trace.csv] [--set dynamic.cr_npr=120000 ...]
//             [--horizons 1,2,3] [--stride 25] [--yaw-only] [--vehicle n008]
//
// The output of a scene depends only on its table, the config and the options, so the batch pipeline can run
// scenes in any order and on any number of workers and still get the same per-scene digests.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stack/vdyn/evaluate.h"
#include "stack/vdyn/vdyn.h"

namespace {

int usage() {
  std::cerr << "usage: vdyn_eval --config FILE (--scene CSV)... [--scene-list FILE] [--out FILE] [--trace FILE]\n"
               "                 [--set section.key=value]... [--horizons 1,2,3] [--stride N] [--yaw-only] [--vehicle ID]\n";
  return 2;
}

void apply_set(dr::vdyn::VehicleConfig& c, const std::string& kv) {
  const auto eq = kv.find('=');
  if (eq == std::string::npos) throw std::runtime_error("--set expects key=value: " + kv);
  const std::string k = kv.substr(0, eq);
  const double v = std::stod(kv.substr(eq + 1));
  if (k == "wheel_radius_m") c.wheel_radius_m = v;
  else if (k == "wheelbase_m") c.dynamic.wheelbase_m = v;
  else if (k == "kinematic.steering_ratio") c.kinematic_steering.ratio = v;
  else if (k == "kinematic.steer_offset_rad") c.kinematic_steering.offset_rad = v;
  else if (k == "dynamic.steering_ratio") c.dynamic_steering.ratio = v;
  else if (k == "dynamic.steer_offset_rad") c.dynamic_steering.offset_rad = v;
  else if (k == "dynamic.lf_m") c.dynamic.lf_m = v;
  else if (k == "dynamic.mass_kg") c.dynamic.mass_kg = v;
  else if (k == "dynamic.yaw_inertia_kgm2") c.dynamic.yaw_inertia_kgm2 = v;
  else if (k == "dynamic.cf_npr") c.dynamic.cf_npr = v;
  else if (k == "dynamic.cr_npr") c.dynamic.cr_npr = v;
  else if (k == "dynamic.min_speed_mps") c.dynamic.min_speed_mps = v;
  else if (k == "fcw_baseline.yaw_rate_decay_s") c.fcw_decay_tau_s = v;
  else if (k == "fcw_baseline.max_curvature") c.fcw_max_curvature = v;
  else throw std::runtime_error("unknown --set key " + k);
}

}  // namespace

int main(int argc, char** argv) {
  std::string config, out_path, trace_path, vehicle;
  std::vector<std::string> scenes, sets;
  dr::vdyn::EvalOptions opt;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--config") config = next();
      else if (a == "--scene") scenes.push_back(next());
      else if (a == "--scene-list") {
        std::ifstream in(next());
        for (std::string line; std::getline(in, line);)
          if (!line.empty()) scenes.push_back(line);
      } else if (a == "--out") out_path = next();
      else if (a == "--trace") trace_path = next();
      else if (a == "--set") sets.push_back(next());
      else if (a == "--stride") opt.stride = std::stoi(next());
      else if (a == "--yaw-only") opt.yaw_only = true;
      else if (a == "--vehicle") vehicle = next();
      else if (a == "--horizons") {
        opt.horizons_s.clear();
        std::stringstream ss(next());
        for (std::string h; std::getline(ss, h, ',');) opt.horizons_s.push_back(std::stod(h));
      } else return usage();
    }
    if (config.empty() || scenes.empty()) return usage();
    dr::vdyn::VehicleConfig cfg = dr::vdyn::load_vehicle_config(config).for_vehicle(vehicle);
    for (const auto& kv : sets) apply_set(cfg, kv);
    std::ofstream file;
    if (!out_path.empty()) file.open(out_path);
    std::ostream& out = out_path.empty() ? std::cout : file;
    std::vector<dr::vdyn::TraceRow> trace;
    const auto t0 = std::chrono::steady_clock::now();
    double recorded_s = 0;
    for (size_t k = 0; k < scenes.size(); ++k) {
      const dr::vdyn::SceneTable s = dr::vdyn::read_scene_csv(scenes[k]);
      const auto j = dr::vdyn::evaluate_scene(s, cfg, opt, (k == 0 && !trace_path.empty()) ? &trace : nullptr);
      recorded_s += j["duration_s"].get<double>();
      out << j.dump() << "\n";
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!trace_path.empty()) {
      std::ofstream tr(trace_path);
      tr << "t_s,v_wheel,r_imu,r_kinematic,r_steady_state,r_dynamic\n";
      for (const auto& r : trace) {
        char b[160];
        std::snprintf(b, sizeof b, "%.2f,%.4f,%.6f,%.6f,%.6f,%.6f\n", r.t, r.v, r.r_imu, r.r_kin, r.r_ss, r.r_dyn);
        tr << b;
      }
    }
    std::fprintf(stderr, "vdyn_eval: %zu scenes, %.1f s recorded, %.3f s wall\n", scenes.size(), recorded_s, wall);
  } catch (const std::exception& e) {
    std::cerr << "vdyn_eval: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
