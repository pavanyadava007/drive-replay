// mpred_bench: latency of one prediction per method on recorded scenes, single thread, as a JSON report.
//
//   mpred_bench --predictors configs/mpred/predictors.json --vehicle-config configs/vdyn/renault_zoe.json
//               --scene-list FILE (lines "path vehicle") [--repeat 3] [--out bench.json]
//
// Every sample of every scene is pushed through the history and the EKF as in the car. From the 50th sample on,
// every method predicts all 7 horizons once per sample; each call is timed on its own with steady_clock (the
// clock overhead is measured and reported, not subtracted). Pin the process to one core (taskset) for stable
// percentiles. These are x86 host timings, not an automotive ECU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "stack/mpred/mpred.h"
#include "stack/mpred/scene.h"
#include "stack/vdyn/vdyn.h"

namespace {

using Clock = std::chrono::steady_clock;

int usage() {
  std::cerr << "usage: mpred_bench --predictors FILE --vehicle-config FILE --scene-list FILE [--repeat N] [--out FILE]\n";
  return 2;
}

nlohmann::json stats(std::vector<double> ns) {
  std::sort(ns.begin(), ns.end());
  auto q = [&](double p) { return ns[static_cast<size_t>(std::min<double>(ns.size() - 1, std::floor(p * (ns.size() - 1) + 0.5)))]; };
  double sum = 0;
  for (double v : ns) sum += v;
  return {{"n", ns.size()}, {"mean_ns", sum / ns.size()}, {"p50_ns", q(0.50)}, {"p90_ns", q(0.90)},
          {"p99_ns", q(0.99)}, {"p999_ns", q(0.999)}, {"max_ns", ns.back()}};
}

double checksum(const dr::mpred::Prediction& p) {
  double s = 0;
  for (const auto& q : p) s += q.x + q.y + q.yaw + q.v;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string predictors, vehicle_config, scene_list, out_path;
  int repeat = 3;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--predictors") predictors = next();
      else if (a == "--vehicle-config") vehicle_config = next();
      else if (a == "--scene-list") scene_list = next();
      else if (a == "--repeat") repeat = std::stoi(next());
      else if (a == "--out") out_path = next();
      else return usage();
    }
    if (predictors.empty() || vehicle_config.empty() || scene_list.empty()) return usage();
    const dr::mpred::PredictorSet ps = dr::mpred::load_predictors(predictors);
    const dr::vdyn::VehicleConfig base = dr::vdyn::load_vehicle_config(vehicle_config);
    struct Scene {
      dr::vdyn::VehicleConfig car;
      std::vector<dr::mpred::Sample> samples;
    };
    std::vector<Scene> scenes;
    std::ifstream list(scene_list);
    for (std::string line; std::getline(list, line);) {
      std::istringstream ls(line);
      std::string path, vehicle;
      ls >> path >> vehicle;
      Scene s{base.for_vehicle(vehicle), {}};
      s.samples = dr::mpred::to_samples(dr::mpred::read_scene_v2(path), s.car);
      scenes.push_back(std::move(s));
    }
    // clock overhead: two consecutive reads
    std::vector<double> overhead;
    for (int i = 0; i < 100000; ++i) {
      const auto a = Clock::now();
      const auto b = Clock::now();
      overhead.push_back(std::chrono::duration<double, std::nano>(b - a).count());
    }
    std::map<std::string, std::vector<double>> t;
    double sink = 0;
    long samples_total = 0;
    auto timed = [&](const std::string& name, auto&& fn) {
      const auto a = Clock::now();
      const dr::mpred::Prediction p = fn();
      const auto b = Clock::now();
      t[name].push_back(std::chrono::duration<double, std::nano>(b - a).count());
      sink += checksum(p);
    };
    for (int rep = 0; rep < repeat; ++rep) {
      for (const Scene& sc : scenes) {
        dr::mpred::History h(ps.ctra_accel);
        dr::mpred::Ekf ekf(ps.ekf);
        const dr::vdyn::LinearSingleTrack dyn(sc.car.dynamic);
        for (size_t k = 0; k < sc.samples.size(); ++k) {
          const dr::mpred::Sample& s = sc.samples[k];
          const auto a = Clock::now();
          h.push(s);
          ekf.step(s);
          const auto b = Clock::now();
          t["history_push_and_ekf_step"].push_back(std::chrono::duration<double, std::nano>(b - a).count());
          ++samples_total;
          if (k < 50) continue;
          timed("cv", [&] { return dr::mpred::predict_cv(h); });
          timed("ctrv", [&] { return dr::mpred::predict_ctrv(h); });
          timed("ctra", [&] { return dr::mpred::predict_ctra(h); });
          timed("kinematic", [&] { return dr::mpred::predict_kinematic(h, sc.car.dynamic.wheelbase_m); });
          timed("dynamic", [&] {
            dr::vdyn::State st;
            st.x = s.x;
            st.y = s.y;
            st.psi = s.yaw;
            const dr::vdyn::Input u{s.d_dyn, s.v};
            dyn.init(st, u, s.r);
            dr::mpred::Prediction p{};
            size_t nx = 0;
            for (int i = 1; i <= dr::mpred::kHorizonSteps.back(); ++i) {
              dyn.step(st, u, u, dr::mpred::kDtInt);
              if (i == dr::mpred::kHorizonSteps[nx]) p[nx++] = dr::mpred::Pose{st.x, st.y, st.psi, s.v};
            }
            return p;
          });
          timed("ekf", [&] { return ekf.forecast(); });
          for (const auto& kv : ps.learned)
            timed(kv.first, [&] { return dr::mpred::predict_learned(kv.second, h, &ekf); });
        }
      }
    }
    nlohmann::json j;
    j["clock_overhead"] = stats(overhead);
    for (auto& kv : t) j["methods"][kv.first] = stats(kv.second);
    j["samples_streamed"] = samples_total;
    j["scenes"] = scenes.size();
    j["repeat"] = repeat;
    j["sizeof"] = {{"History", sizeof(dr::mpred::History)}, {"Ekf", sizeof(dr::mpred::Ekf)},
                   {"Prediction", sizeof(dr::mpred::Prediction)}};
    for (const auto& kv : ps.learned) j["learned_params"][kv.first] = kv.second.n_params();
    j["checksum"] = sink;
    if (out_path.empty()) std::cout << j.dump(1) << "\n";
    else std::ofstream(out_path) << j.dump(1) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "mpred_bench: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
