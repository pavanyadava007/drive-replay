// mpred_eval: run the C++ predictors over recorded scenes as a stream (sample by sample, as in the car) and
// write every prediction, for the Python vs C++ parity check of scripts/reproduce_mpred.py.
//
//   mpred_eval --predictors configs/mpred/predictors.json --vehicle-config configs/vdyn/renault_zoe.json
//              --scene-list FILE (lines "path vehicle") --out preds.f64 [--stride 5] [--history 50]
//              [--min-speed 1.0]
//
// Output: rows of 3 + 7 x 4 doubles: scene index (line in the list), sample index k, method index, then
// x, y, yaw, v for each horizon. The method names are printed as one JSON line on stdout.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stack/mpred/mpred.h"
#include "stack/mpred/scene.h"
#include "stack/vdyn/vdyn.h"

namespace {

int usage() {
  std::cerr << "usage: mpred_eval --predictors FILE --vehicle-config FILE --scene-list FILE --out FILE\n"
               "                  [--stride N] [--history N] [--min-speed MPS]\n";
  return 2;
}

dr::mpred::Prediction dynamic_single_track(const dr::vdyn::LinearSingleTrack& m, const dr::mpred::Sample& s) {
  dr::vdyn::State st;
  st.x = s.x;
  st.y = s.y;
  st.psi = s.yaw;
  const dr::vdyn::Input u{s.d_dyn, s.v};
  m.init(st, u, s.r);
  dr::mpred::Prediction p{};
  size_t next = 0;
  for (int i = 1; i <= dr::mpred::kHorizonSteps.back(); ++i) {
    m.step(st, u, u, dr::mpred::kDtInt);
    if (i == dr::mpred::kHorizonSteps[next]) p[next++] = dr::mpred::Pose{st.x, st.y, st.psi, s.v};
  }
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  std::string predictors, vehicle_config, scene_list, out_path;
  int stride = 5, history = 50;
  double min_speed = 1.0;
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
      else if (a == "--out") out_path = next();
      else if (a == "--stride") stride = std::stoi(next());
      else if (a == "--history") history = std::stoi(next());
      else if (a == "--min-speed") min_speed = std::stod(next());
      else return usage();
    }
    if (predictors.empty() || vehicle_config.empty() || scene_list.empty() || out_path.empty()) return usage();
    if (history < dr::mpred::kMaxLag) throw std::runtime_error("--history must be at least 25 samples");
    const dr::mpred::PredictorSet ps = dr::mpred::load_predictors(predictors);
    const dr::vdyn::VehicleConfig base = dr::vdyn::load_vehicle_config(vehicle_config);
    std::vector<std::string> names{"cv", "ctrv", "ctra", "kinematic", "dynamic", "ekf"};
    for (const auto& kv : ps.learned) names.push_back(kv.first);
    std::ifstream list(scene_list);
    if (!list) throw std::runtime_error("cannot open " + scene_list);
    std::ofstream out(out_path, std::ios::binary);
    long rows = 0;
    int scene_index = 0;
    for (std::string line; std::getline(list, line); ++scene_index) {
      std::istringstream ls(line);
      std::string path, vehicle;
      ls >> path >> vehicle;
      const dr::vdyn::VehicleConfig car = base.for_vehicle(vehicle);
      const dr::vdyn::LinearSingleTrack dyn(car.dynamic);
      const std::vector<dr::mpred::Sample> samples = dr::mpred::to_samples(dr::mpred::read_scene_v2(path), car);
      dr::mpred::History h(ps.ctra_accel);
      dr::mpred::Ekf ekf(ps.ekf);
      for (size_t k = 0; k < samples.size(); ++k) {
        h.push(samples[k]);
        ekf.step(samples[k]);
        if (static_cast<int>(k) < history || static_cast<int>(k) % stride != 0 || samples[k].v < min_speed) continue;
        std::vector<dr::mpred::Prediction> preds{dr::mpred::predict_cv(h), dr::mpred::predict_ctrv(h),
                                                 dr::mpred::predict_ctra(h),
                                                 dr::mpred::predict_kinematic(h, car.dynamic.wheelbase_m),
                                                 dynamic_single_track(dyn, samples[k]), ekf.forecast()};
        for (const auto& kv : ps.learned) preds.push_back(dr::mpred::predict_learned(kv.second, h, &ekf));
        for (size_t m = 0; m < preds.size(); ++m) {
          double row[3 + dr::mpred::kHorizons * 4];
          row[0] = scene_index;
          row[1] = static_cast<double>(k);
          row[2] = static_cast<double>(m);
          for (size_t j = 0; j < dr::mpred::kHorizons; ++j) {
            row[3 + 4 * j] = preds[m][j].x;
            row[4 + 4 * j] = preds[m][j].y;
            row[5 + 4 * j] = preds[m][j].yaw;
            row[6 + 4 * j] = preds[m][j].v;
          }
          out.write(reinterpret_cast<const char*>(row), sizeof row);
          ++rows;
        }
      }
    }
    std::cout << "{\"methods\": [";
    for (size_t m = 0; m < names.size(); ++m) std::cout << (m ? ", " : "") << '"' << names[m] << '"';
    std::cout << "], \"scenes\": " << scene_index << ", \"rows\": " << rows << "}\n";
  } catch (const std::exception& e) {
    std::cerr << "mpred_eval: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
