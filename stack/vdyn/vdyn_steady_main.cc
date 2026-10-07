// vdyn_steady: steady-state yaw rate of the linear single-track model, simulated by the C++ code (RK4, 30 s),
// next to the closed form. Used for the GNU Octave cross-check (tools/vdyn/octave/steady_state.m).
//
//   vdyn_steady --config configs/vdyn/renault_zoe.json > steady.csv
#include <cstdio>
#include <iostream>
#include <string>

#include "stack/vdyn/vdyn.h"

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[1]) != "--config") {
    std::cerr << "usage: vdyn_steady --config FILE\n";
    return 2;
  }
  try {
    const dr::vdyn::VehicleConfig c = dr::vdyn::load_vehicle_config(argv[2]);
    const dr::vdyn::LinearSingleTrack dyn(c.dynamic);
    const double K = c.dynamic.understeer_K();
    std::printf("v_mps,delta_rad,r_cpp_sim,r_cpp_formula\n");
    for (double v : {2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 25.0, 30.0}) {
      for (double delta : {-0.03, 0.01, 0.02, 0.05}) {
        dr::vdyn::State s;
        const dr::vdyn::Input u{delta, v};
        dyn.init(s, u, 0.0);
        for (int i = 0; i < 1500; ++i) dyn.step(s, u, u, 0.02);
        std::printf("%.1f,%.3f,%.12e,%.12e\n", v, delta, s.r,
                    dr::vdyn::linear_steady_yaw_rate(c.dynamic.wheelbase_m, K, delta, v));
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "vdyn_steady: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
