// Tunable parameters. replay_main loads them from the run's resolved config JSON.
#pragma once

#include <string>

namespace dr {

struct Params {
  // radar frontend
  double max_range_m = 80.0;
  int required_ambig_state = 3;  // 3 = unambiguous (nuScenes default filter)
  int max_dyn_prop = 6;
  // clustering
  double cluster_dist_m = 1.5;
  double cluster_dv_mps = 1.5;
  // tracker
  double gate_m = 2.5;
  int confirm_hits = 3;
  int delete_misses = 3;
  double accel_noise = 2.0;   // m/s^2, process noise
  double meas_pos_std = 0.4;  // m
  double meas_vx_std = 0.3;   // m/s
  double meas_vy_std = 1.5;   // m/s, lateral velocity is poorly observed by a doppler radar
  // forward collision warning
  double min_ego_speed_mps = 2.0;
  double corridor_half_width_m = 1.3;
  double front_bumper_m = 3.7;  // rear axle to front bumper
  double min_closing_mps = 0.5;
  double ttc_warn_s = 2.2;
  double ttc_release_s = 2.7;
  double ttc_brake_s = 1.0;
  int warn_confirm_cycles = 2;
  int release_cycles = 3;
};

// Throws std::runtime_error on unknown keys, so a typo in a config never runs silently.
Params load_params(const std::string& json_path);

}  // namespace dr
