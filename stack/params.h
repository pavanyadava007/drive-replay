// Tunable parameters. replay_main loads them from the run's resolved config JSON.
#pragma once

#include <string>

#include "stack/vdyn/vdyn.h"

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
  double max_curvature = 0.2;         // 1/m, clamps yaw_rate / speed at low speed (BUG-0002)
  double yaw_rate_decay_s = 1.0;      // predicted yaw rate decays to zero with this time constant (BUG-0003)
  double oncoming_speed_mps = 1.0;    // targets approaching faster than this over ground are oncoming (BUG-0001)
  double front_bumper_m = 3.7;  // rear axle to front bumper
  double min_closing_mps = 0.5;
  double ttc_warn_s = 2.2;
  double ttc_release_s = 2.7;
  double ttc_brake_s = 1.0;
  int warn_confirm_cycles = 2;
  int release_cycles = 3;
  // path prediction model of the corridor: 0 = yaw rate decaying with yaw_rate_decay_s (default, stack 0.2.0),
  // 1 = identified linear single-track model driven by the steering angle (opt-in, needs /vehicle/can and
  // vehicle_model; falls back to 0 on cycles without a fresh steering angle)
  int path_model = 0;
  double path_horizon_s = 4.0;
  double path_steer_decay_s = 1.0;  // path_model 1: steering angle decays with this time constant (0 = held);
                                    // chosen on the validation scenes, see docs/VEHICLE_DYNAMICS.md
  double steer_max_age_s = 0.1;
  std::string vehicle_model;     // path to configs/vdyn/<vehicle>.json, relative to the working directory
  std::string vehicle_id;        // car whose steering offset applies (nuScenes log vehicle, e.g. n015)
  vdyn::VehicleConfig vehicle;   // loaded from vehicle_model, not a config key itself
};

// Throws std::runtime_error on unknown keys, so a typo in a config never runs silently.
Params load_params(const std::string& json_path);

}  // namespace dr
