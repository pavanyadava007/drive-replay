// Validation of the vehicle dynamics models on one recorded scene (a table written by tools/vdyn/can_extract.py).
//
// Three questions per scene, all answered with sums (n, sum |e|, sum e^2) so that scenes can be pooled and
// bootstrapped later without re-running anything:
//   yaw       yaw rate predicted from steering angle + wheel speed against the IMU yaw rate, by speed bin
//   traj      open-loop ego trajectory from start points every `stride` samples, over each horizon, against
//             the recorded pose: final displacement error, lateral error (normal to the recorded heading)
//             and heading error. "held" models keep the inputs of the start instant (what a function has at
//             run time); "rec" models are fed the recorded future steering and speed (model fidelity only).
//   corridor  lateral offset of the predicted path centre line at fixed look-ahead distances, against the
//             path the car really drove (the question the FCW corridor asks). single_track holds the steering
//             angle; single_track_steer_decay lets it decay with the FCW's yaw-rate time constant (1 s).
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "stack/vdyn/vdyn.h"

namespace dr::vdyn {

struct SceneTable {
  std::string name;
  std::vector<double> t, x, y, yaw, v_pose, steer_sw, r_imu, rpm_rear;
  std::vector<int> drive;
  size_t size() const { return t.size(); }
};

// Reads the CSV written by can_extract.py (header line, columns t_s,x,y,yaw,v_pose,steer_sw,r_imu,rpm_rear,drive).
SceneTable read_scene_csv(const std::string& path);

struct EvalOptions {
  std::vector<double> horizons_s{1.0, 2.0, 3.0};
  int stride = 25;                  // start points every 0.5 s at 50 Hz
  double min_start_speed = 2.0;     // the FCW is active from 2 m/s
  double min_yaw_speed = 0.5;       // yaw-rate samples below this are not scored
  std::vector<double> speed_edges{0.5, 3.0, 6.0, 9.0, 12.0};  // last bin is open ended
  std::vector<double> lookahead_m{10.0, 20.0, 30.0};
  double corridor_horizon_s = 4.0;  // same as the FCW default path_horizon_s
  double corridor_dt = 0.05;
  bool yaw_only = false;            // identification needs only the yaw-rate part
};

struct TraceRow {
  double t, v, r_imu, r_kin, r_ss, r_dyn;
};

// Returns {"scene", "samples", "duration_s", "speed_rmse", "yaw", "traj", "corridor", "digest"}.
nlohmann::json evaluate_scene(const SceneTable& s, const VehicleConfig& c, const EvalOptions& o,
                              std::vector<TraceRow>* trace = nullptr);

std::string fnv1a_hex(const std::string& text);

}  // namespace dr::vdyn
