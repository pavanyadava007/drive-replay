// Message types of the replay stack. Everything the stack consumes comes from a recording;
// everything it produces is written by replay_main and compared by the replay tooling.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dr {

struct EgoState {
  int64_t t_us = 0;
  double x = 0, y = 0, yaw = 0;  // global frame, only used for tracing
  double speed = 0;              // m/s, signed longitudinal
  double yaw_rate = 0;           // rad/s
  // Steering-wheel angle from /vehicle/can (only in recordings that carry it). Never part of the trace or the
  // digests, so recordings without the topic replay exactly as before.
  double steer_sw = 0;           // rad, left positive
  bool steer_valid = false;
};

// One /vehicle/can message (nuScenes CAN bus expansion): steering-wheel angle and rear wheel speed.
struct VehicleCan {
  int64_t t_us = 0;
  double steer_sw = 0;      // rad, left positive
  double wheel_rpm_rear = 0;  // mean rear wheel speed, rpm
};

// Sensor-to-vehicle mounting of the front radar, read from the recording's /calib topic.
struct Extrinsic {
  double tx = 0, ty = 0, yaw = 0;
  bool valid = false;
};

// One ARS408 cluster return in the sensor frame, as recorded.
struct RadarPoint {
  double x = 0, y = 0;             // m
  double vx = 0, vy = 0;           // m/s relative to the sensor (ego motion included)
  double vx_comp = 0, vy_comp = 0; // m/s over ground (ego motion removed by the sensor)
  double rcs = 0;                  // dBsm
  int dyn_prop = 0, ambig_state = 0, invalid_state = 0, id = 0;
};

struct RadarScan {
  int64_t t_us = 0;
  std::vector<RadarPoint> points;
};

// A clustered detection in the vehicle frame (origin at the rear axle, x forward, y left).
struct Detection {
  double x = 0, y = 0;    // m
  double vx = 0, vy = 0;  // m/s relative to ego
  double rcs = 0;
  int n_points = 0;
};

using Vec4 = std::array<double, 4>;
using Mat4 = std::array<std::array<double, 4>, 4>;

struct Track {
  int id = 0;
  Vec4 s{};  // x, y, vx, vy (vehicle frame, relative)
  Mat4 P{};
  int hits = 0;
  int misses = 0;
  int age = 0;             // cycles since birth
  int consecutive_hits = 0;
  bool confirmed = false;
  int in_path_cycles = 0;  // written by the FCW function
};

struct FcwEvent {
  int64_t t_us = 0;
  std::string kind;  // fcw_warning_on, fcw_warning_off, brake_request_on, brake_request_off
  int track_id = -1;
  double ttc = 0, range = 0, closing_speed = 0, ego_speed = 0;
};

}  // namespace dr
