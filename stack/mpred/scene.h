// Scene tables of tools/mpred/extract.py (can_scenes_v2) turned into mpred samples, plus the streaming loop
// shared by the parity tool and the benchmark.
#pragma once

#include <string>
#include <vector>

#include "stack/mpred/mpred.h"
#include "stack/vdyn/vdyn.h"

namespace dr::mpred {

struct SceneRows {
  std::string name;
  std::vector<double> t, x, y, yaw, v_pose, steer_sw, r_imu, rpm_rear, ax_imu, ay_imu, ax_can;
  std::vector<int> drive;
  size_t size() const { return t.size(); }
};

// Header must be t_s,x,y,yaw,v_pose,drive,steer_sw,r_imu,rpm_rear,ax_imu,ay_imu,ax_can.
SceneRows read_scene_v2(const std::string& path);

// Unit conversion with the identified parameters of the car (wheel radius, both steering maps with its offset).
std::vector<Sample> to_samples(const SceneRows& rows, const vdyn::VehicleConfig& car);

}  // namespace dr::mpred
