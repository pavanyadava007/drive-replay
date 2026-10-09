#include "stack/mpred/scene.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace dr::mpred {

SceneRows read_scene_v2(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open scene table " + path);
  SceneRows s;
  const auto slash = path.find_last_of('/');
  s.name = path.substr(slash == std::string::npos ? 0 : slash + 1);
  if (s.name.size() > 4 && s.name.substr(s.name.size() - 4) == ".csv") s.name.resize(s.name.size() - 4);
  std::string line;
  std::getline(in, line);
  if (line != "t_s,x,y,yaw,v_pose,drive,steer_sw,r_imu,rpm_rear,ax_imu,ay_imu,ax_can")
    throw std::runtime_error(path + ": unexpected header (not a tools/mpred/extract.py table)");
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const char* p = line.c_str();
    char* end = nullptr;
    double v[12];
    for (int i = 0; i < 12; ++i) {
      v[i] = std::strtod(p, &end);
      if (end == p) throw std::runtime_error(path + ": bad number in " + line);
      p = (*end == ',') ? end + 1 : end;
    }
    s.t.push_back(v[0]);
    s.x.push_back(v[1]);
    s.y.push_back(v[2]);
    s.yaw.push_back(v[3]);
    s.v_pose.push_back(v[4]);
    s.drive.push_back(static_cast<int>(v[5]));
    s.steer_sw.push_back(v[6]);
    s.r_imu.push_back(v[7]);
    s.rpm_rear.push_back(v[8]);
    s.ax_imu.push_back(v[9]);
    s.ay_imu.push_back(v[10]);
    s.ax_can.push_back(v[11]);
  }
  return s;
}

std::vector<Sample> to_samples(const SceneRows& rows, const vdyn::VehicleConfig& car) {
  std::vector<Sample> out(rows.size());
  const double rpm_to_rad_s = 2.0 * 3.14159265358979323846 / 60.0;
  for (size_t i = 0; i < rows.size(); ++i) {
    Sample& s = out[i];
    s.t = rows.t[i];
    s.x = rows.x[i];
    s.y = rows.y[i];
    s.yaw = rows.yaw[i];
    s.v = rows.rpm_rear[i] * rpm_to_rad_s * car.wheel_radius_m;  // same association as the Python prototype
    s.r = rows.r_imu[i];
    s.d_kin = (rows.steer_sw[i] - car.kinematic_steering.offset_rad) / car.kinematic_steering.ratio;
    s.d_dyn = (rows.steer_sw[i] - car.dynamic_steering.offset_rad) / car.dynamic_steering.ratio;
    s.ax_imu = rows.ax_imu[i];
    s.ax_can = rows.ax_can[i];
  }
  return out;
}

}  // namespace dr::mpred
