#define MCAP_IMPLEMENTATION
#include "stack/recording.h"

#include <cmath>
#include <stdexcept>
#include <string_view>

#include <mcap/reader.hpp>
#include <nlohmann/json.hpp>

namespace dr {
namespace {

double yaw_from_wxyz(const nlohmann::json& q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  return std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

RadarScan parse_radar(const nlohmann::json& j) {
  RadarScan s;
  s.t_us = j.at("t_us").get<int64_t>();
  const auto& f = j.at("fields");
  auto idx = [&](const char* name) {
    for (size_t i = 0; i < f.size(); ++i)
      if (f[i] == name) return static_cast<int>(i);
    throw std::runtime_error(std::string("radar field missing: ") + name);
  };
  const int ix = idx("x"), iy = idx("y"), ivx = idx("vx"), ivy = idx("vy"), ivxc = idx("vx_comp"),
            ivyc = idx("vy_comp"), ircs = idx("rcs"), idyn = idx("dyn_prop"), iamb = idx("ambig_state"),
            iinv = idx("invalid_state"), iid = idx("id");
  for (const auto& p : j.at("points")) {
    RadarPoint r;
    r.x = p[ix]; r.y = p[iy]; r.vx = p[ivx]; r.vy = p[ivy]; r.vx_comp = p[ivxc]; r.vy_comp = p[ivyc];
    r.rcs = p[ircs]; r.dyn_prop = p[idyn]; r.ambig_state = p[iamb]; r.invalid_state = p[iinv]; r.id = p[iid];
    s.points.push_back(r);
  }
  return s;
}

}  // namespace

RecordingStats replay_recording(const std::string& path, int64_t start_us, int64_t end_us, const RecordingHandlers& h) {
  mcap::McapReader reader;
  const auto st = reader.open(path);
  if (!st.ok()) throw std::runtime_error("cannot open recording " + path + ": " + st.message);
  RecordingStats stats;
  mcap::ReadMessageOptions opts;
  opts.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
  const auto on_problem = [](const mcap::Status& s) { throw std::runtime_error("corrupt recording: " + s.message); };
  for (const mcap::MessageView& mv : reader.readMessages(on_problem, opts)) {
    const std::string& topic = mv.channel->topic;
    const int64_t t_us = static_cast<int64_t>(mv.message.logTime / 1000);
    stats.messages += 1;
    if (stats.first_us < 0) stats.first_us = t_us;
    stats.last_us = t_us;
    const bool in_window = t_us >= start_us && (end_us == 0 || t_us <= end_us);
    if (topic == "/calib/radar_front") {
      const auto j = nlohmann::json::parse(std::string_view(reinterpret_cast<const char*>(mv.message.data), mv.message.dataSize));
      Extrinsic e;
      e.tx = j["translation"][0]; e.ty = j["translation"][1]; e.yaw = yaw_from_wxyz(j["rotation_wxyz"]); e.valid = true;
      stats.calib += 1;
      if (h.calib) h.calib(e);
    } else if (!in_window) {
      stats.skipped += 1;
    } else if (topic == "/ego/state") {
      const auto j = nlohmann::json::parse(std::string_view(reinterpret_cast<const char*>(mv.message.data), mv.message.dataSize));
      EgoState e;
      e.t_us = j["t_us"]; e.x = j["x"]; e.y = j["y"]; e.yaw = j["yaw"]; e.speed = j["speed_mps"]; e.yaw_rate = j["yaw_rate_rps"];
      stats.ego += 1;
      if (h.ego) h.ego(e);
    } else if (topic == "/radar/front") {
      const auto j = nlohmann::json::parse(std::string_view(reinterpret_cast<const char*>(mv.message.data), mv.message.dataSize));
      stats.radar += 1;
      if (h.radar) h.radar(parse_radar(j));
    } else if (topic == "/vehicle/can" && h.can) {
      const auto j = nlohmann::json::parse(std::string_view(reinterpret_cast<const char*>(mv.message.data), mv.message.dataSize));
      VehicleCan c;
      c.t_us = j["t_us"]; c.steer_sw = j["steer_sw_rad"]; c.wheel_rpm_rear = j["wheel_rpm_rear"];
      h.can(c);
    } else {
      stats.skipped += 1;
    }
    if (end_us != 0 && t_us > end_us) break;
  }
  reader.close();
  return stats;
}

}  // namespace dr
