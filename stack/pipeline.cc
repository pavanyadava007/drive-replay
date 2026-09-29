#include "stack/pipeline.h"

#include <cstdio>

#include "stack/radar_frontend.h"

namespace dr {

CycleResult Pipeline::on_radar(const RadarScan& scan) {
  CycleResult r;
  if (!have_ego_ || !ext_.valid) return r;
  const double dt = last_radar_us_ < 0 ? 0.0 : (scan.t_us - last_radar_us_) * 1e-6;
  last_radar_us_ = scan.t_us;

  const std::vector<Detection> dets = cluster(to_vehicle_frame(scan, ext_, p_), p_);
  tracker_.step(dets, dt, ego_.yaw_rate);
  r.events = fcw_.step(scan.t_us, tracker_.mutable_tracks(), ego_);
  r.ran = true;

  // Fixed precision: the digest must not depend on printf rounding of the last bits.
  char buf[160];
  std::snprintf(buf, sizeof(buf), "{\"t_us\":%lld,\"ego_v\":%.3f,\"dets\":%zu,\"tracks\":%zu,\"warn\":%d,\"brake\":%d,"
                "\"threat\":%d,\"ttc\":%.3f",
                static_cast<long long>(scan.t_us), ego_.speed, dets.size(), tracker_.tracks().size(),
                fcw_.warning() ? 1 : 0, fcw_.brake() ? 1 : 0, fcw_.threat().track_id,
                fcw_.threat().track_id >= 0 ? fcw_.threat().ttc : -1.0);
  r.trace_line = buf;
  r.trace_line += ",\"confirmed\":[";
  bool first = true;
  for (const Track& t : tracker_.tracks()) {
    if (!t.confirmed) continue;
    std::snprintf(buf, sizeof(buf), "%s[%d,%.2f,%.2f,%.2f,%.2f]", first ? "" : ",", t.id, t.s[0], t.s[1], t.s[2], t.s[3]);
    r.trace_line += buf;
    first = false;
  }
  r.trace_line += "]}";
  return r;
}

}  // namespace dr
