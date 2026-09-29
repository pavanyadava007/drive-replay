#include "stack/fcw.h"

#include <cmath>

namespace dr {

bool Fcw::in_path(const Track& t, const EgoState& /*ego*/) const {
  // straight corridor ahead of the car
  return std::fabs(t.s[1]) < p_.corridor_half_width_m;
}

std::vector<FcwEvent> Fcw::step(int64_t t_us, std::vector<Track>& tracks, const EgoState& ego) {
  std::vector<FcwEvent> events;
  threat_ = Threat{};
  const bool active = ego.speed >= p_.min_ego_speed_mps;
  for (Track& t : tracks) {
    const bool path = in_path(t, ego);
    t.in_path_cycles = path ? t.in_path_cycles + 1 : 0;
    if (!active || !t.confirmed || !path) continue;
    const double range = t.s[0] - p_.front_bumper_m;
    const double closing = -t.s[2];
    if (range <= 0.0 || closing < p_.min_closing_mps) continue;
    const double ttc = range / closing;
    if (ttc < threat_.ttc) threat_ = Threat{t.id, ttc, range, closing};
  }

  auto emit = [&](const char* kind) {
    events.push_back(FcwEvent{t_us, kind, threat_.track_id >= 0 ? threat_.track_id : warned_track_,
                              threat_.track_id >= 0 ? threat_.ttc : 0.0, threat_.range, threat_.closing, ego.speed});
  };

  warn_count_ = threat_.ttc < p_.ttc_warn_s ? warn_count_ + 1 : 0;
  if (!warning_ && warn_count_ >= p_.warn_confirm_cycles) {
    warning_ = true;
    release_count_ = 0;
    warned_track_ = threat_.track_id;
    emit("fcw_warning_on");
  } else if (warning_) {
    release_count_ = threat_.ttc > p_.ttc_release_s ? release_count_ + 1 : 0;
    if (release_count_ >= p_.release_cycles) {
      if (brake_) { brake_ = false; emit("brake_request_off"); }
      warning_ = false;
      emit("fcw_warning_off");
      warned_track_ = -1;
    }
  }
  if (warning_ && !brake_ && threat_.ttc < p_.ttc_brake_s) {
    brake_ = true;
    emit("brake_request_on");
  } else if (brake_ && threat_.ttc > p_.ttc_brake_s + 0.3) {
    brake_ = false;
    emit("brake_request_off");
  }
  return events;
}

}  // namespace dr
