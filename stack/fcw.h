// Forward collision warning: pick the most critical confirmed track in the ego path,
// compute time to collision, and run the warning / brake-request state machine.
#pragma once

#include <memory>
#include <vector>

#include "stack/msgs.h"
#include "stack/params.h"
#include "stack/vdyn/vdyn.h"

namespace dr {

struct Threat {
  int track_id = -1;
  double ttc = 1e9, range = 0, closing = 0;
};

class Fcw {
 public:
  explicit Fcw(const Params& p) : p_(p) {
    if (p_.path_model == 1) model_ = std::make_shared<vdyn::LinearSingleTrack>(p_.vehicle.dynamic);
  }

  std::vector<FcwEvent> step(int64_t t_us, std::vector<Track>& tracks, const EgoState& ego);

  bool warning() const { return warning_; }
  bool brake() const { return brake_; }
  const Threat& threat() const { return threat_; }
  // cycles in which the corridor came from the single-track model (path_model 1 with a fresh steering angle)
  long single_track_cycles() const { return single_track_cycles_; }

 private:
  bool in_path(const Track& t, const EgoState& ego) const;

  Params p_;
  bool warning_ = false, brake_ = false;
  int warn_count_ = 0, release_count_ = 0;
  int warned_track_ = -1;
  Threat threat_;
  // path_model = 1 only: single-track path predicted once per cycle from steering, speed and yaw rate
  std::shared_ptr<const vdyn::Model> model_;
  vdyn::PredictedPath path_;
  bool use_single_track_ = false;
  long single_track_cycles_ = 0;
};

}  // namespace dr
