// Forward collision warning: pick the most critical confirmed track in the ego path,
// compute time to collision, and run the warning / brake-request state machine.
#pragma once

#include <vector>

#include "stack/msgs.h"
#include "stack/params.h"

namespace dr {

struct Threat {
  int track_id = -1;
  double ttc = 1e9, range = 0, closing = 0;
};

class Fcw {
 public:
  explicit Fcw(const Params& p) : p_(p) {}

  std::vector<FcwEvent> step(int64_t t_us, std::vector<Track>& tracks, const EgoState& ego);

  bool warning() const { return warning_; }
  bool brake() const { return brake_; }
  const Threat& threat() const { return threat_; }

 private:
  bool in_path(const Track& t, const EgoState& ego) const;

  Params p_;
  bool warning_ = false, brake_ = false;
  int warn_count_ = 0, release_count_ = 0;
  int warned_track_ = -1;
  Threat threat_;
};

}  // namespace dr
