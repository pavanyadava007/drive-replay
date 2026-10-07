// The stack under test: radar frontend -> tracker -> FCW, driven only by recorded messages.
#pragma once

#include <string>
#include <vector>

#include "stack/fcw.h"
#include "stack/msgs.h"
#include "stack/params.h"
#include "stack/tracker.h"

namespace dr {

struct CycleResult {
  bool ran = false;  // false when ego state or calibration was not available yet
  std::vector<FcwEvent> events;
  std::string trace_line;  // canonical text of the cycle state; feeds the state digest
};

class Pipeline {
 public:
  explicit Pipeline(const Params& p) : p_(p), tracker_(p), fcw_(p) {}

  void on_calibration(const Extrinsic& e) { ext_ = e; }
  void on_ego(const EgoState& e) { ego_ = e; have_ego_ = true; }
  void on_can(const VehicleCan& c) { can_ = c; have_can_ = true; }
  CycleResult on_radar(const RadarScan& scan);
  long single_track_cycles() const { return fcw_.single_track_cycles(); }

 private:
  Params p_;
  Tracker tracker_;
  Fcw fcw_;
  Extrinsic ext_;
  EgoState ego_;
  bool have_ego_ = false;
  VehicleCan can_;
  bool have_can_ = false;
  int64_t last_radar_us_ = -1;
};

}  // namespace dr
