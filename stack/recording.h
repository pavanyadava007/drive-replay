// Reads a drive-replay recording (MCAP, JSON messages) in log-time order.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "stack/msgs.h"

namespace dr {

struct RecordingStats {
  int64_t messages = 0, radar = 0, ego = 0, calib = 0, skipped = 0;
  int64_t first_us = -1, last_us = -1;
};

struct RecordingHandlers {
  std::function<void(const Extrinsic&)> calib;
  std::function<void(const EgoState&)> ego;
  std::function<void(const RadarScan&)> radar;
};

// Replays [start_us, end_us] (0 = open) of the recording through the handlers. Topics the stack does not
// subscribe to (/meta, /gt/objects) are skipped, which is what keeps the ground truth out of the stack.
// Calibration is delivered even when it lies before start_us, as a vehicle would load it at boot.
RecordingStats replay_recording(const std::string& path, int64_t start_us, int64_t end_us, const RecordingHandlers& h);

}  // namespace dr
