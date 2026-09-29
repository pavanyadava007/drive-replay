// Radar frontend: filter invalid returns, move them into the vehicle frame, cluster them into detections.
#pragma once

#include <vector>

#include "stack/msgs.h"
#include "stack/params.h"

namespace dr {

// Filters one scan and returns its points in the vehicle frame (velocities stay relative to ego).
std::vector<RadarPoint> to_vehicle_frame(const RadarScan& scan, const Extrinsic& ext, const Params& p);

// Single-linkage clustering: points closer than cluster_dist_m with similar velocity end in one detection.
std::vector<Detection> cluster(const std::vector<RadarPoint>& pts, const Params& p);

}  // namespace dr
