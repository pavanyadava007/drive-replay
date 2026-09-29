// Multi-object tracker: constant-velocity Kalman filter per track in the vehicle frame,
// global nearest-neighbour association (Hungarian), M-of-N confirmation.
#pragma once

#include <vector>

#include "stack/msgs.h"
#include "stack/params.h"

namespace dr {

class Tracker {
 public:
  explicit Tracker(const Params& p) : p_(p) {}

  // Predict all tracks by dt, compensating the ego rotation, then associate and update.
  void step(const std::vector<Detection>& dets, double dt, double ego_yaw_rate);

  const std::vector<Track>& tracks() const { return tracks_; }
  std::vector<Track>& mutable_tracks() { return tracks_; }

 private:
  void predict(Track& t, double dt, double ego_yaw_rate) const;
  void update(Track& t, const Detection& d) const;

  Params p_;
  std::vector<Track> tracks_;
  int next_id_ = 1;
};

}  // namespace dr
