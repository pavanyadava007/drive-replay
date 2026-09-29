#include "stack/radar_frontend.h"

#include <cmath>
#include <numeric>

namespace dr {

std::vector<RadarPoint> to_vehicle_frame(const RadarScan& scan, const Extrinsic& ext, const Params& p) {
  std::vector<RadarPoint> out;
  out.reserve(scan.points.size());
  const double c = std::cos(ext.yaw), s = std::sin(ext.yaw);
  for (const RadarPoint& in : scan.points) {
    if (in.invalid_state != 0 || in.dyn_prop > p.max_dyn_prop || in.ambig_state != p.required_ambig_state) continue;
    RadarPoint v = in;
    v.x = c * in.x - s * in.y + ext.tx;
    v.y = s * in.x + c * in.y + ext.ty;
    v.vx = c * in.vx - s * in.vy;
    v.vy = s * in.vx + c * in.vy;
    v.vx_comp = c * in.vx_comp - s * in.vy_comp;
    v.vy_comp = s * in.vx_comp + c * in.vy_comp;
    if (std::hypot(v.x, v.y) > p.max_range_m) continue;
    out.push_back(v);
  }
  return out;
}

namespace {
int find(std::vector<int>& parent, int i) {
  while (parent[i] != i) i = parent[i] = parent[parent[i]];
  return i;
}
}  // namespace

std::vector<Detection> cluster(const std::vector<RadarPoint>& pts, const Params& p) {
  const int n = static_cast<int>(pts.size());
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double d = std::hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      const double dv = std::hypot(pts[i].vx - pts[j].vx, pts[i].vy - pts[j].vy);
      if (d < p.cluster_dist_m && dv < p.cluster_dv_mps) {
        const int a = find(parent, i), b = find(parent, j);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);  // smallest index is the root: order-stable
      }
    }
  }
  // Detections come out ordered by their root point index, so the output does not depend on hash order.
  std::vector<int> slot(n, -1);
  std::vector<Detection> dets;
  for (int i = 0; i < n; ++i) {
    const int r = find(parent, i);
    if (slot[r] < 0) { slot[r] = static_cast<int>(dets.size()); dets.emplace_back(); }
    Detection& d = dets[slot[r]];
    d.x += pts[i].x; d.y += pts[i].y; d.vx += pts[i].vx; d.vy += pts[i].vy; d.rcs = std::max(d.rcs, pts[i].rcs);
    d.n_points += 1;
  }
  for (Detection& d : dets) {
    d.x /= d.n_points; d.y /= d.n_points; d.vx /= d.n_points; d.vy /= d.n_points;
  }
  return dets;
}

}  // namespace dr
