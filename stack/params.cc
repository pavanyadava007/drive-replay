#include "stack/params.h"

#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace dr {

Params load_params(const std::string& json_path) {
  std::ifstream in(json_path);
  if (!in) throw std::runtime_error("cannot open config " + json_path);
  const nlohmann::json j = nlohmann::json::parse(in);
  Params p;
#define DR_PARAM(name)                        \
  if (it.key() == #name) {                    \
    p.name = it.value().get<decltype(p.name)>(); \
    continue;                                 \
  }
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key().rfind("_", 0) == 0) continue;  // "_comment" style keys
    DR_PARAM(max_range_m) DR_PARAM(required_ambig_state) DR_PARAM(max_dyn_prop)
    DR_PARAM(cluster_dist_m) DR_PARAM(cluster_dv_mps)
    DR_PARAM(gate_m) DR_PARAM(confirm_hits) DR_PARAM(delete_misses) DR_PARAM(accel_noise)
    DR_PARAM(meas_pos_std) DR_PARAM(meas_vx_std) DR_PARAM(meas_vy_std)
    DR_PARAM(min_ego_speed_mps) DR_PARAM(corridor_half_width_m) DR_PARAM(max_curvature) DR_PARAM(oncoming_speed_mps) DR_PARAM(front_bumper_m)
    DR_PARAM(min_closing_mps) DR_PARAM(ttc_warn_s) DR_PARAM(ttc_release_s) DR_PARAM(ttc_brake_s)
    DR_PARAM(warn_confirm_cycles) DR_PARAM(release_cycles)
    throw std::runtime_error("unknown config key: " + it.key());
  }
#undef DR_PARAM
  return p;
}

}  // namespace dr
