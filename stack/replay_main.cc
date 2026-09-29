// replay_main: runs one recording (or a window of it) through the stack on x86, as fast as it can read.
//
//   replay_main --recording scene-0061.mcap --config resolved_config.json --out runs/x [--start_us N] [--end_us N]
//
// Writes <out>/events.jsonl, <out>/trace.jsonl and <out>/summary.json. The digests in summary.json cover only
// what the stack decided, never wall-clock timing, so two runs of the same drop on the same recording must match.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "stack/build_info.h"
#include "stack/params.h"
#include "stack/pipeline.h"
#include "stack/recording.h"

namespace {

struct Fnv1a {
  uint64_t h = 1469598103934665603ULL;
  void add(const std::string& s) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    h ^= '\n'; h *= 1099511628211ULL;
  }
  std::string hex() const { char b[17]; std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(h)); return b; }
};

std::string event_json(const dr::FcwEvent& e) {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "{\"t_us\":%lld,\"kind\":\"%s\",\"track\":%d,\"ttc\":%.3f,\"range\":%.2f,\"closing\":%.2f,\"ego_v\":%.2f}",
                static_cast<long long>(e.t_us), e.kind.c_str(), e.track_id, e.ttc, e.range, e.closing_speed, e.ego_speed);
  return buf;
}

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}

int usage() {
  std::cerr << "usage: replay_main --recording FILE --config FILE --out DIR [--start_us N] [--end_us N] [--version]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string recording, config, out;
  int64_t start_us = 0, end_us = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
    if (a == "--recording") recording = next();
    else if (a == "--config") config = next();
    else if (a == "--out") out = next();
    else if (a == "--start_us") start_us = std::stoll(next());
    else if (a == "--end_us") end_us = std::stoll(next());
    else if (a == "--version") { std::cout << dr::kVersion << " " << dr::kGitCommit << "\n"; return 0; }
    else return usage();
  }
  if (recording.empty() || config.empty() || out.empty()) return usage();

  try {
    const dr::Params params = dr::load_params(config);
    dr::Pipeline pipeline(params);
    std::filesystem::create_directories(out);
    std::ofstream events(out + "/events.jsonl"), trace(out + "/trace.jsonl");
    Fnv1a ev_digest, state_digest;
    int64_t cycles = 0, idle = 0, n_events = 0;
    std::vector<double> cycle_ms;

    dr::RecordingHandlers h;
    h.calib = [&](const dr::Extrinsic& e) { pipeline.on_calibration(e); };
    h.ego = [&](const dr::EgoState& e) { pipeline.on_ego(e); };
    h.radar = [&](const dr::RadarScan& s) {
      const auto t0 = std::chrono::steady_clock::now();
      const dr::CycleResult r = pipeline.on_radar(s);
      cycle_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      if (!r.ran) { idle += 1; return; }
      cycles += 1;
      trace << r.trace_line << "\n";
      state_digest.add(r.trace_line);
      for (const dr::FcwEvent& e : r.events) {
        const std::string line = event_json(e);
        events << line << "\n";
        ev_digest.add(line);
        n_events += 1;
      }
    };
    const auto wall0 = std::chrono::steady_clock::now();
    const dr::RecordingStats st = dr::replay_recording(recording, start_us, end_us, h);
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "{\n \"stack_version\": \"%s\",\n \"git_commit\": \"%s\",\n \"cycles\": %lld,\n \"idle_cycles\": %lld,\n"
                  " \"events\": %lld,\n \"event_digest\": \"%s\",\n \"state_digest\": \"%s\",\n"
                  " \"messages\": %lld,\n \"radar_msgs\": %lld,\n \"ego_msgs\": %lld,\n"
                  " \"first_us\": %lld,\n \"last_us\": %lld,\n \"start_us\": %lld,\n \"end_us\": %lld,\n"
                  " \"timing\": {\"wall_s\": %.4f, \"cycle_ms_p50\": %.4f, \"cycle_ms_p99\": %.4f, \"cycle_ms_max\": %.4f}\n}\n",
                  dr::kVersion, dr::kGitCommit, static_cast<long long>(cycles), static_cast<long long>(idle),
                  static_cast<long long>(n_events), ev_digest.hex().c_str(), state_digest.hex().c_str(),
                  static_cast<long long>(st.messages), static_cast<long long>(st.radar), static_cast<long long>(st.ego),
                  static_cast<long long>(st.first_us), static_cast<long long>(st.last_us), static_cast<long long>(start_us),
                  static_cast<long long>(end_us), wall_s, pct(cycle_ms, 0.5), pct(cycle_ms, 0.99), pct(cycle_ms, 1.0));
    std::ofstream(out + "/summary.json") << buf;
    std::cout << buf;
  } catch (const std::exception& e) {
    std::cerr << "replay_main: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
