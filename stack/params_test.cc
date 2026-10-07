#include "stack/params.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>

namespace dr {
namespace {

std::string write_tmp(const std::string& body) {
  const std::string path = std::string(std::getenv("TEST_TMPDIR")) + "/cfg.json";
  std::ofstream(path) << body;
  return path;
}

TEST(Params, LoadsKnownKeysAndIgnoresComments) {
  const Params p = load_params(write_tmp(R"({"_comment": "x", "ttc_warn_s": 2.5, "confirm_hits": 4})"));
  EXPECT_DOUBLE_EQ(p.ttc_warn_s, 2.5);
  EXPECT_EQ(p.confirm_hits, 4);
  EXPECT_DOUBLE_EQ(p.ttc_brake_s, Params{}.ttc_brake_s);
}

TEST(Params, RejectsUnknownKeys) {
  EXPECT_THROW(load_params(write_tmp(R"({"ttc_warn": 2.5})")), std::runtime_error);
}

TEST(Params, PathModelIsOptInAndNeedsAVehicleModel) {
  EXPECT_EQ(Params{}.path_model, 0);
  EXPECT_THROW(load_params(write_tmp(R"({"path_model": 1})")), std::runtime_error);
  EXPECT_THROW(load_params(write_tmp(R"({"path_model": 2})")), std::runtime_error);
  const Params p = load_params(write_tmp(R"({"path_model": 0, "vehicle_id": "n015"})"));
  EXPECT_EQ(p.path_model, 0);
}

}  // namespace
}  // namespace dr
