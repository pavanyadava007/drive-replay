#include "stack/hungarian.h"

#include <algorithm>
#include <numeric>
#include <random>

#include <gtest/gtest.h>

namespace dr {
namespace {

double brute_force(const std::vector<std::vector<double>>& c, double forbidden, int* matched) {
  // every injective row->column map (rows <= cols), forbidden pairs left unmatched
  const int rows = c.size(), cols = c[0].size();
  std::vector<int> perm(cols);
  std::iota(perm.begin(), perm.end(), 0);
  double best = 1e18;
  int best_m = 0;
  do {
    double s = 0;
    int m = 0;
    for (int i = 0; i < rows; ++i)
      if (c[i][perm[i]] < forbidden) { s += c[i][perm[i]]; ++m; }
    // more matches first, then lower cost
    if (m > best_m || (m == best_m && s < best)) { best = s; best_m = m; }
  } while (std::next_permutation(perm.begin(), perm.end()));
  *matched = best_m;
  return best;
}

TEST(Hungarian, SimpleSquare) {
  const std::vector<std::vector<double>> c = {{4, 1, 3}, {2, 0, 5}, {3, 2, 2}};
  EXPECT_EQ(solve_assignment(c, 100), (std::vector<int>{1, 0, 2}));
}

TEST(Hungarian, RectangularAndForbidden) {
  const std::vector<std::vector<double>> c = {{0.5, 9.0}, {9.0, 9.0}, {9.0, 0.2}};
  EXPECT_EQ(solve_assignment(c, 2.5), (std::vector<int>{0, -1, 1}));
}

TEST(Hungarian, EmptyInputs) {
  EXPECT_TRUE(solve_assignment({}, 1.0).empty());
  EXPECT_EQ(solve_assignment({{}, {}}, 1.0), (std::vector<int>{-1, -1}));
}

TEST(Hungarian, MatchesBruteForceOnRandomMatrices) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0.0, 4.0);
  for (int trial = 0; trial < 300; ++trial) {
    const int rows = 1 + trial % 5, cols = rows + trial % 3;
    std::vector<std::vector<double>> c(rows, std::vector<double>(cols));
    for (auto& r : c) for (double& x : r) x = u(rng);
    int want_m = 0;
    const double want = brute_force(c, 2.5, &want_m);
    const std::vector<int> got = solve_assignment(c, 2.5);
    double s = 0;
    int m = 0;
    std::vector<int> seen;
    for (int i = 0; i < rows; ++i)
      if (got[i] >= 0) { s += c[i][got[i]]; ++m; seen.push_back(got[i]); }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end()) << "column used twice";
    EXPECT_EQ(m, want_m) << "trial " << trial;
    EXPECT_NEAR(s, want, 1e-9) << "trial " << trial;
  }
}

}  // namespace
}  // namespace dr
