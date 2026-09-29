#include "stack/hungarian.h"

#include <algorithm>
#include <limits>

namespace dr {

std::vector<int> solve_assignment(const std::vector<std::vector<double>>& cost, double forbidden) {
  const int rows = static_cast<int>(cost.size());
  if (rows == 0) return {};
  const int cols = static_cast<int>(cost[0].size());
  std::vector<int> result(rows, -1);
  if (cols == 0) return result;

  // Square matrix; padding cells and forbidden pairs cost `big`, so they are only used when nothing else fits.
  const int n = std::max(rows, cols);
  const double big = forbidden * 4.0 + 1.0;
  auto c = [&](int i, int j) {
    if (i >= rows || j >= cols) return big;
    return cost[i][j] >= forbidden ? big : cost[i][j];
  };

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0), v(n + 1, 0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);  // p[j] = row matched to column j (1-based)
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, inf);
    std::vector<char> used(n + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double cur = c(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  for (int j = 1; j <= n; ++j) {
    const int i = p[j] - 1, col = j - 1;
    if (i < rows && col < cols && cost[i][col] < forbidden) result[i] = col;
  }
  return result;
}

}  // namespace dr
