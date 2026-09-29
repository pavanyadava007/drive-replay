// Minimum-cost assignment (Kuhn-Munkres with potentials, O(n^2 m)).
#pragma once

#include <vector>

namespace dr {

// cost is rows x cols (rows may differ from cols). Pairs whose cost is >= forbidden are never assigned.
// Returns for every row the assigned column, or -1.
std::vector<int> solve_assignment(const std::vector<std::vector<double>>& cost, double forbidden);

}  // namespace dr
