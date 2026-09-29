#include "stack/tracker.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "stack/hungarian.h"

namespace dr {
namespace {

Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int k = 0; k < 4; ++k)
      for (int j = 0; j < 4; ++j) r[i][j] += a[i][k] * b[k][j];
  return r;
}

Mat4 transpose(const Mat4& a) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) r[i][j] = a[j][i];
  return r;
}

// Gauss-Jordan with partial pivoting; the innovation covariance is SPD, so it never fails in practice.
Mat4 inverse(Mat4 a) {
  Mat4 inv{};
  for (int i = 0; i < 4; ++i) inv[i][i] = 1.0;
  for (int col = 0; col < 4; ++col) {
    int piv = col;
    for (int r = col + 1; r < 4; ++r)
      if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
    if (std::fabs(a[piv][col]) < 1e-12) throw std::runtime_error("singular innovation covariance");
    std::swap(a[col], a[piv]);
    std::swap(inv[col], inv[piv]);
    const double d = a[col][col];
    for (int j = 0; j < 4; ++j) { a[col][j] /= d; inv[col][j] /= d; }
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      const double f = a[r][col];
      for (int j = 0; j < 4; ++j) { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
    }
  }
  return inv;
}

}  // namespace

void Tracker::predict(Track& t, double dt, double ego_yaw_rate) const {
  // constant velocity in the (moving) vehicle frame
  Mat4 F{};
  for (int i = 0; i < 4; ++i) F[i][i] = 1.0;
  F[0][2] = dt;
  F[1][3] = dt;
  Vec4 s{t.s[0] + dt * t.s[2], t.s[1] + dt * t.s[3], t.s[2], t.s[3]};
  // the vehicle frame turned by ego_yaw_rate * dt; rotate the state the other way
  const double a = -ego_yaw_rate * dt, c = std::cos(a), sn = std::sin(a);
  t.s = {c * s[0] - sn * s[1], sn * s[0] + c * s[1], c * s[2] - sn * s[3], sn * s[2] + c * s[3]};
  Mat4 R{};
  R[0][0] = c; R[0][1] = -sn; R[1][0] = sn; R[1][1] = c;
  R[2][2] = c; R[2][3] = -sn; R[3][2] = sn; R[3][3] = c;
  const Mat4 RF = mul(R, F);
  Mat4 P = mul(mul(RF, t.P), transpose(RF));
  // white-noise acceleration
  const double q = p_.accel_noise * p_.accel_noise, dt2 = dt * dt, dt3 = dt2 * dt / 2.0, dt4 = dt2 * dt2 / 4.0;
  P[0][0] += q * dt4; P[1][1] += q * dt4; P[0][2] += q * dt3; P[2][0] += q * dt3;
  P[1][3] += q * dt3; P[3][1] += q * dt3; P[2][2] += q * dt2; P[3][3] += q * dt2;
  t.P = P;
}

void Tracker::update(Track& t, const Detection& d) const {
  // H = I: the radar measures position and relative velocity
  Mat4 S = t.P;
  const double r[4] = {p_.meas_pos_std, p_.meas_pos_std, p_.meas_vx_std, p_.meas_vy_std};
  for (int i = 0; i < 4; ++i) S[i][i] += r[i] * r[i];
  const Mat4 K = mul(t.P, inverse(S));
  const Vec4 z{d.x, d.y, d.vx, d.vy};
  Vec4 y{};
  for (int i = 0; i < 4; ++i) y[i] = z[i] - t.s[i];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) t.s[i] += K[i][j] * y[j];
  Mat4 IK{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) IK[i][j] = (i == j ? 1.0 : 0.0) - K[i][j];
  t.P = mul(IK, t.P);
}

void Tracker::step(const std::vector<Detection>& dets, double dt, double ego_yaw_rate) {
  for (Track& t : tracks_) predict(t, dt, ego_yaw_rate);

  std::vector<std::vector<double>> cost(tracks_.size(), std::vector<double>(dets.size()));
  for (size_t i = 0; i < tracks_.size(); ++i)
    for (size_t j = 0; j < dets.size(); ++j)
      cost[i][j] = std::hypot(tracks_[i].s[0] - dets[j].x, tracks_[i].s[1] - dets[j].y) +
                   0.2 * std::fabs(tracks_[i].s[2] - dets[j].vx);
  const std::vector<int> match = solve_assignment(cost, p_.gate_m);

  std::vector<char> used(dets.size(), 0);
  for (size_t i = 0; i < tracks_.size(); ++i) {
    Track& t = tracks_[i];
    t.age += 1;
    if (match[i] >= 0) {
      update(t, dets[match[i]]);
      used[match[i]] = 1;
      t.hits += 1;
      t.consecutive_hits += 1;
      t.misses = 0;
      if (t.consecutive_hits >= p_.confirm_hits) t.confirmed = true;
    } else {
      t.misses += 1;
      t.consecutive_hits = 0;
    }
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) { return t.misses >= p_.delete_misses; }),
                tracks_.end());
  for (size_t j = 0; j < dets.size(); ++j) {
    if (used[j]) continue;
    Track t;
    t.id = next_id_++;
    t.s = {dets[j].x, dets[j].y, dets[j].vx, dets[j].vy};
    const double pv = 1.0, vv = 4.0;
    t.P[0][0] = pv; t.P[1][1] = pv; t.P[2][2] = vv; t.P[3][3] = vv * 4;
    t.hits = 1;
    t.consecutive_hits = 1;
    tracks_.push_back(t);
  }
}

}  // namespace dr
