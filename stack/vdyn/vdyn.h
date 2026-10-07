// Vehicle dynamics models for path prediction and validation against recorded CAN data.
//
// Frame and sign conventions (ISO 8855 style, as nuScenes): x forward, y left, yaw counter-clockwise positive,
// steering angle positive to the left. The reference point of every model is the middle of the rear axle,
// which is also the origin of the nuScenes ego frame, so predicted and recorded positions compare directly.
//
// Models (all integrated with a fixed-step classical RK4; speed is an input, not a state):
//   KinematicBicycle   yaw rate = v tan(delta) / L, no slip
//   LinearSingleTrack  linear dynamic single-track ("bicycle") model with axle cornering stiffnesses; states
//                      lateral velocity at the centre of gravity and yaw rate; steady-state yaw rate
//                      v delta / (L (1 + K v^2)) with K = m (lr/Cf - lf/Cr) / L^2 (understeer gradient)
//   ConstantYawRate    yaw rate held at its initial value (CTRV)
//   ConstantVelocity   straight line at the initial heading
//   YawRateDecay       yaw rate decays to zero with time constant tau; the path assumption of the FCW in
//                      stack 0.2.0 (BUG-0003), including its curvature clamp
// Not modelled: tyre saturation, combined slip, load transfer, roll, pitch, actuator or steering dynamics.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace dr::vdyn {

struct State {
  double x = 0, y = 0;  // rear-axle middle, m
  double psi = 0;       // heading, rad
  double vy = 0;        // lateral velocity at the centre of gravity, m/s (dynamic model only)
  double r = 0;         // yaw rate, rad/s
};

// Inputs at one instant: road-wheel steering angle and longitudinal speed (prescribed, from the wheel speeds).
struct Input {
  double delta = 0;  // rad
  double v = 0;      // m/s
};

struct VehicleParams {
  double wheelbase_m = 2.588;   // Renault Zoe
  double lf_m = 1.165;          // centre of gravity to front axle
  double mass_kg = 1650.0;
  double yaw_inertia_kgm2 = 2740.0;
  double cf_npr = 90000.0;      // front axle cornering stiffness, N/rad
  double cr_npr = 110000.0;     // rear axle cornering stiffness, N/rad
  double min_speed_mps = 1.0;   // below this the dynamic model falls back to kinematic (slip angles undefined)
  double lr() const { return wheelbase_m - lf_m; }
  // Understeer gradient K in s^2/m^2: steady-state yaw rate = v delta / (L (1 + K v^2)).
  double understeer_K() const;
};

// Steering-wheel angle to road-wheel angle: (steer_sw - offset) / ratio.
struct SteeringMap {
  double ratio = 15.0;
  double offset_rad = 0.0;
  double road_wheel(double steer_sw) const { return (steer_sw - offset_rad) / ratio; }
};

// Identified parameter set read from configs/vdyn/<vehicle>.json (written by tools/vdyn/identify.py).
struct VehicleConfig {
  struct Offsets { double kinematic = 0, dynamic = 0; };
  double wheel_radius_m = 0.30;
  SteeringMap kinematic_steering;  // fitted with the kinematic model (K = 0)
  SteeringMap dynamic_steering;    // fitted together with the understeer gradient
  VehicleParams dynamic;
  double fcw_decay_tau_s = 1.0;     // stack 0.2.0 defaults, used by the YawRateDecay baseline
  double fcw_max_curvature = 0.2;
  // steering-wheel sensor offset per car (nuScenes n008, n015); ratio and stiffness are shared
  std::map<std::string, Offsets> offsets_by_vehicle;
  // The config with the offsets of one car; unknown or empty id keeps the shared offsets.
  VehicleConfig for_vehicle(const std::string& id) const;
};

// Throws std::runtime_error when the file cannot be read or a required key is missing.
VehicleConfig load_vehicle_config(const std::string& path);

class Model {
 public:
  virtual ~Model() = default;
  virtual const char* name() const = 0;
  // Advances s by dt with the input moving linearly from u0 to u1 over the step.
  virtual void step(State& s, const Input& u0, const Input& u1, double dt) const = 0;
  // Sets the model-specific internal state (vy, r) from a measured yaw rate before a prediction starts.
  virtual void init(State& s, const Input& u, double measured_yaw_rate) const;
};

class KinematicBicycle : public Model {
 public:
  explicit KinematicBicycle(double wheelbase_m) : L_(wheelbase_m) {}
  const char* name() const override { return "kinematic"; }
  void step(State& s, const Input& u0, const Input& u1, double dt) const override;
  void init(State& s, const Input& u, double measured_yaw_rate) const override;

 private:
  double L_;
};

class LinearSingleTrack : public Model {
 public:
  explicit LinearSingleTrack(const VehicleParams& p) : p_(p) {}
  const char* name() const override { return "dynamic"; }
  void step(State& s, const Input& u0, const Input& u1, double dt) const override;
  void init(State& s, const Input& u, double measured_yaw_rate) const override;
  const VehicleParams& params() const { return p_; }

 private:
  void deriv(const State& s, const Input& u, State& d) const;
  VehicleParams p_;
};

class ConstantYawRate : public Model {
 public:
  const char* name() const override { return "const_yaw_rate"; }
  void step(State& s, const Input& u0, const Input& u1, double dt) const override;
};

class ConstantVelocity : public Model {
 public:
  const char* name() const override { return "const_velocity"; }
  void step(State& s, const Input& u0, const Input& u1, double dt) const override;
  void init(State& s, const Input& u, double measured_yaw_rate) const override;
};

class YawRateDecay : public Model {
 public:
  YawRateDecay(double tau_s, double max_curvature) : tau_(tau_s), max_k_(max_curvature) {}
  const char* name() const override { return "fcw_yaw_decay"; }
  void step(State& s, const Input& u0, const Input& u1, double dt) const override;
  void init(State& s, const Input& u, double measured_yaw_rate) const override;

 private:
  double tau_, max_k_;
};

// Steady-state yaw rates (closed form).
double kinematic_yaw_rate(double wheelbase_m, double delta, double v);
double linear_steady_yaw_rate(double wheelbase_m, double K, double delta, double v);

// Predicted path of the rear axle in the current vehicle frame (origin at the rear axle, x forward), sampled
// every dt over horizon_s. Speed is held; the steering angle is held (steer_decay_s = 0) or decays to zero with
// time constant steer_decay_s (the driver straightens out, as the FCW assumes for the yaw rate). offset(x) returns the lateral position of the path where it
// reaches longitudinal distance x (linear interpolation; straight extrapolation along the last heading).
class PredictedPath {
 public:
  void predict(const Model& m, const Input& u, double measured_yaw_rate, double horizon_s, double dt,
               double steer_decay_s = 0.0);
  double offset(double x) const;
  const std::vector<State>& points() const { return pts_; }

 private:
  std::vector<State> pts_;
};

// The FCW corridor centre line of stack 0.2.0: lateral offset at longitudinal distance x for a yaw rate that
// decays with time constant tau (small-angle form). Moved here unchanged from stack/fcw.cc.
double decay_path_offset(double speed, double yaw_rate, double x, double max_curvature, double tau);

}  // namespace dr::vdyn
