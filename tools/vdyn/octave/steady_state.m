% Independent cross-check of the C++ linear single-track model in GNU Octave (not MATLAB, not Simulink).
%   octave --no-gui steady_state.m <vehicle config json> <csv from vdyn_steady>
% Builds the state-space matrices of the model from the identified parameters, solves the steady state
% 0 = A x + B delta for x = [vy; r] with Octave's linear solver, and compares r with the C++ simulation.
args = argv();
cfg = jsondecode(fileread(args{1}));
d = cfg.dynamic;
m = d.mass_kg; Iz = d.yaw_inertia_kgm2; cf = d.cf_npr; cr = d.cr_npr;
lf = d.lf_m; lr = cfg.wheelbase_m - lf;
T = csvread(args{2}, 1, 0);
worst_sim = 0; worst_formula = 0;
for k = 1:rows(T)
  v = T(k, 1); delta = T(k, 2);
  A = [-(cf + cr) / (m * v), (lr * cr - lf * cf) / (m * v) - v;
       (lr * cr - lf * cf) / (Iz * v), -(lf^2 * cf + lr^2 * cr) / (Iz * v)];
  B = [cf / m; lf * cf / Iz];
  x = -A \ (B * delta);
  worst_sim = max(worst_sim, abs(x(2) - T(k, 3)) / abs(x(2)));
  worst_formula = max(worst_formula, abs(x(2) - T(k, 4)) / abs(x(2)));
end
printf("octave_version %s\n", OCTAVE_VERSION);
printf("cases %d\n", rows(T));
printf("max_rel_diff_cpp_simulation %.3e\n", worst_sim);
printf("max_rel_diff_cpp_closed_form %.3e\n", worst_formula);
