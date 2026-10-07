// Vehicle-dynamics section: reads data/vdyn.json (written by scripts/reproduce_vdyn.py).
(function () {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const DEG = 180 / Math.PI;
  const ci = (c, k = 1, nd = 2) => (c && c.value != null ? `${(c.value * k).toFixed(nd)} <span class="note">[${(c.lo * k).toFixed(nd)}, ${(c.hi * k).toFixed(nd)}]</span>` : "-");
  const LABEL = { const_velocity: "constant velocity", const_yaw_rate: "constant yaw rate", fcw_yaw_decay: "FCW 0.2.0 yaw-rate decay",
    kinematic: "kinematic single track", dynamic: "linear dynamic single track", steady_state: "linear steady state (K)",
    kinematic_rec: "kinematic, recorded inputs", dynamic_rec: "dynamic, recorded inputs", single_track: "dynamic, steering held",
    single_track_steer_decay: "dynamic, steering decays (1 s)" };
  fetch("data/vdyn.json").then((r) => r.json()).then((V) => {
    const yaw = ["kinematic", "steady_state", "dynamic"].map((m) => `<tr><td>${LABEL[m]}</td><td>${ci(V.yaw[m].all, DEG, 3)}</td>` +
      `<td>${ci(V.yaw[m]["3-6"], DEG, 3)}</td><td>${ci(V.yaw[m]["6-9"], DEG, 3)}</td><td>${ci(V.yaw[m]["9-12"], DEG, 3)}</td></tr>`).join("");
    const traj = Object.keys(V.traj).map((m) => `<tr><td>${LABEL[m]}</td>` + ["1", "2", "3"].map((h) => `<td>${ci(V.traj[m][h].lat)}</td>`).join("") +
      `<td>${ci(V.traj[m]["3"].fde)}</td></tr>`).join("");
    const corr = Object.keys(V.corridor).map((m) => `<tr><td>${LABEL[m]}</td>` + ["10", "20", "30"].map((d) => `<td>${ci(V.corridor[m][d])}</td>`).join("") + "</tr>").join("");
    const tp = V.throughput.map((t) => `<tr><td>${t.workers}</td><td>${t.scenes}</td><td>${t.wall_s.toFixed(2)} s</td><td>${t.scenes_per_s.toFixed(0)}</td>` +
      `<td>${Math.round(t.x_realtime).toLocaleString("en")}x</td><td><code>${esc(t.batch_digest)}</code></td></tr>`).join("");
    document.getElementById("vdyn").innerHTML = `
      <p class="note">Split: train ${V.split.train}, val ${V.split.val}, test ${V.split.test} scenes (whole logs, seeded). Test: ${V.test_scenes} scenes, ${V.start_points} trajectory start points.
      Understeer gradient identified on train: ${V.understeer_deg_per_g.toFixed(2)} deg/g.</p>
      <h3>Yaw rate from steering + wheel speed vs IMU (RMSE, deg/s)</h3>
      <div class="panel scroll"><table><tr><th>model</th><th>all speeds</th><th>3-6 m/s</th><th>6-9 m/s</th><th>9-12 m/s</th></tr>${yaw}</table></div>
      <img class="fig" src="data/vdyn_yaw_trace.png" alt="yaw rate trace of one test drive: IMU, kinematic and dynamic model">
      <h3>Open-loop ego prediction (mean error, m)</h3>
      <div class="panel scroll"><table><tr><th>model</th><th>lateral 1 s</th><th>lateral 2 s</th><th>lateral 3 s</th><th>displacement 3 s</th></tr>${traj}</table></div>
      <img class="fig" src="data/vdyn_error_vs_horizon.png" alt="error versus prediction horizon per model">
      <h3>FCW corridor centre line vs driven path (mean absolute error, m)</h3>
      <div class="panel scroll"><table><tr><th>model</th><th>10 m</th><th>20 m</th><th>30 m</th></tr>${corr}</table></div>
      <p class="note">With the opt-in single-track corridor the 13-recording replay suite gives the same warnings as without it: no change, not a measured safety benefit.</p>
      <h3>Reprocessing all scenes</h3>
      <div class="panel scroll"><table><tr><th>workers</th><th>scenes</th><th>wall</th><th>scenes/s</th><th>x real time</th><th>batch digest</th></tr>${tp}</table></div>`;
  }).catch(() => {});
})();
