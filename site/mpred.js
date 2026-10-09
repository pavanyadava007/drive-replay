// Ego motion prediction section: reads data/mpred.json (written by scripts/reproduce_mpred.py).
(function () {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const DEG = 180 / Math.PI;
  const ci = (c, k = 1, nd = 3) => (c && c.value != null ? `${(c.value * k).toFixed(nd)} <span class="note">[${(c.lo * k).toFixed(nd)}, ${(c.hi * k).toFixed(nd)}]</span>` : "-");
  fetch("data/mpred.json").then((r) => r.json()).then((V) => {
    const H = V.horizons_ms;
    const cols = [100, 200, 500].map((h) => H.indexOf(h));
    const methods = Object.keys(V.labels);
    const row = (metric, k, nd) => methods.map((m) => `<tr><td>${esc(V.labels[m])}</td>` +
      cols.map((i) => `<td>${ci(V.acc[m][metric][i], k, nd)}</td>`).join("") + "</tr>").join("");
    const head = `<tr><th>method</th>${[100, 200, 500].map((h) => `<th>${h} ms</th>`).join("")}</tr>`;
    const noise = methods.map((m) => `<tr><td>${esc(V.labels[m])}</td>` + V.noise[m].map((v) => `<td>${v.toFixed(4)}</td>`).join("") + "</tr>").join("");
    const cpp = ["cv", "ctrv", "ctra", "kinematic", "dynamic", "ekf", "ridge", "mlp", "hybrid_ridge", "hybrid_mlp", "history_push_and_ekf_step"]
      .filter((m) => V.bench[m]).map((m) => `<tr><td>${esc(V.labels[m] || "history push + EKF step (per sample)")}</td>` +
        `<td>${V.bench[m].p50_ns.toFixed(0)}</td><td>${V.bench[m].p99_ns.toFixed(0)}</td>` +
        `<td>${V.parity[m] ? V.parity[m].pos_m.toExponential(1) : "-"}</td></tr>`).join("");
    const d = V.data;
    document.getElementById("mpred").innerHTML = `
      <p class="note">Split as v0.3.0 (whole logs, seeded): train ${d.train.scenes} scenes (${d.train.hours.toFixed(2)} h), val ${d.val.scenes}, test ${d.test.scenes} scenes
      (${d.test.hours.toFixed(2)} h, ${d.test.start_points.toLocaleString("en")} start points every 0.1 s). Mean |error| on test, 95 % bootstrap intervals over scenes.</p>
      <h3>Overlay offset 20 m ahead (m): lateral error + 20 m x sin(heading error)</h3>
      <div class="panel scroll"><table>${head}${row("ov20", 1, 4)}</table></div>
      <img class="fig" src="data/mpred_error_vs_horizon.png" alt="position error and overlay offset versus prediction horizon per method">
      <h3>Heading error (deg)</h3>
      <div class="panel scroll"><table>${head}${row("head", DEG, 3)}</table></div>
      <h3>Position error (m)</h3>
      <div class="panel scroll"><table>${head}${row("pos", 1, 4)}</table></div>
      <img class="fig" src="data/mpred_scene_trace.png" alt="yaw rate and signed overlay offset over time in one test scene">
      <h3>Noise sensitivity: overlay offset 20 m at 200 ms (m) by injected noise level</h3>
      <div class="panel scroll"><table><tr><th>method</th>${V.noise_levels.map((l) => `<th>level ${l}</th>`).join("")}</tr>${noise}</table></div>
      <img class="fig" src="data/mpred_noise.png" alt="error versus injected noise level per method">
      <h3>C++17 latency per call (one core, all 7 horizons) and parity with the Python prototype</h3>
      <div class="panel scroll"><table><tr><th>method</th><th>p50 ns</th><th>p99 ns</th><th>max position diff vs Python (m)</th></tr>${cpp}</table></div>
      <p class="note">Open-loop replay of recorded drives (two Renault Zoe cars, Boston and Singapore); x86 host timings, not an automotive ECU.
      Generated at commit <code>${esc(V.commit)}</code>.</p>`;
  }).catch(() => {});
})();
