// Interactive AR-HUD demo of the v0.4.0 motion-prediction benchmark.
// Data: data/hud/*.json written by scripts/export_mpred_demo.py (the real fitted predictors on test scenes) and
// data/mpred.json (report aggregates). No number on this page is computed here except the two formulas of
// tools/mpred/metrics.py: position error = |(lon, lat)| and 20 m overlay offset = lat + 20 sin(heading error).
(function () {
  "use strict";
  const H_MS = [50, 100, 150, 200, 300, 500];
  const OV_D = 20.0;
  const EYE_FWD = 1.0; // driver eye about 1 m ahead of the rear axle (report), height below
  const EYE_H = 1.2;
  const DEG = 180 / Math.PI;
  const ROWS = ["stale", "cv", "ctrv", "dynamic", "ekf", "mlp", "hybrid_ridge", "hybrid_mlp"];
  const SHORT = { stale: "no compensation", cv: "CV", ctrv: "CTRV", dynamic: "dynamic single track", ekf: "EKF-CTRA",
    mlp: "MLP (pure data-driven)", hybrid_ridge: "EKF + ridge", hybrid_mlp: "EKF + MLP" };
  const TINY = { stale: "no comp.", cv: "CV", ctrv: "CTRV", dynamic: "dyn. ST", ekf: "EKF", mlp: "MLP",
    hybrid_ridge: "EKF+ridge", hybrid_mlp: "EKF+MLP" };
  // the driver view is always a night scene: dark-surface steps of the same palette
  const DARK = { stale: "#c3c2b7", cv: "#3987e5", ctrv: "#d95926", dynamic: "#c98500", ekf: "#d55181", mlp: "#199e70",
    hybrid_ridge: "#1f9a1f", hybrid_mlp: "#9085e9" };

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const DEMO = params.get("demo");
  const S = {
    index: null, agg: null, doc: null, cache: {}, p: 0, playing: false, speed: 1, h: 3, level: "0",
    on: new Set((params.get("methods") || "stale,ctrv,hybrid_ridge,hybrid_mlp").split(",").filter((m) => ROWS.includes(m))),
    last: null,
  };
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const color = (m) => css("--s-" + m);

  // ---- theme ------------------------------------------------------------------------------------------------------
  try {
    const saved = localStorage.getItem("hud-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* storage unavailable */ }
  $("theme").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("hud-theme", next); } catch (e) { /* ignore */ }
    draw();
  });

  // ---- data -------------------------------------------------------------------------------------------------------
  function decode(raw) {
    const f = (a, s) => Float64Array.from(a, (v) => v / s);
    const tr = raw.track;
    const d = {
      raw, scene: raw.scene, label: raw.label, rule: raw.rule,
      t: f(tr.t_ms, 1e3), x: f(tr.x_mm, 1e3), y: f(tr.y_mm, 1e3), yaw: f(tr.yaw_1e5, 1e5), v: f(tr.v_cms, 1e2), r: f(tr.r_1e4, 1e4),
      st: f(raw.starts.t_ms, 1e3), k: raw.starts.k, lv: {},
    };
    for (const [lev, L] of Object.entries(raw.levels)) {
      d.lv[lev] = { means: L.means, rows: {} };
      for (const m of ROWS) {
        const R = L.rows[m];
        const lon = f(R.lon, 1e5), lat = f(R.lat, 1e5), head = f(R.head, 1e7);
        const pos = lon.map((v, i) => Math.hypot(v, lat[i]));
        const ov = lat.map((v, i) => v + OV_D * Math.sin(head[i]));
        d.lv[lev].rows[m] = { lon, lat, head, pos, ov };
      }
    }
    // scene-wide maxima for stable scales (per level and horizon)
    d.maxOv = {}; d.maxPos = {};
    for (const lev of Object.keys(d.lv)) {
      d.maxOv[lev] = {}; d.maxPos[lev] = {};
      for (const m of ROWS) {
        const R = d.lv[lev].rows[m];
        d.maxOv[lev][m] = H_MS.map((_, j) => { let a = 0; for (let i = j; i < R.ov.length; i += 6) a = Math.max(a, Math.abs(R.ov[i])); return a; });
        d.maxPos[lev][m] = H_MS.map((_, j) => { let a = 0; for (let i = j; i < R.pos.length; i += 6) a = Math.max(a, R.pos[i]); return a; });
      }
    }
    return d;
  }

  async function loadScene(file) {
    if (!S.cache[file]) S.cache[file] = fetch("data/hud/" + file).then((r) => r.json()).then(decode);
    return S.cache[file];
  }

  // pose at time tau (s, scene clock): linear interpolation of the recorded track, as tools/mpred/data.interp_pose
  function poseAt(d, tau) {
    const t = d.t;
    if (tau <= t[0]) return { x: d.x[0], y: d.y[0], yaw: d.yaw[0], v: d.v[0], i: 0 };
    const n = t.length;
    if (tau >= t[n - 1]) return { x: d.x[n - 1], y: d.y[n - 1], yaw: d.yaw[n - 1], v: d.v[n - 1], i: n - 1 };
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= tau) lo = mid; else hi = mid; }
    const a = (tau - t[lo]) / (t[hi] - t[lo]);
    const L = (k) => d[k][lo] + a * (d[k][hi] - d[k][lo]);
    return { x: L("x"), y: L("y"), yaw: L("yaw"), v: L("v"), i: lo };
  }

  // camera time = playhead + latency; inside a gap between start points (car below 1 m/s or not in drive) the view
  // stays at the display time of the last start point, so the drawn target always belongs to the view
  function camTime(d) {
    const i = startIndex(d, S.p);
    return Math.min(Math.min(S.p, d.st[i] + 0.12) + H_MS[S.h] / 1000, d.t[d.t.length - 1]);
  }

  function startIndex(d, p) {
    let lo = 0, hi = d.st.length - 1;
    if (p <= d.st[0]) return 0;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (d.st[mid] <= p + 1e-9) lo = mid; else hi = mid - 1; }
    return lo;
  }

  // predicted pose of row m at start i, horizon j: truth + (lon, lat) in the true frame, yaw + heading error
  function predPose(d, m, i, j) {
    const R = d.lv[S.level].rows[m];
    const T = poseAt(d, d.st[i] + H_MS[j] / 1000);
    const q = i * 6 + j, c = Math.cos(T.yaw), s = Math.sin(T.yaw);
    return { x: T.x + c * R.lon[q] - s * R.lat[q], y: T.y + s * R.lon[q] + c * R.lat[q], yaw: T.yaw + R.head[q] };
  }

  // ---- canvas helpers ---------------------------------------------------------------------------------------------
  function fit(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g, w, h };
  }
  const niceUp = (v, steps) => steps.find((s) => s >= v) || steps[steps.length - 1];
  function fmtLen(m) { return m >= 1 ? `${+m.toFixed(1)} m` : `${Math.round(m * 100)} cm`; }

  // ---- driver view ----------------------------------------------------------------------------------------------
  // scale sets: no compensation and CV can be metres off in a turn, so they do not set the zoom of the others
  function scaleSet() {
    const core = ROWS.filter((m) => S.on.has(m) && m !== "stale" && m !== "cv");
    return core.length ? core : ROWS.filter((m) => S.on.has(m));
  }

  function drawHud() {
    const d = S.doc; const { g, w, h } = fit($("hudc"));
    const F = (w / 2) / Math.tan(17 / DEG), hy = h * 0.36, cx = w / 2;
    const tCam = camTime(d);
    const cam = poseAt(d, tCam);
    const c = Math.cos(cam.yaw), s = Math.sin(cam.yaw);
    const ex = cam.x + EYE_FWD * c, ey = cam.y + EYE_FWD * s;
    const proj = (wx, wy) => {
      const dx = wx - ex, dy = wy - ey; const f = c * dx + s * dy, l = -s * dx + c * dy;
      if (f < 1.0) return null; return [cx - F * l / f, hy + F * EYE_H / f, f];
    };
    let gr = g.createLinearGradient(0, 0, 0, hy);
    gr.addColorStop(0, "#0b1120"); gr.addColorStop(1, "#2b3548");
    g.fillStyle = gr; g.fillRect(0, 0, w, hy);
    gr = g.createLinearGradient(0, hy, 0, h);
    gr.addColorStop(0, "#15171b"); gr.addColorStop(1, "#24262b");
    g.fillStyle = gr; g.fillRect(0, hy, w, h - hy);
    const line = (pts, dash, col, wdt) => {
      g.save(); g.setLineDash(dash); g.strokeStyle = col; g.lineWidth = wdt; g.beginPath(); let st = false;
      for (const q of pts) { if (!q) { st = false; continue; } if (!st) { g.moveTo(q[0], q[1]); st = true; } else g.lineTo(q[0], q[1]); }
      g.stroke(); g.restore();
    };
    // world-fixed ground grid (5 m), gives the motion and rotation cues
    const G = 5, R = 70, gx0 = Math.floor((ex - R) / G) * G, gy0 = Math.floor((ey - R) / G) * G;
    for (let a = 0; a <= 2 * R / G; a++) {
      const xs = [], ys = [];
      for (let b = 0; b <= 2 * R; b += 2) { xs.push(proj(gx0 + a * G, gy0 + b)); ys.push(proj(gx0 + b, gy0 + a * G)); }
      line(xs, [], "rgba(255,255,255,0.07)", 1); line(ys, [], "rgba(255,255,255,0.07)", 1);
    }
    // lane band from the recorded future path (3.5 m wide)
    const L = [], Rr = [];
    let dist = 0, prev = null;
    for (let i = Math.max(0, cam.i - 10); i < d.t.length && dist < 140; i++) {
      if (prev) dist += Math.hypot(d.x[i] - prev[0], d.y[i] - prev[1]);
      prev = [d.x[i], d.y[i]];
      const nx = -Math.sin(d.yaw[i]), ny = Math.cos(d.yaw[i]);
      L.push(proj(d.x[i] + 1.75 * nx, d.y[i] + 1.75 * ny)); Rr.push(proj(d.x[i] - 1.75 * nx, d.y[i] - 1.75 * ny));
    }
    for (let i = 1; i < L.length; i++) {
      const q = [L[i - 1], L[i], Rr[i], Rr[i - 1]];
      if (q.some((z) => !z)) continue;
      g.beginPath(); q.forEach((z, n) => (n ? g.lineTo(z[0], z[1]) : g.moveTo(z[0], z[1]))); g.closePath();
      g.fillStyle = "#3b3f47"; g.fill(); g.strokeStyle = "#3b3f47"; g.lineWidth = 1; g.stroke();
    }
    line(L, [10, 8], "rgba(235,233,226,0.85)", 2); line(Rr, [10, 8], "rgba(235,233,226,0.85)", 2);
    // illustrative HUD virtual-image frame: 10 x 4 deg, 1.5 to 5.5 deg below the horizon
    const fx = F * Math.tan(5 / DEG), fy0 = hy + F * Math.tan(1.5 / DEG), fy1 = hy + F * Math.tan(5.5 / DEG);
    g.save(); g.strokeStyle = "rgba(255,255,255,0.3)"; g.setLineDash([5, 5]); g.lineWidth = 1;
    g.strokeRect(cx - fx, fy0, 2 * fx, fy1 - fy0); g.restore();
    g.fillStyle = "rgba(255,255,255,0.6)"; g.font = "11px system-ui, sans-serif"; g.textAlign = "right";
    g.fillText("HUD image area (illustrative, 10 x 4 deg)", cx + fx, fy1 + 14); g.textAlign = "left";
    g.fillStyle = "#ffffff"; g.font = `600 ${Math.max(13, Math.round(w / 48))}px system-ui, sans-serif`;
    g.fillText(`${Math.round(cam.v * 3.6)} km/h`, 12, h - 12);
    g.fillStyle = "rgba(255,255,255,0.6)"; g.font = "11px system-ui, sans-serif";
    if (w > 520) g.fillText(`${d.scene}, display time ${(d.st[startIndex(d, S.p)] + H_MS[S.h] / 1000).toFixed(2)} s`, 12, 18);

    // the target (20 m ahead along the true heading at display time) and the arrow of every method
    const i = startIndex(d, S.p), j = S.h;
    const T = poseAt(d, d.st[i] + H_MS[j] / 1000);
    const tc = proj(T.x + OV_D * Math.cos(T.yaw), T.y + OV_D * Math.sin(T.yaw));
    const arrow = (m) => {
      const P = predPose(d, m, i, j);
      const ox = P.x + OV_D * Math.cos(P.yaw), oy = P.y + OV_D * Math.sin(P.yaw);
      const a = proj(ox, oy), b = proj(ox + 2 * Math.cos(P.yaw), oy + 2 * Math.sin(P.yaw));
      if (!a || !b) return;
      const ang = Math.max(-1.2, Math.min(1.2, Math.atan2(b[0] - a[0], a[1] - b[1])));
      const sz = F * 0.6 / a[2];
      g.save(); g.translate(a[0], a[1]); g.rotate(ang);
      g.beginPath(); g.moveTo(-sz * 0.62, sz * 0.25); g.lineTo(0, -sz * 0.45); g.lineTo(sz * 0.62, sz * 0.25);
      g.lineJoin = "round"; g.lineCap = "round";
      if (m === "stale") { g.setLineDash([sz * 0.16, sz * 0.12]); g.lineWidth = Math.max(2.5, sz * 0.14); g.strokeStyle = DARK[m]; g.stroke(); }
      else { g.lineWidth = Math.max(3.5, sz * 0.2); g.strokeStyle = "rgba(0,0,0,0.55)"; g.stroke(); g.lineWidth = Math.max(2.5, sz * 0.14); g.strokeStyle = DARK[m]; g.stroke(); }
      g.restore();
    };
    for (const m of ROWS) if (S.on.has(m)) arrow(m);
    if (tc) {
      const sz = F * 0.6 / tc[2], q = sz * 0.75, e = sz * 0.28;
      g.strokeStyle = "#ffffff"; g.lineWidth = 2;
      g.beginPath();
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        g.moveTo(tc[0] + sx * q, tc[1] + sy * q - sy * e); g.lineTo(tc[0] + sx * q, tc[1] + sy * q); g.lineTo(tc[0] + sx * q - sx * e, tc[1] + sy * q);
      }
      g.stroke();
      g.fillStyle = "#ffffff"; g.font = "600 12px system-ui, sans-serif"; g.textAlign = "center";
      g.fillText("true target", tc[0], tc[1] - q - 6); g.textAlign = "left";
    }
    if (w > 520) {
      g.fillStyle = "rgba(255,255,255,0.6)"; g.font = "11.5px system-ui, sans-serif";
      g.fillText("Simulated view from recorded nuScenes data, not a Mercedes-Benz HUD", 12, 34);
    }
    // legend inside the view
    let ly = 12; g.font = "12.5px system-ui, sans-serif";
    const lx = w - Math.min(178, w * 0.45);
    for (const m of ROWS) if (S.on.has(m)) {
      g.save(); g.strokeStyle = DARK[m]; g.lineWidth = 2.5; if (m === "stale") g.setLineDash([3, 2]);
      g.beginPath(); g.moveTo(lx, ly + 10); g.lineTo(lx + 6, ly + 2); g.lineTo(lx + 12, ly + 10); g.stroke(); g.restore();
      g.fillStyle = "#ecebe5"; g.fillText(SHORT[m], lx + 18, ly + 10); ly += 18;
    }
  }

  // ---- ruler: lateral offset at the anchor, magnified --------------------------------------------------------------
  function drawRuler() {
    const d = S.doc; const cv = $("ruler");
    const rows = ROWS.filter((m) => S.on.has(m));
    const rowH = 19, top = 30;
    cv.style.height = (top + rows.length * rowH + 22) + "px";
    const { g, w, h } = fit(cv);
    g.clearRect(0, 0, w, h);
    const i = startIndex(d, S.p), j = S.h;
    const span = niceUp(Math.max(0.01, ...scaleSet().map((m) => d.maxOv[S.level][m][j])) * 1.05, [0.05, 0.1, 0.2, 0.5, 1, 2, 5]);
    const narrow = w < 520;
    const x0 = narrow ? 78 : 168, x1 = w - (narrow ? 58 : 92), xc = (x0 + x1) / 2;
    const X = (v) => xc + (v / span) * (x1 - x0) / 2;
    g.fillStyle = css("--ink"); g.font = "600 12.5px system-ui, sans-serif";
    g.fillText(narrow ? "Where the arrow lands (cm, + = left)" : "Where the arrow lands: lateral offset from the true target 20 m ahead (cm, + = left)", 0, 13);
    g.font = "11px system-ui, sans-serif"; g.fillStyle = css("--muted"); g.textAlign = "center";
    for (const f of [-1, -0.5, 0, 0.5, 1]) {
      const x = X(f * span);
      g.fillText(`${f > 0 ? "+" : ""}${+(f * span * 100).toFixed(1)}`, x, top - 4);
      g.fillStyle = f === 0 ? css("--ink") : css("--grid");
      g.fillRect(x - (f === 0 ? 1 : 0.5), top, f === 0 ? 2 : 1, rows.length * rowH);
      g.fillStyle = css("--muted");
    }
    g.textAlign = "left";
    rows.forEach((m, r) => {
      const y = top + r * rowH + rowH / 2;
      const v = d.lv[S.level].rows[m].ov[i * 6 + j];
      g.fillStyle = css("--ink2"); g.font = "12px system-ui, sans-serif";
      g.fillText(narrow ? TINY[m] : SHORT[m], 0, y + 4);
      g.strokeStyle = css("--grid"); g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
      const x = Math.max(x0, Math.min(x1, X(v))), clip = x !== X(v);
      g.save(); if (m === "stale") g.setLineDash([4, 3]);
      g.strokeStyle = color(m); g.lineWidth = 2; g.beginPath(); g.moveTo(X(0), y); g.lineTo(x, y); g.stroke(); g.restore();
      g.fillStyle = color(m); g.beginPath();
      if (clip) { const sg = Math.sign(v); g.moveTo(x + sg * 7, y); g.lineTo(x - sg * 3, y - 6); g.lineTo(x - sg * 3, y + 6); g.closePath(); g.fill(); }
      else { g.arc(x, y, 5, 0, 2 * Math.PI); g.fill(); g.strokeStyle = css("--panel"); g.lineWidth = 2; g.stroke(); }
      g.fillStyle = css("--ink"); g.font = "12px " + css("--mono"); g.textAlign = "right";
      g.fillText(`${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}`, w, y + 4); g.textAlign = "left";
    });
    g.fillStyle = css("--muted"); g.font = "11px system-ui, sans-serif";
    g.fillText(narrow ? `scale +/- ${+(span * 100).toFixed(0)} cm, triangles = off scale` : `scale +/- ${+(span * 100).toFixed(0)} cm, set by the predictors in this scene; triangles = off scale`, x0, top + rows.length * rowH + 15);
  }

  // ---- bird's-eye ---------------------------------------------------------------------------------------------
  function drawBev() {
    const d = S.doc; const { g, w, h } = fit($("bevc"));
    g.fillStyle = css("--road"); g.fillRect(0, 0, w, h);
    const i = startIndex(d, S.p), j = S.h;
    const tCam = camTime(d);
    const cam = poseAt(d, tCam);
    const span = 46; const k = w / span; const ox = w / 2, oy = h * 0.66;
    const c = Math.cos(cam.yaw), s = Math.sin(cam.yaw);
    const P = (x, y) => { const dx = x - cam.x, dy = y - cam.y; const f = c * dx + s * dy, l = -s * dx + c * dy; return [ox - l * k, oy - f * k]; };
    // 10 m grid aligned with the car
    g.strokeStyle = css("--grid"); g.lineWidth = 1;
    for (let a = -60; a <= 60; a += 10) {
      g.beginPath(); g.moveTo(ox + a * k, 0); g.lineTo(ox + a * k, h); g.stroke();
      g.beginPath(); g.moveTo(0, oy + a * k); g.lineTo(w, oy + a * k); g.stroke();
    }
    const path = (a, b, col, wdt, dash) => {
      g.save(); g.strokeStyle = col; g.lineWidth = wdt; g.setLineDash(dash || []); g.beginPath();
      for (let q = a; q <= b; q++) { const p = P(d.x[q], d.y[q]); q === a ? g.moveTo(p[0], p[1]) : g.lineTo(p[0], p[1]); }
      g.stroke(); g.restore();
    };
    path(0, cam.i, css("--ink2"), 2.5);
    path(cam.i, d.t.length - 1, css("--muted"), 1.5, [6, 5]);
    const car = (pose, col, dash, fill) => {
      const cy = Math.cos(pose.yaw), sy = Math.sin(pose.yaw);
      const pts = [[3.2, 0.9], [3.2, -0.9], [-0.9, -0.9], [-0.9, 0.9]].map(([f, l]) => P(pose.x + f * cy - l * sy, pose.y + f * sy + l * cy));
      g.save(); g.setLineDash(dash || []); g.beginPath(); pts.forEach((q, n) => (n ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      g.strokeStyle = col; g.lineWidth = 1.5; g.stroke(); g.restore();
    };
    const K = poseAt(d, d.st[i]);
    const T = poseAt(d, d.st[i] + H_MS[j] / 1000);
    car(K, css("--muted"), [4, 3]);
    car(T, css("--truth"), [], "rgba(127,127,127,0.12)");
    // 20 m anchor line and target
    const tgt = P(T.x + OV_D * Math.cos(T.yaw), T.y + OV_D * Math.sin(T.yaw)), tp = P(T.x, T.y);
    g.save(); g.strokeStyle = css("--truth"); g.globalAlpha = 0.45; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(tp[0], tp[1]); g.lineTo(tgt[0], tgt[1]); g.stroke(); g.restore();
    g.strokeStyle = css("--truth"); g.lineWidth = 2; g.beginPath(); g.arc(tgt[0], tgt[1], 6, 0, 2 * Math.PI); g.stroke();
    for (const m of ROWS) if (S.on.has(m)) {
      g.strokeStyle = color(m); g.fillStyle = color(m); g.lineWidth = 2;
      g.save(); if (m === "stale") g.setLineDash([4, 3]);
      g.beginPath(); const a = P(K.x, K.y); g.moveTo(a[0], a[1]);
      let Pj = null;
      for (let q = 0; q <= j; q++) { Pj = predPose(d, m, i, q); const b = P(Pj.x, Pj.y); g.lineTo(b[0], b[1]); }
      g.stroke(); g.restore();
      const e = P(Pj.x, Pj.y), o = P(Pj.x + OV_D * Math.cos(Pj.yaw), Pj.y + OV_D * Math.sin(Pj.yaw));
      g.beginPath(); g.arc(e[0], e[1], 3.5, 0, 2 * Math.PI); g.fill();
      g.beginPath(); g.arc(o[0], o[1], 3.5, 0, 2 * Math.PI); g.fill();
    }
    // scale bar
    g.fillStyle = css("--ink"); g.fillRect(w - 14 - 10 * k, h - 16, 10 * k, 3);
    g.font = "11px system-ui, sans-serif"; g.textAlign = "right"; g.fillText("10 m", w - 14, h - 22); g.textAlign = "left";
    g.fillStyle = css("--muted");
    g.fillText("dashed car: newest pose (sample)", 8, h - 26);
    g.fillText("solid car: true pose at display time", 8, h - 12);
    drawDetail(g, w, h, i, j, T);
  }

  // inset: the predicted rear-axle points around the true pose at display time
  function drawDetail(g, w, h, i, j, T) {
    const d = S.doc;
    const box = Math.min(176, w * 0.4, (h - 80) * 0.62), bx = 8, by = 8;
    const use = scaleSet();
    const half = niceUp(Math.max(0.005, ...use.map((m) => d.maxPos[S.level][m][j])) * 1.05, [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]);
    g.save();
    g.fillStyle = css("--panel"); g.strokeStyle = css("--line"); g.lineWidth = 1;
    g.fillRect(bx, by, box, box + 30); g.strokeRect(bx + 0.5, by + 0.5, box, box + 30);
    g.fillStyle = css("--ink2"); g.font = "600 11px system-ui, sans-serif";
    g.fillText(`Detail, +/- ${fmtLen(half)}`, bx + 6, by + 13);
    const cx = bx + box / 2, cy = by + 18 + box / 2, k = (box / 2 - 8) / half;
    g.strokeStyle = css("--grid"); g.beginPath(); g.moveTo(cx, by + 20); g.lineTo(cx, by + box + 16); g.moveTo(bx + 2, cy); g.lineTo(bx + box - 2, cy); g.stroke();
    g.strokeStyle = css("--truth"); g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, 5, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx, cy - 22); g.stroke();
    for (const m of ROWS) if (S.on.has(m)) {
      const R = d.lv[S.level].rows[m]; const q = i * 6 + j;
      let px = cx - R.lat[q] * k, py = cy - R.lon[q] * k;
      const out = px < bx + 6 || px > bx + box - 6 || py < by + 24 || py > by + box + 12;
      px = Math.max(bx + 6, Math.min(bx + box - 6, px)); py = Math.max(by + 24, Math.min(by + box + 12, py));
      g.fillStyle = color(m); g.strokeStyle = color(m); g.lineWidth = 2;
      if (out) {
        const ang = Math.atan2(py - cy, px - cx);
        g.beginPath(); g.moveTo(px + 7 * Math.cos(ang), py + 7 * Math.sin(ang));
        g.lineTo(px + 6 * Math.cos(ang + 2.4), py + 6 * Math.sin(ang + 2.4)); g.lineTo(px + 6 * Math.cos(ang - 2.4), py + 6 * Math.sin(ang - 2.4));
        g.closePath(); g.fill(); continue;
      }
      g.beginPath(); g.arc(px, py, 4, 0, 2 * Math.PI); g.fill();
      const hd = R.head[q]; g.beginPath(); g.moveTo(px, py); g.lineTo(px - Math.sin(hd) * 22, py - Math.cos(hd) * 22); g.stroke();
    }
    g.fillStyle = css("--muted"); g.font = "10.5px system-ui, sans-serif";
    g.fillText("dot = predicted rear axle", bx + 6, by + box + 26);
    g.restore();
  }

  // ---- metrics -------------------------------------------------------------------------------------------------
  function aggregate(m) {
    const A = S.agg; if (!A || m === "stale") return null;
    if (S.level === "0") { const c = A.acc[m] && A.acc[m].ov20[S.h]; return c ? c.value : null; }
    if (H_MS[S.h] !== 200) return null;
    const li = A.noise_levels.indexOf(+S.level); return li >= 0 ? A.noise[m][li] : null;
  }
  function buildMetrics() {
    const head = `<tr><th class="l">method</th><th class="wide">now<br>pos cm</th><th class="wide">now<br>head deg</th><th class="wide">now<br>20 m cm</th>` +
      `<th class="wide">scene mean<br>pos cm</th><th class="wide">scene mean<br>head deg</th><th>scene mean<br>|20 m| cm</th>` +
      `<th>test split<br>|20 m| cm</th><th class="l sparkh">|20 m offset|<br>over the scene</th></tr>`;
    const body = ROWS.map((m) => `<tr id="r-${m}"><td class="l"><label style="display:flex;gap:6px;align-items:center;cursor:pointer">` +
      `<input type="checkbox" data-m="${m}" ${S.on.has(m) ? "checked" : ""}><span class="sw${m === "stale" ? " dash" : ""}" style="border-color:var(--s-${m})"></span><span class="ln">${SHORT[m]}</span><span class="sn">${TINY[m]}</span></label></td>` +
      `<td class="wide" data-k="pos"></td><td class="wide" data-k="head"></td><td class="wide" data-k="ov"></td>` +
      `<td class="wide" data-k="mpos"></td><td class="wide" data-k="mhead"></td><td data-k="mov"></td><td data-k="agg"></td>` +
      `<td class="l sparkc"><canvas class="spark" data-m="${m}" width="140" height="26"></canvas></td></tr>`).join("");
    $("metrics").innerHTML = head + body;
    $("metrics").querySelectorAll("input[type=checkbox]").forEach((el) => el.addEventListener("change", () => toggle(el.dataset.m, el.checked)));
    $("metrics").querySelectorAll("canvas.spark").forEach((cv) => {
      cv.addEventListener("click", (ev) => { const r = cv.getBoundingClientRect(); seekFrac((ev.clientX - r.left) / r.width); });
      cv.addEventListener("mousemove", (ev) => {
        const d = S.doc; if (!d) return; const r = cv.getBoundingClientRect();
        const i = Math.max(0, Math.min(d.st.length - 1, Math.round(((ev.clientX - r.left) / r.width) * (d.st.length - 1))));
        const v = d.lv[S.level].rows[cv.dataset.m].ov[i * 6 + S.h];
        cv.title = `${SHORT[cv.dataset.m]}: t = ${d.st[i].toFixed(2)} s, 20 m offset ${(v * 100).toFixed(1)} cm (click to jump)`;
      });
    });
  }
  function updateMetrics() {
    const d = S.doc; const i = startIndex(d, S.p), j = S.h;
    const L = d.lv[S.level];
    const shared = Math.max(0.01, ...scaleSet().map((m) => d.maxOv[S.level][m][j]));
    for (const m of ROWS) {
      const tr = $("r-" + m); if (!tr) continue;
      tr.classList.toggle("off", !S.on.has(m));
      const R = L.rows[m], q = i * 6 + j, M = L.means[m];
      const set = (k, v) => { tr.querySelector(`[data-k="${k}"]`).textContent = v; };
      set("pos", (R.pos[q] * 100).toFixed(1));
      set("head", (R.head[q] * DEG).toFixed(3));
      set("ov", `${R.ov[q] >= 0 ? "+" : ""}${(R.ov[q] * 100).toFixed(1)}`);
      set("mpos", (M.pos[j] * 100).toFixed(2));
      set("mhead", (M.head[j] * DEG).toFixed(3));
      set("mov", (M.ov20[j] * 100).toFixed(2));
      const a = aggregate(m); set("agg", a == null ? "-" : (a * 100).toFixed(2));
      const cv = tr.querySelector("canvas.spark"); const g = cv.getContext("2d");
      const W = cv.width, H = cv.height; g.clearRect(0, 0, W, H);
      const n = d.st.length; const top = S.on.has(m) ? shared : d.maxOv[S.level][m][j] || 1;
      g.fillStyle = css("--grid"); g.fillRect(0, H - 1, W, 1);
      g.strokeStyle = color(m); g.lineWidth = 1.5; g.beginPath();
      for (let s = 0; s < n; s++) { const x = (s / (n - 1)) * (W - 2) + 1, y = H - 2 - Math.min(1, Math.abs(R.ov[s * 6 + j]) / top) * (H - 4); s ? g.lineTo(x, y) : g.moveTo(x, y); }
      g.stroke();
      const xp = (i / (n - 1)) * (W - 2) + 1; g.fillStyle = css("--ink"); g.fillRect(xp - 0.5, 0, 1.5, H);
    }
    $("metnote").textContent = `${H_MS[j]} ms, noise level ${S.level}`;
    document.querySelector(".sparkh").title = `one shared scale, 0 to ${(shared * 100).toFixed(0)} cm; no compensation and CV are clipped`;
  }

  // ---- controls ------------------------------------------------------------------------------------------------
  function buildLegend() {
    $("legend").innerHTML = ROWS.map((m) => `<label><input type="checkbox" data-m="${m}" ${S.on.has(m) ? "checked" : ""}>` +
      `<span class="sw${m === "stale" ? " dash" : ""}" style="border-color:var(--s-${m})"></span>${SHORT[m]}</label>`).join("");
    $("legend").querySelectorAll("input").forEach((el) => el.addEventListener("change", () => toggle(el.dataset.m, el.checked)));
  }
  function toggle(m, on) {
    if (on) S.on.add(m); else S.on.delete(m);
    document.querySelectorAll(`input[data-m="${m}"]`).forEach((el) => { el.checked = on; });
    draw();
  }
  function setLatency(h) { S.h = h; $("lat").value = h; $("latv").textContent = `${H_MS[h]} ms`; draw(); }
  function setLevel(l) {
    S.level = l; $("noise").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === l))); draw();
  }
  function setSpeed(v) { S.speed = v; $("speed").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.v === v))); }
  function setPlaying(on) { S.playing = on; $("play").textContent = on ? "Pause" : "Play"; S.last = null; if (on) requestAnimationFrame(tick); }
  function seekFrac(f) { const d = S.doc; if (!d) return; S.p = d.st[0] + Math.max(0, Math.min(1, f)) * (d.st[d.st.length - 1] - d.st[0]); draw(); }

  async function selectScene(file) {
    const d = await loadScene(file);
    S.doc = d;
    $("scene").value = file;
    S.p = Math.max(d.st[0], Math.min(S.p, d.st[d.st.length - 1]));
    const tl = $("time"); tl.min = d.st[0]; tl.max = d.st[d.st.length - 1];
    const e = S.index.scenes.find((x) => x.file === file);
    $("rule").textContent = `${d.scene} (${e.vehicle}, ${e.location}), ${e.start_points} start points. Chosen by rule: ${d.rule}.`;
    draw();
  }

  function draw() {
    if (!S.doc) return;
    const d = S.doc; const i = startIndex(d, S.p);
    $("time").value = S.p;
    const gap = S.p - d.st[i] > 0.15 ? " · no start point here (below 1 m/s or not in drive)" : "";
    $("readout").textContent = DEMO ? `t ${S.p.toFixed(1)} s` : `t ${S.p.toFixed(2)} s · sample ${d.st[i].toFixed(2)} s · display ${(d.st[i] + H_MS[S.h] / 1000).toFixed(2)} s${gap}`;
    drawHud(); drawRuler(); drawBev(); updateMetrics();
    if (DEMO) {
      $("d-lat").innerHTML = H_MS.map((v, n) => `<button type="button" aria-pressed="${n === S.h}">${v} ms</button>`).join("");
      $("d-noise").innerHTML = ["0", "1", "2"].map((v) => `<button type="button" aria-pressed="${v === S.level}">noise ${v}</button>`).join("");
    }
  }

  function tick(ts) {
    if (!S.playing) return;
    if (S.last != null && S.doc) {
      const d = S.doc; S.p += ((ts - S.last) / 1000) * S.speed;
      if (S.p > d.st[d.st.length - 1]) S.p = d.st[0];
      draw();
    }
    S.last = ts; requestAnimationFrame(tick);
  }

  // ---- demo mode (the recorded video): a deterministic script of state per video time --------------------------
  const DEMO_SCENE = "scene-1061.json";
  const SCRIPT = [
    // [start s, caption, state]; playhead p = p0 + (T - start) * rate
    { t: 0, card: "title" },
    { t: 4.5, cap: "An AR-HUD lights its arrow 50 ms to 500 ms after the newest sensor sample. It must be drawn for where the car will be.", p0: 4.0, rate: 0.5, h: 3, level: "0", on: ["stale"] },
    { t: 12.5, cap: "No compensation: drawn from the newest pose, the arrow slides off the target as the car turns right.", p0: 7.6, rate: 0.4, h: 3, level: "0", on: ["stale"] },
    { t: 21.5, cap: "CTRV (constant turn rate and velocity) follows the turn, but overshoots when the yaw rate changes.", p0: 7.6, rate: 0.4, h: 3, level: "0", on: ["stale", "ctrv"] },
    { t: 31, cap: "EKF + MLP: an EKF on the CTRA state plus a small learned correction stays closest to the target.", p0: 7.6, rate: 0.4, h: 3, level: "0", on: ["stale", "ctrv", "hybrid_mlp"] },
    { t: 40.5, cap: "More display latency, larger drift: 100 ms, 200 ms, then 300 ms.", p0: 9.2, rate: 0.12, h: 1, level: "0", on: ["stale", "ctrv", "hybrid_mlp"], hl: "d-lat", steps: [[43.5, 3], [46, 4]] },
    { t: 49.5, cap: "Noisy inputs (level 1): the pure MLP degrades most, EKF + ridge holds up best.", p0: 7.6, rate: 0.4, h: 3, level: "1", on: ["ctrv", "mlp", "hybrid_ridge", "hybrid_mlp"], hl: "d-noise" },
    { t: 60, cap: "Every number comes from the benchmark code and its fitted weights; the test-split column is the report's.", p0: 10.3, rate: 0.0, h: 3, level: "1", on: ["ctrv", "mlp", "hybrid_ridge", "hybrid_mlp"], hl: "p-met" },
    { t: 66.5, card: "end" },
  ];
  const DEMO_END = 78;

  function demoAt(T) {
    let k = 0; for (let n = 0; n < SCRIPT.length; n++) if (SCRIPT[n].t <= T) k = n;
    const sc = SCRIPT[k];
    const cardT = $("card-title"), cardE = $("card-end");
    const fade = (t0, t1) => Math.max(0, Math.min(1, (T - t0) / 0.5, (t1 - T) / 0.5));
    cardT.style.opacity = sc.card === "title" ? fade(-1, SCRIPT[k + 1].t) : 0;
    cardE.style.opacity = sc.card === "end" ? Math.min(1, (T - sc.t) / 0.5) : 0;
    document.querySelectorAll(".hl").forEach((el) => el.classList.remove("hl"));
    if (sc.card) {
      if (sc.card === "title") { const nx = SCRIPT[k + 1]; applyDemo(nx, nx.t); }
      return;
    }
    applyDemo(sc, T);
    const next = SCRIPT[k + 1] ? SCRIPT[k + 1].t : DEMO_END;
    $("captext").textContent = sc.cap;
    $("caption").style.setProperty("--cap-o", Math.max(0, Math.min(1, (T - sc.t) / 0.35, (next - T) / 0.35)).toFixed(3));
    if (sc.hl) $(sc.hl).classList.add("hl");
  }
  function applyDemo(sc, T) {
    S.on = new Set(sc.on);
    document.querySelectorAll("input[data-m]").forEach((el) => { el.checked = S.on.has(el.dataset.m); });
    let h = sc.h; if (sc.steps) for (const [ts, hv] of sc.steps) if (T >= ts) h = hv;
    S.h = h; $("lat").value = h; $("latv").textContent = `${H_MS[h]} ms`;
    S.level = sc.level; $("noise").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === sc.level)));
    S.p = Math.min(sc.p0 + (T - sc.t) * sc.rate, S.doc.st[S.doc.st.length - 1]);
    $("play").textContent = sc.rate > 0 ? "Pause" : "Play";
    draw();
  }

  // ---- boot ----------------------------------------------------------------------------------------------------
  $("play").addEventListener("click", () => setPlaying(!S.playing));
  $("speed").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setSpeed(+b.dataset.v)));
  $("noise").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => setLevel(b.dataset.v)));
  $("lat").addEventListener("input", () => setLatency(+$("lat").value));
  $("time").addEventListener("input", () => { S.p = +$("time").value; draw(); });
  $("scene").addEventListener("change", () => selectScene($("scene").value));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || !S.doc) return;
    if (e.code === "Space" && e.target.tagName !== "BUTTON") { e.preventDefault(); setPlaying(!S.playing); }
    if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
      const d = S.doc; const i = startIndex(d, S.p) + (e.code === "ArrowRight" ? 1 : -1);
      S.p = d.st[Math.max(0, Math.min(d.st.length - 1, i))]; draw();
    }
  });
  let rz = null;
  window.addEventListener("resize", () => { cancelAnimationFrame(rz); rz = requestAnimationFrame(draw); });

  const ready = Promise.all([
    fetch("data/hud/index.json").then((r) => r.json()),
    fetch("data/mpred.json").then((r) => r.json()).catch(() => null),
  ]).then(async ([index, agg]) => {
    S.index = index; S.agg = agg;
    $("scene").innerHTML = index.scenes.map((s) => `<option value="${s.file}">${s.label}</option>`).join("");
    buildLegend(); buildMetrics();
    const want = params.get("scene");
    const first = index.scenes.find((s) => s.scene === want) || index.scenes[0];
    const hp = H_MS.indexOf(+params.get("h")); if (hp >= 0) S.h = hp;
    if (["0", "1", "2"].includes(params.get("noise"))) S.level = params.get("noise");
    $("lat").value = S.h; $("latv").textContent = `${H_MS[S.h]} ms`; setLevel(S.level);
    if (params.get("t")) S.p = +params.get("t");
    await selectScene(DEMO ? DEMO_SCENE : first.file);
    if (DEMO) {
      document.body.classList.add("demo");
      if (DEMO === "portrait") document.body.classList.add("portrait");
      if (!params.get("frame")) {
        const t0 = performance.now();
        const run = () => { const T = (performance.now() - t0) / 1000; demoAt(T % DEMO_END); requestAnimationFrame(run); };
        requestAnimationFrame(run);
      }
    }
  });
  window.HUD = { ready, demoAt, duration: DEMO_END, state: S };
})();
