const COLORS = { "v0.1.0": "--s2", "be3c4e4": "--s4", "v0.2.0": "--s1", "silent-regression": "--s3" };
const LABELS = { "v0.1.0": "v0.1.0", "be3c4e4": "candidate be3c4e4", "v0.2.0": "v0.2.0", "silent-regression": "silent regression" };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const el = (tag, attrs = {}, text) => {
  const e = document.createElementNS(tag === "svg" || ["line", "polyline", "text", "rect", "circle", "g", "path"].includes(tag)
    ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let TL, RES, shown = new Set(["be3c4e4", "v0.2.0"]);

try {
  const t = localStorage.getItem("dr-theme");
  if (t) document.documentElement.dataset.theme = t;
} catch (e) { /* storage unavailable */ }
document.getElementById("theme").onclick = () => {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("dr-theme", next); } catch (e) { /* ignore */ }
  draw();
};

Promise.all([fetch("data/timelines.json").then((r) => r.json()), fetch("data/results.json").then((r) => r.json())])
  .then(([tl, res]) => { TL = tl; RES = res; init(); });

function init() {
  const ci = RES.ci;
  const pass = ci.filter((r) => r.verdict === "PASS").length;
  const tp = RES.throughput;
  const one = tp.find((r) => r.jobs === 1), many = tp[tp.length - 1];
  const cards = [
    [`${ci.length} drops`, `${pass} PASS, ${ci.length - pass} blocked by the replay CI`],
    [`${RES.bugs.length} bugs`, "reproduced from their recording, fix proven on a newer drop"],
    [`${one.x_realtime}x real time`, `one core; ${many.x_realtime}x with ${many.jobs} jobs`],
    [RES.rerun_identical ? "bit-identical" : "NOT identical", `reruns; ${many.runs} parallel runs, ${many.distinct_digests} digests for ${many.recordings} recordings`],
  ];
  document.getElementById("cards").innerHTML = cards.map(([b, s]) => `<div class="panel card"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");

  // CI matrix
  const labels = Object.keys(RES.ci_matrix);
  const scenes = Object.keys(RES.ci_matrix[labels[0]]);
  let h = "<tr><th>recording</th>" + labels.map((l) => `<th>${esc(l)}</th>`).join("") + "</tr>";
  for (const s of scenes) {
    h += `<tr><td>${esc(s)}</td>` + labels.map((l) => {
      const r = RES.ci_matrix[l][s];
      const why = Object.entries(r.checks).filter(([, v]) => v[0] !== "PASS").map(([k, v]) => `${k}: ${v[1]}`).join("\n");
      return `<td><span class="v ${r.verdict}" title="${esc(why || "all checks pass")}">${r.verdict}</span></td>`;
    }).join("") + "</tr>";
  }
  h += "<tr><th>suite</th>" + labels.map((l, i) => `<th><span class="v ${ci[i].verdict}">${ci[i].verdict}</span></th>`).join("") + "</tr>";
  document.getElementById("matrix").innerHTML = h;

  // timeline controls
  const sel = document.getElementById("scene");
  sel.innerHTML = Object.keys(TL).map((s) => `<option value="${s}">${esc(s)} - ${esc(TL[s].description)}</option>`).join("");
  sel.value = "scene-0916";
  sel.onchange = draw;
  const dropBox = document.getElementById("drops");
  dropBox.innerHTML = Object.keys(COLORS).map((d) =>
    `<label class="chk"><input type="checkbox" value="${d}" ${shown.has(d) ? "checked" : ""}>` +
    `<span class="swatch" style="background:var(${COLORS[d]})"></span>${esc(LABELS[d])}</label>`).join(" ") +
    ` <label class="chk"><span class="swatch" style="background:var(--truth)"></span>true TTC (oracle)</label>`;
  dropBox.querySelectorAll("input").forEach((c) => c.onchange = () => { c.checked ? shown.add(c.value) : shown.delete(c.value); draw(); });

  // bugs
  document.getElementById("bugs").innerHTML = Object.values(RES.bug_reports).map((md) => `<div class="panel bug">${mdToHtml(md)}</div>`).join("");
  // preroll
  const pr = RES.preroll;
  document.getElementById("preroll").innerHTML = "<tr><th>case</th><th>warning at</th><th>pre-roll</th><th>window = whole drive</th><th>first warning in window</th></tr>" +
    pr.map((r) => `<tr><td>${esc(r.scene)}</td><td>${r.event_s} s</td><td>${r.preroll_s} s</td>` +
      `<td><span class="v ${r.identical ? "PASS" : "FAIL"}">${r.identical ? "identical" : "diverged"}</span></td><td>${r.first_warning_s ?? "-"} s</td></tr>`).join("");
  document.getElementById("tp").innerHTML = "<tr><th>parallel jobs</th><th>runs</th><th>recorded driving</th><th>wall</th><th>x real time</th><th>distinct digests / recordings</th></tr>" +
    tp.map((r) => `<tr><td>${r.jobs}</td><td>${r.runs}</td><td>${r.recorded_s} s</td><td>${r.wall_s} s</td><td>${r.x_realtime}</td><td>${r.distinct_digests} / ${r.recordings}</td></tr>`).join("");
  document.getElementById("host").textContent = `Generated ${RES.generated} at commit ${RES.commit} on ${RES.cpu}. x86 replay timing only.`;
  draw();
}

function draw() {
  if (!TL) return;
  const s = TL[document.getElementById("scene").value];
  const svg = document.getElementById("chart");
  const W = 1000, H = 330, L = 44, R = 14, T = 14, B = 34;
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const tmax = Math.ceil(s.duration_s), ymax = 6;
  const x = (t) => L + (t / tmax) * (W - L - R);
  const y = (v) => T + (1 - Math.min(v, ymax) / ymax) * (H - T - B);
  for (let v = 0; v <= ymax; v += 1) {
    svg.append(el("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: css("--grid") }));
    svg.append(el("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v}`));
  }
  for (let t = 0; t <= tmax; t += 2) svg.append(el("text", { x: x(t), y: H - 12, "text-anchor": "middle" }, `${t} s`));
  svg.append(el("text", { x: 4, y: 12 }, "TTC s"));
  // FCW threshold
  svg.append(el("line", { x1: L, x2: W - R, y1: y(2.2), y2: y(2.2), stroke: css("--warn"), "stroke-dasharray": "4 4" }));
  svg.append(el("text", { x: W - R, y: y(2.2) - 4, "text-anchor": "end" }, "warn threshold 2.2 s"));
  const line = (pts, color, width, dash) => {
    let seg = [];
    const flush = () => { if (seg.length > 1) svg.append(el("polyline", { points: seg.join(" "), fill: "none", stroke: color, "stroke-width": width, ...(dash ? { "stroke-dasharray": dash } : {}) })); seg = []; };
    for (const [t, v] of pts) { if (v === null || v > ymax * 1.5) flush(); else seg.push(`${x(t).toFixed(1)},${y(v).toFixed(1)}`); }
    flush();
  };
  line(s.true_ttc, css("--truth"), 2, "2 3");
  const ev = [];
  for (const d of Object.keys(COLORS)) {
    if (!shown.has(d) || !s.drops[d]) continue;
    const c = css(COLORS[d]);
    line(s.drops[d].ttc, c, 1.6);
    for (const e of s.drops[d].events.filter((e) => e.kind === "fcw_warning_on")) {
      const v = s.drops[d].verdicts.find((w) => Math.abs(w.t - e.t) < 0.02);
      svg.append(el("line", { x1: x(e.t), x2: x(e.t), y1: T, y2: H - B, stroke: c, "stroke-width": 1.5 }));
      svg.append(el("circle", { cx: x(e.t), cy: y(e.ttc), r: 5, fill: v && v.verdict === "false" ? css("--fail") : css("--pass"), stroke: c, "stroke-width": 2 }));
      ev.push(`<div><b style="color:${c}">${esc(LABELS[d])}</b> ${e.t.toFixed(2)} s warning, stack TTC ${e.ttc.toFixed(2)} s: ` +
        `<span class="${v && v.verdict === "false" ? "FAIL" : "PASS"}">${v ? v.verdict.toUpperCase() : ""}</span>` +
        (v && v.gt.length ? ` (${esc(v.gt[0].category)} ${v.gt[0].gap_m} m ahead, true TTC ${v.gt[0].ttc_s} s)` : v ? " (nothing in the driven path)" : "") + "</div>");
    }
  }
  document.getElementById("desc").textContent = s.description + (s.injected ? `. Injected ${s.injected.kind.replace("_", " ")}, open loop.` : "") +
    " Dotted grey: true TTC of the closest object in the path the car really drove. Dots: warnings (green = justified, red = false).";
  document.getElementById("events").innerHTML = ev.join("") || "<div>No warnings from the selected drops.</div>";
}

function mdToHtml(md) {
  const out = [];
  let table = [];
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\|[-| ]+\|$/.test(r)).map((r) => r.slice(1, -1).split("|").map((c) => c.trim()));
    out.push('<div class="scroll"><table>' + rows.map((r, i) => "<tr>" + r.map((c) => i ? `<td>${inline(c)}</td>` : `<th>${inline(c)}</th>`).join("") + "</tr>").join("") + "</table></div>");
    table = [];
  };
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  for (const line of md.split("\n")) {
    if (line.startsWith("|")) { table.push(line); continue; }
    flushTable();
    if (line.startsWith("# ")) out.push(`<h3>${inline(line.slice(2))}</h3>`);
    else if (line.startsWith("- ")) out.push(`<div>${inline(line)}</div>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  flushTable();
  return out.join("");
}
