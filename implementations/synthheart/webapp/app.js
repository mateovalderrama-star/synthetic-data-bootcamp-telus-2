// SynthHeart app logic, shared by every design in webapp/designs/.
// A design supplies the markup (element ids below) and the look; this file fills in data, runs the predictor
// and animates. Nothing typed into the form leaves the page.
//
// Hooks a design provides (all optional except the form and #result-body):
//   [data-tab="check|compare|scorecard|about"] buttons and [data-panel="..."] sections
//   inputs #age #height #weight #ap_hi #ap_lo (+ hints #h-<name>), .seg[data-name] button groups,
//   input[type=range][data-for="<name>"] sliders, #bmi-out, #examples, #reset-btn, #theme-btn, #test-banner
//   #result-body[data-gauge="arc|ring"], #session, compare / scorecard / about containers (ids below)
(function () {
  "use strict";
  const D = window.SYNTHHEART;
  const P = window.SynthHeartPredictor;
  const model = P.prepare(D.model);
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  // #static (used for screenshots) or the OS "reduce motion" setting turns animations off.
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches || /(^#|&)static/.test(location.hash);
  if (reduced) document.documentElement.classList.add("no-motion");
  const set = (sel, html) => { const e = $(sel); if (e) e.innerHTML = html; return e; };
  const text = (sel, t) => { const e = $(sel); if (e) e.textContent = t; };

  if (D.test_model && $("#test-banner")) $("#test-banner").hidden = false;

  // ---------- motion helpers ----------
  function tween(from, to, ms, step) {
    if (reduced || ms <= 0) { step(to); return; }
    const t0 = performance.now();
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    (function frame(now) {
      const t = Math.min(1, (now - t0) / ms);
      step(from + (to - from) * ease(t));
      if (t < 1) requestAnimationFrame(frame);
    })(t0);
  }
  // Number count-up for any element with data-count="<value>" data-fmt="int|pct1|pct2|dec3|raw"
  function countUp(root) {
    $$("[data-count]", root).forEach((e) => {
      const to = +e.dataset.count, f = e.dataset.fmt || "int";
      const fmt = { int: (v) => Math.round(v).toLocaleString(), pct1: (v) => (v * 100).toFixed(1) + "%", pct2: (v) => (v * 100).toFixed(2) + "%",
        dec3: (v) => v.toFixed(3), raw: (v) => v.toFixed(0) }[f];
      tween(0, to, 900, (v) => { e.textContent = fmt(v); });
    });
  }
  // Staggered entrance for the children of a panel (CSS does the animation: .anim-in > * uses --i)
  function enter(panel) {
    if (!panel) return;
    panel.classList.remove("anim-in");
    $$(":scope > *, :scope .stagger > *", panel).forEach((c, i) => c.style.setProperty("--i", Math.min(i, 12)));
    void panel.offsetWidth;
    panel.classList.add("anim-in");
    countUp(panel);
  }

  // ---------- theme + tabs ----------
  const root = document.documentElement;
  try { const t = localStorage.getItem("synthheart-theme"); if (t) root.dataset.theme = t; } catch (e) {}
  const forced = location.hash.match(/(?:^#|&)(light|dark)(?:&|$)/);  // #light / #dark force a theme (demos, screenshots)
  if (forced) root.dataset.theme = forced[1];
  $$("#theme-btn, [data-theme-toggle]").forEach((b) => b.addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("synthheart-theme", root.dataset.theme); } catch (e) {}
    renderCharts();
    if (current) drawGauge(current.risk, true);
  }));
  function showTab(name) {
    $$("[data-tab]").forEach((x) => { const on = x.dataset.tab === name; x.setAttribute("aria-selected", on); x.classList.toggle("active", on); });
    $$("[data-panel]").forEach((s) => { s.hidden = s.dataset.panel !== name; });
    const panel = $(`[data-panel="${name}"]`);
    if (name !== "check") renderCharts();
    enter(panel);
    document.dispatchEvent(new CustomEvent("synthheart:tab", { detail: name }));
  }
  $$("[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

  // ---------- form ----------
  const NUM = ["age", "height", "weight", "ap_hi", "ap_lo"];
  const SEG = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"];
  const LABEL = { age: "Age", height: "Height", weight: "Weight", ap_hi: "Systolic BP", ap_lo: "Diastolic BP", gender: "Sex",
    cholesterol: "Cholesterol", gluc: "Glucose", smoke: "Smoking", alco: "Alcohol", active: "Physical activity", bmi: "BMI", cardio: "Disease" };
  const LEVEL = ["Normal", "Above normal", "Well above"];
  const SHOW = {
    age: (v) => v + " yrs", height: (v) => v + " cm", weight: (v) => v + " kg", ap_hi: (v) => v + " mmHg", ap_lo: (v) => v + " mmHg",
    gender: (v) => (v ? "Male" : "Female"), cholesterol: (v) => LEVEL[v], gluc: (v) => LEVEL[v],
    smoke: (v) => (v ? "Smoker" : "Non-smoker"), alco: (v) => (v ? "Drinks" : "No alcohol"), active: (v) => (v ? "Active" : "Not active"), bmi: (v) => "BMI " + v,
  };
  const EXAMPLES = [
    { name: "Healthy 40-year-old", short: "Low", v: { age: 40, gender: 0, height: 165, weight: 60, ap_hi: 110, ap_lo: 70, cholesterol: 0, gluc: 0, smoke: 0, alco: 0, active: 1 } },
    { name: "Typical 55-year-old", short: "Typical", v: { age: 55, gender: 1, height: 172, weight: 82, ap_hi: 130, ap_lo: 85, cholesterol: 1, gluc: 0, smoke: 0, alco: 0, active: 1 } },
    { name: "Higher-risk 62-year-old", short: "High", v: { age: 62, gender: 1, height: 170, weight: 95, ap_hi: 160, ap_lo: 100, cholesterol: 2, gluc: 1, smoke: 1, alco: 0, active: 0 } },
  ];
  const state = {};
  const segs = {};
  $$(".seg[data-name]").forEach((g) => {
    segs[g.dataset.name] = g;
    $$("button", g).forEach((b) => b.addEventListener("click", () => { setSeg(g.dataset.name, +b.dataset.v); schedule(); }));
  });
  function setSeg(name, v) {
    state[name] = v;
    if (segs[name]) $$("button", segs[name]).forEach((b) => b.setAttribute("aria-pressed", +b.dataset.v === v));
  }
  function clearSeg(name) { delete state[name]; if (segs[name]) $$("button", segs[name]).forEach((b) => b.setAttribute("aria-pressed", "false")); }
  NUM.forEach((n) => {
    const input = $("#" + n); if (!input) return;
    input.addEventListener("input", () => { syncRange(n); schedule(); });
    $$(`input[type=range][data-for="${n}"]`).forEach((r) => {
      const [lo, hi] = D.typical?.[n] ? [Math.floor(D.typical[n][0] * 0.9), Math.ceil(D.typical[n][1] * 1.08)] : D.ranges[n];
      r.min = Math.max(D.ranges[n][0], lo); r.max = Math.min(D.ranges[n][1], hi);
      r.addEventListener("input", () => { input.value = r.value; schedule(); });
    });
  });
  function syncRange(n) { $$(`input[type=range][data-for="${n}"]`).forEach((r) => { if ($("#" + n).value !== "") r.value = $("#" + n).value; }); }
  function fill(v) {
    NUM.forEach((n) => { if ($("#" + n)) { $("#" + n).value = v[n] ?? ""; syncRange(n); } });
    SEG.forEach((n) => (v[n] === undefined ? clearSeg(n) : setSeg(n, v[n])));
    update();
  }
  const exBox = $("#examples");
  if (exBox) EXAMPLES.forEach((ex, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.dataset.example = i + 1;
    b.innerHTML = `<span class="chip-dot ${["low", "mid", "high"][i]}"></span>${esc(ex.name)}`;
    b.onclick = () => fill(ex.v);
    exBox.append(b);
  });
  $("#reset-btn")?.addEventListener("click", () => fill({}));

  // Validation uses the same medical-range rules that cleaned the real data (synthheart_common.RANGES).
  function readForm() {
    const row = {}; let ok = true;
    NUM.forEach((n) => {
      const input = $("#" + n), hint = $("#h-" + n); if (!input) { ok = false; return; }
      const raw = input.value.trim();
      input.classList.remove("bad");
      if (hint) { hint.className = "hint"; hint.textContent = ""; }
      if (raw === "") { ok = false; return; }
      const v = Number(raw), [lo, hi] = D.ranges[n];
      if (!Number.isFinite(v) || v < lo || v > hi) {
        ok = false; input.classList.add("bad");
        if (hint) { hint.className = "hint err"; hint.textContent = `Enter ${lo}–${hi}`; }
        return;
      }
      row[n] = v;
      const t = D.typical && D.typical[n];
      if (hint && t && (v < t[0] || v > t[1])) { hint.className = "hint warn"; hint.textContent = `Uncommon in training data (${t[0]}–${t[1]})`; }
    });
    if (row.ap_hi !== undefined && row.ap_lo !== undefined && row.ap_hi <= row.ap_lo) {
      ok = false; $("#ap_lo").classList.add("bad");
      const h = $("#h-ap_lo"); if (h) { h.className = "hint err"; h.textContent = "Must be below systolic"; }
    }
    SEG.forEach((n) => { if (state[n] === undefined) ok = false; else row[n] = state[n]; });
    const bmiOut = $("#bmi-out");
    if (row.height && row.weight) {
      row.bmi = P.bmi(row.weight, row.height);
      if (bmiOut) { bmiOut.textContent = row.bmi; bmiOut.dataset.band = row.bmi < 18.5 ? "low" : row.bmi < 25 ? "ok" : row.bmi < 30 ? "mid" : "high"; }
    } else if (bmiOut) { bmiOut.textContent = "–"; delete bmiOut.dataset.band; }
    const done = NUM.filter((n) => row[n] !== undefined).length + SEG.filter((n) => state[n] !== undefined).length;
    $$("[data-progress]").forEach((e) => { e.style.setProperty("--p", done / 11); e.dataset.done = done; });
    text("#progress-text", `${done} of 11 fields`);
    return ok ? row : null;
  }

  // ---------- result ----------
  let current = null, shellBuilt = false, timer = null;
  const BANDS = [[0.35, "low", "Lower risk"], [0.65, "mid", "Moderate risk"], [Infinity, "high", "Higher risk"]];
  const band = (r) => BANDS.find(([cut]) => r < cut);
  function schedule() { clearTimeout(timer); timer = setTimeout(update, 120); }

  function buildShell(body) {
    const ring = body.dataset.gauge === "ring";
    body.innerHTML = `
      <div class="res-head">
        <div class="gauge ${ring ? "ring" : "arc"}" id="gauge" role="img">
          ${ring
            ? `<svg viewBox="0 0 200 200"><circle cx="100" cy="100" r="82" class="g-track" pathLength="100"/><circle cx="100" cy="100" r="82" class="g-fill" pathLength="100" transform="rotate(-90 100 100)"/></svg>`
            : `<svg viewBox="0 0 200 116"><path d="M18 102a82 82 0 0 1 164 0" class="g-track" pathLength="100"/><path d="M18 102a82 82 0 0 1 164 0" class="g-fill" pathLength="100"/></svg>`}
          <div class="g-center"><div class="risk-num" id="risk-num">0%</div><div class="risk-cap">estimated risk</div></div>
        </div>
        <div class="res-meta">
          <span class="band" id="band"></span>
          <p class="res-line" id="res-line"></p>
          <div class="scale"><div class="scale-bar"><i class="scale-pin" id="scale-pin"></i></div><div class="scale-lbl"><span>0</span><span>35</span><span>65</span><span>100</span></div></div>
        </div>
      </div>
      <div class="res-sec"><div class="res-sec-h"><h3>What moved this estimate</h3><span class="muted-sm" id="avg-line"></span></div><div class="drivers" id="drivers"></div></div>
      <div class="res-sec"><div class="res-sec-h"><h3>What if…</h3><span class="muted-sm">one factor changed</span></div><div class="whatifs" id="whatifs"></div></div>
      <div class="res-actions"><input id="code" class="code-input" maxlength="12" placeholder="Patient initials or code" aria-label="Patient initials or code"><button class="btn primary" type="button" id="add-btn">Add to worklist</button></div>
      <p class="fine">Model trained on ${D.rows.synth_scale_1.toLocaleString()} synthetic patients · AUC ${D.utility.tstr_auc.synth_scale_1.toFixed(3)} on ${D.rows.test.toLocaleString()} real held-out patients · bands: lower &lt; 35% ≤ moderate &lt; 65% ≤ higher</p>`;
    $("#add-btn").onclick = addToSession;
    shellBuilt = true;
    body.classList.add("has-result");
  }
  let shownRisk = 0;
  function drawGauge(r, instant) {
    const g = $("#gauge"); if (!g) return;
    const [, cls, label] = band(r);
    g.dataset.band = cls;
    g.setAttribute("aria-label", `Estimated risk ${pct(r)}, ${label}`);
    $(".g-fill", g).style.strokeDashoffset = 100 - r * 100;
    const from = shownRisk; shownRisk = r;
    tween(instant ? r : from, r, 700, (v) => { $("#risk-num").textContent = Math.round(v * 100) + "%"; });
    $("#scale-pin").style.left = (r * 100).toFixed(1) + "%";
  }
  function whatIfs(row) {
    const out = [];
    const tryIt = (icon, label, change) => { const r2 = { ...row, ...change }; r2.bmi = P.bmi(r2.weight, r2.height); out.push([icon, label, P.predict(model, r2).risk]); };
    if (row.ap_hi > 120 || row.ap_lo > 80) tryIt("bp", "Blood pressure at 120/80", { ap_hi: Math.min(row.ap_hi, 120), ap_lo: Math.min(row.ap_lo, 80) });
    if (row.cholesterol > 0) tryIt("lab", "Cholesterol normal", { cholesterol: 0 });
    if (row.bmi > 25) { const w = Math.round(25 * (row.height / 100) ** 2 * 10) / 10; tryIt("scale", `Weight for BMI 25 (${Math.round(w)} kg)`, { weight: w }); }
    if (row.smoke) tryIt("smoke", "Stops smoking", { smoke: 0 });
    if (!row.active) tryIt("run", "Becomes physically active", { active: 1 });
    return out;
  }
  const ICON = {
    bp: '<path d="M12 21s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.5-7 10-7 10z"/>',
    lab: '<path d="M9 3h6M10 3v6L5 19a1.5 1.5 0 0 0 1.3 2h11.4a1.5 1.5 0 0 0 1.3-2L14 9V3"/>',
    scale: '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9a4 4 0 0 1 6 0l-3 3"/>',
    smoke: '<path d="M3 15h13v3H3zM19 15v3M16 8c0-2 2-2 2-4M19 9c0-2 2-2 2-4"/>',
    run: '<circle cx="14" cy="4.5" r="2"/><path d="M8 21l3-6 3 2v4M6 12l3-3 4 1 3 3 3 1"/>',
  };
  const icon = (k) => `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${ICON[k] || ""}</svg>`;

  function update() {
    const row = readForm(), body = $("#result-body");
    if (!body) return;
    if (!row) {
      current = null;
      if (shellBuilt) { body.classList.add("stale"); }
      else body.innerHTML = body.dataset.empty || `<div class="empty"><div class="empty-ico">${icon("bp")}</div><p>Complete the assessment to see the estimate.</p></div>`;
      document.dispatchEvent(new CustomEvent("synthheart:result", { detail: null }));
      return;
    }
    if (!shellBuilt) buildShell(body);
    body.classList.remove("stale");
    const p = P.predict(model, row), [, bc, bl] = band(p.risk);
    const first = !current;
    current = { row, risk: p.risk };
    drawGauge(p.risk, false);
    const b = $("#band"); b.className = "band " + bc; b.textContent = bl;
    text("#res-line", `${SHOW.gender(row.gender)}, ${row.age} yrs · BP ${row.ap_hi}/${row.ap_lo} · BMI ${row.bmi}`);
    text("#avg-line", `vs average patient ${pct(1 / (1 + Math.exp(-p.bias)))}`);
    const drivers = Object.entries(p.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6);
    const maxC = Math.max(0.5, ...drivers.map((d) => Math.abs(d[1])));
    set("#drivers", drivers.map(([f, c]) => {
      const w = Math.abs(c) / maxC * 50, up = c > 0, small = Math.abs(c) < 0.03;
      return `<div class="drv"><div class="drv-name"><b>${LABEL[f]}</b><span>${esc(SHOW[f](row[f]))}</span></div>
        <div class="drv-track"><i class="drv-bar ${up ? "up" : "down"}" style="${up ? "left" : "right"}:50%" data-w="${w.toFixed(1)}"></i></div>
        <div class="drv-dir ${small ? "small" : up ? "up" : "down"}">${small ? "minor" : up ? "▲ raises" : "▼ lowers"}</div></div>`;
    }).join(""));
    const bars = $$("#drivers .drv-bar");
    if (first && !reduced) requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach((e) => { e.style.width = e.dataset.w + "%"; })));
    else bars.forEach((e) => { e.style.transition = "none"; e.style.width = e.dataset.w + "%"; });
    const wi = whatIfs(row);
    set("#whatifs", wi.length ? wi.map(([ic, l, r]) => {
      const d = r - p.risk;
      return `<div class="wi">${icon(ic)}<span class="wi-l">${esc(l)}</span><span class="wi-r"><b>${pct(r)}</b><em class="${d < -0.005 ? "down" : "flat"}">${d < -0.005 ? "−" + Math.abs(d * 100).toFixed(0) + " pts" : "no change"}</em></span></div>`;
    }).join("") : `<div class="wi none">No modifiable factor above target for this patient.</div>`);
    if (first) enter(body);
    document.dispatchEvent(new CustomEvent("synthheart:result", { detail: current }));
  }

  // ---------- worklist (memory only) ----------
  const session = [];
  function addToSession() {
    if (!current) return;
    const code = ($("#code").value || "").trim() || "PT-" + String(session.length + 1).padStart(3, "0");
    session.unshift({ code, time: new Date(), ...current });
    $("#code").value = "";
    renderSession(true);
    const btn = $("#add-btn"); btn.classList.add("ok"); btn.textContent = "Added ✓";
    setTimeout(() => { btn.classList.remove("ok"); btn.textContent = "Add to worklist"; }, 1400);
    text("#worklist-count", session.length);
  }
  function renderSession(animate) {
    const box = $("#session"); if (!box) return;
    text("#worklist-count", session.length || "");
    if (!session.length) { box.innerHTML = `<div class="empty small"><p>No assessments yet. After an estimate, press <b>Add to worklist</b>. The list lives in this tab only and is cleared when it closes.</p></div>`; return; }
    const high = session.filter((s) => s.risk >= 0.65).length, avg = session.reduce((a, s) => a + s.risk, 0) / session.length;
    box.innerHTML = `<div class="kpis stagger">
        <div class="kpi"><span class="kpi-v">${session.length}</span><span class="kpi-k">assessments</span></div>
        <div class="kpi"><span class="kpi-v">${pct(avg)}</span><span class="kpi-k">average risk</span></div>
        <div class="kpi alert"><span class="kpi-v">${high}</span><span class="kpi-k">higher risk ≥ 65%</span></div>
        <div class="kpi"><span class="kpi-v">${session.filter((s) => s.row.ap_hi >= 140).length}</span><span class="kpi-k">systolic ≥ 140</span></div></div>
      <div class="table-wrap"><table class="tbl"><thead><tr><th>Patient</th><th>Time</th><th class="num">Age</th><th>Sex</th><th class="num">BP</th><th class="num">BMI</th><th>Cholesterol</th><th class="num">Risk</th><th></th></tr></thead><tbody>
      ${session.map((s, i) => { const [, bc] = band(s.risk); return `<tr class="${animate && i === 0 ? "row-new" : ""}"><td><b>${esc(s.code)}</b></td><td>${s.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</td>
        <td class="num">${s.row.age}</td><td>${s.row.gender ? "M" : "F"}</td><td class="num">${s.row.ap_hi}/${s.row.ap_lo}</td><td class="num">${s.row.bmi}</td>
        <td>${LEVEL[s.row.cholesterol]}</td><td class="num"><span class="band sm ${bc}">${pct(s.risk)}</span></td>
        <td class="num"><button class="icon-btn" type="button" data-i="${i}" aria-label="Remove ${esc(s.code)}">✕</button></td></tr>`; }).join("")}
      </tbody></table></div>
      <div class="res-actions"><button class="btn" type="button" id="csv-btn">Export CSV</button><button class="btn ghost" type="button" id="clear-btn">Clear worklist</button></div>`;
    $$("button[data-i]", box).forEach((b) => b.onclick = () => { session.splice(+b.dataset.i, 1); renderSession(); });
    $("#clear-btn").onclick = () => { session.length = 0; renderSession(); };
    $("#csv-btn").onclick = () => {
      const cols = ["age", "gender", "height", "weight", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"];
      const lines = [["patient", "time", ...cols, "risk"].join(",")].concat(session.map((s) =>
        [JSON.stringify(s.code), s.time.toISOString(), ...cols.map((c) => s.row[c]), s.risk.toFixed(4)].join(",")));
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })); a.download = "synthheart-worklist.csv";
      a.click(); URL.revokeObjectURL(a.href);
    };
  }

  // ---------- charts ----------
  const ORDER = { "systolic BP": ["<120", "120-139", ">=140"], age: ["<50", "50-59", ">=60"], cholesterol: ["normal", "above normal", "well above"] };
  const NICE = { "<120": "<120", "120-139": "120–139", ">=140": "≥140", "<50": "<50", "50-59": "50–59", ">=60": "≥60", normal: "Normal", "above normal": "Above", "well above": "Well above" };
  function groupedBars(title, curves, order) {
    const real = curves.real, syn = curves.synthetic, keys = order.filter((k) => k in real);
    const W = 320, H = 196, padL = 30, base = H - 30, top = 22, bw = (W - padL - 6) / keys.length;
    let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
    [0, 0.5, 1].forEach((t) => { const y = base - t * (base - top); s += `<line x1="${padL}" x2="${W - 6}" y1="${y}" y2="${y}" class="grid-l"/><text x="${padL - 6}" y="${y + 3}" font-size="9" text-anchor="end" class="ax">${t * 100}%</text>`; });
    keys.forEach((k, i) => {
      const w = Math.min(30, bw * 0.3), x0 = padL + i * bw + (bw - 2 * w - 4) / 2;
      [[real[k], "real", 0], [syn[k], "synth", 1]].forEach(([v, c, j]) => {
        const h = v * (base - top), x = x0 + j * (w + 4), y = base - h;
        s += `<g class="bar-g" style="--d:${(i * 2 + j) * 60}ms"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" class="bar ${c}"><title>${j ? "Synthetic" : "Real"}: ${pct(v, 1)}</title></rect>
          <text x="${x + w / 2}" y="${y - 5}" font-size="9.5" text-anchor="middle" class="val">${Math.round(v * 100)}</text></g>`;
      });
      s += `<text x="${padL + i * bw + bw / 2}" y="${H - 10}" font-size="10.5" text-anchor="middle" class="ax">${esc(NICE[k] || k)}</text>`;
    });
    return s + "</svg>";
  }
  const legend = `<div class="legend"><span><i class="sw real"></i>Real patients</span><span><i class="sw synth"></i>Synthetic patients</span></div>`;

  function renderCharts() {
    const R = D.relationships, F = D.fidelity, G = D.generation;
    set("#compare-tiles", [
      [D.rows.train, "int", "Real training patients", ""],
      [D.rows.synth_scale_1, "int", "Synthetic patients generated", ""],
      [F.cardio_rate.synthetic, "pct1", "Synthetic with disease", `real ${pct(F.cardio_rate.real_train, 1)}`],
      [G ? G.filter.synth_scale_1.pass_rate : 0, "pct2", "Pass medical-sense rules", G ? `${G.filter.synth_scale_1.generated - G.filter.synth_scale_1.kept} rows removed` : ""],
    ].map(([v, f, k, sub]) => `<div class="kpi"><span class="kpi-v" data-count="${v}" data-fmt="${f}">0</span><span class="kpi-k">${k}</span>${sub ? `<span class="kpi-s">${sub}</span>` : ""}</div>`).join(""));
    set("#risk-curves", Object.entries(R.risk_curves).map(([f, c]) =>
      `<div class="card chart-card"><div class="card-h"><h3>Disease rate by ${esc(f)}</h3></div>${legend}${groupedBars(f, c, ORDER[f])}</div>`).join(""));
    const bp = ORDER["systolic BP"], ages = ORDER.age;
    set("#bpage", ["real", "synthetic"].map((src) => {
      const g = R.bp_age_table[src];
      return `<div class="heat-wrap"><div class="eyebrow">${src === "real" ? "Real patients" : "Synthetic patients"}</div><div class="heat">
        <div class="h corner">Age \\ BP</div>${bp.map((b) => `<div class="h">${esc(NICE[b])}</div>`).join("")}
        ${ages.map((a, ai) => `<div class="h">${esc(NICE[a])}</div>` + bp.map((b, bi) => { const v = g[a][b];
          return `<div class="cell" style="--v:${v.toFixed(3)};--d:${(ai * 3 + bi) * 40}ms" data-hi="${v > 0.55}">${Math.round(v * 100)}%</div>`; }).join("")).join("")}
      </div></div>`;
    }).join(""));
    text("#bpage-diff", (R.bp_age_table.max_abs_diff * 100).toFixed(1) + " pts");
    const imp = R.importance, feats = Object.keys(imp.values.real).sort((a, b) => imp.values.real[b] - imp.values.real[a]).slice(0, 8);
    const maxI = Math.max(...feats.map((f) => Math.max(imp.values.real[f], imp.values.synthetic[f])));
    set("#importance", `<div class="legend"><span><i class="sw real"></i>Model trained on real</span><span><i class="sw synth"></i>Model trained on synthetic</span></div>` +
      feats.map((f, i) => `<div class="imp" style="--d:${i * 50}ms"><span class="imp-n">${LABEL[f]}</span>
      <span class="imp-bars"><i class="real" style="--w:${Math.max(0.6, imp.values.real[f] / maxI * 100)}%"></i><i class="synth" style="--w:${Math.max(0.6, imp.values.synthetic[f] / maxI * 100)}%"></i></span>
      <span class="imp-v">${imp.values.real[f].toFixed(3)}</span></div>`).join(""));
    text("#imp-rho", imp.spearman.toFixed(2));
    // correlation-gap matrix
    const M = F.correlation_matrices, cols = ["age", "height", "weight", "ap_hi", "ap_lo", "gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"];
    let maxD = 0;
    const cell = 30, left = 100, top = 96, W = left + cols.length * cell + 4, H = top + cols.length * cell + 4;
    let s = `<svg class="corr" viewBox="0 0 ${W} ${H}" role="img" aria-label="Correlation gap matrix">`;
    cols.forEach((c, i) => {
      s += `<text x="${left + i * cell + cell / 2}" y="${top - 6}" font-size="10" transform="rotate(-50 ${left + i * cell + cell / 2} ${top - 6})" class="ax">${LABEL[c]}</text>`;
      s += `<text x="${left - 6}" y="${top + i * cell + cell / 2 + 3}" font-size="10" text-anchor="end" class="ax">${LABEL[c]}</text>`;
      cols.forEach((r, j) => {
        const d = (M.real[c][r] ?? 0) - (M.synthetic[c][r] ?? 0); if (i !== j) maxD = Math.max(maxD, Math.abs(d));
        const a = i === j ? 1 : Math.min(1, Math.abs(d) / 0.05);
        s += `<rect x="${left + i * cell}" y="${top + j * cell}" width="${cell - 2}" height="${cell - 2}" rx="4" class="${i === j ? "diag" : d >= 0 ? "real" : "synth"}" fill-opacity="${i === j ? 1 : Math.max(0.06, a).toFixed(2)}"><title>${LABEL[c]} × ${LABEL[r]}: ${d.toFixed(3)}</title></rect>`;
        if (i !== j && Math.abs(d) >= 0.01) s += `<text x="${left + i * cell + cell / 2 - 1}" y="${top + j * cell + cell / 2 + 3}" font-size="8.5" text-anchor="middle" class="val">${d.toFixed(2).replace("0.", ".")}</text>`;
      });
    });
    set("#corr", s + "</svg>" + `<div class="legend"><span><i class="sw real"></i>stronger in real</span><span><i class="sw synth"></i>stronger in synthetic</span><span>full colour = gap of 0.05</span></div>`);
    text("#corr-max", maxD.toFixed(3));
    renderScorecard();
  }

  // ---------- scorecard ----------
  const CHECKS = {
    rule_pass_rate: ["Quality", "Pass the medical-sense rules", "Every synthetic patient is checked against the same range rules used to clean real data.", (v) => pct(v, 2)],
    alpha_precision: ["Fidelity", "Look like real patients", "α-precision: share of synthetic patients that sit inside the real data's typical region.", (v) => v.toFixed(3)],
    columns_differing: ["Fidelity", "Columns that differ from real", "Kolmogorov–Smirnov / total-variation test on each of the 12 columns.", (v) => v.toFixed(0) + " of 12"],
    correlation_gap: ["Fidelity", "Relationships between columns", "Overall gap between the real and synthetic correlation matrices.", (v) => v.toFixed(3)],
    tstr_auc: ["Utility", "Useful for a risk model", "Model trained on synthetic patients, tested on real ones (AUC).", (v) => v.toFixed(3)],
    mia_auc: ["Privacy", "Resists membership inference", "An attacker guessing who was in the training data; 0.50 is a coin flip.", (v) => v.toFixed(3)],
    dcr_score: ["Privacy", "Not closer to training patients", "Share of synthetic rows nearer to training than to unseen patients; ideal is the training share.", (v) => v.toFixed(3)],
    exact_copies_vs_holdout: ["Privacy", "No copying of training patients", "Exact matches with seen vs never-seen patients, at the same sample size.", (v) => v.toFixed(1)],
    exact_copies_of_train: ["Privacy", "Exact copies of training patients", "Synthetic rows identical to a training patient on all 12 columns.", (v) => v.toFixed(0)],
  };
  function renderScorecard() {
    const A = D.acceptance, keys = Object.keys(A), passed = keys.filter((k) => A[k].pass).length, all = passed === keys.length;
    set("#sc-count", `<span data-count="${passed}" data-fmt="raw">0</span><small>/${keys.length}</small>`);
    $("#sc-count")?.classList.toggle("all", all);
    const ring = $("#sc-ring"); if (ring) ring.style.setProperty("--p", passed / keys.length);
    text("#sc-sub", `Synthetic patients scored against ${D.rows.train.toLocaleString()} real training patients and ${D.rows.holdout.toLocaleString()} never-seen holdout patients.`);
    set("#sc-table", keys.map((k, i) => { const [area, label, help, f] = CHECKS[k] || ["", k, "", (v) => v]; const a = A[k];
      return `<div class="check ${a.pass ? "pass" : "fail"}" style="--d:${i * 50}ms"><span class="check-ico">${a.pass ? "✓" : "!"}</span>
        <div class="check-main"><span class="eyebrow">${area}</span><b>${esc(label)}</b><span class="check-help">${esc(help)}</span></div>
        <div class="check-val"><b>${f(a.value)}</b><span>target ${esc(a.target.replace(">=", "≥").replace("<=", "≤").replace("+/-", "±"))}</span></div></div>`; }).join(""));
    const U = D.utility, bars = [["Trained on real", U.trtr_auc, "real"], ["Trained on synthetic 1×", U.tstr_auc.synth_scale_1, "synth"], ["Trained on synthetic 2×", U.tstr_auc.synth_scale_2, "synth"]];
    set("#auc-bars", bars.map(([l, v, c], i) => `<div class="hbar" style="--d:${i * 80}ms"><span class="hbar-l">${l}</span><span class="hbar-t"><i class="${c}" style="--w:${((v - 0.5) / 0.5 * 100).toFixed(1)}%"></i></span><b class="hbar-v">${v.toFixed(3)}</b></div>`).join("") +
      `<p class="fine">Scale starts at 0.50 (coin flip). Gap to real: ${U.auc_gap.synth_scale_1.toFixed(4)} AUC. Doubling synthetic data adds only ${(U.tstr_auc.synth_scale_2 - U.tstr_auc.synth_scale_1).toFixed(4)}.</p>`);
    const E = D.privacy.exact_copies, M = D.privacy.mia;
    const rows = [["Patients it learned from", E.train_sample_holdout_size_mean ?? E.train_sample_holdout_size, "synth"], ["Patients it never saw", E.holdout, "real"]];
    if (E.real_holdout_rows_matching_real_train !== undefined) rows.push(["Real vs real (baseline)", E.real_holdout_rows_matching_real_train, "neutral"]);
    const maxE = Math.max(...rows.map((r) => r[1]));
    set("#copies", `<p class="fine top">Exact matches per ${D.rows.holdout.toLocaleString()} patients</p>` +
      rows.map(([l, v, c], i) => `<div class="hbar" style="--d:${i * 80}ms"><span class="hbar-l">${l}</span><span class="hbar-t"><i class="${c}" style="--w:${(v / maxE * 100).toFixed(1)}%"></i></span><b class="hbar-v">${Number.isInteger(v) ? v : v.toFixed(1)}</b></div>`).join("") +
      `<p class="fine">Seen and never-seen patients are matched about equally often: coincidence in coarse data, not memorisation. Membership-inference attack AUC ${M.auc.toFixed(3)} (coin flip = 0.50).</p>`);
  }

  // ---------- about ----------
  function renderAbout() {
    const G = D.generation, B = D.baseline, passed = Object.values(D.acceptance).filter((a) => a.pass).length;
    const steps = [
      ["Clean", `${B ? B.rows_raw.toLocaleString() : "70,000"} public records; impossible values removed with medical-range rules → ${B ? B.rows_clean.toLocaleString() : ""} patients`],
      ["Split", `${D.rows.train.toLocaleString()} train · ${D.rows.holdout.toLocaleString()} holdout · ${D.rows.test.toLocaleString()} locked test`],
      ["Generate", G ? `TabDDPM diffusion model, ${G.diffusion_config.iterations.toLocaleString()} steps on an ${G.gpu} in ${G.train_minutes} min → ${D.rows.synth_scale_1.toLocaleString()} synthetic patients` : "TabDDPM diffusion model"],
      ["Verify", `Fidelity, utility and privacy scorecard: ${passed} of ${Object.keys(D.acceptance).length} checks pass`],
      ["Predict", "Risk model trained only on synthetic patients, running inside this page"],
    ];
    set("#steps", steps.map(([t, d], i) => `<div class="step" style="--d:${i * 90}ms"><span class="step-n">${i + 1}</span><b>${t}</b><p>${esc(d)}</p></div>`).join(""));
    text("#ab-synth", D.rows.synth_scale_1.toLocaleString());
    text("#ab-tstr", D.utility.tstr_auc.synth_scale_1.toFixed(3));
    text("#ab-trtr", D.utility.trtr_auc.toFixed(3));
    text("#ab-acc", pct(D.utility.tstr_accuracy.synth_scale_1, 0));
    text("#ab-built", D.built);
    text("#ab-sc", D.scorecard_file);
    text("#model-chip-auc", "AUC " + D.utility.tstr_auc.synth_scale_1.toFixed(3));
    text("#sc-mini", `${passed}/${Object.keys(D.acceptance).length}`);
  }

  renderSession();
  renderAbout();
  renderCharts();
  window.SynthHeartApp = { fill, update, showTab, examples: EXAMPLES, get current() { return current; } };

  // Deep links: #compare / #scorecard / #about open a tab; #example=1..3 loads an example; &add adds it to the worklist.
  const hash = new URLSearchParams(location.hash.slice(1));
  const tab = [...hash.keys()].find((k) => ["check", "compare", "scorecard", "about"].includes(k));
  showTab(tab || "check");
  const ex = +hash.get("example");
  if (ex >= 1 && ex <= EXAMPLES.length) { fill(EXAMPLES[ex - 1].v); if (hash.has("add")) { $("#code").value = "AB"; addToSession(); } }
  else update();
})();
