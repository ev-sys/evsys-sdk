// Hash-routed SPA: experiments → run detail (scalars + evals + predictions).
const app = document.getElementById("app");
const crumbs = document.getElementById("crumbs");
const COLORS = ["#6ea8fe", "#f0883e", "#3fb950", "#db61a2", "#d29922", "#a371f7"];
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 4) => (typeof v === "number" ? (+v.toFixed(d)).toString() : (v == null ? "—" : esc(v)));

function setCrumbs(parts) {
  crumbs.innerHTML = parts.map((p, i) =>
    (i ? " / " : "") + (p.href ? `<a href="${p.href}">${esc(p.text)}</a>` : esc(p.text))).join("");
}
function err(e) { app.innerHTML = `<div class="error">${esc(e.message || e)}</div>`; }

// -- experiments list --------------------------------------------------------
async function viewExperiments() {
  setCrumbs([{ text: "experiments" }]);
  const exps = await apiGet("/api/experiments");
  if (!exps.length) {
    app.innerHTML = `<h1>Experiments</h1><div class="empty">No runs found in this log dir.
      Run something locally (EVSYS_OFFLINE=1) and refresh.</div>`;
    return;
  }
  const rows = exps.map(e => `<tr class="clickable" data-href="#/exp/${encodeURIComponent(e.id)}">
    <td>${esc(e.name)}</td>
    <td>${e.status ? `<span class="pill ${esc(e.status)}">${esc(e.status)}</span>` : "—"}</td>
    <td class="num">${num(e.best_score)}</td>
    <td class="num">${e.n_runs}</td></tr>`).join("");
  app.innerHTML = `<h1>Experiments</h1>
    <table><thead><tr><th>Name</th><th>Status</th><th class="num">Best score</th>
    <th class="num">Runs</th></tr></thead><tbody>${rows}</tbody></table>`;
  wireRows();
}

// -- experiment detail (its runs) -------------------------------------------
async function viewExperiment(expId) {
  const [exps, runs] = await Promise.all([
    apiGet("/api/experiments"),
    apiGet(`/api/experiments/${encodeURIComponent(expId)}/runs`),
  ]);
  const exp = (exps || []).find(e => e.id === expId) || { name: expId };
  setCrumbs([{ text: "experiments", href: "#/" }, { text: exp.name }]);
  if (!runs.length) { app.innerHTML = `<h1>${esc(exp.name)}</h1><div class="empty">No runs.</div>`; return; }
  const rows = runs.map(r => `<tr class="clickable" data-href="#/run/${encodeURIComponent(r.id)}">
    <td>${esc(r.recipe_kind || r.id)}</td>
    <td>${r.status ? `<span class="pill ${esc(r.status)}">${esc(r.status)}</span>` : "—"}</td>
    <td class="num">${num(r.seed, 0)}</td></tr>`).join("");
  app.innerHTML = `<h1>${esc(exp.name)}</h1>
    <table><thead><tr><th>Run</th><th>Status</th><th class="num">Seed</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  wireRows();
}

// -- run detail (scalars + evals + predictions) -----------------------------
async function viewRun(runId) {
  const detail = await apiGet(`/api/runs/${encodeURIComponent(runId)}`);
  if (!detail) { err(new Error("run not found")); return; }
  const run = detail.run;
  const expId = run.experiment_id;
  setCrumbs([{ text: "experiments", href: "#/" },
    ...(expId ? [{ text: "experiment", href: `#/exp/${encodeURIComponent(expId)}` }] : []),
    { text: run.recipe_kind || runId }]);

  app.innerHTML = `<h1>${esc(run.recipe_kind || "run")} <span class="muted">${esc(runId)}</span></h1>
    <div class="kv"><span>status</span><b>${esc(run.status || "—")}</b>
      <span>seed</span><b>${num(run.seed, 0)}</b></div>
    <h2>Scalars</h2><div id="charts" class="charts"></div>
    <h2>Evals</h2><div id="evals"></div>
    <h2>Predictions</h2><div id="preds"></div>`;

  renderCharts(runId, detail.splits);
  renderEvals(runId);
  renderPredictions(runId);
}

async function renderCharts(runId, splits) {
  const host = document.getElementById("charts");
  const data = await apiGet(`/api/runs/${encodeURIComponent(runId)}/metrics`);
  const names = Object.keys(data.series);
  if (!names.length) { host.innerHTML = `<div class="empty">No metrics logged.</div>`; return; }
  const splitColor = {}; (data.splits || []).forEach((s, i) => splitColor[s] = COLORS[i % COLORS.length]);
  host.innerHTML = names.map(n => `<div class="chart-card"><p class="title">${esc(n)}</p>
    <canvas data-metric="${esc(n)}"></canvas>
    <div class="legend">${(data.splits || []).map(s =>
      `<span><span class="dot" style="background:${splitColor[s]}"></span>${esc(s)}</span>`).join("")}</div>
    </div>`).join("");
  for (const canvas of host.querySelectorAll("canvas")) {
    const n = canvas.dataset.metric;
    const series = Object.entries(data.series[n]).map(([split, points]) =>
      ({ label: split, color: splitColor[split] || COLORS[0], points }));
    drawLineChart(canvas, series);
  }
}

async function renderEvals(runId) {
  const host = document.getElementById("evals");
  const evals = await apiGet(`/api/runs/${encodeURIComponent(runId)}/evals`);
  if (!evals.length) { host.innerHTML = `<div class="empty">No evals.</div>`; return; }
  const cols = ["mean_reward", "pass_rate", "n_tasks", "time_per_task", "tokens_per_task", "cost_per_task"];
  const present = cols.filter(c => evals.some(e => (e.metrics || {})[c] != null));
  const head = ["Benchmark", "Step", ...present].map(h => `<th class="${present.includes(h) ? "num" : ""}">${esc(h)}</th>`).join("");
  const rows = evals.map(e => `<tr><td>${esc(e.benchmark_id || e.model_ref || "—")}</td>
    <td>${e.step == null ? "final" : esc(e.step)}</td>
    ${present.map(c => `<td class="num">${num((e.metrics || {})[c])}</td>`).join("")}</tr>`).join("");
  host.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

async function renderPredictions(runId) {
  const host = document.getElementById("preds");
  const data = await apiGet(`/api/runs/${encodeURIComponent(runId)}/predictions?limit=100`);
  if (!data.total) { host.innerHTML = `<div class="empty">No predictions.</div>`; return; }
  const rows = data.predictions.map(p => {
    const m = p.metadata || {};
    return `<tr><td>${esc(p.task_id)}</td><td class="num">${num(p.sample_idx, 0)}</td>
      <td class="num">${num(p.reward)}</td><td class="num">${num(m.latency_s, 2)}</td>
      <td class="num">${num((m.prompt_tokens || 0) + (m.completion_tokens || 0), 0)}</td>
      <td class="num">${m.cost_usd == null ? "—" : "$" + num(m.cost_usd, 4)}</td>
      <td><details><summary>view</summary><pre>instruction:\n${esc(p.instruction)}\n\nexpected:\n${esc(JSON.stringify(p.expected))}\n\noutput:\n${esc(p.model_output)}</pre></details></td></tr>`;
  }).join("");
  host.innerHTML = `<div class="muted">showing ${data.predictions.length} of ${data.total}</div>
    <table><thead><tr><th>Task</th><th class="num">Sample</th><th class="num">Reward</th>
    <th class="num">Latency s</th><th class="num">Tokens</th><th class="num">Cost</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function wireRows() {
  for (const tr of app.querySelectorAll("tr.clickable"))
    tr.onclick = () => { location.hash = tr.dataset.href.slice(1); };
}

async function router() {
  const h = location.hash.replace(/^#/, "") || "/";
  app.innerHTML = `<div class="loading">loading…</div>`;
  try {
    let m;
    if ((m = h.match(/^\/run\/(.+)$/))) await viewRun(decodeURIComponent(m[1]));
    else if ((m = h.match(/^\/exp\/(.+)$/))) await viewExperiment(decodeURIComponent(m[1]));
    else await viewExperiments();
  } catch (e) { err(e); }
}
window.addEventListener("hashchange", router);
router();
