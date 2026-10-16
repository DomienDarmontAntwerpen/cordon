/*
 * Cordon dashboard logic. No framework, no build step; loaded after the vendored Chart.js.
 * CSP-compliant: never uses innerHTML, only textContent/DOM APIs.
 *
 * Data model: the one data source is GET /metrics, which returns raw ledger
 * rows (ids, costs, tags, savings and guardrail_actions as JSON strings).
 * Every KPI, chart and panel below is an aggregation computed here, in the
 * browser, from those rows - there is no separate aggregation endpoint.
 */

// --- Section: state — the API key lives in sessionStorage only, per tab ---
const state = {
  apiKey: null,
  isAdmin: false,
  tenants: [],        // admin mode only: [{id, name, settings, created_at}]
  tenantId: null,     // admin mode only: the tenant currently viewed, or null for "all"
  ownTenant: null,    // tenant-key mode: this tenant's own settings
  period: "24h",
  refreshTimer: null,
  charts: {},
};

const PERIOD_SECONDS = { "24h": 86400, "7d": 604800, "30d": 2592000 };

// --- Section: sessionStorage — the key, and nothing else sensitive, is ever persisted ---
function loadKey() {
  try {
    return sessionStorage.getItem("cordon_dashboard_key");
  } catch {
    return null;
  }
}
function saveKey(key) {
  try {
    sessionStorage.setItem("cordon_dashboard_key", key);
  } catch {
    /* private-browsing or blocked storage: the session still works, just not across a reload */
  }
}
function clearKey() {
  try {
    sessionStorage.removeItem("cordon_dashboard_key");
  } catch {
    /* ignore */
  }
}

// --- Section: fetch helper — every call is Bearer-authenticated with the stored key ---
async function apiGet(path) {
  const response = await fetch(path, { headers: { Authorization: `Bearer ${state.apiKey}` } });
  if (!response.ok) {
    const error = new Error(`request failed: ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function apiPatch(path, body) {
  const response = await fetch(path, {
    method: "PATCH",
    headers: { Authorization: `Bearer ${state.apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error((data.error && data.error.message) || `request failed: ${response.status}`);
  return data;
}

// --- Section: connect flow — try admin first (tenant listing), then tenant-key metrics ---
async function tryConnect(key) {
  state.apiKey = key;
  try {
    state.tenants = await apiGet("/admin/tenants");
    state.isAdmin = true;
    return true;
  } catch (adminError) {
    if (adminError.status !== 401) throw adminError;
  }
  try {
    await apiGet("/metrics");
    state.isAdmin = false;
    state.ownTenant = await apiGet("/v1/tenant");
    document.getElementById("try-key").value = state.apiKey;  // a team key can try requests directly
    return true;
  } catch {
    state.apiKey = null;
    return false;
  }
}

function showConnectError(message) {
  const el = document.getElementById("connect-error");
  el.textContent = message;
  el.hidden = false;
}

async function handleConnectSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("connect-key-input");
  const key = input.value.trim();
  if (!key) return;
  const ok = await tryConnect(key);
  if (!ok) {
    showConnectError("That key was not accepted. Check it and try again.");
    return;
  }
  saveKey(key);
  enterApp();
}

function lock() {
  clearKey();
  state.apiKey = null;
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  document.getElementById("app").hidden = true;
  document.getElementById("connect-screen").hidden = false;
  document.getElementById("connect-key-input").value = "";
}

// --- Section: entering the app once a key is accepted ---
function enterApp() {
  document.getElementById("connect-screen").hidden = true;
  document.getElementById("app").hidden = false;

  const tenantSelect = document.getElementById("tenant-select");
  if (state.isAdmin && state.tenants.length > 0) {
    tenantSelect.hidden = false;
    tenantSelect.textContent = "";
    const allOption = document.createElement("option");
    allOption.value = "";
    allOption.textContent = "All teams";
    tenantSelect.appendChild(allOption);
    for (const tenant of state.tenants) {
      const option = document.createElement("option");
      option.value = tenant.id;
      option.textContent = tenant.name;
      tenantSelect.appendChild(option);
    }
    tenantSelect.value = "";
  } else {
    tenantSelect.hidden = true;
  }

  refreshAll();
  state.refreshTimer = setInterval(refreshAll, 15000);
}

// --- Section: period / tenant selector wiring ---
function setPeriod(period) {
  state.period = period;
  for (const button of document.querySelectorAll("#period-select .segmented-btn")) {
    button.setAttribute("aria-pressed", String(button.dataset.period === period));
  }
  refreshAll();
}

// --- Section: theme — a per-viewer convenience, not sensitive, so localStorage is fine here ---
function loadTheme() {
  try {
    return localStorage.getItem("cordon_theme_choice");
  } catch {
    return null;
  }
}
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
}
function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const effectiveCurrent = current || "light";  // light unless the viewer picks dark
  const next = effectiveCurrent === "dark" ? "light" : "dark";
  applyTheme(next);
  try {
    localStorage.setItem("cordon_theme_choice", next);
  } catch {
    /* ignore */
  }
}

// --- Section: data fetching — one window for the current period, one for the prior period ---
function currentTenantFilter() {
  return state.isAdmin && state.tenantId ? `&tenant_id=${encodeURIComponent(state.tenantId)}` : "";
}

async function fetchWindow(sinceSeconds) {
  const since = Math.floor(Date.now() / 1000 - sinceSeconds);
  const data = await apiGet(`/metrics?since=${since}${currentTenantFilter()}`);
  return data.requests || [];
}

async function refreshAll() {
  const periodSeconds = PERIOD_SECONDS[state.period];
  let rows, previousRows, dailyRows, monthlyRows;
  try {
    const [allRows, daily, monthly] = await Promise.all([
      fetchWindow(periodSeconds * 2),
      fetchWindow(secondsSinceUtcMidnight()),
      fetchWindow(secondsSinceUtcMonthStart()),
      refreshTeams(),
    ]);
    const splitAt = Date.now() / 1000 - periodSeconds;
    rows = allRows.filter((r) => r.timestamp >= splitAt);
    previousRows = allRows.filter((r) => r.timestamp < splitAt);
    dailyRows = daily;
    monthlyRows = monthly;
  } catch {
    return; // a transient failure just skips this refresh tick; the next one retries
  }

  renderModeBadge();
  renderSettingsCard();
  renderTeamsTable(rows);
  renderKpis(rows, previousRows);
  renderCharts(rows, periodSeconds);
  renderTopRequests(rows);
  renderSecurityPanel(rows);
  renderAlerts(rows, dailyRows, monthlyRows);
  renderBudgetCard(dailyRows, monthlyRows);
  renderRequestsTable(rows);
  document.getElementById("last-updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

// Team settings (mode, features, budgets) are re-read on every refresh, so a change
// made with tools/admin.py or the admin API shows up without logging in again.
async function refreshTeams() {
  if (state.isAdmin) state.tenants = await apiGet("/admin/tenants");
  else state.ownTenant = await apiGet("/v1/tenant");
}

function secondsSinceUtcMidnight() {
  const now = new Date();
  const midnight = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  return (Date.now() - midnight.getTime()) / 1000;
}
function secondsSinceUtcMonthStart() {
  const now = new Date();
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  return (Date.now() - start.getTime()) / 1000;
}

// --- Section: row helpers — parsing the JSON-string columns, consistently ---
function parseJsonColumn(value, fallback) {
  if (!value) return fallback;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}
function rowTags(row) {
  return parseJsonColumn(row.tags_json, []);
}
function rowSavings(row) {
  return parseJsonColumn(row.savings_json, {});
}
// A savings entry is "applied" (really saved) unless it says otherwise; observe mode's
// "would save" figures carry applied: false and are never counted as real savings.
function isApplied(entry) {
  return entry.applied !== false;
}
function wouldSaveEur(row) {
  return sum(Object.values(rowSavings(row)).filter((e) => !isApplied(e)).map((e) => e.eur || 0));
}
function rowGuardrailActions(row) {
  return parseJsonColumn(row.guardrail_actions_json, {});
}
function isServed(row) {
  return row.status !== "blocked";
}
function isOk(row) {
  return row.status === "ok";
}
function sum(values) {
  return values.reduce((a, b) => a + b, 0);
}
function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
  return sorted[index];
}
function formatEur(amount) {
  return `€${amount.toFixed(amount < 1 ? 4 : 2)}`;
}

// --- Section: teams — names, modes and features come from /admin/tenants (or /v1/tenant) ---
function viewingAllTeams() {
  return state.isAdmin && !state.tenantId;
}
function teamById(id) {
  if (!state.isAdmin) return state.ownTenant;
  return state.tenants.find((t) => t.id === id) || null;
}
function teamName(id) {
  const team = teamById(id);
  return team ? team.name : id;
}
function currentTeam() {
  return state.isAdmin ? teamById(state.tenantId) : state.ownTenant;
}

// --- Section: KPI row — six numbers, each with one line saying what it means ---
function computeKpis(rows) {
  const answered = rows.filter((r) => isServed(r) && r.cost_baseline_eur != null);
  const actual = sum(answered.map((r) => r.cost_actual_eur || 0));
  const baseline = sum(answered.map((r) => r.cost_baseline_eur || 0));
  const saved = baseline - actual;
  const okRows = rows.filter(isOk);
  return {
    actual,
    baseline,
    saved,
    pctSaved: baseline > 0 ? (saved / baseline) * 100 : 0,
    wouldSave: sum(rows.map(wouldSaveEur)),
    requests: rows.filter(isServed).length,
    cacheHitRate: okRows.length > 0 ? (okRows.filter((r) => r.cache_type).length / okRows.length) * 100 : 0,
    blocked: rows.filter((r) => r.status === "blocked").length,
    p95Overhead: percentile(okRows.map((r) => r.overhead_ms || 0), 0.95),
  };
}

function renderKpis(rows, previousRows) {
  const now = computeKpis(rows);
  const before = computeKpis(previousRows);
  const cards = [
    { tone: "slate", label: "Cost", value: formatEur(now.actual), hint: "Paid to the AI providers in this period.",
      change: pctChange(now.actual, before.actual), invert: true },
    { tone: "green", label: "Saved", value: `${formatEur(now.saved)} (${now.pctSaved.toFixed(0)}%)`,
      hint: `Without Cordon this would have cost ${formatEur(now.baseline)}.` },
    { tone: "ochre", label: "Could still save", value: formatEur(now.wouldSave),
      hint: now.wouldSave > 0 ? "Found in observe mode. Switch to enforce to get it." : "Nothing extra found in observe mode." },
    { tone: "blue", label: "Requests answered", value: String(now.requests),
      hint: `${now.cacheHitRate.toFixed(0)}% answered from the cache, at no cost.`, change: pctChange(now.requests, before.requests) },
    { tone: "red", label: "Blocked", value: String(now.blocked), hint: "Refused by a limit or a security check. See below for why." },
    { tone: "slate", label: "Delay added by Cordon", value: `${now.p95Overhead.toFixed(0)} ms`, hint: "For 95% of requests, Cordon adds at most this." },
  ];
  const container = document.getElementById("kpi-row");
  container.textContent = "";
  for (const card of cards) container.appendChild(kpiCard(card));
}

function kpiCard(card) {
  const el = document.createElement("div");
  el.className = card.tone ? `kpi-card tone-${card.tone}` : "kpi-card";
  for (const [className, text] of [["kpi-label", card.label], ["kpi-value", card.value], ["kpi-hint", card.hint]]) {
    const part = document.createElement("div");
    part.className = className;
    part.textContent = text;
    el.appendChild(part);
  }
  if (card.change != null && Number.isFinite(card.change)) {
    const change = document.createElement("div");
    const good = card.invert ? card.change <= 0 : card.change >= 0;
    change.className = `kpi-change ${good ? "positive" : "negative"}`;
    change.textContent = `${card.change >= 0 ? "+" : ""}${card.change.toFixed(0)}% vs. the period before`;
    el.appendChild(change);
  }
  return el;
}

// --- Section: mode badge — observe or enforce, for the team being viewed ---
const MODE_NOTES = {
  observe: "Observe mode: Cordon only measures. Every request goes through unchanged; \"Could still save\" shows what enforce would bring.",
  enforce: "Enforce mode: Cordon acts - it hides personal data, blocks attacks, enforces limits and applies the savings that are switched on.",
};

function renderModeBadge() {
  const badge = document.getElementById("mode-badge");
  const note = document.getElementById("mode-note");
  const team = currentTeam();
  const mode = team ? team.settings.mode || "observe" : null;
  badge.hidden = note.hidden = !mode;
  if (!mode) return;
  badge.textContent = mode.toUpperCase();
  badge.className = `mode-badge mode-${mode}`;
  note.textContent = MODE_NOTES[mode] || "";
}

// --- Section: team settings — the admin switches a team's mode and features here ---
const FEATURES = [
  ["trim", "Shorter prompts", "Removes duplicate text, old conversation turns and oversized tool output."],
  ["vision", "Smaller images", "Downscales images to what the model actually needs."],
  ["cache", "Cache (same question)", "Answers an identical question again for free."],
  ["semantic_cache", "Cache (similar question)", "Also answers similar questions from the cache. Off by default: about 1-5% of its hits are wrong."],
  ["routing", "Cheaper model", "Sends a request to a cheaper model where the benchmark proved it, with its check: an answer the check doubts goes to the strong model."],
  ["max_tokens", "Answer length limit", "Caps answers that set no length of their own."],
  ["circuit_breaker", "Agent loop stop", "Stops an agent that keeps asking almost the same question."],
];

function featureLabel(name) {
  return (FEATURES.find(([key]) => key === name) || [name, name])[1];
}

function renderSettingsCard() {
  const team = state.isAdmin ? teamById(state.tenantId) : null;
  document.getElementById("settings-card").hidden = !team;
  if (!team) return;
  const mode = team.settings.mode || "observe";
  for (const button of document.querySelectorAll("#mode-switch .segmented-btn")) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  }
  const enabled = new Set(team.settings.features || []);
  const list = document.getElementById("feature-list");
  list.textContent = "";
  for (const [name, label, description] of FEATURES) list.appendChild(featureItem(name, label, description, enabled.has(name)));
  list.appendChild(settingItem(team, "mask_instructions", "Mask personal data in instructions",
    "For RAG apps that paste documents into the system message. Off by default: it ends the provider's prompt-cache discount."));
}

// A yes/no team setting (not a feature), shown and switched like a feature.
function settingItem(team, name, label, description) {
  const item = featureItem(name, label, description, Boolean(team.settings[name]));
  item.querySelector("input").onchange = (event) => changeTeamSettings(
    { [name]: event.target.checked }, `${label} switched ${event.target.checked ? "on" : "off"} for ${team.name}.`);
  return item;
}

function featureItem(name, label, description, checked) {
  const li = document.createElement("li");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.id = `feature-${name}`;
  box.checked = checked;
  box.onchange = () => toggleFeature(name, box.checked);
  const text = document.createElement("label");
  text.htmlFor = box.id;
  const title = document.createElement("strong");
  title.textContent = label;
  const detail = document.createElement("span");
  detail.className = "feature-description";
  detail.textContent = description;
  text.appendChild(title);
  text.appendChild(detail);
  li.appendChild(box);
  li.appendChild(text);
  return li;
}

// One change at a time: the controls are locked until the server has saved it and the
// page shows the saved settings, so a second click can never overwrite the first change.
async function changeTeamSettings(body, doneText) {
  const status = document.getElementById("settings-status");
  const controls = document.querySelectorAll("#settings-card input, #settings-card button");
  for (const control of controls) control.disabled = true;
  try {
    await apiPatch(`/admin/tenants/${encodeURIComponent(state.tenantId)}`, body);
    status.className = "settings-status ok";
    status.textContent = doneText;
  } catch (error) {
    status.className = "settings-status failed";
    status.textContent = `Not changed: ${error.message}`;
  }
  status.hidden = false;
  await refreshAll();  // re-renders the card from the saved settings
  for (const control of document.querySelectorAll("#settings-card input, #settings-card button")) control.disabled = false;
}

function setMode(mode) {
  const team = teamById(state.tenantId);
  if (!team || (team.settings.mode || "observe") === mode) return;
  if (mode === "enforce" && !window.confirm(`Switch "${team.name}" to enforce? Cordon will start acting on its requests: the features ticked below take effect on the next request.`)) return;
  changeTeamSettings({ mode }, `${team.name} is now in ${mode} mode.`);
}

function toggleFeature(name, on) {
  const team = teamById(state.tenantId);
  if (!team) return;
  // Only features that still exist: a team stored before one was retired must stay editable.
  const features = new Set((team.settings.features || []).filter((f) => FEATURES.some(([known]) => known === f)));
  if (on) features.add(name);
  else features.delete(name);
  const label = FEATURES.find(([n]) => n === name)[1];
  changeTeamSettings({ features: [...features] }, `${label} switched ${on ? "on" : "off"} for ${team.name}.`);
}

// --- Section: teams overview — one row per team, admin "All teams" view only ---
function renderTeamsTable(rows) {
  const card = document.getElementById("teams-card");
  card.hidden = !viewingAllTeams();
  if (card.hidden) return;
  const body = document.getElementById("teams-table-body");
  body.textContent = "";
  for (const team of state.tenants) {
    const k = computeKpis(rows.filter((r) => r.tenant_id === team.id));
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.appendChild(cell(team.name));
    tr.appendChild(modeSwitchCell(team));
    const features = cell((team.settings.features || []).map(featureLabel).join(", ") || "none");
    features.className = "wrap-cell";  // a long feature list wraps instead of pushing the numbers off-screen
    tr.appendChild(features);
    tr.appendChild(cell(String(k.requests)));
    tr.appendChild(cell(String(k.blocked)));
    tr.appendChild(cell(formatEur(k.actual)));
    tr.appendChild(cell(formatEur(k.saved)));
    tr.appendChild(cell(formatEur(k.wouldSave)));
    tr.addEventListener("click", () => selectTeam(team.id));
    body.appendChild(tr);
  }
}

// Observe / Enforce buttons on every team row: the mode is the first thing an admin changes.
function modeSwitchCell(team) {
  const td = document.createElement("td");
  const group = document.createElement("div");
  group.className = "segmented";
  const current = team.settings.mode || "observe";
  for (const mode of ["observe", "enforce"]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "segmented-btn";
    button.textContent = mode === "observe" ? "Observe" : "Enforce";
    button.setAttribute("aria-pressed", String(mode === current));
    button.addEventListener("click", (event) => {
      event.stopPropagation();  // a click on the switch is not a click on the row
      setTeamMode(team, mode);
    });
    group.appendChild(button);
  }
  td.appendChild(group);
  return td;
}

async function setTeamMode(team, mode) {
  if ((team.settings.mode || "observe") === mode) return;
  if (mode === "enforce" && !window.confirm(`Switch "${team.name}" to enforce? Cordon will start acting on its requests.`)) return;
  try {
    await apiPatch(`/admin/tenants/${encodeURIComponent(team.id)}`, { mode });
  } catch (error) {
    window.alert(`Not changed: ${error.message}`);
  }
  await refreshAll();
}

function selectTeam(id) {
  state.tenantId = id;
  document.getElementById("tenant-select").value = id || "";
  refreshAll();
}

function modeCell(mode) {
  const td = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = `mode-badge mode-${mode}`;
  pill.textContent = mode.toUpperCase();
  td.appendChild(pill);
  return td;
}

// null (no comparison shown) when the period before had nothing to compare with.
function pctChange(current, previous) {
  if (previous === 0) return null;
  return ((current - previous) / Math.abs(previous)) * 100;
}

// --- Section: charts ---
function chartColors() {
  const styles = getComputedStyle(document.documentElement);
  return {
    text: styles.getPropertyValue("--text-muted").trim(),
    border: styles.getPropertyValue("--border").trim(),
    accent: styles.getPropertyValue("--accent").trim(),
    alert: styles.getPropertyValue("--alert").trim(),
    palette: ["--accent", "--chart-2", "--chart-3", "--chart-4", "--chart-5", "--chart-6"].map((name) => styles.getPropertyValue(name).trim()),
  };
}

function getOrCreateChart(id, config) {
  if (state.charts[id]) {
    state.charts[id].data = config.data;
    state.charts[id].options = config.options;
    state.charts[id].update();
    return state.charts[id];
  }
  const canvas = document.getElementById(id);
  Chart.defaults.locale = "en-US";  // the same number format as the rest of the page (0.016, not 0,016)
  state.charts[id] = new Chart(canvas, config);
  return state.charts[id];
}

function bucketLabelsAndEdges(periodSeconds) {
  const bucketCount = periodSeconds <= PERIOD_SECONDS["24h"] ? 24 : periodSeconds <= PERIOD_SECONDS["7d"] ? 7 : 30;
  const bucketSeconds = periodSeconds / bucketCount;
  const now = Date.now() / 1000;
  const edges = [];
  for (let i = bucketCount; i >= 0; i--) edges.push(now - i * bucketSeconds);
  const labels = edges.slice(1).map((edge) => {
    const date = new Date(edge * 1000);
    return bucketCount === 24 ? date.toLocaleTimeString([], { hour: "2-digit" }) : date.toLocaleDateString([], { month: "short", day: "numeric" });
  });
  return { edges, labels };
}

function renderCostOverTimeChart(rows, periodSeconds) {
  const { edges, labels } = bucketLabelsAndEdges(periodSeconds);
  const actualByBucket = new Array(labels.length).fill(0);
  const baselineByBucket = new Array(labels.length).fill(0);
  for (const row of rows) {
    if (!isServed(row) || row.cost_baseline_eur == null) continue;
    for (let i = 0; i < labels.length; i++) {
      if (row.timestamp >= edges[i] && row.timestamp < edges[i + 1]) {
        actualByBucket[i] += row.cost_actual_eur || 0;
        baselineByBucket[i] += row.cost_baseline_eur || 0;
        break;
      }
    }
  }
  const colors = chartColors();
  getOrCreateChart("chart-cost-over-time", {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Without Cordon", data: baselineByBucket, borderColor: colors.palette[2], backgroundColor: "transparent", borderDash: [6, 4], tension: 0.3 },
        { label: "Paid", data: actualByBucket, borderColor: colors.accent, backgroundColor: "transparent", tension: 0.3 },
      ],
    },
    options: chartOptions(colors),
  });
}

const TECHNIQUE_LABELS = {
  trim: "Shorter prompts", vision: "Smaller images", cache: "Cache (same question)",
  semantic_cache: "Cache (similar question)", routing: "Cheaper model",
};

function renderSavingsByTechniqueChart(rows) {
  const saved = {};
  const couldSave = {};
  for (const row of rows) {
    for (const [technique, entry] of Object.entries(rowSavings(row))) {
      const target = isApplied(entry) ? saved : couldSave;
      target[technique] = (target[technique] || 0) + (entry.eur || 0);
    }
  }
  const techniques = Object.keys({ ...saved, ...couldSave });
  const colors = chartColors();
  getOrCreateChart("chart-savings-technique", {
    type: "bar",
    data: {
      labels: techniques.map((t) => TECHNIQUE_LABELS[t] || t),
      datasets: [
        { label: "Saved (€)", data: techniques.map((t) => saved[t] || 0), backgroundColor: colors.accent },
        { label: "Could save (€)", data: techniques.map((t) => couldSave[t] || 0), backgroundColor: colors.palette[2] },
      ],
    },
    options: { ...chartOptions(colors), indexAxis: "y" },
  });
}

// All teams: cost per team. One team: cost per tag (the label an app sends in X-Cordon-Tags).
function renderWhereMoneyGoesChart(rows) {
  const byTeam = viewingAllTeams();
  document.getElementById("money-hint").textContent = byTeam
    ? "Cost per team. Click a team in the table above for its details."
    : "Cost per tag - the label an app sends with each request (for example \"faq\" or \"contracts\").";
  const totals = {};
  for (const row of rows) {
    const labels = byTeam ? [teamName(row.tenant_id)] : rowTags(row);
    for (const label of labels.length ? labels : ["(no tag)"]) {
      totals[label] = (totals[label] || 0) + (row.cost_actual_eur || 0);
    }
  }
  const colors = chartColors();
  const labels = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
  getOrCreateChart("chart-where-money-goes", {
    type: "bar",
    data: { labels, datasets: [{ label: "Cost (€)", data: labels.map((l) => totals[l]), backgroundColor: colors.palette[1] }] },
    options: chartOptions(colors),
  });
}

function renderRoutingSplitChart(rows) {
  const counts = {};
  for (const row of rows) {
    if (!row.served_model) continue;
    counts[row.served_model] = (counts[row.served_model] || 0) + 1;
  }
  const colors = chartColors();
  const labels = Object.keys(counts);
  getOrCreateChart("chart-routing-split", {
    type: "doughnut",
    data: { labels, datasets: [{ data: labels.map((l) => counts[l]), backgroundColor: colors.palette }] },
    // No `scales` here: a doughnut has no axes (the shared chartOptions() would draw an x/y grid around it).
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: colors.text } } } },
  });
}

function chartOptions(colors) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: colors.text } } },
    scales: {
      x: { ticks: { color: colors.text }, grid: { color: colors.border } },
      y: { ticks: { color: colors.text }, grid: { color: colors.border } },
    },
  };
}

function renderCharts(rows, periodSeconds) {
  renderCostOverTimeChart(rows, periodSeconds);
  renderSavingsByTechniqueChart(rows);
  renderWhereMoneyGoesChart(rows);
  renderRoutingSplitChart(rows);
}

// --- Section: security & privacy panel ---
function renderStatList(elementId, counts, emptyText) {
  const list = document.getElementById(elementId);
  list.textContent = "";
  const entries = Object.entries(counts).filter(([, count]) => count > 0);
  if (entries.length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyText;
    list.appendChild(li);
    return;
  }
  for (const [label, count] of entries.sort((a, b) => b[1] - a[1])) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("span");
    value.className = "stat-value";
    value.textContent = String(count);
    li.appendChild(name);
    li.appendChild(value);
    list.appendChild(li);
  }
}

const BLOCK_REASON_LABELS = {
  rate_limit_exceeded: "Rate limit",
  daily_cap: "Daily budget cap",
  monthly_budget: "Monthly budget",
  agent_loop_detected: "Agent loop",
  model_not_allowed: "Allowlist (model)",
  model_not_priced: "Allowlist (model)",
  provider_not_allowed: "Allowlist (provider)",
  provider_unavailable: "Allowlist (provider)",
  too_many_messages: "Size (messages)",
  input_too_large: "Size (tokens)",
  too_many_images: "Size (images)",
  image_too_large: "Size (image pixels)",
  pii_blocked: "Personal data (input)",
  injection_blocked: "Prompt injection",
  invalid_tag: "Size (tags)",
};

function renderSecurityPanel(rows) {
  const inputPii = {};
  const outputPii = {};
  let injectionsFlagged = 0;
  let leaksRedacted = 0;
  let pblocked = 0;
  let iblocked = 0;
  const blockedByReason = {};

  for (const row of rows) {
    const actions = rowGuardrailActions(row);
    for (const [key, count] of Object.entries(actions)) {
      if (key.startsWith("input_pii_")) inputPii[key.slice("input_pii_".length)] = (inputPii[key.slice("input_pii_".length)] || 0) + count;
      if (key.startsWith("output_pii_")) outputPii[key.slice("output_pii_".length)] = (outputPii[key.slice("output_pii_".length)] || 0) + count;
      if (key === "injection_flagged") injectionsFlagged += count;
      if (key === "system_prompt_leak_redacted") leaksRedacted += count;
    }
    if (row.status === "blocked" && row.error_code) {
      // Every blocked request is counted here, security blocks included, so this
      // list always agrees with the "blocked" counts in the panels next to it.
      if (row.error_code === "pii_blocked") pblocked += 1;
      if (row.error_code === "injection_blocked") iblocked += 1;
      const label = row.error_code.startsWith("output_blocked_")
        ? "Output (personal data or leak)"
        : BLOCK_REASON_LABELS[row.error_code] || row.error_code;
      blockedByReason[label] = (blockedByReason[label] || 0) + 1;
    }
  }

  renderStatList("security-input-pii", inputPii, "No input PII seen in this period.");
  renderStatList("security-output-pii", outputPii, "No output PII seen in this period.");
  renderStatList(
    "security-injections",
    { "Injections flagged": injectionsFlagged, "Injections blocked": iblocked, "PII blocked": pblocked, "System-prompt leaks redacted": leaksRedacted },
    "No injection or leak activity in this period."
  );
  renderStatList("security-blocked", blockedByReason, "No requests blocked in this period.");
}

// --- Section: alerts ---
function renderAlerts(rows, dailyRows, monthlyRows) {
  const list = document.getElementById("alerts-list");
  list.textContent = "";
  const alerts = [];

  const budget = activeBudget();
  if (budget) {
    const dailyUsed = sum(dailyRows.filter(isServed).map((r) => r.cost_actual_eur || 0));
    const monthlyUsed = sum(monthlyRows.filter(isServed).map((r) => r.cost_actual_eur || 0));
    if (budget.daily > 0) {
      const pct = (dailyUsed / budget.daily) * 100;
      if (pct >= 100) alerts.push({ severity: "alert", text: `Daily budget cap reached (${formatEur(dailyUsed)} / ${formatEur(budget.daily)}).` });
      else if (pct >= 80) alerts.push({ severity: "warn", text: `Daily budget at ${pct.toFixed(0)}% (${formatEur(dailyUsed)} / ${formatEur(budget.daily)}).` });
    }
    if (budget.monthly > 0) {
      const pct = (monthlyUsed / budget.monthly) * 100;
      if (pct >= 100) alerts.push({ severity: "alert", text: `Monthly budget reached (${formatEur(monthlyUsed)} / ${formatEur(budget.monthly)}).` });
      else if (pct >= 80) alerts.push({ severity: "warn", text: `Monthly budget at ${pct.toFixed(0)}% (${formatEur(monthlyUsed)} / ${formatEur(budget.monthly)}).` });
    }
  }

  const hourlyCosts = hourlyBucketCosts(rows);
  if (hourlyCosts.current !== null && hourlyCosts.median > 0 && hourlyCosts.current > hourlyCosts.median * 3) {
    alerts.push({ severity: "alert", text: `Cost anomaly: this hour's cost (${formatEur(hourlyCosts.current)}) is over 3x the 7-day median (${formatEur(hourlyCosts.median)}).` });
  }

  const loopBlocks = rows.filter((r) => r.status === "blocked" && r.error_code === "agent_loop_detected").length;
  if (loopBlocks > 0) alerts.push({ severity: "warn", text: `${loopBlocks} agent-loop request${loopBlocks === 1 ? "" : "s"} blocked in this period.` });

  const injections = rows.filter((r) => r.error_code === "injection_blocked" || rowGuardrailActions(r).injection_flagged);
  if (injections.length > 0) {
    const teams = [...new Set(injections.map((r) => teamName(r.tenant_id)))].join(", ");
    alerts.push({ severity: "alert", text: `${injections.length} prompt-injection attempt${injections.length === 1 ? "" : "s"} in this period (team: ${teams}).` });
  }

  const failOpenEvents = rows.filter((r) => r.error_code === "internal_fail_open").length;
  if (failOpenEvents > 0) alerts.push({ severity: "warn", text: `${failOpenEvents} request${failOpenEvents === 1 ? "" : "s"} fell back to an unprotected passthrough call after an internal failure.` });

  if (alerts.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No active alerts.";
    list.appendChild(li);
    return;
  }
  for (const alert of alerts) {
    const li = document.createElement("li");
    li.className = `severity-${alert.severity}`;
    li.textContent = alert.text;
    list.appendChild(li);
  }
}

function hourlyBucketCosts(rows) {
  const nowHour = Math.floor(Date.now() / 1000 / 3600);
  const byHour = {};
  for (const row of rows) {
    if (!isServed(row) || row.cost_actual_eur == null) continue;
    const hour = Math.floor(row.timestamp / 3600);
    byHour[hour] = (byHour[hour] || 0) + row.cost_actual_eur;
  }
  const pastHours = Object.entries(byHour)
    .filter(([hour]) => Number(hour) !== nowHour)
    .map(([, cost]) => cost);
  const current = byHour[nowHour] ?? null;
  const median = percentile(pastHours, 0.5);
  return { current, median };
}

// --- Section: budget card ---
function activeBudget() {
  if (state.isAdmin) {
    if (!state.tenantId) return null; // "all tenants" has no single budget to show
    const tenant = state.tenants.find((t) => t.id === state.tenantId);
    if (!tenant) return null;
    return { daily: tenant.settings.budget_daily_eur || 0, monthly: tenant.settings.budget_monthly_eur || 0 };
  }
  if (!state.ownTenant) return null;
  return { daily: state.ownTenant.settings.budget_daily_eur || 0, monthly: state.ownTenant.settings.budget_monthly_eur || 0 };
}

function renderBudgetCard(dailyRows, monthlyRows) {
  const container = document.getElementById("budget-card");
  container.textContent = "";
  const budget = activeBudget();
  if (!budget) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = state.isAdmin ? "Select a team to see its budget." : "No budget configured.";
    container.appendChild(p);
    return;
  }

  const dailyUsed = sum(dailyRows.filter(isServed).map((r) => r.cost_actual_eur || 0));
  const monthlyUsed = sum(monthlyRows.filter(isServed).map((r) => r.cost_actual_eur || 0));

  const now = new Date();
  const dayOfMonth = now.getUTCDate();
  const daysInMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 0)).getUTCDate();
  const forecast = dayOfMonth > 0 ? (monthlyUsed / dayOfMonth) * daysInMonth : monthlyUsed;

  container.appendChild(budgetBlock("Today", dailyUsed, budget.daily));
  container.appendChild(budgetBlock("This month", monthlyUsed, budget.monthly, forecast));
}

function budgetBlock(label, used, cap, forecast) {
  const wrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "kpi-label";
  title.textContent = label;
  wrap.appendChild(title);

  const value = document.createElement("div");
  value.className = "kpi-value";
  value.textContent = cap > 0 ? `${formatEur(used)} / ${formatEur(cap)}` : formatEur(used);
  wrap.appendChild(value);

  if (cap > 0) {
    const track = document.createElement("div");
    track.className = "budget-bar-track";
    const fill = document.createElement("div");
    const pct = Math.min(100, (used / cap) * 100);
    fill.className = `budget-bar-fill${used > cap ? " over-budget" : ""}`;
    fill.style.width = `${pct}%`;
    track.appendChild(fill);
    wrap.appendChild(track);
  }

  if (forecast !== undefined) {
    const note = document.createElement("div");
    note.className = "kpi-change";
    note.textContent = `Forecast month-end: ${formatEur(forecast)}${cap > 0 && forecast > cap ? " (over budget)" : ""}`;
    wrap.appendChild(note);
  }

  return wrap;
}

// --- Section: most expensive requests ---
function renderTopRequests(rows) {
  const body = document.getElementById("top-requests-body");
  body.textContent = "";
  const top = rows.filter((r) => (r.cost_actual_eur || 0) > 0).sort((a, b) => b.cost_actual_eur - a.cost_actual_eur).slice(0, 10);
  document.getElementById("top-requests-empty").hidden = top.length > 0;
  for (const row of top) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(new Date(row.timestamp * 1000).toLocaleString()));
    tr.appendChild(cell(teamName(row.tenant_id)));
    tr.appendChild(cell(rowTags(row).join(", ") || "-"));
    tr.appendChild(cell(row.served_model || row.requested_model || "-"));
    tr.appendChild(cell(row.tokens_in != null ? String(row.tokens_in) : "-"));
    tr.appendChild(cell(row.tokens_out != null ? String(row.tokens_out) : "-"));
    tr.appendChild(cell(formatEur(row.cost_actual_eur)));
    body.appendChild(tr);
  }
}

// --- Section: recent requests table ---
function renderRequestsTable(rows) {
  const body = document.getElementById("requests-table-body");
  body.textContent = "";
  const sorted = [...rows].sort((a, b) => b.timestamp - a.timestamp).slice(0, 50);
  document.getElementById("requests-empty").hidden = sorted.length > 0;

  for (const row of sorted) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(new Date(row.timestamp * 1000).toLocaleString()));
    tr.appendChild(cell(teamName(row.tenant_id)));
    tr.appendChild(modeCell(row.mode || "observe"));
    tr.appendChild(cell(rowTags(row).join(", ")));
    tr.appendChild(cell(row.served_model || row.requested_model || ""));
    tr.appendChild(cell(row.cache_type || "-"));
    tr.appendChild(cell(row.cost_actual_eur != null ? formatEur(row.cost_actual_eur) : "-"));
    tr.appendChild(cell(row.cost_baseline_eur != null ? formatEur(row.cost_baseline_eur) : "-"));
    tr.appendChild(cell(row.quality_score != null ? row.quality_score.toFixed(2) : "-"));
    tr.appendChild(cell(Object.keys(rowGuardrailActions(row)).join(", ") || "-"));
    tr.appendChild(statusCell(row));
    body.appendChild(tr);
  }
}

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}
function statusCell(row) {
  const td = document.createElement("td");
  const pill = document.createElement("span");
  pill.className = `status-pill status-${row.status}`;
  pill.textContent = row.status;
  td.appendChild(pill);
  return td;
}

// --- Section: wiring ---
document.getElementById("connect-form").addEventListener("submit", handleConnectSubmit);
document.getElementById("lock-btn").addEventListener("click", lock);
document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
document.getElementById("tenant-select").addEventListener("change", (event) => selectTeam(event.target.value || null));
for (const button of document.querySelectorAll("#period-select .segmented-btn")) {
  button.addEventListener("click", () => setPeriod(button.dataset.period));
}
for (const button of document.querySelectorAll("#mode-switch .segmented-btn")) {
  button.addEventListener("click", () => setMode(button.dataset.mode));
}

applyTheme(loadTheme());
setPeriod("24h");

const existingKey = loadKey();
if (existingKey) {
  tryConnect(existingKey).then((ok) => {
    if (ok) enterApp();
    else clearKey();
  });
}


// --- Section: try a request — one prompt through Cordon, with what Cordon did ---
function euro(value) {
  return value === null || value === undefined || value === "" ? "-" : `€${Number(value).toFixed(6)}`;
}

// Two earlier turns sent with the prompt when "support conversation" is ticked: a chat app sends its history.
const TRY_HISTORY = [
  { role: "user", content: "I ordered a jacket last week and it has not arrived." },
  { role: "assistant", content: "Sorry to hear that. Your order shipped on Monday and should arrive within two working days." },
];
const KIND_LABELS = {
  chat: "a single question", conversation: "a conversation, sent with its earlier turns", classification: "classifying a text",
  summarization: "summarizing a long text", extraction: "extracting data", code: "code", reasoning: "arithmetic / reasoning",
};

// What the quality check did, in words: X-Cordon-Check names the rule or the check that decided.
function checkVerdict(check, score, escalated) {
  const sentOn = "asked the strong model (gpt-4o) instead";
  if (check === "rules_only") return "rules only (this route was proven without the ML score): the answer passes - kept";
  if (check === "classifier") return `ML score ${score.toFixed(2)}: ${escalated ? `in doubt - ${sentOn}` : "the answer passes - kept"}`;
  if (check === "no_confidence" || check === "error") return `could not be checked - ${sentOn}`;
  return `rule "${(check || "").replaceAll("_", " ")}" fired - ${sentOn}`;
}

function tryRequestSteps(header) {
  const found = (header("X-Cordon-Guardrails") || "").split(",").filter(Boolean);
  const masked = found.filter((f) => f.startsWith("input_pii_")).map((f) => f.replace("input_pii_", "").replace("=", " x"));
  const route = header("X-Cordon-Route") || "-";
  const escalated = header("X-Cordon-Fallback") === "true";
  const quality = header("X-Cordon-Quality");
  const check = header("X-Cordon-Check");
  const cached = header("X-Cordon-Cache") && header("X-Cordon-Cache") !== "none";
  const cost = Number(header("X-Cordon-Cost") || 0);
  const baseline = Number(header("X-Cordon-Baseline-Cost") || 0);
  const steps = [
    ["1. Protect input", masked.length ? `masked before sending: ${masked.join(", ")}` : "no personal data found"],
    ["2. Cache", cached ? "answered from the cache - no model called" : "new question"],
  ];
  if (!cached) {
    const cheaper = route !== "gpt-4o" || escalated;
    const [kind, difficulty] = (header("X-Cordon-Kind") || "").split("/");
    if (kind) steps.push(["3. Kind of request", `${KIND_LABELS[kind] || kind} (${difficulty})`]);
    steps.push(["4. Pick model", cheaper ? "a cheaper model, proven as good for this kind of request (gpt-4.1-mini)" : `${route} - no cheaper model is proven for this kind of request`]);
    if (quality !== null) steps.push(["5. Quality check", checkVerdict(check, Number(quality), escalated)]);
    steps.push(["6. Answered by", route]);
  }
  steps.push(["Cost", `${euro(cost)} instead of ${euro(baseline)} without Cordon` + (baseline > 0 ? ` - ${Math.round((1 - cost / baseline) * 100)}% saved` : "")]);
  return steps;
}

function showTryFacts(response, body) {
  const list = document.getElementById("try-facts");
  list.replaceChildren();
  for (const [label, value] of tryRequestSteps((name) => response.headers.get(name))) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  document.getElementById("try-answer").textContent = body.choices?.[0]?.message?.content || "";
  const cost = Number(response.headers.get("X-Cordon-Cost") || 0);
  const baseline = Number(response.headers.get("X-Cordon-Baseline-Cost") || 0);
  const saving = document.getElementById("try-saving");
  saving.hidden = !(baseline > 0);
  if (baseline > 0) {
    document.getElementById("try-saving-big").textContent = `${Math.round((1 - cost / baseline) * 100)}% saved`;
    document.getElementById("try-saving-detail").textContent =
      `${euro(cost)} paid instead of ${euro(baseline)} without Cordon, for this request.`;
  }
  document.getElementById("try-result").hidden = false;
}

function tryMessages(prompt) {
  const history = document.getElementById("try-history").checked ? TRY_HISTORY : [];
  return [...history, { role: "user", content: prompt }];
}

async function sendTryRequest(event) {
  event.preventDefault();
  const key = document.getElementById("try-key").value.trim();
  const prompt = document.getElementById("try-prompt").value.trim();
  const error = document.getElementById("try-error");
  const button = document.getElementById("try-send");
  error.hidden = true;
  if (!key || !prompt) {
    error.textContent = "A team key and a prompt are needed.";
    error.hidden = false;
    return;
  }
  button.disabled = true;
  try {
    const started = performance.now();
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: "gpt-4o", messages: tryMessages(prompt) }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || `HTTP ${response.status}`);
    showTryFacts(response, body);
    showTryStats(response, body, performance.now() - started);
    refreshAll();
  } catch (err) {
    error.textContent = `Cordon refused or failed: ${err.message}`;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
}

document.getElementById("try-form").addEventListener("submit", sendTryRequest);


// --- Section: tabs - show one part of the dashboard at a time ---
function showTab(name) {
  for (const button of document.querySelectorAll(".tab")) button.classList.toggle("active", button.dataset.tabTarget === name);
  for (const section of document.querySelectorAll("[data-tab]")) section.classList.toggle("tab-hidden", section.dataset.tab !== name);
  for (const chart of Object.values(state.charts)) chart.resize();  // charts drawn while hidden have no size
  try { localStorage.setItem("cordon-tab", name); } catch { /* storage may be unavailable */ }
}

for (const button of document.querySelectorAll(".tab")) button.addEventListener("click", () => showTab(button.dataset.tabTarget));
let savedTab = "overview";
try { savedTab = localStorage.getItem("cordon-tab") || "overview"; } catch { /* storage may be unavailable */ }
showTab(savedTab);


// Speed and size of the one request, under the saving: what the user waited, and Cordon's own share of it.
function showTryStats(response, body, elapsedMs) {
  const overhead = Number(response.headers.get("X-Cordon-Overhead-Ms") || 0);
  const usage = body.usage || {};
  const stats = [
    ["Answered in", `${(elapsedMs / 1000).toFixed(2)} s`],
    ["Cordon's own time", `${overhead.toFixed(0)} ms`],
    ["Tokens in / out", `${usage.prompt_tokens ?? "-"} / ${usage.completion_tokens ?? "-"}`],
    ["Model", response.headers.get("X-Cordon-Route") || "-"],
  ];
  const container = document.getElementById("try-stats");
  container.replaceChildren();
  for (const [label, value] of stats) {
    const box = document.createElement("div");
    box.className = "try-stat";
    const name = document.createElement("div");
    name.className = "try-stat-label";
    name.textContent = label;
    const number = document.createElement("div");
    number.className = "try-stat-value";
    number.textContent = value;
    box.append(name, number);
    container.appendChild(box);
  }
  container.hidden = false;
}
