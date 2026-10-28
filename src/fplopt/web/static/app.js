"use strict";

/* ---------- helpers ---------- */
const $app = document.getElementById("app");
const POSITIONS = ["GK", "DEF", "MID", "FWD"];
let STATUS = null;

const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 1) => (n == null || Number.isNaN(Number(n)) ? "–" : Number(n).toFixed(d));
const money = (n) => (n == null ? "–" : `£${Number(n).toFixed(1)}m`);
const pct = (n, d = 1) => (n == null ? "–" : `${Number(n).toFixed(d)}%`);
const signed = (n, d = 1) => (n > 0 ? "+" : n < 0 ? "−" : "±") + Math.abs(n).toFixed(d);
const isDiff = (p) => STATUS && p.ownership != null && p.ownership < STATUS.ownership_max_pct;

async function api(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}
const spinner = () => '<div class="spinner" role="status" aria-label="Loading"></div>';
const errorBox = (e) => `<div class="error" role="alert">${esc(e.message || e)}</div>`;
const query = (params) => {
  const q = Object.entries(params).filter(([, v]) => v !== "" && v != null).map(([k, v]) => `${k}=${encodeURIComponent(v)}`);
  return q.length ? `?${q.join("&")}` : "";
};

/* ---------- status bar ---------- */
function deadlineHtml(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const when = d.toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  const ms = d - Date.now();
  if (ms <= 0) return `deadline passed (${esc(when)})`;
  const h = Math.floor(ms / 3.6e6);
  const left = h >= 24 ? `${Math.floor(h / 24)}d ${h % 24}h` : `${h}h ${Math.floor((ms % 3.6e6) / 6e4)}m`;
  return `<span class="hide-sm">deadline ${esc(when)} · </span>${left} left`;
}
const shortDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "–");

async function loadStatus() {
  try {
    STATUS = await api("/api/status");
    document.getElementById("gw-chip").innerHTML = `<b>GW${STATUS.gameweek}</b> · ${deadlineHtml(STATUS.deadline)}`;
    document.getElementById("footer").innerHTML =
      `Live FPL data ${esc(shortDate(STATUS.data_updated))} · predictions ${esc(shortDate(STATUS.predictions_updated))} · ` +
      `model trained on ${esc((STATUS.model_train_seasons || []).join(", ") || "–")}. ` +
      "Independent fan project using public FPL data; not affiliated with the Premier League.";
  } catch (e) {
    document.getElementById("gw-chip").textContent = "Data not ready";
    $app.innerHTML = `<div class="empty"><h2>No data yet</h2><p>${esc(e.message)}</p></div>`;
    throw e;
  }
}

/* ---------- shared components ---------- */
const SHIRT = '<svg viewBox="0 0 46 42" aria-hidden="true"><path d="M15 2 7 6 1 15l7 4 2-3v24h26V16l2 3 7-4-6-9-8-4c-2 4-5 6-8 6s-6-2-8-6Z"/></svg>';

function fixtureLabel(p) {
  const f = p.fixtures || [];
  const now = f.filter((x) => x.gw === STATUS.gameweek);
  if (!now.length) return "blank";
  const first = `${now[0].opponent} (${now[0].home ? "H" : "A"})`;
  return now.length > 1 ? `${first} +${now.length - 1}` : first;
}

function fixtureChips(p, n = 3) {
  const chips = [];
  for (let gw = STATUS.gameweek; gw < STATUS.gameweek + n; gw++) {
    const games = (p.fixtures || []).filter((f) => f.gw === gw);
    if (!games.length) chips.push('<span class="blank" title="No fixture">–</span>');
    games.forEach((f) => chips.push(
      `<span class="d${esc(f.difficulty)}" title="GW${gw}: ${esc(f.opponent)} ${f.home ? "home" : "away"}, difficulty ${esc(f.difficulty)}">${esc(f.opponent)} ${f.home ? "H" : "a"}</span>`));
  }
  return `<span class="fdr">${chips.join("")}</span>`;
}

function token(p, plan, opts = {}) {
  const badge = plan.captain.id === p.id ? '<span class="badge c" title="Captain">C</span>'
    : plan.vice_captain.id === p.id ? '<span class="badge v" title="Vice-captain">V</span>' : "";
  const flag = p.status !== "a" ? `<span class="flag" title="${esc(p.news || "Flagged")}">!</span>` : "";
  const diff = isDiff(p) ? `<span class="tag-diff" title="Owned by ${pct(p.ownership)} of managers">DIFF</span>` : "";
  return `<div class="player ${p.position === "GK" ? "gk" : ""}" title="${esc(p.name)} · ${esc(p.team)} · ${money(p.price)} · owned ${pct(p.ownership)}">
    ${opts.order ? `<div class="order">${opts.order}</div>` : ""}
    <div class="shirt">${SHIRT}<span class="club">${esc(p.team)}</span>${badge}${flag}${diff}</div>
    <div class="pname">${esc(p.name)}</div>
    <div class="pmeta"><b>${fmt(p.projection)}</b><span class="sep"> · </span><span class="fx">${esc(fixtureLabel(p))}</span></div>
  </div>`;
}

function pitch(plan) {
  const rows = POSITIONS.map((pos) => `<div class="pitch-row">${plan.xi.filter((p) => p.position === pos).map((p) => token(p, plan)).join("")}</div>`);
  const bench = plan.bench.map((p, i) => token(p, plan, { order: i === 0 ? "GK" : i })).join("");
  return `<div class="pitch" aria-label="Starting XI">${rows.join("")}</div>
    <div class="bench" aria-label="Bench">${bench}</div><div class="bench-label">Bench · substitution order</div>`;
}

const stat = (label, value, sub = "") =>
  `<div class="card stat"><div class="label">${label}</div><div class="value num">${value}</div><div class="sub">${sub}</div></div>`;

/* ---------- My team ---------- */
const teamState = { result: null, selected: 0 };

function renderTeamPage() {
  const saved = store.get("teamId") || "";
  $app.innerHTML = `
    <div class="page-head">
      <div><h1>Your next move</h1>
        <p>Enter your FPL team ID (the number in your Points page URL). The optimizer compares 0, 1 and 2 transfers
        for GW${STATUS.gameweek} under the real rules (hits, selling prices, 3 per club) and picks your XI and captain.</p></div>
    </div>
    <div class="card card-pad">
      <form class="search" id="team-form">
        <label class="sr-only" for="team-id">Team ID</label>
        <input id="team-id" inputmode="numeric" pattern="[0-9]*" placeholder="Team ID, e.g. 1234567" value="${esc(saved)}" style="flex:1;min-width:180px">
        <button type="submit" id="team-go">Analyse team</button>
        <button type="button" class="ghost" id="squad-go">Best new squad</button>
      </form>
      <details class="options"><summary>Options</summary>
        <div class="options-grid">
          <div class="field"><label for="opt-proj">Projections</label>
            <select id="opt-proj"><option value="model">This model</option><option value="ep_next">FPL ep_next</option></select></div>
          <div class="field"><label for="opt-bonus">Differential bonus: <span id="bonus-val">0%</span></label>
            <input type="range" id="opt-bonus" min="0" max="0.3" step="0.05" value="0"></div>
          <div class="field"><label for="opt-ft">Free transfers</label>
            <select id="opt-ft"><option value="">Work it out</option>${[0, 1, 2, 3, 4, 5].map((n) => `<option>${n}</option>`).join("")}</select></div>
          <div class="field"><label for="opt-max">Compare up to</label>
            <select id="opt-max">${[1, 2, 3].map((n) => `<option value="${n}" ${n === STATUS.max_transfers ? "selected" : ""}>${n} transfer${n > 1 ? "s" : ""}</option>`).join("")}</select></div>
        </div>
      </details>
    </div>
    <div id="team-result" style="margin-top:20px"></div>`;
  const bonus = document.getElementById("opt-bonus");
  bonus.addEventListener("input", () => { document.getElementById("bonus-val").textContent = `${Math.round(bonus.value * 100)}%`; });
  document.getElementById("team-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = document.getElementById("team-id").value.trim();
    if (!/^\d+$/.test(id)) { document.getElementById("team-result").innerHTML = errorBox("Team IDs are numbers."); return; }
    store.set("teamId", id);
    runTeam(id);
  });
  document.getElementById("squad-go").addEventListener("click", () => runTeam(null));
  if (teamState.result) drawTeamResult();
  else if (saved) runTeam(saved);
  else document.getElementById("team-result").innerHTML =
    '<div class="card empty"><h2>No team loaded</h2><p>Enter your team ID, or build the best £100m squad from scratch.</p></div>';
}

async function runTeam(teamId) {
  const out = document.getElementById("team-result");
  const params = {
    projection: document.getElementById("opt-proj").value,
    differential_bonus: document.getElementById("opt-bonus").value,
    free_transfers: document.getElementById("opt-ft").value,
    max_transfers: document.getElementById("opt-max").value,
  };
  out.innerHTML = spinner();
  document.querySelectorAll("#team-form button").forEach((b) => { b.disabled = true; });
  try {
    if (teamId) teamState.result = await api(`/api/team/${teamId}${query(params)}`);
    else teamState.result = await api(`/api/squad${query({ projection: params.projection, differential_bonus: params.differential_bonus })}`);
    teamState.selected = teamState.result.recommended;
    drawTeamResult();
  } catch (e) {
    out.innerHTML = errorBox(e);
  } finally {
    document.querySelectorAll("#team-form button").forEach((b) => { b.disabled = false; });
  }
}

function drawTeamResult() {
  const r = teamState.result;
  const plan = r.options[teamState.selected];
  const isTeam = r.mode === "transfers";
  const flagged = isTeam ? r.team.squad.filter((p) => p.status !== "a") : [];
  const base = r.options[0].net_points;

  const stats = isTeam
    ? stat("Expected points", fmt(plan.net_points), plan.hits ? `${fmt(plan.projected_points)} before a −${plan.hit_cost} hit` : `GW${r.gameweek}, captain counted twice`)
      + stat("Transfers", plan.n_transfers, plan.hits ? `${plan.hits} hit${plan.hits > 1 ? "s" : ""}` : `${r.team.free_transfers} free available`)
      + stat("Bank after", money(plan.bank), `now ${money(r.team.bank)}`)
      + stat("Captain", esc(plan.captain.name), `${fmt(plan.captain.projection)} pts × 2 · vice ${esc(plan.vice_captain.name)}`)
    : stat("Expected points", fmt(plan.projected_points), `GW${r.gameweek}, captain counted twice`)
      + stat("Squad cost", money(100 - plan.bank), `${money(plan.bank)} in the bank`)
      + stat("Formation", plan.formation, "best XI from 15")
      + stat("Captain", esc(plan.captain.name), `${fmt(plan.captain.projection)} pts × 2`);

  const options = isTeam ? `
    <div class="card"><div class="card-head"><h2>Transfer options</h2><span class="muted small">click a row to see it</span></div>
      <div class="card-body"><table>
        <thead><tr><th>#</th><th>Moves</th><th class="r">Hits</th><th class="r">Net</th><th class="r">vs 0</th></tr></thead>
        <tbody>${r.options.map((o, i) => `
          <tr class="clickable ${i === teamState.selected ? "selected" : ""}" data-option="${i}">
            <td class="num">${o.n_transfers}</td>
            <td>${o.transfers.map((t) => `${esc(t.out.name)} → <b>${esc(t.in.name)}</b>`).join("<br>") || '<span class="muted">Roll the transfer</span>'}
              ${i === r.recommended ? ' <span class="pill">recommended</span>' : ""}</td>
            <td class="r num">${o.hits ? `−${o.hit_cost}` : "0"}</td>
            <td class="r num">${fmt(o.net_points)}</td>
            <td class="r num ${o.net_points - base > 0 ? "gain" : ""}">${signed(o.net_points - base)}</td>
          </tr>`).join("")}</tbody></table>
        <p class="note">Each extra transfer must add ${fmt(0.5)}+ projected points to be recommended; otherwise the free transfer is banked.</p>
      </div></div>` : "";

  const moves = isTeam ? `
    <div class="card"><div class="card-head"><h2>${plan.n_transfers ? "Transfers" : "No transfers"}</h2>
      <span class="muted small">${r.team.free_transfers_inferred ? "free transfers worked out from your history" : "free transfers set by you"}</span></div>
      <div class="card-body">${plan.transfers.length ? plan.transfers.map((t) => `
        <div class="transfer">
          <div class="who out"><b>${esc(t.out.name)}</b><span>${esc(t.out.team)} · sells ${money(r.team.squad.find((s) => s.id === t.out.id)?.selling_price)}</span></div>
          <div class="arrow">→</div>
          <div class="who"><b>${esc(t.in.name)}</b><span>${esc(t.in.team)} · ${money(t.in.price)} · ${pct(t.in.ownership)} owned</span></div>
          <div class="gain-cell r"><span class="gain">${signed(t.gain)}</span> <span class="muted small">pts</span></div>
        </div>`).join("") : '<p class="muted">Your squad is already the best available for this gameweek: bank the free transfer.</p>'}
        ${flagged.length ? `<p class="note">Flagged in your squad: ${flagged.map((p) => `<span class="pill warn" title="${esc(p.news)}">${esc(p.name)}</span>`).join(" ")}</p>` : ""}
      </div></div>` : "";

  document.getElementById("team-result").innerHTML = `
    ${isTeam ? `<h2 style="margin-bottom:12px">${esc(r.team.name)} <span class="muted small">· ${esc(r.team.manager)} · projections: ${esc(r.projection)}</span></h2>` : ""}
    <div class="stats">${stats}</div>
    <div class="grid-team" style="margin-top:16px">
      <div>${pitch(plan)}<p class="note">${plan.formation} · numbers are expected points for GW${r.gameweek}${r.differential_bonus ? ` · differential bonus +${Math.round(r.differential_bonus * 100)}%` : ""}. <span class="pill diff">DIFF</span> = owned by under ${STATUS.ownership_max_pct}%.</p></div>
      <div class="stack">${moves}${options}</div>
    </div>`;
  document.querySelectorAll("tr[data-option]").forEach((tr) => tr.addEventListener("click", () => {
    teamState.selected = Number(tr.dataset.option);
    drawTeamResult();
  }));
}

/* ---------- Differentials ---------- */
const diffState = { position: "", maxOwn: null, maxPrice: "" };

function renderDifferentialsPage() {
  diffState.maxOwn = diffState.maxOwn ?? STATUS.ownership_max_pct;
  $app.innerHTML = `
    <div class="page-head">
      <div><h1>Differentials for GW${STATUS.gameweek}</h1>
        <p>Players owned by few managers whom the model expects to score well, ranked by expected points (already scaled by
        FPL's chance-of-playing flags). "Why" lists the features pushing each prediction up, compared with a typical regular starter.</p></div>
    </div>
    <div class="card card-pad" style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end">
      <div class="field"><label>Position</label><div class="segmented" id="pos-seg">
        ${["", ...POSITIONS].map((p) => `<button type="button" data-pos="${p}" class="${p === diffState.position ? "on" : ""}">${p || "All"}</button>`).join("")}
      </div></div>
      <div class="field"><label for="max-own">Owned by under</label>
        <select id="max-own">${[2, 5, 10, 15, 20].map((n) => `<option value="${n}" ${Number(diffState.maxOwn) === n ? "selected" : ""}>${n}%</option>`).join("")}</select></div>
      <div class="field"><label for="max-price">Max price</label>
        <select id="max-price"><option value="">Any</option>${[4.5, 5, 5.5, 6, 7, 8, 10].map((n) => `<option value="${n}" ${String(diffState.maxPrice) === String(n) ? "selected" : ""}>£${n.toFixed(1)}m</option>`).join("")}</select></div>
    </div>
    <div id="diff-result" style="margin-top:16px"></div>`;
  document.querySelectorAll("#pos-seg button").forEach((b) => b.addEventListener("click", () => { diffState.position = b.dataset.pos; renderDifferentialsPage(); }));
  document.getElementById("max-own").addEventListener("change", (e) => { diffState.maxOwn = e.target.value; loadDifferentials(); });
  document.getElementById("max-price").addEventListener("change", (e) => { diffState.maxPrice = e.target.value; loadDifferentials(); });
  loadDifferentials();
}

async function loadDifferentials() {
  const out = document.getElementById("diff-result");
  out.innerHTML = spinner();
  try {
    const data = await api(`/api/differentials${query({ limit: 25, max_ownership: diffState.maxOwn, position: diffState.position, max_price: diffState.maxPrice })}`);
    if (!data.players.length) { out.innerHTML = '<div class="card empty"><h2>No players match</h2><p>Loosen the filters.</p></div>'; return; }
    const why = (p) => esc(p.reasons).replaceAll("; ", "<br>");
    out.innerHTML = `<div class="card">
      <div class="table-wide"><table>
        <thead><tr><th>#</th><th>Player</th><th class="r">Price</th><th class="r">Owned</th><th>Next 3</th><th class="r">xPts</th><th class="r">Haul</th><th class="hide-sm">Why</th></tr></thead>
        <tbody>${data.players.map((p, i) => `<tr>
          <td class="num muted">${i + 1}</td>
          <td><div class="player-cell"><b>${esc(p.name)} ${p.status !== "a" ? `<span class="pill warn" title="${esc(p.news)}">${esc(p.status)}</span>` : ""}</b><span>${esc(p.team)} · ${esc(p.position)}</span></div></td>
          <td class="r num">${money(p.price)}</td>
          <td class="r num">${pct(p.ownership)}</td>
          <td>${fixtureChips(p)}</td>
          <td class="r num"><b>${fmt(p.projection, 2)}</b></td>
          <td class="r num">${pct(p.p_haul * 100, 0)}</td>
          <td class="hide-sm"><div class="reasons">${why(p)}</div></td></tr>`).join("")}</tbody></table></div>
      <div class="dcards">${data.players.map((p, i) => `<div class="dcard">
          <div class="dcard-top"><div class="player-cell"><b>${i + 1}. ${esc(p.name)}</b><span>${esc(p.team)} · ${esc(p.position)} · ${money(p.price)}</span></div>
            <div style="text-align:right"><b class="num">${fmt(p.projection, 2)}</b><div class="muted small">xPts</div></div></div>
          <div class="dcard-nums"><span>Owned <b>${pct(p.ownership)}</b></span><span>Haul <b>${pct(p.p_haul * 100, 0)}</b></span></div>
          ${fixtureChips(p)}<div class="reasons">${why(p)}</div></div>`).join("")}</div>
    </div>
    <p class="note">Fixture colours are FPL's published difficulty (display only; the model rates opponents from their recent results).</p>`;
  } catch (e) {
    out.innerHTML = errorBox(e);
  }
}

/* ---------- Mini-league ---------- */
function renderLeaguePage() {
  const league = store.get("leagueId") || "";
  const team = store.get("teamId") || "";
  $app.innerHTML = `
    <div class="page-head">
      <div><h1>Mini-league</h1>
        <p>Standings plus what your rivals own. League ownership comes from the current squads of the top entries,
        so you can see the league's template, the threats you don't cover, and strong picks nobody has.</p></div>
    </div>
    <div class="card card-pad">
      <form class="search" id="league-form">
        <div class="field" style="flex:1;min-width:160px"><label for="league-id">League ID</label>
          <input id="league-id" inputmode="numeric" placeholder="e.g. 123456" value="${esc(league)}"></div>
        <div class="field" style="flex:1;min-width:160px"><label for="league-team">Your team ID (optional)</label>
          <input id="league-team" inputmode="numeric" placeholder="highlights you" value="${esc(team)}"></div>
        <div class="field"><label for="league-sample">Rivals to scan</label>
          <select id="league-sample">${[10, 20, 30, 50].map((n) => `<option ${n === 20 ? "selected" : ""}>${n}</option>`).join("")}</select></div>
        <button type="submit" style="align-self:flex-end">Load league</button>
      </form>
    </div>
    <div id="league-result" style="margin-top:20px"></div>`;
  document.getElementById("league-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = document.getElementById("league-id").value.trim();
    const me = document.getElementById("league-team").value.trim();
    if (!/^\d+$/.test(id) || (me && !/^\d+$/.test(me))) { document.getElementById("league-result").innerHTML = errorBox("IDs are numbers."); return; }
    store.set("leagueId", id);
    if (me) store.set("teamId", me);
    runLeague(id, me);
  });
  if (league) runLeague(league, team);
  else document.getElementById("league-result").innerHTML =
    '<div class="card empty"><h2>No league loaded</h2><p>Your league IDs are in the URL of each league page on the FPL site.</p></div>';
}

function ownershipList(title, subtitle, players, emptyText) {
  return `<div class="card"><div class="card-head"><h2>${title}</h2><span class="muted small">${subtitle}</span></div>
    <div class="card-body">${players.length ? `<table><thead><tr><th>Player</th><th>League</th><th class="r">xPts</th></tr></thead><tbody>
      ${players.map((p) => `<tr><td><div class="player-cell"><b>${esc(p.name)}</b><span>${esc(p.team)} · ${esc(p.position)} · ${pct(p.ownership)} overall</span></div></td>
        <td style="width:38%"><div class="num small">${pct(p.league_ownership, 0)}</div><div class="bar"><i style="width:${Math.min(100, p.league_ownership)}%"></i></div></td>
        <td class="r num"><b>${fmt(p.projection)}</b></td></tr>`).join("")}</tbody></table>` : `<p class="muted">${emptyText}</p>`}</div></div>`;
}

async function runLeague(id, teamId) {
  const out = document.getElementById("league-result");
  out.innerHTML = spinner();
  try {
    const sample = document.getElementById("league-sample").value;
    const d = await api(`/api/league/${id}${query({ team_id: teamId, sample })}`);
    const move = (s) => (s.last_rank && s.last_rank !== s.rank
      ? (s.rank < s.last_rank ? `<span class="gain" title="Up from ${s.last_rank}">▲</span>` : `<span class="loss" title="Down from ${s.last_rank}">▼</span>`)
      : '<span class="muted">–</span>');
    out.innerHTML = `
      <h2 style="margin-bottom:12px">${esc(d.league.name)} <span class="muted small">· after GW${d.gameweek} · ownership from ${d.sampled_entries} rivals</span></h2>
      <div class="grid-team">
        <div class="card"><div class="card-head"><h2>Standings</h2><span class="muted small">top ${d.standings.length}</span></div>
          <div class="card-body"><table><thead><tr><th>#</th><th></th><th>Team</th><th class="r">GW</th><th class="r">Total</th></tr></thead><tbody>
          ${d.standings.map((s) => `<tr class="${s.is_you ? "you" : ""}"><td class="num">${s.rank}</td><td>${move(s)}</td>
            <td><div class="player-cell"><b>${esc(s.entry_name)}</b><span>${esc(s.manager)}</span></div></td>
            <td class="r num">${s.gw_points}</td><td class="r num"><b>${s.total}</b></td></tr>`).join("")}</tbody></table></div></div>
        <div class="stack">
          ${d.you ? ownershipList("Threats", "rivals own, you don't", d.you.threats, "You cover everything your rivals own.") : ""}
          ${d.you ? ownershipList("Your edge", "your players few rivals own", d.you.your_differentials, "—") : ""}
          ${ownershipList("Untapped", `strong picks owned by ≤${STATUS.ownership_max_pct}% of rivals`, d.untapped, "Nothing obvious.")}
          ${ownershipList("League template", "most owned by rivals", d.template.slice(0, 10), "—")}
        </div>
      </div>`;
  } catch (e) {
    out.innerHTML = errorBox(e);
  }
}

/* ---------- router ---------- */
const ROUTES = { team: renderTeamPage, differentials: renderDifferentialsPage, league: renderLeaguePage };

function route() {
  const name = (location.hash.replace(/^#\/?/, "") || "team").split("?")[0];
  const page = ROUTES[name] ? name : "team";
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.route === page));
  ROUTES[page]();
}

window.addEventListener("hashchange", route);
loadStatus().then(route).catch(() => {});
