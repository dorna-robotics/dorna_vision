// system.js — the System page on a vision Pi: this unit, and its `upgrade`.
//
// The workspace's System page (its gui/orchestrator/web/admin/system.js),
// for the vision Pi: the same four views, the same rules, the same look.
// Offered only where the unit declares the vision server the owner
// (/etc/dorna/unit.json "system": "vision" — the vision Pi's upgrade writes
// it; a Jetson declares "workspace" and upgrades from the workspace's page).
//
// NOTHING HERE NAMES A PART. The parts, their labels, folders and branches
// come from the upgrade's own record (server/system.py RECORD, written by
// each part of the upgrade repo); the progress steps come from the
// `==> <part>` lines the upgrade prints.
//
// Four views in one card: the UNIT (what is installed, how each part
// compares with what the upgrade installs), the CONFIRM (every risk a
// warning, the exact command), the PROGRESS (the steps, the live log —
// followed through the server's restart and the reboot), and the RESULT
// (the record of the run). Nothing is greyed out for the unit's state; the
// server refuses only "not this unit's", "not installed" and "already
// running".

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function toast(msg, kind = "ok") {
  const area = $("toastArea");
  if (!area) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  area.appendChild(el);
  setTimeout(() => el.remove(), 3500);
  el.addEventListener("click", () => el.remove());
}
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${r.status} ${r.statusText}`);
  return j;
}

// This browser started an upgrade: {started_at, before: {part: commit}} —
// kept across the restart and the reboot, cleared at the result.
const KEY = "vision_system_upgrade";

let S = null;            // the last /api/system state
let run = null;          // the upgrade being followed
let pollTimer = null;
let shown = false;

const host = () => $("sys");
const warn = (html, bad = false) =>
  `<div class="ver-warn${bad ? " bad" : ""}"><span class="bang">!</span><div>${html}</div></div>`;
const short = h => (h || "").slice(0, 7);
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const fmtWhen = iso => {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};
const fmtDur = s => {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
};
const fmtTook = s => {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), r = s % 60;
  return h ? `${h} h ${m} min` : m ? (r && m < 10 ? `${m} min ${r} s` : `${m} min`) : `${r} s`;
};
const fmtUp = s => {
  if (s == null) return "—";
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return d ? `${plural(d, "day")} ${h} h` : h ? `${h} h ${m} min` : `${m} min`;
};
const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || "null"); } catch { return null; } };
const save = v => { try { v ? localStorage.setItem(KEY, JSON.stringify(v)) : localStorage.removeItem(KEY); } catch {} };

function card(title, meta, body, foot = "") {
  host().innerHTML = `<div class="sys-card">
    <div class="sys-head"><span>${esc(title)}</span><div class="spacer"></div>${meta ? `<span class="mono">${esc(meta)}</span>` : ""}</div>
    <div class="sys-body">${body}</div>
    <div class="sys-foot" id="sysFoot">${foot}</div></div>`;
}
function onFoot(handlers) {
  for (const [id, fn] of Object.entries(handlers)) $(id)?.addEventListener("click", fn);
}

// ── a part, as its row and its warnings ─────────────────────────────────
function partNow(p) {
  const n = p.now;
  if (!p.dir) return { runs: "—", installs: "re-applied every upgrade", pill: "" };
  if (!n || !n.exists) return { runs: "not installed", installs: p.branch, pill: `<span class="pill off">new install</span>` };
  if (!n.git) return { runs: "not a git folder", installs: p.branch, pill: `<span class="pill bad">replaced</span>` };
  const runs = `${n.branch} ${short(n.head && n.head.hash)}`;
  let pill;
  if (n.reachable === false) pill = `<span class="pill off">can't check</span>`;
  else if (n.switches) pill = `<span class="pill warn">branch changes</span>`;
  else if (n.behind) pill = `<span class="pill warn">${plural(n.behind, "update")}</span>`;
  else pill = `<span class="pill ok">up to date</span>`;
  if ((n.changes || []).length || (n.untracked || []).length) pill += ` <span class="pill warn">local changes</span>`;
  return { runs, installs: n.target_branch || p.branch, pill };
}

function partWarnings(p) {
  const n = p.now, name = `<b>${esc(p.label || p.name)}</b> (<code>${esc(p.dir)}</code>)`;
  const out = [];
  if (!p.dir || !n || !n.exists) return out;
  if (!n.git) { out.push(warn(`${name} is not a git folder — the upgrade deletes it and clones it fresh.`, true)); return out; }
  if (n.reachable === false) out.push(warn(`${name} can't be updated right now — no connection to its source.<div class="ver-sub">${esc(n.fetch_error)}</div>`, true));
  if (n.switches) out.push(warn(`${name} is on branch <code>${esc(n.branch)}</code> — the upgrade runs <code>${esc(n.target_branch)}</code> from now on (<code>${esc(n.branch)}</code> stays as a branch, not run).`));
  const c = n.changes || [], u = n.untracked || [];
  if (c.length) out.push(warn(`${name}: ${plural(c.length, "uncommitted change")} DISCARDED (<code>git reset --hard</code>).`
    + `<div class="ver-sub">${esc(c.slice(0, 6).join(", "))}${c.length > 6 ? ` … +${c.length - 6}` : ""}</div>`, true));
  if (u.length) out.push(warn(`${name}: ${plural(u.length, "untracked file")} DELETED (<code>git clean -fd</code>) — ignored files are kept.`
    + `<div class="ver-sub">${esc(u.slice(0, 6).join(", "))}${u.length > 6 ? ` … +${u.length - 6}` : ""}</div>`, true));
  if ((n.lost_commits || []).length) out.push(warn(`${name}: ${plural(n.lost_commits.length, "local commit")} on <code>${esc(n.branch)}</code> that were never pushed — the upgrade leaves them behind.`
    + `<div class="ver-sub">${n.lost_commits.slice(0, 4).map(x => esc(`${x.hash} ${x.subject}`)).join("<br>")}</div>`, true));
  return out;
}

function statusPill(s) {
  if (!s.helper) return `<span class="pill bad">upgrade not installed</span>`;
  if (s.running) return `<span class="pill warn">upgrading</span>`;
  if (!s.record) return `<span class="pill off">not upgraded from here yet</span>`;
  if (s.to_update.length) return `<span class="pill warn">${plural(s.to_update.length, "part")} to update</span>`;
  if (s.parts.some(p => p.dir && p.now && p.now.reachable === false)) return `<span class="pill off">can't check for updates</span>`;
  return `<span class="pill ok">up to date</span>`;
}

// ── the UNIT ────────────────────────────────────────────────────────────
function viewUnit() {
  const s = S, r = s.record;
  let rows = `<dt>Status</dt><dd>${statusPill(s)}</dd>
    <dt>Upgrade tool</dt><dd>${s.helper
      ? `installed${r && r.upgrade && r.upgrade.branch ? `<div class="ver-sub">channel ${esc(r.upgrade.branch)}</div>` : ""}`
      : `not installed<div class="ver-sub">install it once from a terminal — the upgrade repo's README</div>`}</dd>`;
  if (r && r.error) rows += `<dt>Last upgrade</dt><dd>${warn(esc(r.error), true)}</dd>`;
  else if (r) rows += `<dt>Last upgrade</dt><dd>${esc(fmtWhen(r.finished))}`
    + `<div class="ver-sub">${r.result === "finished" ? "finished" : `stopped — ${esc(r.reason || "")}`} · took ${esc(fmtTook((new Date(r.finished) - new Date(r.started)) / 1000))}</div></dd>`;
  rows += `<dt>Uptime</dt><dd>${esc(fmtUp(s.uptime_s))}</dd>`;

  let table;
  if (s.parts.length) {
    table = `<table class="sys-t"><tr><th>Part</th><th>Folder</th><th>Installed</th><th>Upgrade to</th></tr>`
      + s.parts.map(p => {
        const x = partNow(p);
        const failed = p.exit != null && p.exit !== 0;
        return `<tr><td class="what">${esc(p.label || p.name)}`
          + `${failed ? `<div class="sub bad"><span class="msg-mark sm" data-level="bad" aria-hidden="true">!</span>failed last time (exit ${esc(p.exit)})</div>` : ""}</td>`
          + `<td class="h">${p.dir ? esc(p.dir.split("/").pop()) : `<span class="sub">—</span>`}</td>`
          + `<td class="h">${esc(x.runs)}</td>`
          + `<td>${p.dir ? `<span class="h">${esc(x.installs)}</span> · ` : ""}${x.pill || esc(x.installs)}</td></tr>`;
      }).join("") + `</table>`;
  } else {
    table = `<div class="ver-sub">The parts are listed here after the next upgrade.</div>`;
  }
  card("Installed", "", `<dl class="ver-rows">${rows}</dl>${table}`,
    `<div class="spacer"></div><button class="btn" id="btnSysCheck">Check again</button>`
    + (s.running ? `<button class="btn btn-primary" id="btnSysFollow">Show progress</button>`
       : s.helper ? `<button class="btn btn-primary" id="btnSysUpgrade">Upgrade…</button>` : ""));
  onFoot({ btnSysCheck: () => refresh(true), btnSysUpgrade: viewConfirm,
           btnSysFollow: () => follow(load() || { started_at: S.now - 3 * 3600, before: {} }) });
}

// A unit whose System page is the workspace's (a Jetson): said, not offered.
function viewNotOwned() {
  card("System", "", `<div class="ver-sub">This unit is upgraded from the workspace's System page — it upgrades every part here, the vision server included.</div>`);
}

// ── the CONFIRM ─────────────────────────────────────────────────────────
function viewConfirm() {
  const s = S;
  let html = `<div class="ver-sub">Installs the latest version of every part below, then restarts the vision Pi. Takes several minutes — keep this page open.</div>`;
  const clients = s.clients || [], cams = s.cameras || [];
  if (clients.length) html += warn(`<b>${plural(clients.length, "client")} connected</b> (this page included) — ${esc(clients.join(", "))}. `
    + `The upgrade restarts the vision server: a workspace using it loses its cameras and detections until the Pi is back.`, clients.length > 1);
  if (cams.length) html += warn(`<b>${plural(cams.length, "camera")} open</b> — ${esc(cams.join(", "))}. They close, and open again only when a client adds them.`);
  html += warn(`<b>The vision Pi reboots</b> at the end — every open page and every workspace using it lose it for about a minute.`);
  if (!s.record) html += warn(`This unit has not been upgraded from here before, so the changes to each part can't be listed yet. They will be next time.`);
  for (const p of s.parts) html += partWarnings(p).join("");
  html += warn(`If something fails to start, the Pi is <b>not restarted</b>. If this page can't reconnect, check the log over SSH: <code>journalctl -u ${esc(s.unit)}</code>.`);
  html += `<div class="ver-label">Will run</div><ul class="ver-cmds sh"><li>systemd-run --unit=${esc(s.unit)} --collect ${esc(s.helper_path)}</li></ul>`;
  card("Upgrade", "", html,
    `<div class="spacer"></div><button class="btn" id="btnSysCancel">Cancel</button><button class="btn btn-warn" id="btnSysGo">Upgrade and reboot</button>`);
  onFoot({ btnSysCancel: viewUnit, btnSysGo: start });
}

async function start() {
  const go = $("btnSysGo");
  if (go) { go.disabled = true; go.textContent = "Starting…"; }
  const before = {};
  for (const p of S.parts) if (p.now && p.now.head) before[p.name] = { branch: p.now.branch, hash: p.now.head.hash };
  try {
    const r = await api("/api/system", { method: "POST", body: JSON.stringify({ confirmed: true }) });
    const v = { started_at: r.started_at, before };
    save(v);
    follow(v);
  } catch (e) {
    toast(String(e.message || e), "bad");
    if (go) { go.disabled = false; go.textContent = "Upgrade and reboot"; }
  }
}

// ── the PROGRESS — the journal, through the restart and the reboot ──────
function follow(v) {
  clearTimeout(pollTimer);
  run = { ...v, lines: [], cursor: "", marks: [], down: null, lastOk: Date.now() };
  poll();
}

function steps() {
  const names = (S && S.parts || []).map(p => p.name);
  for (const m of run.marks) if (!names.includes(m.name) && !["services", "reboot"].includes(m.name)) names.push(m.name);
  names.push("services", "reboot");
  const label = n => n === "services" ? "Restart services" : n === "reboot" ? "Reboot"
    : ((S && S.parts || []).find(p => p.name === n) || {}).label || n;
  const at = Object.fromEntries(run.marks.map((m, i) => [m.name, i]));
  const cur = run.marks.length ? run.marks[run.marks.length - 1].name : null;
  const now = Date.now() / 1000;
  return `<ul class="sys-steps">${names.map(n => {
    const i = at[n];
    if (i === undefined) return `<li><span class="ic"></span>${esc(label(n))}<span class="t"></span></li>`;
    const t0 = run.marks[i].t, t1 = i + 1 < run.marks.length ? run.marks[i + 1].t : now;
    const done = n !== cur;
    return `<li class="${done ? "done" : "now"}"><span class="ic">${done ? "✓" : ""}</span>${esc(label(n))}<span class="t">${fmtDur(t1 - t0)}</span></li>`;
  }).join("")}</ul>`;
}

function viewProgress() {
  const el = Date.now() / 1000 - run.started_at;
  const meta = `started ${new Date(run.started_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${fmtDur(el)}`;
  const last = run.marks.length ? run.marks[run.marks.length - 1].name : "";
  if (run.down) {
    const off = fmtDur((Date.now() - run.down) / 1000);
    const [k, h, sub] = last === "reboot"
      ? ["Rebooting", "The vision Pi is restarting", "Everything is installed."]
      : last === "services"
      ? ["Reconnecting", "Restarting the services", "The connection drops while the new version starts."]
      : ["Reconnecting", "Connection lost", "The upgrade keeps running on the Pi."];
    card("Upgrading", meta, `<div class="sys-big"><div class="ring"></div>
        <div class="sys-k">${k} · ${off}</div>
        <h2>${h}</h2>
        <div class="ver-sub">${sub} This page reconnects on its own.</div></div>${steps()}`);
    return;
  }
  const keep = host().querySelector(".sys-log");
  const stick = !keep || keep.scrollTop + keep.clientHeight >= keep.scrollHeight - 8;
  card("Upgrading", meta, `${steps()}<div class="ver-label">Live log — journalctl -u ${esc(S ? S.unit : "dorna-upgrade")}</div>
    <pre class="sys-log" id="sysLog">${esc(run.lines.slice(-400).map(l => l.text).join("\n")) || "Waiting for the first line…"}</pre>`,
    `<span class="ver-sub">Keep this page open — it reconnects on its own when the Pi restarts.</span>`);
  const log = $("sysLog");
  if (log && stick) log.scrollTop = log.scrollHeight;
}

async function poll() {
  try {
    const r = await api(`/api/system/log?since=${encodeURIComponent(run.started_at)}&cursor=${encodeURIComponent(run.cursor)}`);
    run.down = null; run.lastOk = Date.now();
    run.cursor = r.cursor || run.cursor;
    for (const l of r.lines) {
      run.lines.push(l);
      const m = /^==> (\S+)/.exec(l.text);
      if (m) run.marks.push({ name: m[1], t: l.t });
    }
    if (run.lines.length > 4000) run.lines.splice(0, run.lines.length - 4000);
    // Over: the job has stopped and said something — or never said
    // anything in 20 s (it failed before its first line).
    if (!r.running && (run.lines.length || Date.now() / 1000 - run.started_at > 20)) {
      S = await api("/api/system?fetch=0");
      if (!S.running) return viewResult();
    }
  } catch (e) {
    run.down = run.down || Date.now();
  }
  if (!shown) return;                     // the page left: picked up again by onShow
  viewProgress();
  pollTimer = setTimeout(poll, run.down ? 2500 : 1500);
}

// ── the RESULT — the record of this run ─────────────────────────────────
function viewResult() {
  const r = S.record;
  const ours = r && !r.error && new Date(r.finished).getTime() / 1000 >= run.started_at - 5;
  const tail = run.lines.slice(-30).map(l => l.text).join("\n");
  const full = run.lines.map(l => l.text).join("\n");
  const actions = `<button class="btn" id="btnSysLog">Full log</button><div class="spacer"></div>`;
  if (!ours) {
    card("Upgrade stopped", "", `<div class="sys-big"><div class="bang">!</div><h2>The upgrade did not finish</h2>
      <div class="ver-sub">It stopped early — most often no network, so the download failed. Nothing was installed after that point; the last lines below say why.</div></div>
      <div class="ver-label">Its last lines</div><pre class="sys-log">${esc(tail) || "(no output)"}</pre>`,
      actions + `<button class="btn" id="btnSysCheck">Check again</button><button class="btn btn-primary" id="btnSysAgain">Upgrade again…</button>`);
  } else {
    const ok = r.result === "finished";
    const failed = S.parts.filter(p => p.exit != null && p.exit !== 0);
    const rows = S.parts.filter(p => p.dir).map(p => {
      const b = (run.before || {})[p.name];
      const now = p.commit ? `${p.branch} ${short(p.commit)}` : "—";
      const same = b && p.commit && b.branch === p.branch && p.commit.startsWith(b.hash);
      return `<tr><td class="what">${esc(p.label || p.name)}${p.exit ? `<div class="sub bad"><span class="msg-mark sm" data-level="bad" aria-hidden="true">!</span>failed (exit ${esc(p.exit)})</div>` : ""}</td>`
        + `<td class="h">${b ? esc(`${b.branch} ${short(b.hash)}`) : `<span class="sub">not installed</span>`}</td>`
        + `<td class="h">${esc(now)}${same ? ` <span class="sub">same</span>` : ""}</td></tr>`;
    }).join("");
    const head = ok
      ? `<div class="sys-big"><div class="tick">✓</div><h2>${failed.length ? "Upgraded, with failures" : "Upgraded"}</h2>
         <div class="ver-sub">The vision Pi rebooted and the server is back.${failed.length ? ` ${plural(failed.length, "part")} failed — see below and the full log.` : ""}</div></div>`
      : `<div class="sys-big"><div class="bang">!</div><h2>Stopped before the reboot</h2>
         <div class="ver-sub">${esc(r.reason || "")}. The parts before it are installed; the Pi was NOT rebooted.</div></div>`;
    card(ok ? "Upgrade finished" : "Upgrade stopped",
      `${new Date(r.started).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} → ${new Date(r.finished).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${fmtDur((new Date(r.finished) - new Date(r.started)) / 1000)}`,
      head + (rows ? `<table class="sys-t"><tr><th>Part</th><th>Before</th><th>Now</th></tr>${rows}</table>` : "")
        + (ok ? "" : `<div class="ver-label">Its last lines</div><pre class="sys-log">${esc(tail)}</pre>`),
      actions + (ok ? `<button class="btn btn-primary" id="btnSysDone">Done</button>`
                    : `<button class="btn" id="btnSysCheck">Check again</button><button class="btn btn-primary" id="btnSysAgain">Upgrade again…</button>`));
  }
  save(null);
  run = null;
  onFoot({
    btnSysDone: () => refresh(true), btnSysCheck: () => refresh(true),
    btnSysAgain: async () => { await refresh(true); viewConfirm(); },
    btnSysLog: () => {
      const h = host();
      const pre = h.querySelector(".sys-log") || (() => {
        h.querySelector(".sys-body").insertAdjacentHTML("beforeend", `<pre class="sys-log"></pre>`);
        return h.querySelector(".sys-log");
      })();
      pre.textContent = full;
      pre.style.maxHeight = "60vh";
    },
  });
}

// ── load ────────────────────────────────────────────────────────────────
async function refresh(fetchRemote) {
  card("Installed", "", `<div class="ver-busy">${fetchRemote ? "Checking for updates…" : "Loading…"}</div>`);
  try {
    S = await api(`/api/system?fetch=${fetchRemote ? 1 : 0}`);
  } catch (e) {
    card("Installed", "", warn(esc(String(e.message || e)), true),
      `<div class="spacer"></div><button class="btn btn-primary" id="btnSysCheck">Check again</button>`);
    onFoot({ btnSysCheck: () => refresh(true) });
    return;
  }
  if (!S.owned) return viewNotOwned();
  viewUnit();
}

export async function onShow() {
  shown = true;
  if (run) return poll();                 // following one: carry on
  // An upgrade this browser started — still running, or over while the
  // page was away (the restart, the reboot): follow it to its result.
  const mine = load();
  if (mine) {
    try { S = await api("/api/system?fetch=0"); } catch {}
    follow(mine);
    return;
  }
  await refresh(true);
}

export function onHide() {
  shown = false;
  clearTimeout(pollTimer);
}
