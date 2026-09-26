// files.js — the Files page: browse the captures folder.
//
// Same grammar as the workspace orchestrator's file browser (its
// files.js): breadcrumbs, one row per entry, a resizable preview pane on
// the right. Here the root is the capture folder (server --captures,
// default ~/captures), and the preview is built for IMAGES — the point
// preview pane stays hidden until a file is clicked, exactly like the
// workspace's; an image shows in it, ↑/↓ steps through the folder.
//
// Folders download as a zip (streamed by the server — a folder of
// thousands of captures never sits in memory). Delete takes a file or
// an EMPTY folder only, same rule as the workspace.

const API = "/api/files";
const IMG_EXT = /\.(jpe?g|png|bmp|gif|webp|tiff?)$/i;

const ICON = {
  folder: '<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/>',
  file:   '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>',
  image:  '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
  down:   '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
  up:     '<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  trash:  '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6"/>',
  newdir: '<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/>',
  zip:    '<path d="M21 8v13H3V8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/>',
  reload: '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
};

const svg = (d, size = 14) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtSize(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

function fmtWhen(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const sameDay = d.toDateString() === new Date().toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return sameDay ? time : `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${time}`;
}

function toast(msg, kind = "ok") {
  const area = document.querySelector("#toastArea");
  if (!area) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  area.appendChild(el);
  setTimeout(() => el.remove(), 3500);
  el.addEventListener("click", () => el.remove());
}

const url = (params) => `${API}?${new URLSearchParams(params)}`;
const ROOT = "captures";

// ── the folder socket (server: fslive.py, twin of the workspace's) ──
// Listing, New folder and Delete go over one WebSocket, answered by id;
// the server PUSHES changes to the folder on screen — a new capture
// appears the moment it is written, without reloading. File BYTES stay
// on HTTP (preview, download, zip, upload): streamed, with the
// browser's own download handling and image cache. The socket is open
// while the Files page is on screen, closed when it is left.
class FolderSocket {
  constructor(url, onEvent) {
    this.url = url; this.onEvent = onEvent;
    this.seq = 0; this.wait = new Map(); this.closed = false; this.backoff = 500;
    this.ready = this._connect();
  }
  _connect() {
    return new Promise((resolve) => {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      ws.onopen = () => { this.backoff = 500; resolve(); this.onEvent({ ev: "open" }); };
      ws.onmessage = (m) => {
        let d; try { d = JSON.parse(m.data); } catch { return; }
        if (d.id != null && this.wait.has(d.id)) {
          const { ok, fail } = this.wait.get(d.id); this.wait.delete(d.id);
          d.ok ? ok(d) : fail(new Error(d.error || "request failed"));
        } else if (d.ev) this.onEvent(d);
      };
      ws.onclose = () => {
        for (const { fail } of this.wait.values()) fail(new Error("connection lost"));
        this.wait.clear();
        if (this.closed) return;
        setTimeout(() => { if (!this.closed) this.ready = this._connect(); }, this.backoff);
        this.backoff = Math.min(this.backoff * 2, 8000);
      };
    });
  }
  async request(op, args) {
    await this.ready;
    const id = ++this.seq;
    return new Promise((ok, fail) => {
      this.wait.set(id, { ok, fail });
      this.ws.send(JSON.stringify({ id, op, ...args }));
    });
  }
  close() { this.closed = true; try { this.ws.close(); } catch {} }
}

const socketUrl = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/files`;
const byOrder = (a, b) => (a.dir === b.dir ? b.mtime - a.mtime : a.dir ? -1 : 1);

// PUT one file as the raw request body (streamed to disk by the server).
function putFile(u, file, onPct) {
  return new Promise((ok, fail) => {
    const x = new XMLHttpRequest();
    x.open("PUT", u);
    x.upload.onprogress = (e) => { if (e.lengthComputable) onPct(Math.round((e.loaded / e.total) * 100)); };
    x.onload = () => {
      let d = {}; try { d = JSON.parse(x.responseText); } catch {}
      x.status < 300 ? ok(d) : fail(new Error(d.error || `upload failed (${x.status})`));
    };
    x.onerror = () => fail(new Error("upload failed — connection lost"));
    x.send(file);
  });
}

// ── state ──────────────────────────────────────────────────────────
let path = "";          // folder shown, relative to the captures root
let entries = [];       // its listing
let selected = null;    // the entry in the preview
let active = false;     // page on screen (keyboard nav only then)
let wired = false;
let sock = null;        // the folder socket, while the page is on screen
const rows = new Map(); // path -> its row element

const $ = (s) => document.querySelector(`#filesPage ${s}`);

// ── listing ────────────────────────────────────────────────────────
// One "open" per folder; afterwards the server pushes what changed in it
// (onEvent) and only those rows are touched.
let opening = 0;
async function load() {
  const list = $(".fb-list");
  crumbs();
  const mine = ++opening;
  list.innerHTML = `<div class="fb-loading">${[0, 1, 2].map(() => '<div class="fb-skel"></div>').join("")}</div>`;
  let data;
  try {
    data = (await sock.request("open", { root: ROOT, path })).folder;
  } catch (err) {
    if (mine !== opening) return;
    list.innerHTML = `<div class="fb-error">${svg(ICON.file)} ${esc(err.message)}</div>`;
    return;
  }
  if (mine !== opening) return;
  entries = data.entries;
  $(".fb-path").textContent = data.abs;
  $(".fb-path").title = data.abs + (data.live ? "" : " — not live: reload to refresh");
  // keep the preview when the selected file is still here (reload)
  if (selected && !entries.some((e) => e.path === selected.path)) select(null);
  render();
}

function render() {
  const list = $(".fb-list");
  rows.clear();
  if (!entries.length) {
    list.innerHTML = `<div class="fb-empty">Nothing here yet — <b>Upload</b> adds files, or drop them here.</div>`;
    return;
  }
  const frag = document.createDocumentFragment();
  for (const e of entries) frag.appendChild(row(e));
  list.innerHTML = "";
  list.appendChild(frag);
  markSelected();
}

// What the server pushes for the folder on screen: rows added, changed
// or gone — applied in place, order kept.
function onEvent(d) {
  if (d.ev === "open" && rows.size) { load(); return; }   // reconnected: resync
  if (d.root !== ROOT || d.path !== path) return;
  if (d.ev === "gone") { go(path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ""); return; }
  if (d.ev !== "change") return;
  if (d.reset) { entries = d.upsert.sort(byOrder); render(); return; }
  const rm = new Set(d.remove || []);
  const up = new Map((d.upsert || []).map((e) => [e.path, e]));
  entries = entries.filter((e) => !rm.has(e.path) && !up.has(e.path))
                   .concat([...up.values()]).sort(byOrder);
  if (!entries.length || !rows.size) { render(); return; }
  for (const p of rm) { rows.get(p)?.remove(); rows.delete(p); }
  for (const [p, e] of up) { rows.get(p)?.remove(); rows.delete(p); row(e); }
  const frag = document.createDocumentFragment();          // one reorder pass
  for (const e of entries) frag.appendChild(rows.get(e.path));
  $(".fb-list").appendChild(frag);
  if (selected) {
    if (rm.has(selected.path)) select(null);
    else if (up.has(selected.path)) select(up.get(selected.path));
    else markSelected();
  }
}

function row(e) {
  const isImg = !e.dir && IMG_EXT.test(e.name);
  const r = document.createElement("div");
  r.className = "fb-row";
  r.dataset.path = e.path;
  r.setAttribute("role", "option");
  r.innerHTML = `
    <span class="fb-ic ${e.dir ? "is-dir" : ""}">${svg(e.dir ? ICON.folder : isImg ? ICON.image : ICON.file, 15)}</span>
    <span class="fb-name">${esc(e.name)}</span>
    <span class="fb-size">${e.dir ? "" : fmtSize(e.size)}</span>
    <span class="fb-when">${fmtWhen(e.mtime)}</span>
    <span class="fb-acts"></span>`;
  const acts = r.querySelector(".fb-acts");
  const dl = document.createElement("a");
  dl.className = "btn btn-ghost btn-sm btn-icon";
  dl.title = e.dir ? `Download ${e.name} as a zip` : `Download ${e.name}`;
  dl.href = e.dir ? url({ path: e.path, zip: 1 }) : url({ path: e.path, download: 1 });
  dl.setAttribute("download", e.dir ? `${e.name}.zip` : e.name);
  dl.innerHTML = svg(e.dir ? ICON.zip : ICON.down, 13);
  dl.addEventListener("click", (ev) => ev.stopPropagation());
  acts.appendChild(dl);
  const del = document.createElement("button");
  del.className = "btn btn-ghost btn-sm btn-icon fb-del";
  del.title = e.dir ? `Delete the folder ${e.name}` : `Delete ${e.name}`;
  del.innerHTML = svg(ICON.trash, 13);
  del.addEventListener("click", (ev) => { ev.stopPropagation(); remove(e.path); });
  acts.appendChild(del);
  r.addEventListener("click", () => {
    const cur = entries.find((x) => x.path === e.path) || e;
    if (cur.dir) { go(cur.path); } else { select(cur); }
  });
  rows.set(e.path, r);
  return r;
}

function go(p) {
  path = p;
  select(null);
  load();
}

function crumbs() {
  const wrap = $(".fb-crumbs");
  wrap.innerHTML = "";
  const parts = path ? path.split("/") : [];
  const mk = (label, target, last) => {
    const b = document.createElement("button");
    b.className = "fb-crumb" + (last ? " is-last" : "");
    b.type = "button";
    b.textContent = label;
    if (!last) b.addEventListener("click", () => go(target));
    wrap.appendChild(b);
    if (!last) wrap.insertAdjacentHTML("beforeend", '<span class="fb-sep">/</span>');
  };
  mk("Captures", "", parts.length === 0);
  parts.forEach((p, i) => mk(p, parts.slice(0, i + 1).join("/"), i === parts.length - 1));
}

// ── selection + preview ────────────────────────────────────────────
function markSelected() {
  document.querySelectorAll("#filesPage .fb-row").forEach((r) =>
    r.classList.toggle("is-selected", !!selected && r.dataset.path === selected.path));
}

function showPane(on) {
  $(".fb-preview").hidden = !on;
  $(".fb-split").hidden = !on;
}

function select(e) {
  selected = e;
  markSelected();
  const pv = $(".fb-preview");
  if (!e) { showPane(false); pv.innerHTML = ""; return; }
  showPane(true);
  document.querySelector(`#filesPage .fb-row[data-path="${CSS.escape(e.path)}"]`)
    ?.scrollIntoView({ block: "nearest" });
  const head = `<div class="fb-pv-head"><span class="fb-pv-name">${esc(e.name)}</span>` +
    `<span class="fb-pv-meta">${fmtSize(e.size)} · ${fmtWhen(e.mtime)}</span></div>`;
  if (!IMG_EXT.test(e.name)) {
    pv.innerHTML = head + `<div class="fb-error">Cannot preview this file</div>`;
    return;
  }
  // mtime in the URL: a file overwritten in place shows its new content
  pv.innerHTML = head + `<div class="fb-pv-scroll fb-pv-img"><img alt="${esc(e.name)}"
    src="${url({ path: e.path, raw: 1, t: e.mtime })}"/></div>`;
  pv.querySelector("img").onerror = () => {
    pv.querySelector(".fb-pv-img").outerHTML = `<div class="fb-error">Could not load the image</div>`;
  };
}

function step(delta) {
  const files = entries.filter((x) => !x.dir);
  if (!files.length) return;
  const i = selected ? files.findIndex((x) => x.path === selected.path) : -1;
  const j = i < 0 ? (delta > 0 ? 0 : files.length - 1)
                  : Math.min(files.length - 1, Math.max(0, i + delta));
  select(files[j]);
}

// ── actions ────────────────────────────────────────────────────────
// New folder / Delete go over the socket; the list updates from the
// change the server then pushes — no reload.
async function remove(p) {
  const e = entries.find((x) => x.path === p);
  if (!e) return;
  const ok = await window.confirmDialog({
    title: e.dir ? `Delete the folder “${e.name}”?` : `Delete “${e.name}”?`,
    message: e.dir ? "Only an empty folder can be deleted." : "This cannot be undone.",
    confirm: "Delete", variant: "danger", icon: "remove",
  });
  if (!ok) return;
  try {
    await sock.request("delete", { root: ROOT, path: e.path });
  } catch (err) { toast(err.message, "bad"); }
}

// Each file is its own PUT, streamed to disk by the server; the button
// shows which file and how far.
async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const btn = $(".fb-upload");
  const label = btn.innerHTML;
  btn.disabled = true;
  let failed = 0;
  try {
    for (const [k, f] of files.entries()) {
      const tag = files.length > 1 ? `${k + 1}/${files.length} ` : "";
      try {
        await putFile(`${API}/upload?${new URLSearchParams({ path, name: f.name })}`, f,
          (pct) => { btn.innerHTML = `<span class="fb-spin"></span> Uploading ${tag}${pct}%`; });
      } catch (err) { failed++; toast(`${f.name}: ${err.message}`, "bad"); }
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = label;
  }
  return failed;
}

function wire() {
  if (wired) return;
  wired = true;
  const input = $(".fb-file");
  $(".fb-upload").addEventListener("click", () => input.click());
  input.addEventListener("change", () => { upload(input.files).then(() => { input.value = ""; }); });

  $(".fb-mkdir").addEventListener("click", async () => {
    const name = window.prompt("New folder name");
    if (!name) return;
    try {
      await sock.request("mkdir", { root: ROOT, path: path ? `${path}/${name}` : name });
    } catch (err) { toast(err.message, "bad"); }
  });

  $(".fb-zip").addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = url({ path, zip: 1 });
    a.setAttribute("download", "");
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  $(".fb-reload").addEventListener("click", () => load());

  // Drop files anywhere on the panel to upload them into this folder.
  const panel = $(".fb-panel");
  let depth = 0;
  panel.addEventListener("dragenter", (ev) => {
    if (![...(ev.dataTransfer?.types || [])].includes("Files")) return;
    ev.preventDefault(); depth++; panel.classList.add("is-dropping");
  });
  panel.addEventListener("dragover", (ev) => {
    if ([...(ev.dataTransfer?.types || [])].includes("Files")) ev.preventDefault();
  });
  panel.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) panel.classList.remove("is-dropping");
  });
  panel.addEventListener("drop", (ev) => {
    ev.preventDefault(); depth = 0; panel.classList.remove("is-dropping");
    upload(ev.dataTransfer.files);
  });

  // Resizable split — the width survives a reload, like the workspace's.
  const SPLIT_KEY = "vf_split_pct";
  const body = $(".fb-body");
  const split = $(".fb-split");
  const pv = $(".fb-preview");
  const read = () => {
    const v = parseFloat(localStorage.getItem(SPLIT_KEY) || "");
    return Number.isFinite(v) ? Math.min(75, Math.max(20, v)) : 46;
  };
  pv.style.width = `${read()}%`;
  split.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    split.setPointerCapture(ev.pointerId);
    split.classList.add("is-dragging");
    body.classList.add("is-resizing");
    const move = (e) => {
      const r = body.getBoundingClientRect();
      if (!r.width) return;
      const pct = Math.min(75, Math.max(20, ((r.right - e.clientX) / r.width) * 100));
      pv.style.width = `${pct}%`;
      localStorage.setItem(SPLIT_KEY, String(Math.round(pct)));
    };
    const up = () => {
      split.releasePointerCapture(ev.pointerId);
      split.classList.remove("is-dragging");
      body.classList.remove("is-resizing");
      split.removeEventListener("pointermove", move);
      split.removeEventListener("pointerup", up);
    };
    split.addEventListener("pointermove", move);
    split.addEventListener("pointerup", up);
  });
  split.addEventListener("dblclick", () => {
    localStorage.removeItem(SPLIT_KEY);
    pv.style.width = "46%";
  });

  // ↑/↓ (and ←/→) step through the folder's files; Backspace goes up.
  document.addEventListener("keydown", (ev) => {
    if (!active || ev.target.closest("input, textarea, select, [contenteditable]")) return;
    if (document.querySelector(".confirm-overlay.show")) return;
    if (ev.key === "ArrowDown" || ev.key === "ArrowRight") { ev.preventDefault(); step(+1); }
    else if (ev.key === "ArrowUp" || ev.key === "ArrowLeft") { ev.preventDefault(); step(-1); }
    else if (ev.key === "Backspace" && path) {
      ev.preventDefault();
      go(path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "");
    }
  });
}

// ── page hooks ─────────────────────────────────────────────────────
export function onShow() {
  wire();
  active = true;
  if (!sock) sock = new FolderSocket(socketUrl(), onEvent);
  load();
}

export function onHide() {
  active = false;
  if (sock) { sock.close(); sock = null; }   // the server stops watching
  rows.clear();
}
