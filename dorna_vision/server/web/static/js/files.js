// files.js — the Files page: browse the captures folder.
//
// Same grammar as the workspace orchestrator's file browser (its
// files.js): breadcrumbs, one row per entry, a resizable preview pane on
// the right. Here the root is the capture folder (server --captures,
// default ~/captures), and the preview is built for IMAGES — the point
// of the page is reviewing a dataset: click an image and it fills the
// pane, ↑/↓ steps through the folder.
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

// ── state ──────────────────────────────────────────────────────────
let path = "";          // folder shown, relative to the captures root
let entries = [];       // its listing
let selected = null;    // the entry in the preview
let active = false;     // page on screen (keyboard nav only then)
let wired = false;

const $ = (s) => document.querySelector(`#filesPage ${s}`);

// ── listing ────────────────────────────────────────────────────────
async function load() {
  const list = $(".fb-list");
  crumbs();
  list.innerHTML = `<div class="fb-loading">${[0, 1, 2].map(() => '<div class="fb-skel"></div>').join("")}</div>`;
  let data;
  try {
    const resp = await fetch(url({ path }));
    data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Could not read the folder");
  } catch (err) {
    list.innerHTML = `<div class="fb-error">${svg(ICON.file)} ${esc(err.message)}</div>`;
    return;
  }
  entries = data.entries;
  $(".fb-path").textContent = data.abs + (path ? `/${path}` : "");
  $(".fb-path").title = $(".fb-path").textContent;
  const nImg = entries.filter((e) => !e.dir && IMG_EXT.test(e.name)).length;
  const nDir = entries.filter((e) => e.dir).length;
  $(".fb-count").textContent = [
    nDir ? `${nDir} folder${nDir === 1 ? "" : "s"}` : "",
    `${entries.length - nDir} file${entries.length - nDir === 1 ? "" : "s"}`,
    nImg && nImg !== entries.length - nDir ? `${nImg} image${nImg === 1 ? "" : "s"}` : "",
  ].filter(Boolean).join(" · ");

  // keep the preview when the selected file is still here (reload)
  if (selected && !entries.some((e) => e.path === selected.path)) select(null);

  if (!entries.length) {
    list.innerHTML = `<div class="fb-empty">Nothing here yet — <b>Upload</b> adds files, or drop them here.</div>`;
    return;
  }
  list.innerHTML = "";
  for (const e of entries) list.appendChild(row(e));
  markSelected();
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
  del.addEventListener("click", (ev) => { ev.stopPropagation(); remove(e); });
  acts.appendChild(del);
  r.addEventListener("click", () => {
    if (e.dir) { go(e.path); } else { select(e); }
  });
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

function select(e) {
  selected = e;
  markSelected();
  const pv = $(".fb-preview");
  if (!e) {
    pv.innerHTML = `<div class="fb-pv-empty">${svg(ICON.image, 28)}<span>Select an image to view it</span>
      <span class="fb-pv-hint">↑ ↓ step through the folder</span></div>`;
    return;
  }
  const rowEl = document.querySelector(`#filesPage .fb-row[data-path="${CSS.escape(e.path)}"]`);
  rowEl?.scrollIntoView({ block: "nearest" });
  const files = entries.filter((x) => !x.dir);
  const idx = files.findIndex((x) => x.path === e.path);
  const head = `<div class="fb-pv-head">
      <span class="fb-pv-name" title="${esc(e.name)}">${esc(e.name)}</span>
      <span class="fb-pv-meta"><span class="fb-pv-dims"></span>${fmtSize(e.size)} · ${fmtWhen(e.mtime)} · ${idx + 1} / ${files.length}</span>
      <a class="btn btn-ghost btn-sm btn-icon" title="Download ${esc(e.name)}"
         href="${url({ path: e.path, download: 1 })}" download="${esc(e.name)}">${svg(ICON.down, 13)}</a>
    </div>`;
  if (!IMG_EXT.test(e.name)) {
    pv.innerHTML = head + `<div class="fb-pv-empty">${svg(ICON.file, 28)}<span>No preview for this file type</span></div>`;
    return;
  }
  pv.innerHTML = head + `<div class="fb-pv-stage is-loading"><img alt="${esc(e.name)}"/></div>`;
  const stage = pv.querySelector(".fb-pv-stage");
  const img = stage.querySelector("img");
  img.onload = () => {
    stage.classList.remove("is-loading");
    const dims = pv.querySelector(".fb-pv-dims");
    if (dims) dims.textContent = `${img.naturalWidth}×${img.naturalHeight} · `;
  };
  img.onerror = () => {
    stage.classList.remove("is-loading");
    stage.innerHTML = `<div class="fb-error">Could not load the image</div>`;
  };
  // mtime in the URL: a file overwritten in place shows its new content
  img.src = url({ path: e.path, raw: 1, t: e.mtime });
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
async function post(action, body, isForm = false) {
  const u = `${API}/${action}${isForm ? `?${new URLSearchParams({ path })}` : ""}`;
  const resp = await fetch(u, isForm
    ? { method: "POST", body }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const d = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(d.error || `${action} failed`);
  return d;
}

async function remove(e) {
  const ok = await window.confirmDialog({
    title: e.dir ? `Delete the folder “${e.name}”?` : `Delete “${e.name}”?`,
    message: e.dir ? "Only an empty folder can be deleted." : "This cannot be undone.",
    confirm: "Delete", variant: "danger", icon: "remove",
  });
  if (!ok) return;
  try {
    await post("delete", { path: e.path });
    if (selected && selected.path === e.path) select(null);
    toast(`Deleted ${e.name}`, "ok");
    load();
  } catch (err) { toast(err.message, "bad"); }
}

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const btn = $(".fb-upload");
  const label = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="fb-spin"></span> Uploading ${files.length}…`;
  try {
    const fd = new FormData();
    for (const f of files) fd.append("file", f);
    const d = await post("upload", fd, true);
    toast(`Uploaded ${d.saved.length} file${d.saved.length === 1 ? "" : "s"}`, "ok");
    load();
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    btn.disabled = false;
    btn.innerHTML = label;
  }
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
      await post("mkdir", { path: path ? `${path}/${name}` : name });
      load();
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
    return Number.isFinite(v) ? Math.min(75, Math.max(25, v)) : 55;
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
      const pct = Math.min(75, Math.max(25, ((r.right - e.clientX) / r.width) * 100));
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
    pv.style.width = "55%";
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
  if (!selected) select(null);
  load();
}

export function onHide() {
  active = false;
}
