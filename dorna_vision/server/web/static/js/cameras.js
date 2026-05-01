// Cameras page — list / add (with kwargs textarea) / remove / capture-on-demand.

const $  = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));

const DEFAULT_KWARGS = {
  mode: "bgrd",
  stream: { width: 848, height: 480, fps: 15 },
};

let _vc = null;
let _devices = [];
const _capturedUrls = new Map();   // sn -> object URL (revoked on next swap)
const _prevAttached = new Map();   // sn -> last seen `attached` boolean (for transition detection on manual refresh)
const _health     = new Map();     // sn -> { state, msg, ts }   live device-protocol state pushed by the server


// Combined "is the camera usable right now?" check. Prefers the
// device-protocol health pushed by the bus when known; falls back to the
// USB-attached flag from camera_list when the bus hasn't said anything
// yet (e.g. when --no-mqtt or before the first WS event arrives).
function _isUsable(d) {
  const h = _health.get(d.serial_number);
  if (h && h.state) return h.state === "ok";
  return !!d.attached;
}

// ── helpers ─────────────────────────────────────────────────────────

function isPageActive() {
  const sec = document.querySelector('section[data-page="cameras"]');
  return !!sec && sec.classList.contains("active");
}

function escHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function toast(msg, kind = "ok") {
  const area = $("#toastArea");
  if (!area) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  area.appendChild(el);
  setTimeout(() => el.remove(), 3500);
  el.addEventListener("click", () => el.remove());
}

// ── render ──────────────────────────────────────────────────────────

function statusPill(d) {
  // Health badge driven by Camera's device-protocol state ("ok" | "down" |
  // "recovering"), pushed by the server over the WS device_state event.
  // Falls back to the USB-attached flag when no live state has been seen
  // yet (--no-mqtt mode, or the brief window before the first WS push).
  const h = _health.get(d.serial_number);
  const state = (h && h.state) ? h.state : (d.attached ? "ok" : "down");
  let klass, dotClass, title;
  if (state === "ok") {
    klass = "connected"; dotClass = "ok pulse"; title = "OK";
  } else if (state === "recovering") {
    klass = "recovering"; dotClass = "warn pulse"; title = "Recovering";
  } else {
    klass = "missing"; dotClass = "bad"; title = (h && h.msg) || "Down";
  }
  return `<span class="cc-status ${klass}" title="${escHtml(title)}"><span class="dot ${dotClass}"></span></span>`;
}

function cardHTML(d) {
  // Only added cameras get cards. Available devices are reachable via the Add modal.
  const sn = d.serial_number;
  const name = d.name || "RealSense camera";

  // Inline the cached JPEG blob URL straight into the <img> so re-renders
  // don't flicker. Without this, innerHTML rebuilds an empty <img> and the
  // browser paints one blank frame before src is reassigned.
  const cached = _capturedUrls.get(sn);
  const usable = _isUsable(d);

  const thumb = usable ? `
    <div class="cc-thumb">
      <img alt="capture" data-cam-thumb="${escHtml(sn)}"
           ${cached ? `src="${cached}" style="display:block"` : `style="display:none"`}/>
      <div class="cc-thumb-empty" ${cached ? `style="display:none"` : ""}>Click <strong>Capture</strong> to grab a frame</div>
      <button class="cc-expand" data-act="expand" title="Expand" ${cached ? `style="display:flex"` : `style="display:none"`}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
      </button>
      <span class="cc-dim-overlay" data-meta="dim" ${cached ? "" : `style="display:none"`}>—</span>
    </div>` : `
    <div class="cc-thumb cc-thumb-warn">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <div class="cc-thumb-warn-title">Camera ${escHtml((_health.get(sn) || {}).state || "down")}</div>
      <div class="cc-thumb-warn-sub">${escHtml((_health.get(sn) || {}).msg || "Reconnect the USB cable to resume.")}</div>
    </div>`;

  const k  = (key) => `<span class="cc-mk">${key}</span>`;
  const v  = (value, dataMeta = "") =>
    `<strong${dataMeta ? ` data-meta="${dataMeta}"` : ""}>${value}</strong>`;
  const sep = ` <span class="cc-sep">·</span> `;

  // One compact mono line — identity + live params.
  const meta = `
    <div class="cc-meta-lines">
      <div class="cc-meta-line">
        ${k("sn")} ${v(escHtml(sn))}${d.usb_type ? `${sep}${k("usb")} ${v(escHtml(d.usb_type))}` : ""}${sep}${k("type")} ${v("—", "mode")}${sep}${k("fps")} ${v("—", "fps")}
      </div>
    </div>`;

  return `
    <div class="cam-card is-added ${usable ? "" : "is-missing"}" data-sn="${escHtml(sn)}">
      <div class="cc-head">
        <div class="cc-info">
          <div class="cc-name">${escHtml(name)}</div>
        </div>
        ${statusPill(d)}
      </div>
      ${meta}
      <div class="cc-body">${thumb}</div>
      <div class="cc-foot">
        <button class="btn btn-sm" data-act="capture" ${usable ? "" : "disabled"}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
          Capture
        </button>
        ${(() => {
          if (usable) return "";
          const busy = (_health.get(sn) || {}).state === "recovering";
          return `<button class="btn btn-sm btn-primary" data-act="recover" ${busy ? "disabled" : ""}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
            ${busy ? "Recovering…" : "Recover"}
          </button>`;
        })()}
        <div class="spacer"></div>
        <button class="btn btn-danger btn-sm" data-act="remove">Remove</button>
      </div>
    </div>
  `;
}

// Header search query — narrows the visible grid by serial / name. Empty
// string = show all. Stored at module scope so render() and the input
// handler share the same source of truth.
let _searchQuery = "";

function _matchesSearch(d, q) {
  if (!q) return true;
  const hay = `${d.serial_number || ""} ${d.name || ""}`.toLowerCase();
  return hay.includes(q);
}

function _updateCount(visible, total) {
  const el = $("#camCount");
  if (el) el.textContent = `${visible} of ${total}`;
}

function render() {
  const grid = $("#camGrid");
  if (!grid) return;
  const added   = _devices.filter(d => d.added);
  const visible = added.filter(d => _matchesSearch(d, _searchQuery));
  _updateCount(visible.length, added.length);

  if (!added.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◫</div>
        <h3>No cameras yet</h3>
        <p>Click <strong>Add Camera</strong> to get started.</p>
      </div>`;
    return;
  }
  if (!visible.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◫</div>
        <h3>No matches</h3>
        <p>No cameras match <strong>${escHtml(_searchQuery)}</strong>.</p>
      </div>`;
    return;
  }
  grid.innerHTML = visible.map(cardHTML).join("");
  // wire per-card actions
  $$(".cam-card", grid).forEach(card => {
    const sn = card.dataset.sn;
    const d = visible.find(x => x.serial_number === sn);
    _bindCardHandlers(card, sn);
    if (d && _isUsable(d)) {
      refreshMeta(sn, card);   // one-shot on render; mode/fps don't change after connect
    }
  });
}

// (cardHTML now inlines the cached blob URL directly into the <img>, so
//  there's no separate "restore on render" step — eliminates the flicker
//  caused by innerHTML rebuilding an empty <img>.)

// ── data ────────────────────────────────────────────────────────────

async function refreshList() {
  if (!_vc?.isConnected()) return;
  try {
    const fresh = await _vc.cameraList();
    // Detect missing→connected transitions for added cameras so we can
    // warm the pipeline (the camera module's recovery only fires inside
    // get_all(), not on attribute reads like get_temp()).
    const recovered = [];
    for (const d of fresh) {
      if (!d.added) continue;
      const wasAttached = _prevAttached.get(d.serial_number);
      if (wasAttached === false && d.attached) recovered.push(d.serial_number);
      _prevAttached.set(d.serial_number, d.attached);
    }
    _devices = fresh;
    render();
    for (const sn of recovered) {
      const card = document.querySelector(`.cam-card[data-sn="${CSS.escape(sn)}"]`);
      if (card) {
        // Capture triggers Camera._recover() server-side, then refresh meta.
        captureFrame(sn, card).then(() => refreshMeta(sn, card));
      }
    }
  } catch (e) {
    console.error("camera_list failed", e);
  }
}

async function refreshMeta(sn, card) {
  if (!_vc?.isConnected()) return;
  const cam = _vc.camera(sn);
  // Connect-time attributes only — both are stored on the Camera and don't
  // change after connect. The dim chip is the image overlay; refreshed by
  // captureFrame from the actual JPEG shape.
  const [mode, stream] = await Promise.allSettled([
    cam.mode(),
    cam.stream(),
  ]);
  const setVal = (k, v) => {
    const el = card.querySelector(`[data-meta="${k}"]`);
    if (el) el.textContent = v;
  };
  setVal("mode", mode.status === "fulfilled" && mode.value ? String(mode.value) : "—");
  if (stream.status === "fulfilled" && stream.value && typeof stream.value === "object") {
    const s = stream.value;
    setVal("dim", (s.width != null && s.height != null) ? `${s.width} × ${s.height}` : "—");
    setVal("fps", s.fps != null ? String(s.fps) : "—");
  } else {
    setVal("dim", "—");
    setVal("fps", "—");
  }
}

async function captureFrame(sn, card) {
  if (!_vc?.isConnected()) return;
  const btn = card.querySelector('[data-act="capture"]');
  if (btn) btn.disabled = true;
  try {
    const { json, binary } = await _vc.cameraGetImg(sn, "color_img", 75);
    if (!binary) throw new Error("no image returned");
    const url = URL.createObjectURL(new Blob([binary], { type: "image/jpeg" }));
    const prev = _capturedUrls.get(sn);
    _capturedUrls.set(sn, url);
    const img = card.querySelector(`img[data-cam-thumb="${sn}"]`);
    if (img) {
      img.src = url;
      img.style.display = "block";
      const empty  = card.querySelector(".cc-thumb-empty");
      const expand = card.querySelector('[data-act="expand"]');
      const dim    = card.querySelector(".cc-dim-overlay");
      if (empty)  empty.style.display = "none";
      if (expand) expand.style.display = "flex";
      if (dim) {
        // Use the JPEG's actual dimensions returned by the server (from the
        // raw frame — accurate even if Camera.connect requested something else).
        const shape = json && Array.isArray(json.shape) ? json.shape : null;
        if (shape && shape.length >= 2) {
          dim.textContent = `${shape[1]} × ${shape[0]}`;
        }
        dim.style.display = "inline-block";
      }
    }
    if (prev) URL.revokeObjectURL(prev);
  } catch (e) {
    toast(`Capture failed: ${e.message || e}`, "bad");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function expandCapture(sn) {
  const url = _capturedUrls.get(sn);
  if (!url) { toast("Nothing captured yet", "warn"); return; }
  const overlay = $("#imgLightbox");
  const img = $("#imgLightboxImg");
  const cap = $("#imgLightboxCap");
  if (!overlay || !img) return;
  img.src = url;
  if (cap) cap.textContent = `SN ${sn}`;
  overlay.dataset.sn = sn;
  // Make sure the "Capture again" button is visible — the playground
  // expand flow hides it because the playground doesn't have a single SN.
  $("#imgLightboxCapture")?.removeAttribute("hidden");
  overlay.classList.add("show");
}

async function lightboxRecapture() {
  const sn = $("#imgLightbox")?.dataset.sn;
  if (!sn) return;
  const card = document.querySelector(`.cam-card[data-sn="${CSS.escape(sn)}"]`);
  if (!card) return;
  await captureFrame(sn, card);
  const url = _capturedUrls.get(sn);
  if (url) $("#imgLightboxImg").src = url;
}

async function removeCamera(sn) {
  if (!_vc?.isConnected()) return;
  const ok = await window.confirmDialog({
    title: "Remove camera?",
    message: `Camera ${sn} will be released. Any client still using it will be disconnected.`,
    confirm: "Remove",
    variant: "danger",
    icon: "remove",
  });
  if (!ok) return;

  // Wait for the server to actually release the camera (closes the pyrealsense
  // pipeline, drops USB handles) — then sync the UI from authoritative state.
  try {
    await _vc.cameraRemove(sn);
    const url = _capturedUrls.get(sn);
    if (url) { URL.revokeObjectURL(url); _capturedUrls.delete(sn); }
    _prevAttached.delete(sn);
    toast(`Removed ${sn}`, "ok");
    await refreshList();
  } catch (e) {
    toast(`Remove failed: ${e.message || e}`, "bad");
  }
}

// ── add modal ───────────────────────────────────────────────────────

function populateAddModalDropdown(presetSn) {
  const sel = $("#camFieldSn");
  if (!sel) return;
  const candidates = _devices.filter(d => d.attached && !d.added);
  sel.innerHTML = candidates.length
    ? candidates.map(d => `<option value="${escHtml(d.serial_number)}">${escHtml(d.serial_number)} — ${escHtml(d.name || "RealSense")}</option>`).join("")
    : `<option value="">(no available cameras)</option>`;
  if (presetSn && candidates.find(d => d.serial_number === presetSn)) {
    sel.value = presetSn;
  }
}

function openAddModal(presetSn) {
  const overlay = $("#camModalOverlay");
  const ta  = $("#camFieldKwargs");
  const err = $("#camFieldKwargsErr");

  populateAddModalDropdown(presetSn);

  // Show the JSON template as placeholder; keep the value empty so user can
  // either submit empty (uses Camera.connect defaults) or paste/edit their own.
  ta.placeholder = JSON.stringify(DEFAULT_KWARGS, null, 2);
  ta.classList.remove("input-error");
  err.style.display = "none";
  err.textContent = "";

  overlay.classList.add("show");
}

async function modalRefresh() {
  // Full refresh — fetch device list, re-render the card grid (so any
  // attached/missing changes show), then update the dropdown from the
  // freshly fetched list.
  const btn = $("#camModalRefresh");
  if (btn) btn.disabled = true;
  try {
    await refreshList();
    populateAddModalDropdown();
  } finally {
    if (btn) btn.disabled = false;
  }
}

function closeAddModal() { $("#camModalOverlay")?.classList.remove("show"); }

function parseKwargs() {
  const ta  = $("#camFieldKwargs");
  const err = $("#camFieldKwargsErr");
  const raw = ta.value.trim();
  if (!raw) return {};
  try {
    const obj = JSON.parse(raw);
    if (obj === null || typeof obj !== "object" || Array.isArray(obj)) {
      throw new Error("must be a JSON object");
    }
    ta.classList.remove("input-error");
    err.style.display = "none";
    err.textContent = "";
    return obj;
  } catch (e) {
    ta.classList.add("input-error");
    err.style.display = "block";
    err.textContent = `Invalid JSON: ${e.message}`;
    throw e;
  }
}

async function submitAddModal() {
  const sn = $("#camFieldSn").value;
  if (!sn) { toast("Pick a camera first", "warn"); return; }
  let kwargs;
  try { kwargs = parseKwargs(); }
  catch { return; }

  closeAddModal();
  try {
    await _vc.cameraAdd(sn, kwargs);
    toast(`Added ${sn}`, "ok");
    await refreshList();
    // Grab and display the first frame so the card lands populated.
    const card = document.querySelector(`.cam-card[data-sn="${CSS.escape(sn)}"]`);
    if (card) captureFrame(sn, card);
  } catch (e) {
    toast(`Add failed: ${e.message || e}`, "bad");
  }
}

// ── lifecycle ───────────────────────────────────────────────────────

export function init(vc) {
  _vc = vc;
  _wireDeviceStateChannel();
  $("#camAddBtn")?.addEventListener("click", () => openAddModal());

  // Header search — re-render on every keystroke. Trim + lowercase
  // once here so the matcher's fast path is a plain substring check.
  $("#camSearch")?.addEventListener("input", (e) => {
    _searchQuery = e.target.value.trim().toLowerCase();
    render();
  });
  $("#camModalRefresh")?.addEventListener("click", modalRefresh);
  $("#camModalClose")?.addEventListener("click", closeAddModal);
  $("#camModalCancel")?.addEventListener("click", closeAddModal);
  $("#camModalConfirm")?.addEventListener("click", submitAddModal);
  $("#camModalOverlay")?.addEventListener("click", (e) => {
    if (e.target.id === "camModalOverlay") closeAddModal();
  });

  // Lightbox close handlers (any click outside the image dismisses)
  $("#imgLightboxClose")?.addEventListener("click", () => $("#imgLightbox")?.classList.remove("show"));
  $("#imgLightboxCapture")?.addEventListener("click", lightboxRecapture);
  $("#imgLightbox")?.addEventListener("click", (e) => {
    if (e.target.id === "imgLightbox") $("#imgLightbox").classList.remove("show");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#imgLightbox")?.classList.remove("show");
  });
}

async function recoverCamera(sn) {
  if (!_vc?.isConnected()) return;
  try {
    toast(`Recovering ${sn}…`, "ok");
    const r = await _vc.cameraRecover(sn, { timeout: 60000 });
    if (r?.ok) toast(`Recovered ${sn}`, "ok");
    else      toast(`Recover failed: ${r?.msg || "unknown"}`, "bad");
    // The server pushes device_state events as the recovery progresses, so
    // the badge updates automatically. Refresh attached/added flags too.
    await refreshList();
  } catch (e) {
    toast(`Recover failed: ${e.message || e}`, "bad");
  }
}

// ── live device-state subscription ──────────────────────────────────
// Called once during init() — wires the api.js event channel into the
// per-card UI. Server pushes one event per state transition; we update
// _health, then rebuild the card so badge / thumbnail / footer / class
// all reflect the new health together. The cached blob URL is re-inlined
// by cardHTML, so the captured frame doesn't flicker.
function _wireDeviceStateChannel() {
  if (!_vc) return;
  _vc.onEvent("device_state", (evt) => {
    const sn = evt.serial_number;
    if (!sn) return;
    _health.set(sn, { state: evt.state, msg: evt.msg, ts: evt.ts });

    // When a camera goes down, the previously captured thumbnail no
    // longer represents reality (the pipeline will be torn down and
    // rebuilt, possibly with a different view). Invalidate the cache so
    // the disconnected-warning thumbnail shows immediately and, after
    // recovery, the user sees a clean "Click Capture" prompt instead of
    // the stale frame from before the drop.
    if (evt.state === "down") {
      const cached = _capturedUrls.get(sn);
      if (cached) {
        try { URL.revokeObjectURL(cached); } catch {}
        _capturedUrls.delete(sn);
      }
    }

    if (!isPageActive()) return;
    const card = document.querySelector(`.cam-card[data-sn="${CSS.escape(sn)}"]`);
    if (!card) return;
    const d = _devices.find(x => x.serial_number === sn);
    if (!d) return;

    const tmp = document.createElement("div");
    tmp.innerHTML = cardHTML(d);
    const newCard = tmp.firstElementChild;
    if (!newCard) return;
    card.replaceWith(newCard);
    _bindCardHandlers(newCard, sn);
  });
}

// Per-card handler binding, used by render() and the live event channel.
function _bindCardHandlers(card, sn) {
  card.querySelector('[data-act="remove"]')?.addEventListener("click", () => removeCamera(sn));
  card.querySelector('[data-act="capture"]')?.addEventListener("click", () => captureFrame(sn, card));
  card.querySelector('[data-act="expand"]')?.addEventListener("click", () => expandCapture(sn));
  card.querySelector('[data-act="recover"]')?.addEventListener("click", () => recoverCamera(sn));
}

export function onShow() {
  // One-shot fetch when the page opens. No background polling.
  refreshList();
}

export function onHide() {
  // Captured frames are kept in cache so re-opening the page restores the last shot.
}
