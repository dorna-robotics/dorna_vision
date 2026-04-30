// Phase 1 — skeleton wiring: connect to the WS, refresh server info + counts,
// switch between sections via the sidebar, and surface connection state.

import { VisionClient, VisionServerError } from "/static/js/api.js";
import * as Cameras    from "/static/js/cameras.js";
import * as Playground from "/static/js/playground.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const vc = new VisionClient();
window.vc = vc;   // for ad-hoc poking from devtools

const PAGE_HOOKS = {
  // refreshHome re-fetches the camera + detection counts. Without this,
  // the Home stats only update once on boot, so adding a detection in
  // the playground (or adding/removing a camera) leaves the stats stale.
  home:       { onShow: () => refreshHome() },
  cameras:    { onShow: () => Cameras.onShow(),    onHide: () => Cameras.onHide() },
  playground: { onShow: () => Playground.onShow(), onHide: () => Playground.onHide() },
};
let _currentRoute = null;

// ── connection pill ──────────────────────────────────────────────
function setConn(state, label) {
  const pill = $("#connPill");
  const dot  = $("#connDot");
  pill.classList.remove("ok", "warn", "bad", "off");
  dot.classList.remove("ok", "warn", "bad", "pulse");
  if (state === "ok") {
    pill.classList.add("ok");
    dot.classList.add("ok", "pulse");
  } else if (state === "warn") {
    pill.classList.add("warn");
    dot.classList.add("warn");
  } else if (state === "bad") {
    pill.classList.add("bad");
    dot.classList.add("bad");
  } else {
    pill.classList.add("off");
  }
  $("#connLabel").textContent = label;
}

// ── routing (hash-based) ─────────────────────────────────────────
function showPage(name) {
  if (_currentRoute && _currentRoute !== name) {
    PAGE_HOOKS[_currentRoute]?.onHide?.();
  }
  $$("section.page").forEach(s => s.classList.toggle("active", s.dataset.page === name));
  $$(".app-nav-link").forEach(a => a.classList.toggle("active", a.dataset.route === name));
  const titleMap = { home: "Home", cameras: "Cameras", playground: "Playground" };
  $("#pageTitle").textContent = titleMap[name] || "Home";
  _currentRoute = name;
  PAGE_HOOKS[name]?.onShow?.();
}

const ROUTES = ["home", "cameras", "playground"];

function currentRoute() {
  const r = location.pathname.replace(/^\/+|\/+$/g, "") || "home";
  return ROUTES.includes(r) ? r : "home";
}

// Intercept clicks on sidebar nav links so we don't full-reload the page.
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-route]");
  if (!a) return;
  const href = a.getAttribute("href");
  if (!href || !href.startsWith("/")) return;
  e.preventDefault();
  if (location.pathname !== href) history.pushState(null, "", href);
  showPage(currentRoute());
});

// Browser back / forward
window.addEventListener("popstate", () => showPage(currentRoute()));

// ── refresh card hints + footer meta ─────────────────────────────
async function refreshHome() {
  // Server version for the muted footer.
  try {
    const hello = await vc.hello();
    const v = $("#srvVersion");
    if (v) v.textContent = hello.version ? `v${hello.version}` : "?";
  } catch (e) { /* leave dash */ }

  try {
    const [cams, dets] = await Promise.all([
      vc.cameraList().catch(() => []),
      vc.detectionList().catch(() => []),
    ]);
    // Count only ADDED cameras (matching what shows in the Cameras
    // section grid). USB-attached but unmanaged devices don't count.
    const camCount = cams.filter(c => c.added).length;
    const detCount = dets.length;

    const c = $("#homeCardCameras");
    if (c) {
      c.textContent = camCount === 0
        ? "Manage camera connections"
        : `${camCount} ${camCount === 1 ? "camera" : "cameras"} connected`;
    }
    const p = $("#homeCardPlayground");
    if (p) {
      p.textContent = detCount === 0
        ? "Build and test detections"
        : `${detCount} ${detCount === 1 ? "detection" : "detections"} active`;
    }
  } catch (e) { /* leave defaults */ }
}

// ── boot ─────────────────────────────────────────────────────────
(async function boot() {
  setConn("warn", "connecting…");
  vc.on((ev) => {
    if (ev === "close") setConn("bad", "disconnected");
  });
  // Wire page modules with the shared client
  Cameras.init(vc);
  Playground.init(vc);
  // Show the initial page (idempotent before connect — onShow may need to retry)
  showPage(currentRoute());
  try {
    await vc.connect();
    setConn("ok", "connected");
    refreshHome();
    // Re-show the current page so its onShow runs against a live connection
    PAGE_HOOKS[_currentRoute]?.onShow?.();
  } catch (e) {
    setConn("bad", `failed: ${e.message || e}`);
    console.error(e);
  }
})();

// expose helpers for later phases
window.refreshHome = refreshHome;
