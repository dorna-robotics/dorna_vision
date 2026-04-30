// Theme + Fullscreen — shared across all pages
// Include via: <script src="/vendor/theme.js"></script>

const KEY = "orch_theme";

const SUN = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/><line x1="4.22" y1="4.22" x2="6.34" y2="6.34"/><line x1="17.66" y1="17.66" x2="19.78" y2="19.78"/><line x1="4.22" y1="19.78" x2="6.34" y2="17.66"/><line x1="17.66" y1="6.34" x2="19.78" y2="4.22"/></svg>';
const MOON = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

function setTheme(theme) {
  localStorage.setItem(KEY, theme);
  document.documentElement.setAttribute("data-theme", theme);

  // Notify embedded 3D viewer iframe (orchestrator workspace page)
  var viewer = document.getElementById("ws3dFrame");
  if (viewer && viewer.contentWindow) {
    viewer.contentWindow.postMessage({ type: "theme", value: theme }, "*");
  }

  // Notify other listeners
  window.dispatchEvent(new StorageEvent("storage", { key: KEY, newValue: theme }));

  var btn = document.getElementById("btnTheme");
  if (btn) {
    btn.title     = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    btn.innerHTML = theme === "dark" ? SUN : MOON;
  }
}

// Apply on load
setTheme(localStorage.getItem(KEY) || "dark");

// ── Fullscreen ──────────────────────────────────────────────────────────

var FS_EXPAND = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';
var FS_SHRINK = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';

function updateFsButton() {
  var btn = document.getElementById("btnFullscreen");
  if (!btn) return;
  var isFs = !!document.fullscreenElement;
  btn.title = isFs ? "Exit fullscreen" : "Fullscreen";
  btn.innerHTML = isFs ? FS_SHRINK : FS_EXPAND;
}

document.addEventListener("fullscreenchange", updateFsButton);

// ── Wire up buttons ─────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
  var btnTheme = document.getElementById("btnTheme");
  if (btnTheme) {
    btnTheme.addEventListener("click", function() {
      var current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  var btnFs = document.getElementById("btnFullscreen");
  if (btnFs) {
    btnFs.addEventListener("click", function() {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    });
    updateFsButton();
  }
});
