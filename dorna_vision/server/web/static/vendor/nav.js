// App Navigation — collapse/expand sidebar, mobile burger
// Include via: <script src="/vendor/nav.js"></script>

(function() {
  var nav = document.querySelector(".app-nav");
  var overlay = document.querySelector(".app-nav-overlay");
  var collapse = document.querySelector(".app-nav-collapse");
  var burger = document.getElementById("btnBurger");
  var KEY = "nav_expanded";

  if (!nav) return;

  // Restore saved state (desktop only) — collapsed by default, expanded if saved
  if (window.innerWidth > 768 && localStorage.getItem(KEY) === "1") {
    nav.classList.add("expanded");
  }
  // Remove instant-load class so transitions work for user interactions
  requestAnimationFrame(function() {
    document.documentElement.classList.remove("nav-expanded");
  });

  // Desktop: collapse/expand toggle
  if (collapse) {
    collapse.addEventListener("click", function() {
      nav.classList.toggle("expanded");
      localStorage.setItem(KEY, nav.classList.contains("expanded") ? "1" : "0");
    });
  }

  // Mobile: burger opens sidebar as overlay
  function mobileOpen()  { nav.classList.add("mobile-open"); overlay && overlay.classList.add("show"); }
  function mobileClose() { nav.classList.remove("mobile-open"); overlay && overlay.classList.remove("show"); }

  if (burger) {
    burger.addEventListener("click", function() {
      nav.classList.contains("mobile-open") ? mobileClose() : mobileOpen();
    });
  }
  if (overlay) {
    overlay.addEventListener("click", mobileClose);
  }

  document.addEventListener("keydown", function(e) {
    if (e.key === "Escape") mobileClose();
  });
})();
