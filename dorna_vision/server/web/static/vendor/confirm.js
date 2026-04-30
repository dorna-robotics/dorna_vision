// Shared confirm dialog — iOS-style alert that replaces window.confirm()
// Include via: <script src="/vendor/confirm.js"></script>
// Usage: if (!await confirmDialog({ title, message, confirm, variant, icon })) return;

(function() {
  var ICONS = {
    kill: '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>',
    remove: '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
    end: '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6" rx="0.5"/></svg>',
    warning: '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  };

  var _overlay = null;

  function _ensure() {
    if (_overlay) return _overlay;
    var el = document.createElement("div");
    el.className = "confirm-overlay";
    el.innerHTML =
      '<div class="confirm-dialog">' +
        '<div class="confirm-icon"></div>' +
        '<div class="confirm-title"></div>' +
        '<div class="confirm-message"></div>' +
        '<label class="confirm-remember"><input type="checkbox" class="confirm-remember-cb"><span>Don\'t ask me again</span></label>' +
        '<div class="confirm-actions"><button class="btn confirm-cancel">Cancel</button><button class="btn confirm-ok">Confirm</button></div>' +
      '</div>';
    document.body.appendChild(el);
    _overlay = el;
    el.addEventListener("click", function(e) {
      if (e.target === el) el.querySelector(".confirm-cancel").click();
    });
    return el;
  }

  window.confirmDialog = function(opts) {
    opts = opts || {};
    var title = opts.title || "Are you sure?";
    var message = opts.message || "";
    var confirmLabel = opts.confirm || "Confirm";
    var cancelLabel = opts.cancel || "Cancel";
    var variant = opts.variant || "danger";
    var icon = opts.icon || "warning";
    var remember = opts.remember || null;

    if (remember && localStorage.getItem("confirm_skip_" + remember) === "1") {
      return Promise.resolve(true);
    }

    return new Promise(function(resolve) {
      var el = _ensure();
      var dialog = el.querySelector(".confirm-dialog");
      var iconEl = el.querySelector(".confirm-icon");
      var titleEl = el.querySelector(".confirm-title");
      var msgEl = el.querySelector(".confirm-message");
      var rememberLabel = el.querySelector(".confirm-remember");
      var rememberCb = el.querySelector(".confirm-remember-cb");
      var cancelBtn = el.querySelector(".confirm-cancel");
      var okBtn = el.querySelector(".confirm-ok");

      iconEl.innerHTML = ICONS[icon] || ICONS.warning;
      iconEl.dataset.variant = variant;
      titleEl.textContent = title;
      msgEl.textContent = message;
      cancelBtn.textContent = cancelLabel;
      okBtn.textContent = confirmLabel;

      rememberLabel.style.display = remember ? "" : "none";
      rememberCb.checked = false;

      okBtn.className = "btn confirm-ok " + (variant === "danger" ? "confirm-ok-danger" : "btn-primary");

      dialog.classList.remove("confirm-out");
      el.classList.add("show");
      void dialog.offsetWidth;
      okBtn.focus();

      function cleanup(result) {
        if (result && remember && rememberCb.checked) {
          localStorage.setItem("confirm_skip_" + remember, "1");
        }
        dialog.classList.add("confirm-out");
        el.classList.add("hiding");
        setTimeout(function() { el.classList.remove("show", "hiding"); dialog.classList.remove("confirm-out"); }, 180);
        cancelBtn.removeEventListener("click", onCancel);
        okBtn.removeEventListener("click", onOk);
        document.removeEventListener("keydown", onKey);
        resolve(result);
      }

      function onCancel() { cleanup(false); }
      function onOk() { cleanup(true); }
      function onKey(e) {
        if (e.key === "Escape") cleanup(false);
        if (e.key === "Enter") cleanup(true);
      }

      cancelBtn.addEventListener("click", onCancel);
      okBtn.addEventListener("click", onOk);
      document.addEventListener("keydown", onKey);
    });
  };
})();
