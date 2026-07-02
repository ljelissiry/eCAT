(function () {
  const minZoom = 50;
  const maxZoom = 200;
  const zoomStep = 10;
  const defaultZoom = 100;

  function zoomTarget() {
    return document.querySelector(".ecat-workspace");
  }

  function zoomValue() {
    return document.getElementById("ecat-zoom-value");
  }

  function clampZoom(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return defaultZoom;
    }
    return Math.min(maxZoom, Math.max(minZoom, Math.round(numeric / zoomStep) * zoomStep));
  }

  function syncButtons(percent) {
    document.querySelectorAll("[data-ecat-zoom-action]").forEach(function (button) {
      const action = button.dataset.ecatZoomAction;
      button.disabled = (action === "out" && percent <= minZoom) || (action === "in" && percent >= maxZoom);
    });
  }

  function applyZoom(percent, persist) {
    const resolved = clampZoom(percent);
    const scale = resolved / 100;
    const target = zoomTarget();
    document.documentElement.style.setProperty("--ecat-ui-zoom", String(scale));
    if (target) {
      target.dataset.ecatZoom = String(resolved);
    }
    const label = zoomValue();
    if (label) {
      label.textContent = `${resolved}%`;
    }
    syncButtons(resolved);
    if (persist) {
      try {
        localStorage.setItem("ecat-ui-zoom", String(resolved));
      } catch (_error) {
        // Ignore blocked storage.
      }
    }
    return resolved;
  }

  function editableTarget(target) {
    if (!target) {
      return false;
    }
    const tag = String(target.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || target.isContentEditable;
  }

  function currentZoom() {
    const label = zoomValue();
    if (!label) {
      return defaultZoom;
    }
    return clampZoom(String(label.textContent || "").replace("%", ""));
  }

  function initZoomControls() {
    const buttons = document.querySelectorAll("[data-ecat-zoom-action]");
    if (!buttons.length || document.body.dataset.ecatZoomReady === "1") {
      return;
    }
    document.body.dataset.ecatZoomReady = "1";
    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        const action = button.dataset.ecatZoomAction;
        const current = currentZoom();
        if (action === "out") {
          applyZoom(current - zoomStep, true);
        } else if (action === "in") {
          applyZoom(current + zoomStep, true);
        }
      });
    });
    document.addEventListener("keydown", function (event) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey || editableTarget(event.target)) {
        return;
      }
      const key = event.key;
      const current = currentZoom();
      if (key === "+" || key === "=") {
        event.preventDefault();
        applyZoom(current + zoomStep, true);
      } else if (key === "-" || key === "_") {
        event.preventDefault();
        applyZoom(current - zoomStep, true);
      } else if (key === "0") {
        event.preventDefault();
        applyZoom(defaultZoom, true);
      }
    });
    let saved = defaultZoom;
    try {
      saved = localStorage.getItem("ecat-ui-zoom") || defaultZoom;
    } catch (_error) {
      saved = defaultZoom;
    }
    applyZoom(saved, false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initZoomControls);
  } else {
    initZoomControls();
  }
  setInterval(initZoomControls, 1000);
})();
