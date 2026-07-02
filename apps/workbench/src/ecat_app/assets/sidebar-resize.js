(function () {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function initSidebarResize() {
    const shell = document.querySelector("#ecat-app-shell");
    const sidebar = document.querySelector(".ecat-sidebar");
    const handle = document.querySelector("#ecat-sidebar-resizer");
    if (!sidebar || !handle || handle.dataset.ecatReady === "1") {
      return;
    }
    handle.dataset.ecatReady = "1";

    let dragging = false;
    let startX = 0;
    let startWidth = 0;

    function currentZoom() {
      const cssZoom = Number(
        getComputedStyle(document.documentElement).getPropertyValue("--ecat-ui-zoom")
      );
      if (Number.isFinite(cssZoom) && cssZoom > 0) {
        return cssZoom;
      }
      const workspace = document.querySelector(".ecat-workspace");
      const storedPercent = Number(workspace && workspace.dataset.ecatZoom);
      if (Number.isFinite(storedPercent) && storedPercent > 0) {
        return storedPercent / 100;
      }
      return 1;
    }

    function isCollapsed() {
      return shell && shell.classList.contains("ecat-sidebar-collapsed");
    }

    function syncCollapsedWidth() {
      if (isCollapsed()) {
        if (sidebar.style.width) {
          sidebar.dataset.ecatLastWidth = sidebar.style.width;
        }
        sidebar.style.width = "";
      } else if (sidebar.dataset.ecatLastWidth && !sidebar.style.width) {
        sidebar.style.width = sidebar.dataset.ecatLastWidth;
      }
    }

    handle.addEventListener("mousedown", function (event) {
      if (isCollapsed()) {
        return;
      }
      dragging = true;
      startX = event.clientX;
      startWidth = sidebar.getBoundingClientRect().width / currentZoom();
      document.body.classList.add("ecat-resizing-sidebar");
      event.preventDefault();
    });

    window.addEventListener("mousemove", function (event) {
      if (!dragging) {
        return;
      }
      const width = clamp(startWidth + (event.clientX - startX) / currentZoom(), 260, 520);
      sidebar.style.width = width + "px";
      sidebar.dataset.ecatLastWidth = sidebar.style.width;
    });

    window.addEventListener("mouseup", function () {
      if (!dragging) {
        return;
      }
      dragging = false;
      startX = 0;
      startWidth = 0;
      document.body.classList.remove("ecat-resizing-sidebar");
    });

    if (shell) {
      new MutationObserver(syncCollapsedWidth).observe(shell, { attributes: true, attributeFilter: ["class"] });
    }
    syncCollapsedWidth();
  }

  document.addEventListener("DOMContentLoaded", initSidebarResize);
  setInterval(initSidebarResize, 1000);
})();
