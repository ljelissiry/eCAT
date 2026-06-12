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
      document.body.classList.add("ecat-resizing-sidebar");
      event.preventDefault();
    });

    window.addEventListener("mousemove", function (event) {
      if (!dragging) {
        return;
      }
      const width = clamp(event.clientX, 260, 520);
      sidebar.style.width = width + "px";
      sidebar.dataset.ecatLastWidth = sidebar.style.width;
    });

    window.addEventListener("mouseup", function () {
      if (!dragging) {
        return;
      }
      dragging = false;
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
