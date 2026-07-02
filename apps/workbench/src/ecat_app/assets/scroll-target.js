(function () {
  function cardForTarget(target) {
    if (!target) {
      return null;
    }
    var key = String(target).split(":")[0];
    if (key === "import") {
      return document.getElementById("ecat-object-table");
    }
    if (key === "plot") {
      return document.getElementById("ecat-plot-card") || document.getElementById("ecat-default-plot");
    }
    if (key === "analysis") {
      return document.getElementById("ecat-analysis-results") || document.getElementById("ecat-analysis-results-content");
    }
    if (key === "model-program") {
      return document.getElementById("ecat-model-result-program") || document.getElementById("ecat-model-results");
    }
    if (key === "model-simulation") {
      return document.getElementById("ecat-model-result-simulation") || document.getElementById("ecat-model-results");
    }
    if (key === "model-fit") {
      return document.getElementById("ecat-model-result-fit") || document.getElementById("ecat-model-results");
    }
    if (key === "model") {
      return document.getElementById("ecat-model-results");
    }
    return null;
  }

  function currentZoom() {
    var cssZoom = Number(
      getComputedStyle(document.documentElement).getPropertyValue("--ecat-ui-zoom")
    );
    if (Number.isFinite(cssZoom) && cssZoom > 0) {
      return cssZoom;
    }
    var workspace = document.querySelector(".ecat-workspace");
    var storedPercent = Number(workspace && workspace.dataset.ecatZoom);
    if (Number.isFinite(storedPercent) && storedPercent > 0) {
      return storedPercent / 100;
    }
    return 1;
  }

  function scrollMainToTarget(target) {
    var main = document.querySelector(".ecat-main");
    if (!main || !main.contains(target)) {
      return;
    }
    var mainRect = main.getBoundingClientRect();
    var targetRect = target.getBoundingClientRect();
    var delta = targetRect.top - mainRect.top;
    var top = Math.max(0, main.scrollTop + delta / currentZoom() - 12);
    main.scrollTo({ top: top, behavior: "smooth" });
  }

  function scrollToSignal(signal) {
    var target = cardForTarget(signal);
    if (!target) {
      return;
    }
    scrollMainToTarget(target);
    target.classList.add("ecat-command-highlight");
    window.setTimeout(function () {
      target.classList.remove("ecat-command-highlight");
    }, 1200);
  }

  function installObserver() {
    var signal = document.getElementById("ecat-scroll-target");
    if (!signal) {
      window.setTimeout(installObserver, 250);
      return;
    }
    var observer = new MutationObserver(function () {
      scrollToSignal(signal.textContent || "");
    });
    observer.observe(signal, { childList: true, characterData: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installObserver);
  } else {
    installObserver();
  }
})();
