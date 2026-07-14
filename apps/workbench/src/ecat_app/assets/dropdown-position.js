(function () {
  var frame = null;
  var lastInteractedDropdownRoot = null;

  function visibleRect(element) {
    if (!element || !element.getBoundingClientRect) {
      return null;
    }
    var rect = element.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      return null;
    }
    return rect;
  }

  function dropdownRoots() {
    return Array.prototype.slice.call(document.querySelectorAll(".dash-dropdown, .Select"));
  }

  function dropdownIsOpen(root) {
    if (!root) {
      return false;
    }
    if (root.classList && root.classList.contains("is-open")) {
      return true;
    }
    return Boolean(
      root.querySelector(
        '[aria-expanded="true"], [data-state="open"], .Select-menu-outer, .dash-dropdown-menu'
      )
    );
  }

  function activeDropdownRoot() {
    var active = document.activeElement;
    if (!active || !active.closest) {
      return null;
    }
    return active.closest(".dash-dropdown, .Select");
  }

  function rootFromEventTarget(target) {
    if (!target || !target.closest) {
      return null;
    }
    return target.closest(".dash-dropdown, .Select");
  }

  function rememberDropdownRoot(root) {
    if (root && document.body.contains(root)) {
      lastInteractedDropdownRoot = root;
    }
  }

  function handleDropdownInteraction(event) {
    rememberDropdownRoot(rootFromEventTarget(event.target));
    schedulePositioning();
  }

  function popperWrappers() {
    return Array.prototype.slice
      .call(document.querySelectorAll("[data-radix-popper-content-wrapper]"))
      .filter(function (wrapper) {
        return Boolean(
          wrapper.querySelector(
            '.dash-dropdown-menu, .dash-dropdown-content, .dash-dropdown-options, [role="listbox"]'
          )
        );
      });
  }

  function currentWorkspaceZoom() {
    var cssZoom = Number(
      getComputedStyle(document.documentElement).getPropertyValue("--ecat-ui-zoom")
    );
    if (Number.isFinite(cssZoom) && cssZoom > 0) {
      return cssZoom;
    }
    var workspace = document.querySelector(".ecat-workspace");
    var workspaceZoom = Number(workspace && getComputedStyle(workspace).zoom);
    if (Number.isFinite(workspaceZoom) && workspaceZoom > 0) {
      return workspaceZoom;
    }
    return 1;
  }

  function coordinateScaleForWrapper(wrapper) {
    if (wrapper && wrapper.closest(".ecat-workspace")) {
      return currentWorkspaceZoom();
    }
    return 1;
  }

  function scoreAnchor(anchorRect, popperRect, preferOpen) {
    var anchorCenter = (anchorRect.left + anchorRect.right) / 2;
    var popperCenter = (popperRect.left + popperRect.right) / 2;
    var vertical = Math.min(
      Math.abs(anchorRect.bottom - popperRect.top),
      Math.abs(anchorRect.top - popperRect.bottom)
    );
    var horizontal = Math.abs(anchorCenter - popperCenter);
    var overlap = Math.max(
      0,
      Math.min(anchorRect.right, popperRect.right) - Math.max(anchorRect.left, popperRect.left)
    );
    return vertical * 4 + horizontal - overlap - (preferOpen ? 10000 : 0);
  }

  function bestAnchorForWrapper(wrapper) {
    var rememberedRect = visibleRect(lastInteractedDropdownRoot);
    if (rememberedRect) {
      return rememberedRect;
    }

    var activeRoot = activeDropdownRoot();
    var activeRect = visibleRect(activeRoot);
    if (activeRect) {
      return activeRect;
    }

    var popperRect = visibleRect(wrapper);
    if (!popperRect) {
      return null;
    }

    var best = null;
    var bestScore = Infinity;
    dropdownRoots().forEach(function (root) {
      var rect = visibleRect(root);
      if (!rect) {
        return;
      }
      var score = scoreAnchor(rect, popperRect, dropdownIsOpen(root));
      if (score < bestScore) {
        best = rect;
        bestScore = score;
      }
    });
    return best;
  }

  function placeWrapper(wrapper, anchorRect) {
    var viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
    var viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    var wrapperRect = visibleRect(wrapper);
    var scale = coordinateScaleForWrapper(wrapper);
    var maxHeight = Math.max(160, viewportHeight * 0.6);
    var menuHeight = Math.min(wrapperRect ? wrapperRect.height : maxHeight, maxHeight);
    var top = anchorRect.bottom;
    if (viewportHeight && top + menuHeight > viewportHeight && anchorRect.top > menuHeight) {
      top = anchorRect.top - menuHeight;
    }
    var left = Math.max(0, Math.min(anchorRect.left, Math.max(0, viewportWidth - anchorRect.width)));

    wrapper.dataset.ecatPositioned = "1";
    setImportantStyle(wrapper, "position", "fixed");
    setImportantStyle(wrapper, "transform", "none");
    setImportantStyle(wrapper, "left", left / scale + "px");
    setImportantStyle(wrapper, "top", top / scale + "px");
    setImportantStyle(wrapper, "width", anchorRect.width / scale + "px");
    setImportantStyle(wrapper, "min-width", anchorRect.width / scale + "px");
    setImportantStyle(wrapper, "z-index", "10000");
  }

  function setImportantStyle(element, property, value) {
    if (
      element.style.getPropertyValue(property) !== value ||
      element.style.getPropertyPriority(property) !== "important"
    ) {
      element.style.setProperty(property, value, "important");
    }
  }

  function positionDropdowns() {
    popperWrappers().forEach(function (wrapper) {
      var anchorRect = bestAnchorForWrapper(wrapper);
      if (anchorRect) {
        placeWrapper(wrapper, anchorRect);
      }
    });
  }

  function schedulePositioning() {
    if (frame !== null) {
      return;
    }
    frame = window.requestAnimationFrame(function () {
      frame = null;
      positionDropdowns();
      window.requestAnimationFrame(positionDropdowns);
    });
  }

  function installDropdownPositioning() {
    document.addEventListener("ecat:layout-resized", schedulePositioning);
    window.addEventListener("resize", schedulePositioning);
    window.addEventListener("scroll", schedulePositioning, true);
    document.addEventListener("pointerdown", handleDropdownInteraction, true);
    document.addEventListener("mousedown", handleDropdownInteraction, true);
    document.addEventListener("focusin", handleDropdownInteraction, true);
    document.addEventListener("click", schedulePositioning, true);
    document.addEventListener("keydown", schedulePositioning, true);
    new MutationObserver(schedulePositioning).observe(document.body, {
      attributes: true,
      attributeFilter: ["class", "style", "data-state", "aria-expanded"],
      childList: true,
      subtree: true,
    });
    schedulePositioning();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installDropdownPositioning);
  } else {
    installDropdownPositioning();
  }
})();
