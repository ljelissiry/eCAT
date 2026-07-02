(function () {
  function plotFrameFor(target) {
    return target ? target.closest(".ecat-plot-frame") : null;
  }

  function mediaFor(target) {
    var frame = plotFrameFor(target);
    if (!frame) {
      return null;
    }
    return frame.querySelector("img.ecat-plot") || frame.querySelector("iframe.ecat-animation-frame");
  }

  function saveSourceFor(button) {
    var frame = plotFrameFor(button);
    if (!frame) {
      return null;
    }
    return frame.dataset.ecatSaveSrc || null;
  }

  function extensionFor(src) {
    if (src.indexOf("data:text/html") === 0) {
      return "html";
    }
    if (src.indexOf("image/svg+xml") === 5 || src.indexOf("image/svg+xml") !== -1) {
      return "svg";
    }
    if (src.indexOf("image/jpeg") !== -1) {
      return "jpg";
    }
    return "png";
  }

  function statusFor(button) {
    var frame = plotFrameFor(button);
    if (!frame) {
      return null;
    }
    var status = frame.querySelector(".ecat-plot-action-status");
    if (!status) {
      status = document.createElement("span");
      status.className = "ecat-plot-action-status";
      status.setAttribute("aria-live", "polite");
      var actions = frame.querySelector(".ecat-plot-actions");
      if (actions) {
        actions.appendChild(status);
      }
    }
    return status;
  }

  function setButtonState(button, state, message) {
    if (!button) {
      return;
    }
    button.classList.remove("is-working", "is-success", "is-error");
    if (state) {
      button.classList.add("is-" + state);
      button.dataset.ecatPlotState = state;
    } else {
      delete button.dataset.ecatPlotState;
    }
    var status = statusFor(button);
    if (status) {
      status.textContent = message || "";
    }
    if (button.dataset.ecatPlotStateTimer) {
      window.clearTimeout(Number(button.dataset.ecatPlotStateTimer));
      delete button.dataset.ecatPlotStateTimer;
    }
    if (state === "success" || state === "error") {
      button.dataset.ecatPlotStateTimer = String(
        window.setTimeout(function () {
          button.classList.remove("is-working", "is-success", "is-error");
          delete button.dataset.ecatPlotState;
          if (status) {
            status.textContent = "";
          }
        }, 2800)
      );
    }
  }

  function dataUriToBlob(src) {
    var parts = src.split(",");
    if (parts.length < 2 || src.indexOf("data:") !== 0) {
      return null;
    }
    var mimeMatch = parts[0].match(/^data:([^;]+)(;base64)?/);
    var mimeType = mimeMatch ? mimeMatch[1] : "application/octet-stream";
    if (mimeMatch && mimeMatch[2]) {
      var binary = window.atob(parts.slice(1).join(","));
      var bytes = new Uint8Array(binary.length);
      for (var index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return new Blob([bytes], { type: mimeType });
    }
    return new Blob([decodeURIComponent(parts.slice(1).join(","))], { type: mimeType });
  }

  function browserDownloadImage(src) {
    var link = document.createElement("a");
    var objectUrl = null;
    try {
      var blob = dataUriToBlob(src);
      if (blob && window.URL && typeof window.URL.createObjectURL === "function") {
        objectUrl = window.URL.createObjectURL(blob);
      }
    } catch (_error) {
      objectUrl = null;
    }
    link.href = objectUrl || src;
    link.download = "ecat-plot." + extensionFor(src);
    document.body.appendChild(link);
    link.click();
    link.remove();
    if (objectUrl) {
      window.setTimeout(function () {
        window.URL.revokeObjectURL(objectUrl);
      }, 1000);
    }
  }

  async function saveImage(src) {
    var filename = "ecat-plot." + extensionFor(src);
    if (typeof fetch === "function") {
      try {
        var response = await fetch("/ecat-app/save-plot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ src: src, filename: filename })
        });
        if (response.ok) {
          var data = await response.json();
          return {
            message: data.message || "Saved to Downloads",
            filename: data.filename || filename,
            path: data.path || ""
          };
        }
      } catch (_error) {
        // Fall through to the browser download path.
      }
    }
    browserDownloadImage(src);
    return { message: "Download requested", filename: filename, path: "" };
  }

  function saveMessage(result) {
    if (!result) {
      return "Download requested";
    }
    if (result.filename && result.message === "Saved to Downloads") {
      return result.message + ": " + result.filename;
    }
    return result.message || "Download requested";
  }

  async function copyImage(src) {
    if (src.indexOf("data:text/html") === 0) {
      return saveImage(src);
    }
    if (!navigator.clipboard || !window.ClipboardItem || typeof fetch !== "function") {
      return saveImage(src);
    }
    var response = await fetch(src);
    var blob = await response.blob();
    await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-ecat-plot-action]");
    if (!button) {
      return;
    }
    if (button.dataset.ecatPlotAction === "refresh") {
      var refresh = document.getElementById("ecat-replot");
      if (refresh) {
        event.preventDefault();
        event.stopPropagation();
        setButtonState(button, "working", "Refreshing plot");
        refresh.click();
        window.setTimeout(function () {
          setButtonState(button, "success", "Replot requested");
        }, 250);
      }
      return;
    }
    var media = mediaFor(button);
    if (!media || !media.src) {
      setButtonState(button, "error", "No plot found");
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (button.dataset.ecatPlotAction === "save") {
      setButtonState(button, "working", "Preparing download");
      saveImage(saveSourceFor(button) || media.src).then(function (result) {
        setButtonState(button, "success", saveMessage(result));
      }).catch(function () {
        try {
          window.open(media.src, "_blank", "noopener,noreferrer");
          setButtonState(button, "success", "Opened plot in new tab");
        } catch (_fallbackError) {
          setButtonState(button, "error", "Save failed");
        }
      });
    } else if (button.dataset.ecatPlotAction === "copy") {
      setButtonState(button, "working", "Copying plot");
      copyImage(media.src).then(function (result) {
        setButtonState(button, "success", result ? saveMessage(result) : "Copied");
      }).catch(function () {
        saveImage(saveSourceFor(button) || media.src).then(function (result) {
          setButtonState(button, "success", saveMessage(result));
        }).catch(function () {
          setButtonState(button, "error", "Copy failed");
        });
      });
    }
  });
})();
