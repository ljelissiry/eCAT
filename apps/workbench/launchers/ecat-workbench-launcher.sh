#!/bin/zsh

set -u

# Install command for source checkouts: pip install -e .

LAUNCHER_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${TMPDIR:-/tmp}/ecat-workbench-launch.log"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ecat-matplotlib}"

mkdir -p "$MPLCONFIGDIR" "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

log() {
  print -r -- "$*" >> "$LOG_FILE"
}

alert_failure() {
  local message="$1"
  /usr/bin/osascript - "$message" "$LOG_FILE" <<'OSA' >/dev/null 2>&1 || true
on run argv
  set messageText to item 1 of argv
  set logPath to item 2 of argv
  display alert "eCAT Workbench could not start" message messageText & return & return & "Log: " & logPath
end run
OSA
}

install_failure_message() {
  if [ -n "${REPO_ROOT:-}" ]; then
    print -r -- "Install or update the app dependencies from the eCAT repository folder:"
    print -r -- ""
    print -r -- "cd \"$REPO_ROOT\""
    print -r -- "python3 -m pip install -e ."
  else
    print -r -- "Install or update the app dependencies from the eCAT repository folder:"
    print -r -- ""
    print -r -- "python3 -m pip install -e ."
  fi
  print -r -- ""
  print -r -- "If you use a specific Python, set ECAT_PYTHON to that interpreter."
}

find_repo_root() {
  local search_dir="$LAUNCHER_DIR"
  for _ in 1 2 3 4 5 6 7 8; do
    if [ -f "$search_dir/apps/workbench/app.py" ]; then
      print -r -- "$search_dir"
      return 0
    fi
    search_dir="$(dirname "$search_dir")"
  done
  return 1
}

python_has_app_deps() {
  local python_bin="$1"
  [ -x "$python_bin" ] || return 1
  "$python_bin" - <<'PY' >> "$LOG_FILE" 2>&1
import importlib.util
missing = [
    module
    for module in ("numpy", "dash", "dash_ag_grid", "webview")
    if importlib.util.find_spec(module) is None
]
if missing:
    print("Missing modules:", ", ".join(missing))
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_installed_command() {
  if command -v ecat-app >/dev/null 2>&1; then
    log "Trying installed ecat-app command."
    ecat-app --port 0 >> "$LOG_FILE" 2>&1
    return $?
  fi
  return 127
}

run_repo_python() {
  local repo_root="$1"
  local candidates=()

  if [ -n "${ECAT_PYTHON:-}" ]; then
    candidates+=("$ECAT_PYTHON")
  fi
  candidates+=(
    "$repo_root/.venv/bin/python"
    "$repo_root/venv/bin/python"
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
  )

  for python_bin in "${candidates[@]}"; do
    log "Checking Python candidate: $python_bin"
    if python_has_app_deps "$python_bin"; then
      log "Launching with $python_bin"
      "$python_bin" "$repo_root/apps/workbench/app.py" --port 0 >> "$LOG_FILE" 2>&1
      return $?
    fi
  done
  return 127
}

log "eCAT Workbench launch started."
if [ -n "${ECAT_LAUNCHER_ENTRY:-}" ]; then
  log "$ECAT_LAUNCHER_ENTRY"
fi
log "Launcher directory: $LAUNCHER_DIR"
log "Matplotlib config: $MPLCONFIGDIR"

run_installed_command
installed_status=$?
if [ "$installed_status" -eq 0 ]; then
  exit 0
fi
if [ "$installed_status" -ne 127 ]; then
  log "Installed ecat-app failed with status $installed_status."
fi

REPO_ROOT="$(find_repo_root || true)"
if [ -n "$REPO_ROOT" ]; then
  log "Repository root: $REPO_ROOT"
  run_repo_python "$REPO_ROOT"
  repo_status=$?
  if [ "$repo_status" -eq 0 ]; then
    exit 0
  fi
  log "Repository launcher failed with status $repo_status."
else
  log "Could not find repository root."
fi

install_failure_message >> "$LOG_FILE"
alert_failure "$(install_failure_message)"
exit 1
