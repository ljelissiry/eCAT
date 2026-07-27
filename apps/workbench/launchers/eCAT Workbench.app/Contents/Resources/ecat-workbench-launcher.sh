#!/bin/zsh

set -u

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
  print -r -- "Install or update the app dependencies from the eCAT repository folder:"
  print -r -- ""
  print -r -- "python3 -m pip install -e \".[app]\""
  print -r -- ""
  print -r -- "If you use a specific Python, set ECAT_PYTHON to that interpreter."
}

log "eCAT Workbench bundled fallback launch started."
if [ -n "${ECAT_LAUNCHER_ENTRY:-}" ]; then
  log "$ECAT_LAUNCHER_ENTRY"
fi
log "Matplotlib config: $MPLCONFIGDIR"

if command -v ecat-app >/dev/null 2>&1; then
  log "Trying installed ecat-app command."
  ecat-app --port 0 >> "$LOG_FILE" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  log "Installed ecat-app failed with status $status."
else
  log "Installed ecat-app command was not found."
fi

install_failure_message >> "$LOG_FILE"
alert_failure "$(install_failure_message)"
exit 1
