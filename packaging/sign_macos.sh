#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${1:-dist/standalone/eCAT Workbench.app}"
ZIP_PATH="${2:-dist/standalone/eCAT Workbench-macOS.zip}"

: "${APPLE_SIGN_IDENTITY:?Set APPLE_SIGN_IDENTITY to your Developer ID Application identity.}"
: "${APPLE_ID:?Set APPLE_ID to the Apple ID used for notarization.}"
: "${APPLE_TEAM_ID:?Set APPLE_TEAM_ID to your Apple developer team ID.}"
: "${APPLE_APP_PASSWORD:?Set APPLE_APP_PASSWORD to an app-specific password or keychain profile password.}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "App not found: $APP_PATH" >&2
  exit 1
fi

xattr -cr "$APP_PATH" 2>/dev/null || true
find -H "$APP_PATH" -name ".DS_Store" -delete 2>/dev/null || true
find -H "$APP_PATH" -name "._*" -delete 2>/dev/null || true
find -H "$APP_PATH" -exec xattr -d com.apple.FinderInfo {} + 2>/dev/null || true

codesign \
  --force \
  --deep \
  --options runtime \
  --timestamp \
  --sign "$APPLE_SIGN_IDENTITY" \
  "$APP_PATH"

codesign --verify --deep --strict --verbose=2 "$APP_PATH"

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

xcrun notarytool submit "$ZIP_PATH" \
  --apple-id "$APPLE_ID" \
  --team-id "$APPLE_TEAM_ID" \
  --password "$APPLE_APP_PASSWORD" \
  --wait

xcrun stapler staple "$APP_PATH"
spctl --assess --type execute --verbose=4 "$APP_PATH"

echo "Signed and notarized: $APP_PATH"
