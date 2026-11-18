#!/usr/bin/env bash
# Builds RumbleBar.app from app/RumbleBar.swift and signs it.
#
# Signing: uses RUMBLEBAR_CODE_SIGN_IDENTITY if set, otherwise the first
# "Apple Development" (then "Developer ID Application") identity found. A stable
# identity keeps Microphone/Accessibility approvals across rebuilds. Set
# RUMBLEBAR_ALLOW_ADHOC_SIGNING=1 to fall back to ad-hoc signing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT/RumbleBar.app"
PYTHON="${RUMBLEBAR_PYTHON:-$ROOT/.venv/bin/python}"
SERVER="$ROOT/server/rumble_server.py"
PORT="${RUMBLEBAR_PORT:-8765}"

[[ -x "$PYTHON" ]] || { echo "Missing $PYTHON — run scripts/setup.sh first" >&2; exit 1; }

identity="${RUMBLEBAR_CODE_SIGN_IDENTITY:-}"
if [[ -z "$identity" ]]; then
  ids="$(security find-identity -v -p codesigning 2>/dev/null || true)"
  identity="$(sed -nE 's/.*"(Apple Development: [^"]+)".*/\1/p' <<<"$ids" | head -1)"
  [[ -n "$identity" ]] || identity="$(sed -nE 's/.*"(Developer ID Application: [^"]+)".*/\1/p' <<<"$ids" | head -1)"
fi
if [[ -z "$identity" ]]; then
  if [[ "${RUMBLEBAR_ALLOW_ADHOC_SIGNING:-0}" == "1" ]]; then
    identity="-"
    echo "warning: ad-hoc signing; macOS may re-ask for permissions after each rebuild" >&2
  else
    echo "No code-signing identity found. Set RUMBLEBAR_CODE_SIGN_IDENTITY or RUMBLEBAR_ALLOW_ADHOC_SIGNING=1" >&2
    exit 1
  fi
fi

BUILD="$ROOT/.build"
mkdir -p "$BUILD"
swiftc -O -swift-version 5 -target arm64-apple-macos13.0 \
  -framework AppKit -framework AVFoundation -framework Carbon -framework ServiceManagement \
  "$ROOT/app/RumbleBar.swift" -o "$BUILD/RumbleBar"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BUILD/RumbleBar" "$APP/Contents/MacOS/RumbleBar"
cp "$ROOT/app/Info.plist" "$APP/Contents/Info.plist"
plutil -replace RumbleBarPython -string "$PYTHON" "$APP/Contents/Info.plist"
plutil -replace RumbleBarServerScript -string "$SERVER" "$APP/Contents/Info.plist"
plutil -replace RumbleBarPort -integer "$PORT" "$APP/Contents/Info.plist"

codesign --force --options runtime --entitlements "$ROOT/app/RumbleBar.entitlements" --sign "$identity" "$APP"
codesign --verify --strict "$APP"
echo "Built $APP (signed: $identity)"
