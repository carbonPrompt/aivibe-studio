#!/usr/bin/env bash
# Build dist/aiVibeStudio-Setup-<version>.exe from the committed tree (git archive HEAD) on a Windows build
# machine with Inno Setup 6, then leak-check its inputs and output. The build machine is the ssh alias in
# AIVIBE_BUILD_HOST; LEAKGATE (optional) points at a leak gate script to run over the output.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AIVIBE_BUILD_HOST:?set AIVIBE_BUILD_HOST to the ssh alias of the Windows build machine}"
ver=$(sed -nE 's/^VERSION = "(.+)"/\1/p' studio/__init__.py)
read -r uv_ver uv_sha < installer/uv.version
mkdir -p installer/bin dist
if ! echo "$uv_sha  installer/bin/uv.zip" | sha256sum -c --status 2>/dev/null; then
  curl -fsSL -o installer/bin/uv.zip "https://github.com/astral-sh/uv/releases/download/$uv_ver/uv-x86_64-pc-windows-msvc.zip"
  echo "$uv_sha  installer/bin/uv.zip" | sha256sum -c --quiet
fi
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
git archive HEAD | tar -x -C "$stage"
# leak check of everything we wrote, before the sha256-pinned upstream uv.exe joins it
if [ -n "${LEAKGATE:-}" ]; then
  python3 "$LEAKGATE" "$stage"
fi
mkdir -p "$stage/installer/bin"
unzip -o -q installer/bin/uv.zip uv.exe -d "$stage/installer/bin"

ssh -o LogLevel=ERROR "$AIVIBE_BUILD_HOST" "powershell -NoProfile -Command \"Remove-Item -Recurse -Force C:\\aivibe-build\\src -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force C:\\aivibe-build\\src | Out-Null\""
tar -C "$stage" -cf - . | ssh -o LogLevel=ERROR "$AIVIBE_BUILD_HOST" "C:\\Windows\\System32\\tar.exe -xf - -C C:/aivibe-build/src"
ssh -o LogLevel=ERROR "$AIVIBE_BUILD_HOST" "& \"\$env:LOCALAPPDATA\\Programs\\Inno Setup 6\\ISCC.exe\" /Q /DAppVer=$ver C:\\aivibe-build\\src\\installer\\aivibe.iss; exit \$LASTEXITCODE"
out="dist/aiVibeStudio-Setup-$ver.exe"
scp -q -o LogLevel=ERROR "$AIVIBE_BUILD_HOST:C:/aivibe-build/src/dist/aiVibeStudio-Setup-$ver.exe" "$out"

# and the readable text of what came out
if [ -n "${LEAKGATE:-}" ]; then
  python3 "$LEAKGATE" "$out"
fi
echo "built $out ($(du -h "$out" | cut -f1)), sha256 $(sha256sum "$out" | cut -c1-64)"
