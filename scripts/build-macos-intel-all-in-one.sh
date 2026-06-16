#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  cat >&2 <<EOF
This script must be run on an Intel macOS builder (x86_64).

Current machine architecture: $ARCH

For a fully bundled Intel build, use one of:
  1. a real Intel Mac, or
  2. GitHub Actions macos-13 runner, or
  3. an x86_64 macOS VM/builder.

Do not build the Intel installer on Apple Silicon with arm64 Homebrew binaries;
PyInstaller, Python native wheels, ffprobe/fpcalc, Torch, and Essentia must all be x86_64.
EOF
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build-macos-all-in-one.sh"
