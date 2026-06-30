#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/desktop"
TAURI_DIR="$DESKTOP_DIR/src-tauri"
RESOURCES_DIR="$TAURI_DIR/resources"
MODELS_SOURCE="${MUSICIDX_MODELS_SOURCE:-$ROOT_DIR/.musicidx-models}"
PYINSTALLER_WORK_DIR="$ROOT_DIR/build/pyinstaller"
PYINSTALLER_SPEC_DIR="$ROOT_DIR/build/pyinstaller-spec"
COPIED_DYLIB_MARKER_DIR="$ROOT_DIR/build/copied-dylibs"
PACKAGE_WITH_ML="${MUSICIDX_PACKAGE_WITH_ML:-0}"
PACKAGE_VENV="${MUSICIDX_PACKAGE_VENV:-$ROOT_DIR/build/package-venv}"
if [[ -n "${MUSICIDX_PACKAGE_PYTHON:-}" ]]; then
  PACKAGE_PYTHON="$MUSICIDX_PACKAGE_PYTHON"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PACKAGE_PYTHON="/opt/homebrew/bin/python3.11"
elif [[ -x /usr/local/bin/python3.11 ]]; then
  PACKAGE_PYTHON="/usr/local/bin/python3.11"
else
  PACKAGE_PYTHON="3.11"
fi
UV_EXTRA_ARGS=(--python "$PACKAGE_PYTHON" --extra semantic)
PYINSTALLER_ML_COLLECT_ARGS=()
if [[ "$PACKAGE_WITH_ML" == "1" || "$PACKAGE_WITH_ML" == "true" || "$PACKAGE_WITH_ML" == "yes" ]]; then
  UV_EXTRA_ARGS+=(--extra ml)
  PYINSTALLER_ML_COLLECT_ARGS+=(--collect-all essentia)
fi

is_system_dylib() {
  local dep="$1"
  [[ "$dep" == /usr/lib/* || "$dep" == /System/Library/* ]]
}

expand_loader_path() {
  local value="$1"
  local loader="$2"
  local loader_dir
  loader_dir="$(dirname "$loader")"
  value="${value//@loader_path/$loader_dir}"
  value="${value//@executable_path/$loader_dir}"
  printf '%s\n' "$value"
}

resolve_dylib() {
  local dep="$1"
  local loader="$2"
  if [[ "$dep" == /* && -f "$dep" ]]; then
    printf '%s\n' "$dep"
    return 0
  fi

  if [[ "$dep" == @loader_path/* || "$dep" == @executable_path/* ]]; then
    local expanded
    expanded="$(expand_loader_path "$dep" "$loader")"
    if [[ -f "$expanded" ]]; then
      printf '%s\n' "$expanded"
      return 0
    fi
  fi

  local name="${dep##*/}"
  local rpath
  while IFS= read -r rpath; do
    [[ -z "$rpath" ]] && continue
    local expanded
    expanded="$(expand_loader_path "$rpath" "$loader")/$name"
    if [[ -f "$expanded" ]]; then
      printf '%s\n' "$expanded"
      return 0
    fi
  done < <(otool -l "$loader" 2>/dev/null | awk '/cmd LC_RPATH/{getline; getline; print $2}')

  local candidate
  for candidate in \
    "/opt/homebrew/lib/$name" \
    "/usr/local/lib/$name" \
    /opt/homebrew/opt/*/lib/"$name" \
    /usr/local/opt/*/lib/"$name"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

rewrite_and_copy_dylibs() {
  local target="$1"
  local mode="$2"
  local dep

  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    is_system_dylib "$dep" && continue

    local source=""
    if ! source="$(resolve_dylib "$dep" "$target")"; then
      printf 'Warning: could not resolve dylib dependency for %s: %s\n' "$target" "$dep" >&2
      continue
    fi

    local base dest new_ref
    base="$(basename "$source")"
    dest="$RESOURCES_DIR/lib/$base"
    if [[ "$mode" == "bin" ]]; then
      new_ref="@loader_path/../lib/$base"
    else
      new_ref="@loader_path/$base"
    fi

    install_name_tool -change "$dep" "$new_ref" "$target" 2>/dev/null || true

    local marker
    marker="$COPIED_DYLIB_MARKER_DIR/$base"
    if [[ ! -e "$marker" ]]; then
      mkdir -p "$COPIED_DYLIB_MARKER_DIR"
      : > "$marker"
      cp "$source" "$dest"
      chmod u+w "$dest"
      install_name_tool -id "@loader_path/$base" "$dest" 2>/dev/null || true
      rewrite_and_copy_dylibs "$dest" "lib"
    fi
  done < <(otool -L "$target" 2>/dev/null | tail -n +2 | awk '{print $1}')
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This all-in-one installer script currently targets macOS only." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to build the Python sidecar. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to build the Tauri frontend." >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust/Cargo is required to build the Tauri app." >&2
  exit 1
fi

if ! command -v install_name_tool >/dev/null 2>&1; then
  echo "install_name_tool is required. Install Xcode Command Line Tools." >&2
  exit 1
fi

if [[ ! -d "$MODELS_SOURCE" ]]; then
  echo "Model directory not found: $MODELS_SOURCE" >&2
  echo "Set MUSICIDX_MODELS_SOURCE=/path/to/.musicidx-models or download models first." >&2
  exit 1
fi

FFMPEG_SOURCE="${MUSICIDX_FFMPEG_SOURCE:-$(command -v ffmpeg || true)}"
FFPROBE_SOURCE="${MUSICIDX_FFPROBE_SOURCE:-$(command -v ffprobe || true)}"
FPCALC_SOURCE="${MUSICIDX_FPCALC_SOURCE:-$(command -v fpcalc || true)}"
if [[ -z "$FFMPEG_SOURCE" || ! -x "$FFMPEG_SOURCE" ]]; then
  echo "ffmpeg not found. Install/provide it or set MUSICIDX_FFMPEG_SOURCE=/path/to/ffmpeg." >&2
  exit 1
fi
if [[ -z "$FFPROBE_SOURCE" || ! -x "$FFPROBE_SOURCE" ]]; then
  echo "ffprobe not found. Install/provide it or set MUSICIDX_FFPROBE_SOURCE=/path/to/ffprobe." >&2
  exit 1
fi
if [[ -z "$FPCALC_SOURCE" || ! -x "$FPCALC_SOURCE" ]]; then
  echo "fpcalc not found. Install/provide it or set MUSICIDX_FPCALC_SOURCE=/path/to/fpcalc." >&2
  exit 1
fi

mkdir -p "$RESOURCES_DIR/musicidx-bin" "$RESOURCES_DIR/models" "$RESOURCES_DIR/bin" "$RESOURCES_DIR/lib"

printf '\n==> Cleaning previous packaged resources\n'
find "$RESOURCES_DIR/musicidx-bin" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
find "$RESOURCES_DIR/models" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
find "$RESOURCES_DIR/bin" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
find "$RESOURCES_DIR/lib" -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
rm -rf "$PYINSTALLER_WORK_DIR" "$PYINSTALLER_SPEC_DIR" "$COPIED_DYLIB_MARKER_DIR"

printf '\n==> Building Python CLI sidecar with PyInstaller\n'
if [[ ${#PYINSTALLER_ML_COLLECT_ARGS[@]} -gt 0 ]]; then
  printf '    Python extras: semantic + ml\n'
else
  printf '    Python extras: semantic\n'
fi
printf '    Package Python: %s\n' "$PACKAGE_PYTHON"
printf '    Package venv:   %s\n' "$PACKAGE_VENV"
UV_PROJECT_ENVIRONMENT="$PACKAGE_VENV" uv run \
  "${UV_EXTRA_ARGS[@]}" \
  --with pyinstaller \
  pyinstaller \
  --noconfirm \
  --clean \
  --onedir \
  --name musicidx \
  --distpath "$RESOURCES_DIR/musicidx-bin" \
  --workpath "$PYINSTALLER_WORK_DIR" \
  --specpath "$PYINSTALLER_SPEC_DIR" \
  --paths "$ROOT_DIR/src" \
  --collect-all musicidx \
  --collect-all rich \
  --collect-all typer \
  --collect-all numpy \
  --collect-all scipy \
  --collect-all soundfile \
  --collect-all _soundfile_data \
  --collect-all librosa \
  --collect-all sklearn \
  --collect-all torch \
  --collect-all transformers \
  --collect-all sentence_transformers \
  "${PYINSTALLER_ML_COLLECT_ARGS[@]}" \
  "$ROOT_DIR/packaging/musicidx_cli_entry.py"

chmod +x "$RESOURCES_DIR/musicidx-bin/musicidx/musicidx"

printf '\n==> Copying local models into app resources\n'
rsync -a --delete --exclude '.DS_Store' "$MODELS_SOURCE/" "$RESOURCES_DIR/models/"
touch "$RESOURCES_DIR/models/.gitkeep"

printf '\n==> Copying audio helper binaries into app resources\n'
cp "$FFMPEG_SOURCE" "$RESOURCES_DIR/bin/ffmpeg"
cp "$FFPROBE_SOURCE" "$RESOURCES_DIR/bin/ffprobe"
cp "$FPCALC_SOURCE" "$RESOURCES_DIR/bin/fpcalc"
chmod +x "$RESOURCES_DIR/bin/ffmpeg" "$RESOURCES_DIR/bin/ffprobe" "$RESOURCES_DIR/bin/fpcalc"

printf '\n==> Bundling ffmpeg/ffprobe/fpcalc dylib dependencies\n'
rewrite_and_copy_dylibs "$RESOURCES_DIR/bin/ffmpeg" "bin"
rewrite_and_copy_dylibs "$RESOURCES_DIR/bin/ffprobe" "bin"
rewrite_and_copy_dylibs "$RESOURCES_DIR/bin/fpcalc" "bin"
touch "$RESOURCES_DIR/lib/.gitkeep"

printf '\n==> Installing frontend dependencies\n'
npm --prefix "$DESKTOP_DIR" install

printf '\n==> Building Tauri macOS bundle\n'
npm --prefix "$DESKTOP_DIR" run tauri:build -- --bundles app,dmg

printf '\n==> Done. Bundles are under:\n'
printf '    %s\n' "$TAURI_DIR/target/release/bundle"

printf '\nCopy the generated .dmg to your friend’s Mac. If macOS blocks it, right-click Open.\n'
printf 'This build is architecture-specific. Build on Apple Silicon for Apple Silicon Macs, or Intel for Intel Macs.\n'
