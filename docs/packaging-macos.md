# macOS all-in-one packaging

This project can build a large self-contained macOS `.dmg` intended for copying to another Mac.

The all-in-one bundle includes:

- the Tauri desktop app
- a PyInstaller-built `musicidx` Python CLI backend
- installed Python dependencies with semantic search support by default
- optional ML extras only when explicitly requested and compatible wheels are available
- local `.musicidx-models` files
- `ffmpeg`, `ffprobe`, and `fpcalc`
- best-effort bundled Homebrew dylib dependencies for `ffmpeg`/`ffprobe`/`fpcalc`

The packaged app stores its runtime DB in the macOS app-data directory, not the repo:

```text
~/Library/Application Support/local.musicidx.desktop/musicidx.sqlite
```

## Build on the target CPU architecture

The generated app is architecture-specific because the bundle includes Python native wheels, PyInstaller output, `ffprobe`/`fpcalc`, Torch, Essentia, and dylibs.

- Build on Apple Silicon for Apple Silicon Macs.
- Build on Intel for Intel Macs.
- Universal app packaging is a later task.

For an Intel-based friend's laptop, use an Intel macOS builder. Do not build the Intel installer from an Apple Silicon machine with `/opt/homebrew` binaries; that produces an arm64 app and arm64 helper binaries.

## Prerequisites on the build Mac

```bash
xcode-select --install
brew install ffmpeg chromaprint
```

Also install:

- Rust/Cargo
- Node/npm
- `uv`
- a framework/shared-library Python 3.11 for PyInstaller, for example Homebrew `python@3.11`

Avoid pyenv Pythons built without `--enable-shared`/framework support for packaging; PyInstaller will fail with:

```text
Python was built without a shared library, which is required by PyInstaller.
```

The packaging script uses a dedicated `build/package-venv` and prefers `/opt/homebrew/bin/python3.11` or `/usr/local/bin/python3.11` when present. Override if needed:

```bash
MUSICIDX_PACKAGE_PYTHON=/path/to/python3.11 npm --prefix desktop run package:mac:all-in-one
```

Make sure models exist locally:

```text
.musicidx-models/
.musicidx-models/all-MiniLM-L6-v2
```

The packaging scripts build with semantic support only by default. This avoids fragile `essentia-tensorflow` wheel availability issues. Search, metadata repair, embeddings, health checks, and desktop playback do not require the ML extra.

To build the full bundle with local Essentia ML tagging included, use:

```bash
MUSICIDX_PACKAGE_WITH_ML=1 npm --prefix desktop run package:mac:all-in-one
```

This requires compatible `essentia-tensorflow` wheels for the build Python and architecture. The project pins the ML extra to `essentia-tensorflow==2.1b6.dev1389` because newer `2.1b6.dev1438` wheels are currently CPython 3.14-only on PyPI.

## Build

From the repository root, for the current Mac architecture:

```bash
npm --prefix desktop run package:mac:all-in-one
```

For an Intel Mac build, run this on an Intel Mac or x86_64 macOS builder:

```bash
npm --prefix desktop run package:mac:intel:all-in-one
```

Equivalent direct commands:

```bash
./scripts/build-macos-all-in-one.sh
./scripts/build-macos-intel-all-in-one.sh
```

Outputs are under:

```text
desktop/src-tauri/target/release/bundle/
```

The `.dmg` is the file to copy to your friend's Mac. Confirm it is Intel/x86_64 before copying:

```bash
file desktop/src-tauri/target/release/bundle/macos/MusicIdx.app/Contents/MacOS/MusicIdx
file desktop/src-tauri/resources/bin/ffprobe
file desktop/src-tauri/resources/bin/fpcalc
```

Each should report `x86_64` for an Intel build.

## Optional custom paths

If your models are somewhere else:

```bash
MUSICIDX_MODELS_SOURCE=/path/to/.musicidx-models \
  npm --prefix desktop run package:mac:all-in-one
```

If you want to provide custom/self-contained helper binaries:

```bash
MUSICIDX_FFMPEG_SOURCE=/path/to/ffmpeg \
MUSICIDX_FFPROBE_SOURCE=/path/to/ffprobe \
MUSICIDX_FPCALC_SOURCE=/path/to/fpcalc \
  npm --prefix desktop run package:mac:all-in-one
```

## Gatekeeper warning

This build is unsigned/not notarized by default. On your friend's Mac, macOS may block it.

For informal testing:

1. Open the `.dmg`.
2. Drag MusicIdx to Applications.
3. Right-click MusicIdx → Open.
4. Confirm the warning.

For public distribution, add Apple Developer signing and notarization later.

## Notes

The script bundles Homebrew `ffmpeg`/`ffprobe`/`fpcalc` dylib dependencies using `otool` and `install_name_tool`. This is best-effort and should be tested on a clean Mac account/machine before sharing widely.

Before sharing an installer, run `musicidx index-health --json` inside the packaged app flow or with the same DB/model paths to confirm DB/model separation, profile v2 coverage, current embeddings, context-fit coverage, and failed/quarantined track warnings.

If the bundled audio helper binaries fail on the target Mac, rebuild using known self-contained/static `ffmpeg`, `ffprobe`, and `fpcalc` binaries via `MUSICIDX_FFMPEG_SOURCE`, `MUSICIDX_FFPROBE_SOURCE`, and `MUSICIDX_FPCALC_SOURCE`.
