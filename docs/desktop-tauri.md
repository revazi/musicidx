# MusicIdx Tauri Desktop App

The desktop app is a thin cross-platform UI over the existing local `musicidx` CLI. The Python CLI remains the source of truth for scanning, indexing, search, export, eval, and feedback.

## Architecture

```text
Tauri web UI
  -> Rust command: run_musicidx_stream(args, cwd, settings)
  -> local musicidx CLI process
  -> local SQLite DB + local audio/model files
```

No audio files are uploaded. The UI calls the same JSON CLI contracts documented in `docs/ui-json-contracts.md`.

## Prerequisites

- Node.js + npm
- Rust + Cargo
- A working MusicIdx CLI
- `ffprobe` and `fpcalc` installed/on `PATH` for full indexing

Check the CLI from the repo root:

```bash
uv run musicidx doctor
```

## First run

From the repo root:

```bash
cd desktop
npm install
```

Then start Tauri dev mode. If `musicidx` is installed on `PATH`:

```bash
npm run tauri:dev
```

If you want the desktop app to call the repo-local CLI through `uv`, use:

```bash
MUSICIDX_CLI_PATH=uv \
MUSICIDX_CLI_PREFIX_ARGS="run musicidx" \
npm run tauri:dev
```

On Windows PowerShell:

```powershell
$env:MUSICIDX_CLI_PATH = "uv"
$env:MUSICIDX_CLI_PREFIX_ARGS = "run musicidx"
npm run tauri:dev
```

In the app, set **Working directory** to the MusicIdx repo/library directory. This matters because the CLI currently defaults to project-local paths like:

```text
./musicidx.sqlite
./.musicidx-models/
```

When a working directory contains `.env`, the Rust wrapper passes those variables to the child `musicidx` command unless they are already set in the desktop app environment. This lets the app reuse local values such as `GEMINI_API_KEY`, `MUSICIDX_FFPROBE_PATH`, and `MUSICIDX_FPCALC_PATH` during development.

## Current UI capabilities

The Tauri UI can:

- run `doctor --json`
- use a minimal React + Tailwind + local shadcn-style component UI with subtle purple accents
- support System, Dark, and Light themes from Settings; System follows OS preference
- show DB info from advanced indexing actions
- use a separate settings page opened from the header gear icon
- keep semantic/embedding model configuration on the settings page
- choose a working directory with a native directory picker
- choose a music folder with a native directory picker
- choose DB/model/export paths where useful
- run a cancellable low-impact/adaptive indexing pipeline from a floating bottom-left icon button
- poll the configured music folder at the user-selected interval while the app is open and auto-index detected changes
- refresh derived metadata/fingerprints/features/tags/profiles/embeddings when an existing file is modified
- keep indexing setup and individual scan/metadata/fingerprint/basic/tag/embed actions in Settings
- show manual and background pipeline progress by step, runtime, counts, and memory diagnostics in a floating progress panel with a Cancel button
- enable/disable background auto-indexing and choose a 1/5/10/30/60 minute check interval from Settings
- choose an indexing resource profile: auto, low, balanced, or full
- run quick basic audio analysis in adaptive chunks to lower peak RAM
- run ML tag analysis in adaptive subprocess batches to lower peak RAM
- inspect failed/quarantined tracks and reset them for retry from advanced indexing actions
- stream command stdout/stderr into an expandable live/raw output panel
- parse natural-language intent
- run concise JSON search with explanations
- render search result cards
- save good/bad/neutral feedback from result cards
- run the starter eval set from advanced indexing actions
- export M3U playlists from a floating bottom-right icon button
- store desktop-only settings in browser local storage

## Settings

The settings panel can override development values without editing `.env`:

```text
CLI path
CLI prefix args
MUSICIDX_DB_PATH
MUSICIDX_MODELS_PATH
MUSICIDX_FFPROBE_PATH
MUSICIDX_FPCALC_PATH
semantic/embedding model
indexing resource profile
background auto-indexing toggle
GEMINI_API_KEY
LLM provider/model
```

For repo-local development, this is usually enough:

```text
CLI path:        uv
CLI prefix args: run musicidx
Working dir:    /path/to/musicidx
```

## Current limitations

This is still an early wrapper scaffold.

- Packaging the Python CLI sidecar is not implemented yet.
- App-data DB/model locations are not implemented yet.
- Long-running CLI commands stream output and can be cancelled, but there is no per-track structured progress percentage yet.
- Feedback buttons call the CLI one rating at a time; batch feedback is not implemented yet.
- Settings are local to the webview/localStorage and not a formal app config file yet.

## Next Tauri task

Decide and implement the Python sidecar packaging approach for Windows/macOS builds.
