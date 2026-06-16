# MusicIdx UI JSON Contracts

This document captures the JSON-oriented CLI commands that the cross-platform desktop UI can call. Treat these as the preferred integration surface for the Tauri wrapper.

## Principles

- The desktop UI should call `musicidx` commands with `--json` or `--format json`.
- Audio files are never uploaded.
- The UI should treat file paths as opaque strings and normalize/display them per platform.
- Prefer concise search JSON for result lists.
- Keep terminal table formatting out of UI flows.

## Core commands

### Health/status

```bash
musicidx doctor --json
musicidx db-info --json
musicidx models path --json
musicidx models list --json
```

Expected use:

- show setup status
- detect missing `ffprobe` / `fpcalc`
- detect local model availability
- show DB path and indexed counts

### Indexing workflow

```bash
musicidx scan <folder> --json
musicidx metadata --missing-only --json
musicidx fingerprint --missing-only --json
musicidx analyze-basic --quick --chunked --chunk-sec auto --workers auto --resource-profile auto --json
musicidx analyze-tags --missing-only --workers auto --resource-profile auto --subprocess-batches --batch-size auto --json
musicidx embed --batch-size auto --resource-profile auto --json
```

Expected use:

- run each indexing step from a UI action or setup wizard
- use adaptive low-impact defaults for one-click indexing
- use `scan <folder> --json` for app-open background polling
- run derived indexing steps when `added + modified > 0`; `missing > 0` only needs the scan result to update library state
- treat `modified > 0` as requiring refresh; scan invalidates stale derived rows for changed files so `--missing-only` steps can rebuild them
- handle `root_missing: true` as a warning state for a previously indexed folder that is currently unavailable; active tracks under that root have been marked missing
- display command summaries and errors
- show runtime diagnostics from `duration_sec`, `peak_memory_mb`, `child_peak_memory_mb`, and `diagnostics`
- keep long-running steps cancellable at the process/process-tree level

Common indexing diagnostics fields:

| Field | Meaning |
| --- | --- |
| `duration_sec` | Wall-clock duration for the command. |
| `peak_memory_mb` | Best-effort peak RSS for the CLI process. |
| `child_peak_memory_mb` | Best-effort peak RSS for child processes, when available. Useful for `ffprobe`, `fpcalc`, and tag subprocess batches. |
| `diagnostics.started_at` / `finished_at` | UTC timestamps for the command. |
| `diagnostics.peak_memory_source` | Platform API used for memory measurement. |
| `chunked` / `chunk_sec` | Basic-analysis chunking settings when `analyze-basic --chunked` is used. |
| `root_missing` | Scan-only flag. `true` means the scanned root was known from a previous scan but is currently unavailable, so active tracks under it were marked missing. |

### Missing and failed tracks

```bash
musicidx missing --json
musicidx prune-missing --track-id <id> --json
musicidx prune-missing --all --json
musicidx failed --json
musicidx failed --quarantined-only --json
musicidx retry-failed --track-id <id> --json
musicidx retry-failed --all --json
```

Expected use:

- show tracks marked missing after file removal or unavailable indexed roots
- allow a user to prune missing database rows without deleting music files
- show tracks skipped because of repeated decode/indexing failures
- allow a user to retry a fixed/replaced file
- prevent corrupt files from being retried on every indexing run

`missing --json` returns `{ db_path, count, missing }`; each item includes `id`, `path`, optional metadata, `root_path`, and `missing_at`. `prune-missing` returns `{ db_path, pruned, track_id }`.

### Track inspection

```bash
musicidx tags --track-id <id> --json
musicidx search-text "ambient" --json
musicidx search-semantic "ambient" --json
```

Expected use:

- inspect stored tags/features around a selected search result
- provide lower-level debug/search screens if needed

### Natural-language parsing

```bash
musicidx parse "chill bar" --json
musicidx parse "chill bar" --llm --llm-provider gemini --json
```

Expected use:

- preview parsed intent
- debug local vs LLM-assisted interpretation

### Search results

Preferred UI search command:

```bash
musicidx search "chill bar" --format json --concise --limit 10 --explain
```

Important top-level fields:

| Field | Meaning |
| --- | --- |
| `db_path` | SQLite DB used for the search. |
| `query` | Original user query. |
| `parser` | Parser mode, for example `dynamic` or `dynamic+gemini`. |
| `llm_error` | LLM failure message when `--llm` fallback occurred. |
| `intent` | Compact parsed intent. |
| `diagnostics` | Candidate counts, ranking weights, semantic errors. |
| `results` | Ranked result list. |

Important result fields:

| Field | Meaning |
| --- | --- |
| `track_id` | Stable local track ID. |
| `path` | Local file path. |
| `title` / `artist` / `album` / `genre` | Display metadata when available. |
| `score` | Final local ranking score. |
| `why` | Human-readable explanations when `--explain` is used. |
| `scores` | Compact score components. |
| `matched_tags` | Top matched ML/local tags. |

### Evaluation and feedback

```bash
musicidx eval eval/search_queries.json --limit 10 --json
musicidx judge "chill bar" --limit 10
musicidx feedback --track-id <id> --query "chill bar" --rating good --json
```

Expected use:

- run repeatable search-quality checks before/after ranking changes
- collect local good/bad judgments from users
- use feedback-aware ranking in later searches

`judge` is interactive and is mainly a CLI/manual workflow. The Tauri UI uses the non-interactive `feedback` command for result-card good/bad/neutral buttons.

### Playlist/export

```bash
musicidx search "ambient background" --format m3u > ambient.m3u
musicidx export "chill bar" --limit 25 --out chill-bar.m3u
musicidx export "chill bar" --format json --out chill-bar.json
musicidx export "chill bar" --format csv --out chill-bar.csv
```

Expected use:

- export selected or generated result sets
- provide playlist downloads/saves from the UI
- use `--absolute-paths` or `--relative-paths` when the target player requires it

## Cross-platform notes

- Windows paths may contain backslashes and drive letters; do not parse paths manually in the UI.
- macOS paths may require app permission/bookmarks in a packaged app.
- For packaged apps, DB/model locations should eventually move to platform app-data directories.
- During CLI-first development, project-local paths are acceptable and easier to inspect.
