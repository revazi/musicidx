# AGENTS.md

## Project
MusicIdx MVP

## Goal
Build MusicIdx as a local-first CLI plus Tauri desktop app that scans a directory of music tracks, analyzes metadata/audio features/ML tags/embeddings, stores everything in SQLite, and lets the user search the library with natural-language queries such as “chill bar music,” “shower music,” or “melancholic songs.”

The CLI remains the source-of-truth engine. The desktop app is a thin UI over the same local CLI/database.

## Current implementation status — 2026-06-17

Implemented and expected:

- Typer CLI with SQLite/FTS5, scanner, metadata, fingerprints, basic features, Essentia tags, profile text, semantic embeddings, hybrid search, export, eval, feedback, missing-track handling, and diagnostics.
- Tauri desktop UI for indexing/search/settings/playback, including app-open background auto-indexing with cancellation.
- Background indexing is app-open polling, not a daemon. It scans for added/modified/missing/root-missing changes and runs derived indexing only when needed.
- Missing files are marked with `missing_at`; pruning is explicit via `prune-missing`.
- Modified files invalidate stale derived metadata/fingerprints/features/tags/profiles/embeddings.
- Basic audio analysis should use full-track chunked analysis by default. Do **not** use quick/first-120s analysis automatically; `--quick` is only an explicit CLI escape hatch.
- Semantic search uses sentence-transformers profile-text embeddings. Hybrid search should report `semantic_candidate_count > 0` and `semantic_error: null` when semantics are active.
- Optional LLM intent parsing supports Gemini/OpenAI. It must always try to produce usable structured music intent for vague/slang/conversational queries unless the user explicitly asks not to search.
- Natural-language sorting is first-class via `sort_by`, e.g. highest BPM, slowest, most energetic, least aggressive.
- User-facing result `score` is normalized relative relevance (`1.0` for the top returned result). Raw weighted ranking score is exposed as `raw_score`/breakdown for diagnostics.

## Product summary
Build an MVP command-line app where a user can:
- select or provide a local music directory
- scan supported audio files
- extract metadata such as title, artist, album, genre, duration, codec, and technical info
- analyze basic audio features such as BPM, energy, brightness, loudness proxy, and mood-related signals
- store all indexed information in a local SQLite database
- search the library using regular language
- receive ranked track recommendations with short explanations
- export search results as playlists, especially M3U

Example target commands:

```bash
musicidx init
musicidx doctor
musicidx scan ~/Music
musicidx metadata
musicidx analyze-basic
musicidx search "Give me 10 tracks for a chill bar" --limit 10 --explain
musicidx export "shower music" --limit 25 --out shower.m3u
```

## Current stack direction
- Language: Python
- CLI framework: Typer
- Terminal output: Rich
- Database: SQLite
- Full-text search: SQLite FTS5
- Metadata extraction: ffprobe JSON output first
- Audio analysis: librosa first
- Optional music tagging later: Essentia models
- Optional LLM intent parsing: Gemini/OpenAI today; local providers such as Ollama can be added later
- Semantic embeddings: sentence-transformers over deterministic track-profile text; brute-force NumPy cosine search for now
- Desktop app: Tauri wrapper over the CLI/helper engine

## Core engineering principles
- Keep the MVP simple
- Prefer boring, maintainable solutions
- Make the smallest useful change possible
- Keep the CLI usable independently from the desktop UI
- Prefer synchronous request/response flows first
- Keep background behavior bounded and explicit; current background indexing is app-open polling only
- Keep dependencies minimal
- Keep indexing and search behavior transparent and debuggable
- Prefer deterministic parsing and scoring before adding LLM behavior
- Optimize for readability and iteration speed
- Treat every feature as local-first by default

## Working rules
- Only do the task that was requested
- Before coding, provide a short plan
- After completing the requested task, stop
- No file changes without explicit confirmation
- Do not jump ahead to later phases
- Do not add unrelated improvements
- Do not perform broad refactors unless explicitly asked
- If something is ambiguous, choose the simplest sensible option and state it clearly
- Keep the implementation aligned with the current phase
- Do not introduce new services, daemons, workers, or UI layers unless requested
- Do not add heavyweight ML models unless the task explicitly asks for them

## Architecture constraints
- Local-first only
- No cloud indexing
- No remote audio upload
- No user account system
- No web app unless explicitly requested
- Do not add a new desktop framework; the current desktop app is Tauri
- No Celery
- No Redis
- No Docker unless explicitly requested
- No Kubernetes
- No microservices
- No event-driven architecture
- No permanent daemon/background worker system unless explicitly requested
- No external hosted database
- No mandatory external LLM API
- No telemetry unless explicitly requested

## CLI conventions
- Expose commands through a single `musicidx` CLI
- Use Typer for command definitions
- Use Rich for readable terminal output
- Support `--json` output for commands that may later be called by a desktop app
- Keep commands predictable and composable
- Prefer explicit command names over clever shortcuts
- Commands should fail gracefully with clear error messages
- Long-running commands should show progress where practical
- Do not crash the whole process because one audio file is corrupt or unsupported

## Expected core commands
Start with only the commands required by the current task or phase.

Likely command progression:

```bash
musicidx --help
musicidx doctor
musicidx init
musicidx db-info
musicidx scan <directory>
musicidx metadata
musicidx search-text <query>
musicidx analyze-basic
musicidx parse <query>
musicidx search <query>
musicidx export <query>
```

Do not implement commands from later phases unless explicitly asked.

## Backend/module conventions
- Keep code under `src/musicidx/`
- Keep modules simple and practical
- Prefer straightforward functions and small service modules
- Avoid plugin-style internal architectures
- Avoid premature abstractions
- Prefer explicit code over clever code
- Keep database access understandable
- Keep scoring logic separate from CLI command handlers
- Keep analysis logic separate from database persistence
- Use dataclasses and explicit allow-list validation unless a stronger schema library is clearly needed
- Add tests for behavior that can regress easily

Suggested project structure:

```text
musicidx/
  pyproject.toml
  README.md
  AGENTS.md
  src/musicidx/
    __init__.py
    cli.py
    config.py
    db.py
    migrations.py
    scanner.py
    metadata.py
    fingerprint.py
    analyzer/
      __init__.py
      basic_features.py
      embeddings.py
      essentia_models.py
    search/
      __init__.py
      intent.py
      llm.py
      ranker.py
      explain.py
    export.py
    models.py
    logging.py
  tests/
    test_db.py
    test_scanner.py
    test_metadata.py
    test_intent.py
    test_ranker.py
```

Only create directories and files needed for the current task.

## Database conventions
- Use SQLite as the local source of truth
- Default CLI development DB is project/current-working-directory `musicidx.sqlite`; packaged desktop builds should use app-data paths
- Keep migrations simple and idempotent
- Enable foreign keys
- Prefer WAL mode for the local index database
- Do not delete missing files immediately; mark them as missing
- Keep schema changes explicit
- Avoid complex ORM layers unless explicitly requested
- Use SQLite FTS5 for text search
- Do not add vector search infrastructure until embeddings are actually implemented

Default database location for packaged desktop-app development:

```text
~/Library/Application Support/MusicIdx/index.sqlite
```

A configurable database path is acceptable for development and tests.

## Audio file support
Supported extensions for scanning:

```text
.mp3
.flac
.m4a
.aac
.wav
.aiff
.aif
.ogg
.opus
.alac
.wv
```

Scanning should be idempotent. Re-running a scan over the same unchanged directory must not create duplicate tracks.

## Metadata implementation guidance
- Use `ffprobe` JSON output first
- Keep metadata extraction replaceable but do not over-abstract it
- Normalize common tag aliases
- Store useful technical info such as codec, sample rate, bit rate, channels, and duration
- Generate a simple text profile for each track after metadata extraction
- Keep profile generation deterministic and inspectable

Common metadata fields:

```text
title
artist
album
album_artist
genre
date/year
track_number
disc_number
duration_sec
codec
sample_rate
bit_rate
channels
```

## Audio analysis guidance
- Start with basic deterministic audio features
- Use `librosa` first
- Analyze in mono with a consistent sample rate
- Use full-track chunked analysis by default; do not use quick/first-120s sampling automatically
- Keep `--quick` only as an explicit CLI escape hatch for manual experiments
- Store raw features and normalized derived values where useful
- Make analysis resumable by skipping already-analyzed tracks with the current analysis version
- Corrupt files should be recorded as analysis failures and skipped

Initial useful features:

```text
BPM / tempo
RMS energy
spectral centroid
spectral flatness
spectral rolloff
zero crossing rate
MFCC mean/std
chroma profile
rough key/mode estimate
brightness proxy
aggression proxy
danceability proxy
```

## AI / search implementation guidance
- Keep first versions heuristic and easy to inspect
- Prefer deterministic intent parsing as the fallback path
- LLM-backed parsing is optional and should improve intent extraction only
- The LLM should parse user intent, not invent track recommendations
- The ranking system must only return tracks that exist in the local database
- Keep scoring logic transparent and debuggable
- Separate query parsing, retrieval, scoring, and explanation logic
- Validate all LLM output with a strict schema
- Always provide a non-LLM fallback

Natural-language queries should be converted into structured intent such as:

```json
{
  "limit": 10,
  "context": "chill_bar",
  "energy": [0.25, 0.6],
  "valence": [0.35, 0.8],
  "tempo_bpm": [70, 115],
  "prefer_tags": ["chill", "lounge", "warm", "downtempo"],
  "avoid_tags": ["aggressive", "chaotic", "very loud"],
  "diversity": {
    "max_tracks_per_artist": 2
  }
}
```

## Ranking guidance
Use a simple, inspectable hybrid ranking system.

Candidate sources may include:
- metadata/full-text search
- generated track profile text
- audio feature ranges
- tag matches
- semantic profile embeddings when available
- optional/query-aware user feedback

Initial scoring can combine:

```text
semantic/profile match
feature fit
mood/tag fit
metadata text match
diversity penalty
```

Keep ranking weights configurable or easy to adjust. Do not hardcode subjective assumptions in too many places.

## Playlist/export guidance
- M3U export is the first priority
- JSON output is used by the desktop app
- CSV export can be added when requested
- Do not build full playlist management until requested

## Desktop app guidance
- The desktop app is implemented in `desktop/` with Tauri and should remain a thin layer over the CLI/database.
- Design CLI JSON output so the desktop can call it reliably.
- Keep the engine independent from terminal formatting.
- Do not introduce another desktop UI framework without explicit direction.

## Privacy and local-first rules
- Do not upload audio files anywhere
- Do not send track metadata to remote APIs unless explicitly requested
- Do not require hosted LLM APIs; Gemini/OpenAI are optional intent parsers only
- Do not add telemetry or analytics unless explicitly requested
- Any optional model download behavior must be explicit
- Keep user library paths local and private

## File and project hygiene
- Add only the files needed for the current task
- Keep naming clear and predictable
- Update docs when the task changes public behavior or setup
- Avoid dead code, placeholder abstractions, and speculative structure
- Avoid adding dependencies that are not used by the current task
- Keep tests focused on the behavior being implemented
- Prefer small commits/changesets organized around one task

## Expected output for each task
Return:
1. Plan
2. Changes made
3. Files created/edited
4. Commands to run
5. Verification steps
6. Decisions/tradeoffs

## Non-goals for now
- desktop UI before the CLI works
- cloud sync
- streaming service integrations
- user accounts
- social features
- recommendation feeds
- enterprise-grade infrastructure
- advanced async processing
- multi-tenant architecture
- plugin frameworks
- deep analytics
- production-grade scaling work before the MVP exists
- direct audio-to-text embedding models before basic search quality is proven
- commercial model packaging before licenses are reviewed
