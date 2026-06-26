# musicidx Implementation Phases

This document defines the phased implementation plan for a local-first CLI tool, and later a cross-platform desktop app, that indexes a local music library and enables natural-language music search such as:

- “Give me a list of 10 tracks, I’m today playing in a chill bar”
- “Shower music”
- “I’m a little bit sad and wanna listen to melancholic music”

The recommended approach is:

```text
local music library
  -> scanner
  -> metadata extraction
  -> audio analysis
  -> mood/tag/feature profiles
  -> SQLite index
  -> natural-language intent parser
  -> hybrid search/ranking
  -> CLI + Tauri desktop UI
```

The LLM should not be responsible for listening to or analyzing raw audio. It should translate natural-language user requests into structured search intent. The search/ranking system should then retrieve real tracks from the local database.

---

## Current implementation status — 2026-06-25

This document began as a phased plan; the current implementation is ahead of several early-phase assumptions.

Implemented now:

- Python/Typer CLI with SQLite/FTS5, scanner, metadata, metadata repair, fingerprints, full-track chunked audio features, Essentia tags, profile text, semantic embeddings, hybrid search, export, eval, feedback, missing-track handling, index health, and diagnostics.
- Tauri desktop app over the CLI/database with settings, indexing, search, feedback, in-app playback, missing-track actions, index health, search diagnostics, suggestions, and app-open background auto-indexing.
- Background auto-indexing is app-open polling, not a daemon. It marks missing/root-missing state and runs derived indexing for added/modified files.
- Basic audio analysis defaults to full-track chunked mode. Do **not** use quick/first-120s analysis automatically; `--quick` is only an explicit CLI option.
- Semantic search uses sentence-transformers profile-text embeddings and participates in hybrid ranking when `semantic_candidate_count > 0` and `semantic_error` is null.
- Optional LLM intent parsing uses Gemini/OpenAI. LLM hints are validated by guardrails before affecting local ranking; broad/suspicious hints fall back to dynamic local parsing.
- Natural-language sort intent is structured through `sort_by` and supports feature sorting such as highest BPM, slowest, most energetic, least aggressive, most danceable, brightest, and darkest.
- User-facing result `score` is a calibrated raw relevance score, not normalized to the top result. `raw_score`, confidence labels, evidence, warnings, and breakdowns are exposed for diagnostics.

---

## Recommended Build Order

```text
1. SQLite DB + scanner
2. Metadata + FTS search
3. Basic audio features
4. Profile text generation
5. Semantic embeddings over profile text
6. Rule-based natural-language parser
7. Local LLM structured parser
8. Hybrid ranking
9. M3U/JSON/CSV export
10. Feedback/evaluation loop
11. Optional Essentia ML tags
12. Optional CLAP/MERT audio embeddings
13. Cross-platform desktop wrapper
14. Post-UI search quality tuning
15. Post-UI explanation polish
16. Post-UI playlist and saved-search workflow
```

---

## Architecture

```text
                 ┌──────────────────────┐
                 │ User Music Directory │
                 └──────────┬───────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│ Scanner                                                │
│ - finds audio files                                    │
│ - detects changes                                      │
│ - marks missing/deleted files                          │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Analyzer                                               │
│ - metadata: title, artist, album, genre, duration      │
│ - fingerprints/checksums                               │
│ - audio features: BPM, key, loudness, energy, etc.     │
│ - ML tags: mood, genre, instrumental, danceability     │
│ - embeddings                                           │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Local Index DB                                         │
│ SQLite tables + FTS + optional vector index            │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Search Engine                                          │
│ - LLM parses query into structured intent              │
│ - SQL/FTS/vector retrieval                             │
│ - feature-based ranking                                │
│ - diversity rules                                      │
└──────────┬────────────────────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ CLI + Tauri desktop app                                │
│ - search results                                       │
│ - explanations                                         │
│ - M3U/JSON export                                      │
│ - feedback loop                                        │
│ - in-app playback                                      │
└───────────────────────────────────────────────────────┘
```

---

## Recommended Stack

```text
Language:         Python
CLI:              Typer + Rich
Database:         SQLite
Full-text search: SQLite FTS5
Vector search:    Start with NumPy brute force; later sqlite-vec
Metadata:         ffprobe JSON first; TagLib/lofty later if needed
Audio analysis:   librosa first; Essentia optional later
LLM intent:       optional Gemini/OpenAI today; local providers later if needed
Embeddings:       sentence-transformers for track-profile text
Packaging:        uv for development; PyInstaller/Tauri sidecar packaging scaffolding
Desktop app:      Tauri wrapper calling CLI/helper engine
```

---

# Phase 0 — Product Constraints and Repo Setup

## Goal

Create a local-first CLI project with a clean architecture that can later become a cross-platform desktop app backend.

## Steps for Pi Coding Agent

1. Create repository structure:

```text
musicidx/
  pyproject.toml
  README.md
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
      essentia_models.py
      embeddings.py
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
    test_scanner.py
    test_db.py
    test_intent.py
    test_ranker.py
```

2. Use `pyproject.toml` with these core dependencies:

```text
typer
rich
pydantic
numpy
scipy
librosa
soundfile
sentence-transformers
pytest
ruff
```

3. Add optional dependencies:

```text
ollama
sqlite-vec
essentia
```

4. Add CLI entry point:

```bash
musicidx
```

5. Add config file support:

```text
~/.config/musicidx/config.toml
```

6. Add default DB path:

```text
~/Library/Application Support/MusicIdx/index.sqlite
```

7. Implement structured logging.

## Acceptance Criteria

These commands must work:

```bash
musicidx --help
musicidx doctor
musicidx init
```

`musicidx doctor` should report:

```text
SQLite available
FFmpeg/ffprobe available or missing
fpcalc available or missing
Ollama available or missing
Essentia available or missing
Embedding model available or missing
```

---

# Phase 1 — SQLite Database and Migrations

## Goal

Create the persistent local music index.

## Steps for Pi Coding Agent

1. Implement `db.py`:

```python
connect_db(path: Path) -> sqlite3.Connection
init_db(conn) -> None
apply_migrations(conn) -> None
```

2. Implement migrations using numbered SQL files or Python migration functions.

3. Add tables:

```text
library_roots
tracks
audio_features
track_tags
track_profiles
tracks_fts
embeddings
search_events
feedback
```

4. Enable SQLite settings:

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
```

5. Add DB utility commands:

```bash
musicidx init
musicidx db-info
musicidx reset --yes
```

## Initial Schema

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS library_roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    root_id INTEGER,
    path TEXT NOT NULL UNIQUE,
    path_hash TEXT NOT NULL,
    extension TEXT,
    file_size INTEGER,
    file_mtime_ns INTEGER,
    content_hash TEXT,
    chromaprint TEXT,

    title TEXT,
    artist TEXT,
    album TEXT,
    album_artist TEXT,
    genre TEXT,
    date TEXT,
    track_number TEXT,
    disc_number TEXT,

    duration_sec REAL,
    codec TEXT,
    sample_rate INTEGER,
    bit_rate INTEGER,
    channels INTEGER,

    analysis_version INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT,
    analyzed_at TEXT,
    missing_at TEXT,
    last_error TEXT,

    FOREIGN KEY(root_id) REFERENCES library_roots(id)
);

CREATE TABLE IF NOT EXISTS audio_features (
    track_id TEXT PRIMARY KEY,

    bpm REAL,
    key_name TEXT,
    mode TEXT,

    loudness_integrated REAL,
    loudness_range REAL,
    dynamic_range REAL,

    energy REAL,
    valence REAL,
    danceability REAL,
    acousticness REAL,
    instrumentalness REAL,
    vocalness REAL,
    speechiness REAL,
    aggression REAL,
    brightness REAL,

    spectral_centroid_mean REAL,
    spectral_centroid_std REAL,
    spectral_flatness_mean REAL,
    spectral_rolloff_mean REAL,
    zero_crossing_rate_mean REAL,

    mfcc_mean_json TEXT,
    mfcc_std_json TEXT,

    raw_features_json TEXT,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS track_tags (
    track_id TEXT NOT NULL,
    source TEXT NOT NULL,
    tag TEXT NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL,

    PRIMARY KEY(track_id, source, tag),
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS track_profiles (
    track_id TEXT PRIMARY KEY,
    profile_text TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS tracks_fts
USING fts5(
    track_id UNINDEXED,
    title,
    artist,
    album,
    genre,
    profile_text
);

CREATE TABLE IF NOT EXISTS embeddings (
    track_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    text TEXT,
    updated_at TEXT NOT NULL,

    PRIMARY KEY(track_id, kind, model),
    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS search_events (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    parsed_intent_json TEXT,
    result_track_ids_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    search_event_id TEXT,
    track_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,

    FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
```

## Acceptance Criteria

Running `musicidx init` twice is safe and idempotent.

Tests verify:

```text
tables exist
foreign keys work
FTS table can insert and search
migrations do not duplicate
```

---

# Phase 2 — Directory Scanner

## Goal

Scan a local music directory and track added, changed, and missing files.

## Supported Extensions

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

## Steps for Pi Coding Agent

1. Implement:

```python
scan_library(root_path: Path, conn) -> ScanSummary
```

2. For each audio file, collect:

```text
absolute path
file size
mtime ns
path hash
extension
```

3. Generate stable track IDs:

```text
First scan: UUID
Rescan: match by path
Moved file detection: match by content_hash or chromaprint later
```

4. Mark missing files instead of deleting immediately:

```text
missing_at = current timestamp
```

5. Add command:

```bash
musicidx scan ~/Music
```

6. Add options:

```bash
musicidx scan ~/Music --full-hash
musicidx scan ~/Music --follow-symlinks
musicidx scan ~/Music --dry-run
musicidx scan ~/Music --json
```

## Acceptance Criteria

A test directory with fake audio filenames should produce:

```text
N added
N unchanged
N modified
N missing
```

Rescanning unchanged files should not duplicate rows.

---

# Phase 3 — Metadata Extraction

## Goal

Extract track metadata and technical audio information.

## Recommended First Implementation

Use `ffprobe` JSON output:

```bash
ffprobe -v error \
  -show_format \
  -show_streams \
  -print_format json \
  /path/to/file.mp3
```

## Steps for Pi Coding Agent

1. Implement:

```python
extract_metadata(path: Path) -> TrackMetadata
```

2. Parse:

```text
title
artist
album
album_artist
genre
date/year
track number
disc number
duration
codec
sample rate
bit rate
channels
```

3. Normalize common tag aliases:

```text
album_artist / albumartist / Album Artist
track / tracknumber / TRACKNUMBER
date / year / originaldate
```

4. Store metadata in `tracks`.

5. Add command:

```bash
musicidx metadata
musicidx metadata --track-id <id>
musicidx metadata --missing-only
musicidx metadata --json
```

6. Generate `track_profiles.profile_text`.

Example:

```text
Artist: Air. Title: La Femme d'Argent. Album: Moon Safari.
Genre: downtempo/electronic. Duration: 7:08.
```

7. Sync `tracks_fts`.

## Acceptance Criteria

Given a tagged MP3/FLAC/M4A fixture, metadata fields are extracted and searchable.

Example:

```bash
musicidx search-text "Nick Drake"
```

returns matching tracks through FTS.

---

# Phase 4 — Fingerprinting and Duplicate Detection

## Goal

Detect duplicate or near-duplicate tracks and moved files.

## Steps for Pi Coding Agent

1. Implement `fingerprint.py`.

2. Prefer `fpcalc` if installed:

```bash
fpcalc -json /path/to/file
```

3. Store:

```text
chromaprint
fingerprint_duration
```

4. Add commands:

```bash
musicidx fingerprint
musicidx duplicates
```

5. Duplicate logic:

```text
same chromaprint + similar duration = duplicate candidate
same content_hash = exact duplicate
same artist/title/duration = possible duplicate
```

## Acceptance Criteria

`musicidx duplicates` shows grouped candidates and does not delete anything.

---

# Phase 5 — Basic Audio Feature Extraction

## Goal

Create useful audio descriptors before adding heavier ML models.

## Features to Extract First

```text
tempo/BPM
RMS energy
spectral centroid
spectral flatness
spectral rolloff
zero crossing rate
MFCC mean/std
chroma profile
rough key/mode estimate
brightness
dynamic range proxy
```

## Steps for Pi Coding Agent

1. Implement:

```python
analyze_basic_features(path: Path) -> AudioFeatures
```

2. Decode/resample consistently:

```text
mono
22050 Hz
full-track chunked mode by default
quick/first-120s mode only when explicitly passed on the CLI
```

3. Add command:

```bash
musicidx analyze-basic
musicidx analyze-basic --chunked --chunk-sec auto --workers auto --resource-profile auto
musicidx analyze-basic --chunked --chunk-sec 300 --resource-profile full
musicidx analyze-basic --track-id <id>
musicidx analyze-basic --json
```

4. Normalize features to 0–1 where possible:

```text
energy
brightness
aggression
danceability proxy
acousticness proxy
```

5. Store raw and normalized values.

6. Update `track_profiles.profile_text` with feature descriptions.

Example:

```text
Low energy, slow tempo around 78 BPM, dark tone, low brightness,
likely calm or melancholic.
```

## Acceptance Criteria

The analyzer should skip files that are already analyzed with the current `analysis_version`.

Corrupt or unreadable files should not crash indexing; they should be marked with an analysis error.

---

# Phase 6 — Mood, Genre, and Music Tags

## Goal

Move from raw features to meaningful human labels.

## Recommended Path

Start with optional Essentia models after the basic indexer works. Treat all model licensing as something to verify before commercial distribution.

## Steps for Pi Coding Agent

1. Implement optional analyzer:

```python
analyze_essentia_tags(path: Path) -> list[TrackTag]
```

2. Add model manager:

```bash
musicidx models list
musicidx models install essentia-basic
musicidx models path
```

3. Store tags in `track_tags`:

```text
source = essentia
tag = "melancholic"
score = 0.82
```

4. Initial useful tag groups:

```text
mood: happy, sad, melancholic, relaxed, aggressive, party
genre/style: jazz, rock, electronic, classical, ambient, hip-hop
function: danceable, acoustic, instrumental, vocal
texture: bright, dark, soft, noisy
```

5. Build derived fields:

```text
energy
valence
danceability
acousticness
instrumentalness
vocalness
aggression
```

6. Update profile text.

Example:

```text
Tags: melancholic 0.81, acoustic 0.74, vocal 0.88, relaxed 0.66.
```

## Acceptance Criteria

`musicidx analyze-tags` adds tags and updates track profiles.

`musicidx tags --track-id <id>` displays model outputs.

---

# Phase 7 — Text Embeddings over Track Profiles

## Goal

Support semantic search over generated track descriptions.

The MVP should use text embeddings on generated profile text. This is easier and more reliable than direct audio-text search at the beginning.

## Steps for Pi Coding Agent

1. Implement:

```python
embed_track_profile(track_id: str) -> np.ndarray
embed_query(query: str) -> np.ndarray
```

2. Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

3. Store vector as float32 BLOB in `embeddings`.

4. For MVP, load all vectors into memory and cosine-rank them.

Reasonable memory expectation:

```text
100,000 tracks × 384 dims × 4 bytes ≈ 154 MB
```

So brute-force NumPy search is acceptable for many personal libraries.

5. Later add `sqlite-vec` if needed.

6. Add commands:

```bash
musicidx embed
musicidx search-semantic "melancholic acoustic songs"
```

## Acceptance Criteria

Semantic search should find relevant tracks even when the query does not match exact tags or metadata.

---

# Phase 8 — Natural-Language Intent Parser

## Goal

Turn user text into structured search parameters.

## First Version

Implement a rule-based parser before using the LLM. This gives deterministic fallback behavior.

Examples:

```text
"10 tracks" -> limit = 10
"chill" -> energy low/medium, aggression low
"sad" -> valence low, melancholic tags
"bar" -> avoid harsh/aggressive/chaotic, prefer groovy/warm
"shower" -> upbeat/high energy/positive
```

## LLM Version

Use optional Gemini/OpenAI intent parsing to produce structured JSON hints. Validate and sanitize every field in code. If the LLM is unavailable or invalid, fall back to the dynamic local parser.

The LLM prompt must always try to produce usable music search intent, even for vague, slang, typo-filled, or conversational queries. Empty hints are only appropriate when the user explicitly asks not to search.

## SearchIntent JSON Schema

```json
{
  "type": "object",
  "properties": {
    "limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100
    },
    "contexts": {
      "type": "array",
      "items": { "type": "string" }
    },
    "prefer_tag_concepts": {
      "type": "array",
      "items": { "type": "string" }
    },
    "avoid_tag_concepts": {
      "type": "array",
      "items": { "type": "string" }
    },
    "feature_preferences": {
      "type": "object",
      "description": "field -> very_low|low|low_mid|mid|mid_high|high|very_high"
    },
    "sort_by": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "field": { "type": "string" },
          "direction": { "type": "string", "enum": ["asc", "desc"] }
        }
      }
    },
    "notes": {
      "type": "string"
    }
  },
  "required": []
}
```

## LLM Prompt

```text
You are a music search intent parser.

Return only JSON matching the provided schema.

Do not invent track names.
Do not recommend music.
Only translate the user's request into search hints.
Always produce a usable music search intent, even for vague/slang/conversational text.
Use sort_by for explicit ordering requests such as highest BPM or least aggressive.

User request object contains the query, aggregate library profile, allowed contexts,
allowed feature fields/levels, and allowed sort fields/directions.
```

## Steps for Pi Coding Agent

1. Implement:

```python
parse_intent_rule_based(query: str) -> SearchIntent
parse_intent_llm(query: str) -> SearchIntent
parse_intent(query: str) -> SearchIntent
```

2. Add fallback order:

```text
LLM available and valid JSON -> merge sanitized LLM hints into dynamic parser
LLM unavailable/invalid -> dynamic local parser only
```

3. Add command:

```bash
musicidx parse "shower music"
musicidx parse "shower music" --llm
musicidx parse "shower music" --no-llm
```

4. Validate and sanitize all LLM output with strict allow-lists.

5. Clamp unsafe or out-of-range values.

## Acceptance Criteria

These commands must return valid JSON:

```bash
musicidx parse "Give me 10 tracks for a chill bar"
musicidx parse "Shower music"
musicidx parse "I'm sad and want melancholic songs"
```

---

# Phase 9 — Hybrid Search and Ranking

## Goal

Produce useful ranked results.

## Retrieval Pipeline

```text
1. Parse natural-language query into SearchIntent.
2. Load local candidate rows, tags, FTS scores, semantic scores when available, and feedback scores.
3. Score each candidate with semantic/tag/feature/text/feedback components.
4. Filter weak fallback candidates; subjective queries prefer tag/text/semantic evidence when available.
5. Apply explicit sort intent if requested, otherwise apply artist diversity.
6. Return top N with calibrated raw relevance scores, confidence labels, and warnings.
7. Explain why each track matched, including semantic-only/low-confidence cases.
```

## Ranking Formula

Start with configurable weights:

```text
weighted_score =
    semantic_weight * semantic_score
  + tag_weight      * tag_score
  + context_weight  * context_score
  + feature_weight  * feature_score
  + text_weight     * text_score
  + metadata_weight * metadata_score
  + feedback_weight * feedback_score

User-facing result.score is calibrated by dividing the weighted score by the active
weight budget. It is not normalized to the top returned result. `raw_score`, score
components, confidence, evidence, and warnings remain available for diagnostics.
```

## Feature Scoring

For target ranges:

```python
def range_score(value, low, high, softness=0.15):
    if value is None:
        return 0.4
    if low <= value <= high:
        return 1.0
    distance = min(abs(value - low), abs(value - high))
    return max(0.0, 1.0 - distance / softness)
```

For tag matching:

```text
tag_score =
    sum(score(tag) * intent.prefer_tags[tag])
  - sum(score(tag) * intent.avoid_tags[tag])
```

For diversity:

```text
Do not return 10 tracks from the same artist.
Penalize near-duplicates.
Penalize multiple tracks from the same album unless requested.
Optionally prefer previously unplayed or less recently returned tracks.
```

## Steps for Pi Coding Agent

1. Implement:

```python
search(query: str, limit: int | None) -> SearchResults
```

2. Implement candidate retrieval:

```python
get_semantic_candidates(intent, query, k=200)
get_fts_candidates(query, k=200)
get_feature_candidates(intent, k=500)
get_tag_candidates(intent, k=500)
```

3. Implement ranking:

```python
score_track(track, intent, query_embedding) -> ScoreBreakdown
```

4. Implement explanations:

```text
Matched because:
- energy 0.38 fits requested chill range
- tags include relaxed, downtempo, warm
- BPM 92 fits bar/lounge context
- low aggression
```

5. Add command:

```bash
musicidx search "chill bar" --limit 10 --explain
```

6. Add formats:

```bash
--format table
--format json
--format m3u
```

## Acceptance Criteria

Search should never return tracks that do not exist on disk unless `--include-missing` is passed.

Search should not crash if some tracks have missing features.

---

# Phase 10 — Export and Playlist Workflow

## Goal

Make search output usable by DJs and listeners.

## Steps for Pi Coding Agent

1. Add M3U export:

```bash
musicidx export "chill bar" --limit 25 --out chill_bar.m3u
```

2. Add JSON export:

```bash
musicidx export "melancholic" --format json --out melancholic.json
```

3. Add CSV export:

```bash
musicidx export "shower music" --format csv --out shower.csv
```

4. Add playlist preview:

```bash
musicidx playlist-preview "chill bar"
```

5. Add options:

```bash
--absolute-paths
--relative-paths
```

## Acceptance Criteria

Generated M3U opens in common music players.

---

# Phase 11 — Evaluation Harness

## Goal

Improve search quality deliberately instead of guessing.

## Steps for Pi Coding Agent

1. Add test query file:

```yaml
queries:
  - id: chill_bar
    text: "Give me 10 tracks for a chill bar"
    expected_tags:
      - chill
      - relaxed
      - downtempo
    avoid_tags:
      - aggressive
      - metal
      - chaotic

  - id: shower
    text: "Shower music"
    expected_tags:
      - upbeat
      - energetic
      - happy

  - id: melancholic
    text: "I'm sad and want melancholic music"
    expected_tags:
      - sad
      - melancholic
      - reflective
```

2. Add manual judgment command:

```bash
musicidx judge "chill bar"
```

This should show results one by one:

```text
Good match? [y/n/s]
```

3. Store feedback in `feedback`.

4. Add evaluation command:

```bash
musicidx eval
```

5. Metrics:

```text
precision@10
average rating
tag coverage
diversity score
duplicate rate
```

6. Tune ranking weights from feedback.

## Acceptance Criteria

A user can run 20 searches, mark results good/bad, and see ranking improve.

---

# Phase 12 — Watch Mode and Incremental Analysis

## Goal

Keep the index up to date.

## Steps for Pi Coding Agent

1. Add watch command:

```bash
musicidx watch ~/Music
```

2. Use filesystem events where available.

3. Debounce changes.

4. Queue jobs:

```text
metadata extraction
fingerprinting
basic features
ML tags
embeddings
```

5. Add job table if needed:

```sql
CREATE TABLE analysis_jobs (
    id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

6. Add command:

```bash
musicidx status
```

## Acceptance Criteria

Adding a new file while watch mode is running eventually makes it searchable.

---

# Phase 13 — Optional Direct Audio-Text Embeddings

## Goal

Improve vague semantic matching after the baseline system works.

This should not be part of the MVP. Add it only after the basic feature/tag/profile search works and you have an evaluation harness.

## Steps for Pi Coding Agent

1. Add experimental analyzer:

```bash
musicidx analyze-audio-embeddings --model clap
```

2. Store:

```text
embedding kind = audio_clap
model = selected checkpoint
```

3. At search time:

```text
query text -> CLAP text embedding
compare with audio CLAP embeddings
```

4. Compare quality against profile-text embeddings.

## Acceptance Criteria

Keep this only if it improves evaluation metrics.

---

# Phase 13 — Cross-Platform Desktop Wrapper

## Goal

Add a desktop UI that works on Windows and macOS while keeping the Python CLI/search engine as the local-first backend.

## Recommended Cross-Platform Architecture

Start with:

```text
Cross-platform desktop UI
  -> local helper CLI: musicidx --json
  -> SQLite index
  -> local models/audio files selected by user
```

Recommended wrapper options:

```text
Tauri + Python sidecar
  Pros: lightweight, modern, Windows/macOS/Linux support
  Cons: Rust/Tauri packaging complexity

Electron + Python sidecar
  Pros: mature, easiest cross-platform desktop packaging
  Cons: larger app size and higher memory use

PySide6/Qt
  Pros: Python-native, can call engine directly or via CLI
  Cons: UI polish and packaging ML/audio dependencies need care
```

Default recommendation:

```text
Use Tauri for the long-term app if packaging proves manageable.
Use Electron if the priority is the fastest Windows/macOS MVP.
Keep the CLI/helper JSON contract stable either way.
```

Later, only if needed:

```text
Desktop UI
  -> bundled helper service or sidecar engine
  -> SQLite + local models
```

## Desktop App Features, v1

```text
Select music folder
Show indexing progress
Show analyzed track count
Natural-language search box
Result list with title/artist/reason
Open file in file manager/Finder/Explorer
Play preview
Export playlist
Feedback buttons: good / bad / too energetic / too sad / not chill
```

## Cross-Platform Concerns

```text
Windows and macOS filesystem paths differ; keep paths normalized in JSON.
Bundle or detect ffmpeg/ffprobe and fpcalc per platform.
Package Python, native audio/ML dependencies, and local models carefully.
Store SQLite DB in the platform-appropriate app data directory for packaged apps.
Keep the CLI/helper as a stable boundary before rewriting engine internals.
Do not require admin privileges for normal indexing/search.
```

## Acceptance Criteria

A non-technical user on Windows or macOS can select a music folder, index it, search it, inspect explanations, and export a playlist without using the terminal.

---

# Post-UI Phases

These phases should happen after the first cross-platform desktop UI can index a folder, run a search, and display results. The goal is to tune quality with real interaction instead of guessing from CLI output alone.

## Phase 14 — Post-UI Search Quality Tuning

### Goal

Improve ranking quality once search results can be reviewed quickly in the UI.

### Steps for Pi Coding Agent

1. Create a small repeatable query set for common use cases:

```text
chill bar
shower music
focus coding ambient
sad reflective songs
party electronic
aggressive workout
sleepy calm
```

2. Add configurable ranking weights for the main scoring signals:

```text
semantic/profile similarity
ML mood/genre tags
audio feature fit
text/FTS match
artist/album diversity
```

3. Improve context mappings for high-value intents:

```text
bar
shower
focus
party
sleep
workout
sad / melancholic
```

4. Calibrate feature ranges and tag boosts using actual library results.

5. Use the evaluation harness and UI feedback to compare before/after ranking changes.

### Acceptance Criteria

For the saved query set, top-10 results should visibly improve and regressions should be easy to detect.

---

## Phase 15 — Post-UI Explanation Polish

### Goal

Make result explanations shorter, prettier, and useful in both CLI and UI.

### Steps for Pi Coding Agent

1. Convert verbose score details into concise reason summaries:

```text
Matched chill/bar mood, low-medium energy, downtempo tags.
```

2. Show only the most useful evidence first:

```text
top matched tags
top feature matches
semantic/profile match when available
main penalty, if any
```

3. Keep machine-readable explanation data available for JSON/UI consumers.

4. Ensure `--explain` remains readable in terminal table output.

5. Ensure `--format json --concise` stays compact enough for UI/API use.

### Acceptance Criteria

Users can understand why a result appeared without reading raw score breakdowns.

---

## Phase 16 — Post-UI Playlist and Saved-Search Workflow

### Goal

Make good search results easy to reuse, export, and refine.

### Steps for Pi Coding Agent

1. Add saved searches:

```bash
musicidx saved-searches add chill-bar "chill bar" --limit 25
musicidx saved-searches list
musicidx saved-searches run chill-bar
```

2. Add saved playlists generated from searches:

```bash
musicidx playlists save chill-bar "chill bar" --limit 25
musicidx playlists list
musicidx playlists show chill-bar
```

3. Polish playlist export:

```bash
musicidx playlists export chill-bar --format m3u --out chill-bar.m3u
musicidx search "ambient background" --format m3u --absolute-paths
```

4. Support UI-friendly JSON contracts for saved searches and playlists.

5. Add lightweight commands for editing or deleting saved items.

### Acceptance Criteria

A user can search, save the result set, rerun it later, and export it as an M3U playlist from either CLI or UI.

---

# Example Query Interpretations

## Query: “Give me a list of 10 tracks, I’m today playing in a chill bar”

Expected intent:

```json
{
  "limit": 10,
  "context": "chill_bar",
  "energy": [0.25, 0.6],
  "valence": [0.35, 0.8],
  "tempo_bpm": [70, 115],
  "danceability": [0.3, 0.75],
  "aggression": [0.0, 0.25],
  "prefer_tags": [
    "chill",
    "lounge",
    "downtempo",
    "soul",
    "jazz",
    "warm",
    "groovy",
    "soft vocals",
    "instrumental"
  ],
  "avoid_tags": [
    "metal",
    "punk",
    "hardcore",
    "aggressive",
    "chaotic",
    "very loud",
    "screaming"
  ],
  "diversity": {
    "max_tracks_per_artist": 1,
    "max_tracks_per_album": 1
  }
}
```

## Query: “Shower music”

Expected intent:

```json
{
  "limit": 20,
  "context": "shower",
  "energy": [0.55, 0.95],
  "valence": [0.55, 1.0],
  "tempo_bpm": [95, 150],
  "danceability": [0.45, 1.0],
  "prefer_tags": [
    "upbeat",
    "pop",
    "dance",
    "singalong",
    "feel good",
    "energetic"
  ],
  "avoid_tags": [
    "ambient",
    "sad",
    "drone",
    "very quiet",
    "sleep"
  ]
}
```

## Query: “I’m a little bit sad and wanna listen to melancholic music”

Expected intent:

```json
{
  "limit": 10,
  "context": "melancholic_listening",
  "energy": [0.05, 0.5],
  "valence": [0.0, 0.45],
  "tempo_bpm": [50, 115],
  "prefer_tags": [
    "sad",
    "melancholic",
    "reflective",
    "acoustic",
    "piano",
    "ambient",
    "minor",
    "intimate"
  ],
  "avoid_tags": [
    "party",
    "aggressive",
    "happy",
    "very energetic"
  ]
}
```

---

# Coding-Agent Task Prompts

## Task 1 — Build Phases 0–2

```text
Build Phase 0–2 of the musicidx project.

Requirements:
- Create a Python CLI package called musicidx.
- Use Typer for CLI and Rich for output.
- Use SQLite as the local DB.
- Implement commands:
  - musicidx --help
  - musicidx doctor
  - musicidx init
  - musicidx db-info
  - musicidx scan <directory>
- Implement SQLite migrations for:
  - library_roots
  - tracks
  - audio_features
  - track_tags
  - track_profiles
  - tracks_fts
  - embeddings
  - search_events
  - feedback
- Scanner must recursively find supported audio files:
  .mp3, .flac, .m4a, .aac, .wav, .aiff, .aif, .ogg, .opus, .alac, .wv
- Scanner must store:
  path, path_hash, file_size, file_mtime_ns, extension, indexed_at.
- Scanner must be idempotent.
- Missing files must be marked with missing_at instead of deleted.
- Add pytest tests for DB creation and scanner idempotency.
- Do not implement audio analysis yet.
- Return JSON output when --json is passed.
```

## Task 2 — Build Phase 3 Metadata Extraction

```text
Build Phase 3 metadata extraction.

Requirements:
- Add metadata.py.
- Use ffprobe JSON output if available.
- Implement musicidx metadata command.
- Extract title, artist, album, album_artist, genre, date, track number,
  disc number, duration, codec, sample rate, bit rate, channels.
- Normalize common metadata tag aliases.
- Store extracted metadata in tracks table.
- Generate profile_text and profile_json in track_profiles.
- Sync tracks_fts with title, artist, album, genre, and profile_text.
- Add musicidx search-text <query> using SQLite FTS5.
- Add tests using small fixture files if available; otherwise mock ffprobe JSON.
```

## Task 3 — Build Phase 5 Basic Audio Analysis

```text
Build Phase 5 basic audio analysis.

Requirements:
- Add analyzer/basic_features.py.
- Implement musicidx analyze-basic.
- Use librosa to compute:
  BPM, RMS energy, spectral centroid mean/std, spectral flatness mean,
  spectral rolloff mean, zero crossing rate mean, MFCC mean/std,
  chroma profile, rough key/mode estimate.
- Normalize derived fields:
  energy, brightness, aggression proxy, danceability proxy.
- Store results in audio_features.
- Update track profile text after analysis.
- Add full-track chunked analysis with --chunked, --chunk-sec, and --max-chunks.
- Keep --quick only as an explicit CLI escape hatch; do not use it by default.
- Add --workers and --resource-profile options.
- Corrupt files must be recorded as analysis errors and skipped, not crash.
```

## Task 4 — Build Phases 8–9 Search

```text
Build Phase 8–9 search.

Requirements:
- Add search/intent.py, search/ranker.py, search/explain.py.
- Implement musicidx parse <query>.
- Implement deterministic rule-based parser for:
  chill, bar, shower, sad, melancholic, party, workout, focus, sleep.
- Implement musicidx search <query>.
- Retrieve candidates from:
  tracks_fts
  track_tags
  audio_features
  optional embeddings if present
- Rank using weighted hybrid score and expose calibrated raw relevance as user-facing score; do not normalize the top result to 1.0.
- Support structured sort_by for natural-language sorting such as highest BPM.
- Add --limit, --json, --explain, --format table/json/m3u.
- Enforce diversity:
  default max 2 tracks per artist.
- Search must only return non-missing files by default.
```

## Task 5 — Add Optional LLM Structured Intent Parsing

```text
Add optional LLM structured intent parsing.

Requirements:
- Add search/llm.py.
- Support Gemini/OpenAI providers from config; local providers can be added later.
- Prompt model to return only JSON matching the intent-hints schema.
- Always ask the LLM to produce usable music intent for vague/slang/conversational queries unless the user explicitly asks not to search.
- Validate/sanitize with strict allow-lists.
- If LLM is unavailable or invalid, fall back to the dynamic local parser.
- Add musicidx parse/search --llm and --no-llm.
- Add tests using mocked Gemini/OpenAI responses.
```

---

# Final Recommendation

Keep the CLI/database/search engine reliable and keep the Tauri desktop app as a thin wrapper over it. The MVP does not need deep raw-audio text models. It needs:

```text
reliable scanning
metadata extraction
basic MIR features
generated track profiles
semantic profile embeddings
structured query parsing
tunable hybrid ranking
playlist export
feedback loop
```

The product becomes genuinely useful once the feedback loop is added. For subjective queries like “chill bar” or “shower music,” user corrections will eventually outperform a generic model.
