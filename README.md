# MusicIdx

MusicIdx is a local-first CLI for indexing a music library in SQLite and searching it using metadata, audio features, ML mood/genre tags, and semantic profile embeddings.

Implemented so far:

- SQLite index and migrations
- Recursive audio-file scanner
- Metadata extraction with `ffprobe`
- SQLite FTS5 text search
- Fingerprinting with `fpcalc`/Chromaprint
- Duplicate and possible moved-file candidate reporting
- Basic audio feature extraction with `librosa`
- Optional local Essentia ML mood/genre tagging
- Optional semantic embeddings over enriched track profiles
- Dynamic library-aware natural-language parsing and hybrid search
- Optional Gemini/OpenAI intent parsing hints with local DB-only ranking
- M3U/JSON/CSV playlist-style search export

Not implemented yet:

- Feedback/evaluation loop
- Cross-platform desktop UI

## Local-first behavior

MusicIdx stores data locally in SQLite. It does not upload audio or metadata.

Default database path:

```text
./musicidx.sqlite
```

Default local ML model directory:

```text
./.musicidx-models/
```

Both can be overridden:

```bash
MUSICIDX_DB_PATH=/path/to/index.sqlite musicidx db-info
MUSICIDX_MODELS_PATH=/path/to/models musicidx models list
```

Most commands also support explicit paths:

```bash
musicidx init --db /path/to/index.sqlite
musicidx analyze-tags --models-path /path/to/models
```

## Installation

System tools for full local analysis:

macOS:

```bash
brew install ffmpeg chromaprint
```

Windows:

```powershell
# Install FFmpeg and Chromaprint/fpcalc, then ensure both are on PATH.
where ffprobe
where fpcalc
```

These provide:

- `ffprobe` for metadata/technical audio extraction
- `fpcalc` for Chromaprint fingerprinting

Verify on macOS/Linux:

```bash
which ffprobe
which fpcalc
musicidx doctor
```

If binaries are installed outside `PATH`, configure them explicitly:

```bash
MUSICIDX_FFPROBE_PATH=/path/to/ffprobe musicidx metadata
MUSICIDX_FPCALC_PATH=/path/to/fpcalc musicidx fingerprint
```

Development install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Optional ML tag support:

```bash
pip install -e '.[dev,ml]'
```

The `ml` extra uses the currently published Essentia TensorFlow pre-release build because no stable `2.1` wheel is available on PyPI yet.

Optional semantic embedding support:

```bash
pip install -e '.[dev,semantic]'
```

Everything:

```bash
pip install -e '.[dev,ml,semantic]'
```

Check local capabilities:

```bash
musicidx doctor
musicidx doctor --json
```

## Supported audio extensions

Scanner support:

```text
.mp3 .flac .m4a .aac .wav .aiff .aif .ogg .opus .alac .wv
```

## Quick start workflow

Index a library:

```bash
musicidx init
musicidx scan /path/to/music
musicidx metadata
musicidx analyze-basic --quick
musicidx analyze-tags
musicidx embed
```

Search with current implemented commands:

```bash
musicidx search-text "Nick Drake"
musicidx search-text "ambient relaxing"
musicidx search-semantic "chill atmospheric background music"
musicidx parse "chill bar"
musicidx search "chill bar" --limit 10 --explain
```

Inspect the database:

```bash
musicidx db-info
```

## Commands

### Help and diagnostics

```bash
musicidx --help
musicidx doctor
musicidx db-info
```

### Initialize database

```bash
musicidx init
musicidx init --json
musicidx init --db ./my-index.sqlite
```

Running `init` repeatedly is safe.

### Scan a directory

```bash
musicidx scan /path/to/music
musicidx scan /path/to/music --json
musicidx scan /path/to/music --dry-run
musicidx scan /path/to/music --full-hash
musicidx scan /path/to/music --follow-symlinks
```

Scanning is idempotent. Re-scanning unchanged files does not duplicate rows.

Deleted files are not removed from the database immediately. They are marked with `missing_at`.

Moved files currently appear as:

- old path marked missing
- new path added

After fingerprinting or full hashes, `musicidx duplicates` can report likely moved-file candidates.

### Extract metadata

Requires `ffprobe` from FFmpeg for real files. On macOS use `brew install ffmpeg`; on Windows install FFmpeg and ensure `ffprobe` is on `PATH`.

If needed, override the binary path:

```bash
MUSICIDX_FFPROBE_PATH=/path/to/ffprobe musicidx metadata
```

```bash
musicidx metadata
musicidx metadata --missing-only
musicidx metadata --track-id <track-id>
musicidx metadata --json
```

Extracted fields include:

- title
- artist
- album
- album artist
- genre
- date/year
- track/disc number
- duration
- codec/sample rate/bit rate/channels

Metadata also generates `track_profiles.profile_text` and syncs `tracks_fts`.

### Text search

```bash
musicidx search-text "Nick Drake"
musicidx search-text "ambient" --limit 20
musicidx search-text "electronic ambient" --json
musicidx search-text "ambient" --include-missing
```

This uses SQLite FTS5 over metadata and generated profile text.

### Fingerprint tracks

Requires `fpcalc` from Chromaprint for real files. On macOS use `brew install chromaprint`; on Windows install Chromaprint/fpcalc and ensure `fpcalc` is on `PATH`.

If needed, override the binary path:

```bash
MUSICIDX_FPCALC_PATH=/path/to/fpcalc musicidx fingerprint
```

```bash
musicidx fingerprint
musicidx fingerprint --missing-only
musicidx fingerprint --track-id <track-id>
musicidx fingerprint --json
```

Fingerprints store Chromaprint values in `tracks.chromaprint`.

### Duplicate and moved-file candidates

```bash
musicidx duplicates
musicidx duplicates --json
musicidx duplicates --exclude-missing
musicidx duplicates --duration-tolerance 5
```

Duplicate groups may be detected by:

- same content hash
- same Chromaprint and similar duration
- same artist/title and similar duration

No files are deleted or merged automatically.

### Basic audio analysis

Uses `librosa` to compute deterministic audio descriptors.

```bash
musicidx analyze-basic
musicidx analyze-basic --quick
musicidx analyze-basic --workers 4
musicidx analyze-basic --track-id <track-id>
musicidx analyze-basic --json
```

Computed features include:

- BPM
- energy proxy
- brightness proxy
- aggression proxy
- danceability proxy
- spectral centroid mean/std
- spectral flatness
- spectral rolloff
- zero crossing rate
- MFCC mean/std
- chroma profile
- rough key/mode estimate

Corrupt or unreadable files are recorded in `tracks.last_error` and skipped.

## ML mood/genre tags

ML tagging is optional and local. It uses local Essentia/TensorFlow model files. MusicIdx does not download model files automatically.

Check model path:

```bash
musicidx models path
```

Default:

```text
./.musicidx-models/
```

List configured models:

```bash
musicidx models list
musicidx models list --json
```

Run tag analysis:

```bash
musicidx analyze-tags
musicidx analyze-tags --workers 2
musicidx analyze-tags --missing-only
musicidx analyze-tags --track-id <track-id>
musicidx analyze-tags --min-score 0.15
musicidx analyze-tags --json
```

Show stored tags:

```bash
musicidx tags --track-id <track-id>
musicidx tags --track-id <track-id> --json
```

Stored tags go into `track_tags` and are added to `track_profiles.profile_text`, making them available to FTS and embeddings.

### Essentia model manifest

Create:

```text
.musicidx-models/manifest.json
```

Example using Discogs EffNet embedding plus genre and mood/theme classifier heads:

```json
{
  "models": [
    {
      "name": "genre-discogs400",
      "kind": "genre",
      "profile": "effnet_classifier",
      "sample_rate": 16000,
      "embedding_model": "discogs-effnet-bs64-1.pb",
      "embedding_output": "PartitionedCall:1",
      "classifier_model": "genre_discogs400-discogs-effnet-1.pb",
      "classifier_input": "serving_default_model_Placeholder",
      "classifier_output": "PartitionedCall:0",
      "labels_file": "genre_discogs400-discogs-effnet-1.json",
      "top_k": 10,
      "min_score": 0.0
    },
    {
      "name": "moodtheme-jamendo",
      "kind": "mood",
      "profile": "effnet_classifier",
      "sample_rate": 16000,
      "embedding_model": "discogs-effnet-bs64-1.pb",
      "embedding_output": "PartitionedCall:1",
      "classifier_model": "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
      "classifier_input": "model/Placeholder",
      "classifier_output": "model/Sigmoid",
      "labels_file": "mtg_jamendo_moodtheme-discogs-effnet-1.json",
      "top_k": 10,
      "min_score": 0.0
    }
  ]
}
```

Supported manifest profiles:

- `musicnn_classifier`
- `effnet_classifier`
- `direct_2d`

Set `min_score` to `0.0` if you want best-guess tags for every track. This stores the top `top_k` predictions even when model confidence is low. Use a higher `min_score` only if you prefer sparse, higher-confidence tags.

Review model licenses before commercial use.

## Semantic profile embeddings

Semantic profile search is optional. It embeds enriched `track_profiles.profile_text`, which may include:

- metadata
- audio feature descriptors
- ML mood/genre tags

Install semantic dependencies:

```bash
pip install -e '.[dev,semantic]'
```

Generate embeddings:

```bash
musicidx embed
musicidx embed --refresh
musicidx embed --batch-size 64
musicidx embed --track-id <track-id>
musicidx embed --model sentence-transformers/all-MiniLM-L6-v2
musicidx embed --json
```

Search semantically:

```bash
musicidx search-semantic "chill atmospheric music"
musicidx search-semantic "relaxing ambient background" --limit 10
musicidx search-semantic "upbeat shower music" --json
musicidx search-semantic "melancholic reflective songs" --include-missing
```

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Note: the first run may download the sentence-transformers model if it is not already cached. For offline/local-only operation, pre-cache the model or pass a local model path with `--model`.

## Dynamic natural-language search

`musicidx parse` and `musicidx search` use the analyzed local library to generate dynamic search intent. The parser combines:

- the user's query
- actual local tags in `track_tags`
- local audio-feature distributions
- available profile embeddings
- broad listening-context priors such as chill, bar, shower, focus, sleep, party, workout, sad/melancholic

Parse a query:

```bash
musicidx parse "Give me 10 tracks for a chill bar"
musicidx parse "shower music" --json
```

Search with hybrid ranking:

```bash
musicidx search "chill bar" --limit 10 --explain
musicidx search "shower music" --format json
musicidx search "shower music" --format json --concise
musicidx search "focus ambient background" --format m3u
musicidx search "melancholic reflective songs" --semantic-model .musicidx-models/all-MiniLM-L6-v2
```

Candidate scoring uses available signals:

- semantic profile similarity, if embeddings exist for the selected model
- ML mood/genre tag matches
- audio feature range fit
- SQLite FTS/profile text matches
- simple artist diversity, max 2 tracks per artist by default

Unknown queries still work through FTS and semantic/profile matching even when no context prior is detected.

### Optional Gemini/OpenAI intent parsing

By default, parsing is local and deterministic. You can explicitly add LLM intent hints with `--llm`. Gemini is the default provider.

Set your Gemini API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Optional Gemini model override:

```bash
export MUSICIDX_GEMINI_MODEL=gemini-1.5-flash
```

Use LLM-assisted parsing/search:

```bash
musicidx parse "Give me 10 tracks for a chill bar" --llm --json
musicidx search "shower music" --llm --limit 10 --explain
musicidx search "focus music for coding" --llm --format json --concise
```

OpenAI remains available as an explicit provider:

```bash
export OPENAI_API_KEY=your_key_here
musicidx search "chill bar" --llm --llm-provider openai --llm-model gpt-4o-mini
```

LLM behavior:

- sends the query plus aggregate library profile only
- does not send audio files
- does not send full track lists
- does not allow the LLM to recommend invented tracks
- falls back to dynamic local parsing if the selected LLM provider is unavailable or returns invalid JSON
- final ranking always uses only tracks from the local SQLite database

## Example: current test-music workflow

```bash
musicidx scan test-music
musicidx analyze-tags
musicidx tags --track-id <track-id>
musicidx search-text ambient
musicidx embed
musicidx search-semantic "relaxing ambient atmosphere"
```

Example stored ML tag from a local test run:

```json
{
  "source": "essentia:genre-discogs400",
  "tag": "electronic---ambient",
  "score": 0.173762
}
```

## Playlist/export workflow

You can export ranked search results directly:

```bash
musicidx search "ambient background" --format m3u > ambient.m3u
musicidx export "chill bar" --limit 25 --out chill-bar.m3u
musicidx export "chill bar" --format json --out chill-bar.json
musicidx export "chill bar" --format csv --out chill-bar.csv
```

Path options:

```bash
musicidx export "chill bar" --out chill-bar.m3u --absolute-paths
musicidx export "chill bar" --out chill-bar.m3u --relative-paths
```

The dedicated `export` command uses the same local parser/ranker as `search`, and supports Gemini/OpenAI intent hints with `--llm`.

For future desktop-wrapper integration notes, see `docs/ui-json-contracts.md`. A starter search-quality query set lives in `eval/search_queries.json`.

## JSON output

Commands intended for automation support `--json`, including:

```bash
musicidx doctor --json
musicidx init --json
musicidx db-info --json
musicidx scan /path/to/music --json
musicidx metadata --json
musicidx search-text "ambient" --json
musicidx fingerprint --json
musicidx duplicates --json
musicidx analyze-basic --json
musicidx analyze-tags --json
musicidx tags --track-id <track-id> --json
musicidx embed --json
musicidx search-semantic "ambient" --json
musicidx parse "chill bar" --json
musicidx parse "chill bar" --llm --json
musicidx search "chill bar" --json
musicidx search "chill bar" --json --concise
musicidx search "chill bar" --llm --json
musicidx search "chill bar" --llm --json --concise
musicidx export "chill bar" --json
musicidx models list --json
```

## CLI flag reference

### Global patterns

Most data-producing commands support:

| Flag | Meaning |
| --- | --- |
| `--db PATH` | Use a specific SQLite database instead of `./musicidx.sqlite` or `MUSICIDX_DB_PATH`. |
| `--json` | Print machine-readable JSON. |

### `musicidx scan <directory>`

| Flag | Meaning |
| --- | --- |
| `--full-hash` | Compute SHA-256 content hashes while scanning. Slower, but useful for exact duplicate/move detection. |
| `--follow-symlinks` | Follow symlinked directories/files while scanning. |
| `--dry-run` | Show scan counts without writing DB changes. |
| `--json` | Print scan summary as JSON. |

### `musicidx metadata`

| Flag | Meaning |
| --- | --- |
| `--track-id ID` | Extract metadata for one track only. |
| `--missing-only` | Process only tracks missing metadata/profile data. |
| `--json` | Print summary as JSON. |

### `musicidx search-text <query>`

| Flag | Meaning |
| --- | --- |
| `--limit N` | Limit number of FTS results. |
| `--include-missing` | Include tracks marked missing from disk. |
| `--json` | Print search results as JSON. |

### `musicidx fingerprint`

| Flag | Meaning |
| --- | --- |
| `--track-id ID` | Fingerprint one track only. |
| `--missing-only` | Process only tracks without stored fingerprints. |
| `--json` | Print summary as JSON. |

### `musicidx duplicates`

| Flag | Meaning |
| --- | --- |
| `--include-missing / --exclude-missing` | Include/exclude missing tracks. Including missing tracks helps detect moved-file candidates. |
| `--duration-tolerance SECONDS` | Duration tolerance for grouping possible duplicates. Default: `3.0`. |
| `--json` | Print duplicate groups as JSON. |

### `musicidx analyze-basic`

| Flag | Meaning |
| --- | --- |
| `--track-id ID` | Analyze one track only. |
| `--quick` | Analyze only the first 120 seconds. |
| `--workers N` | Number of analysis worker threads. |
| `--json` | Print summary as JSON. |

### `musicidx analyze-tags`

| Flag | Meaning |
| --- | --- |
| `--models-path PATH` | Use a specific local Essentia model directory. |
| `--track-id ID` | Analyze ML tags for one track only. |
| `--missing-only` | Process only tracks without stored Essentia tags. |
| `--min-score FLOAT` | Runtime minimum tag score. Manifest model `min_score` may also apply. Use `0.0` for best guesses. |
| `--workers N` | Number of tag-analysis worker threads. |
| `--json` | Print summary as JSON. |

### `musicidx tags --track-id <id>`

| Flag | Meaning |
| --- | --- |
| `--track-id ID` | Required track ID to inspect. |
| `--json` | Print stored tags as JSON. |

### `musicidx embed`

| Flag | Meaning |
| --- | --- |
| `--track-id ID` | Embed one track profile only. |
| `--model NAME_OR_PATH` | Sentence-transformers model name or local path. Example: `.musicidx-models/all-MiniLM-L6-v2`. |
| `--batch-size N` | Embedding batch size. |
| `--refresh` | Recompute even when stored embedding text is current. |
| `--json` | Print summary as JSON. |

### `musicidx search-semantic <query>`

| Flag | Meaning |
| --- | --- |
| `--model NAME_OR_PATH` | Embedding model name/path matching stored embeddings. |
| `--limit N` | Limit number of semantic results. |
| `--include-missing` | Include tracks marked missing from disk. |
| `--json` | Print results as JSON. |

### `musicidx parse <query>`

| Flag | Meaning |
| --- | --- |
| `--limit N` | Override parsed result limit. |
| `--semantic-model NAME_OR_PATH` | Embedding model name/path to consider when checking semantic availability. |
| `--include-missing` | Build intent while including missing tracks in library statistics. |
| `--llm / --no-llm` | Enable/disable LLM intent hints. Default: disabled. |
| `--llm-provider PROVIDER` | LLM provider. Supported: `gemini`, `openai`. Default: `gemini`. |
| `--llm-model MODEL` | Provider model override. Gemini default comes from `MUSICIDX_GEMINI_MODEL`. |
| `--llm-timeout SECONDS` | LLM request timeout. |
| `--json` | Print parsed intent as JSON. |

### `musicidx search <query>`

| Flag | Meaning |
| --- | --- |
| `--limit N` | Limit number of returned tracks, 1–100. |
| `--semantic-model NAME_OR_PATH` | Embedding model name/path to use if matching embeddings exist. |
| `--include-missing` | Include tracks marked missing from disk. Default: false. |
| `--explain` | Include human-readable explanation lines per result. |
| `--format table/json/m3u` | Output as Rich table, JSON, or M3U playlist. Default: `table`. |
| `--concise` | Shorter JSON output. Omits full library profile and verbose breakdowns. Useful with `--format json` or `--json`. |
| `--llm / --no-llm` | Enable/disable LLM intent hints. Default: disabled. |
| `--llm-provider PROVIDER` | LLM provider. Supported: `gemini`, `openai`. Default: `gemini`. |
| `--llm-model MODEL` | Provider model override. |
| `--llm-timeout SECONDS` | LLM request timeout. |
| `--json` | Shortcut for `--format json`. Can be combined with `--concise`. |

Examples:

```bash
musicidx search "chill bar" --limit 5 --explain
musicidx search "shower music" --format json --concise
musicidx search "focus music" --llm --llm-provider gemini --limit 10 --explain
musicidx search "ambient background" --format m3u > ambient.m3u
```

### `musicidx export <query>`

| Flag | Meaning |
| --- | --- |
| `--out PATH`, `-o PATH` | Write export output to a file instead of stdout. |
| `--limit N` | Limit number of exported tracks, 1–100. |
| `--semantic-model NAME_OR_PATH` | Embedding model name/path to use if matching embeddings exist. |
| `--include-missing` | Include tracks marked missing from disk. Default: false. |
| `--format m3u/json/csv` | Export format. Default: `m3u`. |
| `--absolute-paths` | Export absolute track paths. |
| `--relative-paths` | Export paths relative to the output file or cwd. |
| `--llm / --no-llm` | Enable/disable LLM intent hints. Default: disabled. |
| `--llm-provider PROVIDER` | LLM provider. Supported: `gemini`, `openai`. Default: `gemini`. |
| `--llm-model MODEL` | Provider model override. |
| `--llm-timeout SECONDS` | LLM request timeout. |
| `--json` | Shortcut for `--format json`. |

Examples:

```bash
musicidx export "chill bar" --limit 25 --out chill-bar.m3u
musicidx export "chill bar" --format csv --out chill-bar.csv --absolute-paths
musicidx export "focus music" --llm --format json --out focus.json
```

### `musicidx models path`

| Flag | Meaning |
| --- | --- |
| `--models-path PATH` | Show an explicit model path instead of the default. |
| `--json` | Print paths as JSON. |

### `musicidx models list`

| Flag | Meaning |
| --- | --- |
| `--models-path PATH` | List models from a specific local model directory. |
| `--json` | Print model status as JSON. |

## Development checks

```bash
ruff check .
pytest -q
```

Or, without installing into the active Python environment:

```bash
PYTHONPATH=src uv run --no-project --with pytest python -m pytest -q
uv run --no-project --with ruff ruff check .
```

## Privacy notes

- Audio files stay local.
- Metadata stays local.
- SQLite DB is local.
- Essentia model inference is local.
- Semantic embeddings are stored locally.
- No telemetry is implemented.

When `--llm` is used, MusicIdx sends the user query and aggregate library profile to the selected LLM provider, Gemini by default, for intent parsing. It does not send audio files or full track lists. Do not use `--llm` if you want a fully local-only run.

The other exception is optional dependency/model installation: tools like `pip`, `uv`, or `sentence-transformers` may download packages/models if you request them and they are not already cached.
