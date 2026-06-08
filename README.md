# MusicIdx

MusicIdx is a local-first CLI for indexing a music library in SQLite.

Current phase: **Phases 0–2** — project setup, SQLite database, and directory scanner.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Commands

```bash
musicidx --help
musicidx doctor
musicidx init
musicidx db-info
musicidx scan /path/to/music
musicidx metadata
musicidx search-text "Nick Drake"
musicidx fingerprint
musicidx duplicates
musicidx analyze-basic --quick
musicidx models path
musicidx models list
musicidx analyze-tags
musicidx tags --track-id <id>
musicidx embed
musicidx search-semantic "chill atmospheric music"
```

The default database path is project-local, relative to the current working directory:

```text
./musicidx.sqlite
```

Override it with either:

```bash
musicidx init --db /path/to/index.sqlite
```

or the environment variable:

```bash
MUSICIDX_DB_PATH=/path/to/index.sqlite musicidx init
```

## ML tag models

ML mood/genre tagging is optional and local. Install the optional Essentia extra if it is available for your platform:

```bash
pip install -e '.[dev,ml]'
```

Model files are not downloaded automatically. Put local Essentia/TensorFlow model files under:

```bash
musicidx models path
```

and add a `manifest.json` file. Example shape:

```json
{
  "models": [
    {
      "name": "mood-basic",
      "kind": "mood",
      "profile": "musicnn_classifier",
      "embedding_model": "msd-musicnn-1.pb",
      "embedding_output": "model/dense/BiasAdd",
      "classifier_model": "mood_happy-msd-musicnn-1.pb",
      "classifier_input": "model/Placeholder",
      "classifier_output": "model/Sigmoid",
      "labels": ["not happy", "happy"],
      "top_k": 2,
      "min_score": 0.2
    }
  ]
}
```

## Semantic profile search

Profile embeddings are optional and use enriched profile text, including metadata, audio features, and ML tags.

Install the optional semantic extra:

```bash
pip install -e '.[dev,semantic]'
```

Then generate embeddings and search:

```bash
musicidx embed
musicidx search-semantic "chill atmospheric music"
```
