"""SQLite migrations for MusicIdx."""

from __future__ import annotations

INITIAL_SCHEMA_SQL = """
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
"""

ADD_FINGERPRINT_DURATION_SQL = """
ALTER TABLE tracks ADD COLUMN fingerprint_duration REAL;
"""

ADD_FAILURE_QUARANTINE_SQL = """
ALTER TABLE tracks ADD COLUMN error_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tracks ADD COLUMN last_error_at TEXT;
ALTER TABLE tracks ADD COLUMN quarantined_at TEXT;
ALTER TABLE tracks ADD COLUMN quarantine_reason TEXT;
"""

MIGRATIONS = [
    (1, "initial_schema", INITIAL_SCHEMA_SQL),
    (2, "add_fingerprint_duration", ADD_FINGERPRINT_DURATION_SQL),
    (3, "add_failure_quarantine", ADD_FAILURE_QUARANTINE_SQL),
]
