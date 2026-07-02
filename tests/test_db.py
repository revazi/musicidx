from __future__ import annotations

import array
import sqlite3

import chromaprint
import pytest

from musicidx.db import CORE_TABLES, apply_migrations, connect_db, init_db, table_exists
from musicidx.migrations import (
    add_decoded_chromaprint,
    add_metadata_provenance,
    add_versioned_track_profiles,
)


def _encoded_test_fingerprint() -> str:
    payload = array.array("I", [1, 2, 3, 4])
    return chromaprint.encode_fingerprint(payload, 1).decode("ascii")


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)

        for table in CORE_TABLES:
            assert table_exists(conn, table), table
    finally:
        conn.close()


def test_foreign_keys_are_enforced(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
                VALUES ('missing-track', 'text', '{}', '2026-01-01T00:00:00+00:00')
                """
            )
    finally:
        conn.close()


def test_fts_table_can_insert_and_search(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
            VALUES (
                'track-1', 'Pink Moon', 'Nick Drake', 'Pink Moon', 'folk',
                'melancholic acoustic'
            )
            """
        )
        rows = conn.execute(
            "SELECT track_id FROM tracks_fts WHERE tracks_fts MATCH ?",
            ("Nick",),
        ).fetchall()

        assert [row["track_id"] for row in rows] == ["track-1"]
    finally:
        conn.close()


def test_metadata_provenance_migration_tolerates_existing_columns(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        add_metadata_provenance(conn)
        add_metadata_provenance(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tracks)")}
        assert "artist_norm" in columns
        assert table_exists(conn, "track_metadata_claims")
    finally:
        conn.close()


def test_versioned_profile_migration_tolerates_existing_columns(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        add_versioned_track_profiles(conn)
        add_versioned_track_profiles(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(track_profiles)")}
        assert "embedding_text" in columns
        assert "profile_schema_version" in columns
        assert "source_fingerprint" in columns
    finally:
        conn.close()


def test_decoded_chromaprint_migration_backfills_existing_rows(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        conn.execute(
            """
            INSERT INTO tracks (
                id, path, path_hash, extension, file_size, file_mtime_ns,
                chromaprint, indexed_at
            ) VALUES (
                'track-1', '/tmp/a.mp3', 'hash-track-1', '.mp3', 1, 1, ?,
                '2026-01-01T00:00:00+00:00'
            )
            """,
            (_encoded_test_fingerprint(),),
        )
        conn.execute(
            """
            UPDATE tracks
            SET chromaprint_algorithm = NULL,
                chromaprint_frames = NULL,
                chromaprint_frame_count = NULL
            WHERE id = 'track-1'
            """
        )

        add_decoded_chromaprint(conn)

        row = conn.execute(
            """
            SELECT chromaprint_algorithm, chromaprint_frames, chromaprint_frame_count
            FROM tracks WHERE id = 'track-1'
            """
        ).fetchone()
        assert row["chromaprint_algorithm"] == 1
        assert row["chromaprint_frames"] is not None
        assert row["chromaprint_frame_count"] == 4
    finally:
        conn.close()


def test_migrations_are_idempotent(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        apply_migrations(conn)
        apply_migrations(conn)

        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert [row["version"] for row in rows] == [1, 2, 3, 4, 5, 6, 7]
    finally:
        conn.close()
