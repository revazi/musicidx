from __future__ import annotations

import sqlite3

import pytest

from musicidx.db import CORE_TABLES, apply_migrations, connect_db, init_db, table_exists


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


def test_migrations_are_idempotent(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        apply_migrations(conn)
        apply_migrations(conn)

        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert [row["version"] for row in rows] == [1, 2]
    finally:
        conn.close()
