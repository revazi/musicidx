from __future__ import annotations

import shutil

import pytest

from musicidx.db import connect_db, init_db, utc_now
from musicidx.scanner import scan_library


def test_scanner_is_idempotent_and_marks_missing(tmp_path):
    root = tmp_path / "library"
    nested = root / "nested"
    nested.mkdir(parents=True)
    track_a = root / "a.mp3"
    track_b = nested / "b.FLAC"
    ignored = root / "notes.txt"
    track_a.write_bytes(b"audio-a")
    track_b.write_bytes(b"audio-b")
    ignored.write_text("not audio")

    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)

        first = scan_library(root, conn)
        assert first.added == 2
        assert first.unchanged == 0
        assert first.modified == 0
        assert first.missing == 0
        assert _track_count(conn) == 2

        second = scan_library(root, conn)
        assert second.added == 0
        assert second.unchanged == 2
        assert second.modified == 0
        assert second.missing == 0
        assert _track_count(conn) == 2

        track_a.write_bytes(b"audio-a changed")
        third = scan_library(root, conn)
        assert third.added == 0
        assert third.unchanged == 1
        assert third.modified == 1
        assert third.missing == 0
        assert _track_count(conn) == 2

        track_b.unlink()
        fourth = scan_library(root, conn)
        assert fourth.added == 0
        assert fourth.unchanged == 1
        assert fourth.modified == 0
        assert fourth.missing == 1
        assert _track_count(conn) == 2

        missing_at = conn.execute(
            "SELECT missing_at FROM tracks WHERE path = ?",
            (str(track_b.resolve()),),
        ).fetchone()["missing_at"]
        assert missing_at is not None
    finally:
        conn.close()


def test_scan_dry_run_does_not_write_rows(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "song.ogg").write_bytes(b"audio")

    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        summary = scan_library(root, conn, dry_run=True)

        assert summary.added == 1
        assert _track_count(conn) == 0
    finally:
        conn.close()


def test_scan_marks_tracks_missing_when_known_root_disappears(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "a.mp3").write_bytes(b"audio-a")
    (root / "b.flac").write_bytes(b"audio-b")

    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        scan_library(root, conn)
        shutil.rmtree(root)

        summary = scan_library(root, conn)

        assert summary.root_missing is True
        assert summary.missing == 2
        assert _missing_track_count(conn) == 2

        second = scan_library(root, conn)
        assert second.root_missing is True
        assert second.missing == 0
        assert _missing_track_count(conn) == 2
    finally:
        conn.close()


def test_scan_unknown_missing_root_still_errors(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        with pytest.raises(FileNotFoundError):
            scan_library(tmp_path / "not-indexed", conn)
    finally:
        conn.close()


def test_modified_file_invalidates_derived_outputs(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    track = root / "song.mp3"
    track.write_bytes(b"old-audio")

    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        scan_library(root, conn)
        track_row = conn.execute(
            "SELECT id FROM tracks WHERE path = ?",
            (str(track.resolve()),),
        ).fetchone()
        track_id = track_row["id"]
        _seed_derived_outputs(conn, track_id)

        track.write_bytes(b"new-audio-with-different-size")
        summary = scan_library(root, conn)

        assert summary.modified == 1
        row = conn.execute(
            """
            SELECT chromaprint, fingerprint_duration, title, duration_sec,
                   codec, analysis_version, analyzed_at
            FROM tracks
            WHERE id = ?
            """,
            (track_id,),
        ).fetchone()
        assert row["chromaprint"] is None
        assert row["fingerprint_duration"] is None
        assert row["title"] is None
        assert row["duration_sec"] is None
        assert row["codec"] is None
        assert row["analysis_version"] == 0
        assert row["analyzed_at"] is None
        assert _related_count(conn, "audio_features", track_id) == 0
        assert _related_count(conn, "track_tags", track_id) == 0
        assert _related_count(conn, "track_profiles", track_id) == 0
        assert _related_count(conn, "embeddings", track_id) == 0
        assert _related_count(conn, "tracks_fts", track_id) == 0
    finally:
        conn.close()


def _seed_derived_outputs(conn, track_id: str) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE tracks
        SET chromaprint = 'old-fingerprint', fingerprint_duration = 12.3,
            title = 'Old title', duration_sec = 12.3, codec = 'mp3',
            analysis_version = 999, analyzed_at = ?
        WHERE id = ?
        """,
        (now, track_id),
    )
    conn.execute(
        "INSERT INTO audio_features (track_id, updated_at) VALUES (?, ?)",
        (track_id, now),
    )
    conn.execute(
        "INSERT INTO track_tags (track_id, source, tag, score, updated_at) VALUES (?, ?, ?, ?, ?)",
        (track_id, "essentia:test", "old", 0.9, now),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (track_id, "old profile", "{}", now),
    )
    conn.execute(
        """
        INSERT INTO embeddings (track_id, kind, model, dim, vector, text, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (track_id, "profile", "test-model", 1, b"old", "old profile", now),
    )
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (track_id, "Old title", None, None, None, "old profile"),
    )
    conn.commit()


def _related_count(conn, table: str, track_id: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE track_id = ?", (track_id,)).fetchone()
    return int(row[0])


def _missing_track_count(conn) -> int:
    row = conn.execute("SELECT COUNT(*) FROM tracks WHERE missing_at IS NOT NULL").fetchone()
    return int(row[0])


def _track_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
