from __future__ import annotations

import json

from typer.testing import CliRunner

from musicidx.cli import app
from musicidx.db import connect_db, init_db
from musicidx.missing import list_missing_tracks, prune_missing_tracks


def test_list_missing_tracks(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        root_id = _insert_root(conn, tmp_path / "library")
        _insert_track(conn, "missing-1", tmp_path / "library" / "gone.mp3", root_id, missing=True)
        _insert_track(conn, "active-1", tmp_path / "library" / "active.mp3", root_id, missing=False)
        conn.commit()

        missing = list_missing_tracks(conn)

        assert [track.id for track in missing] == ["missing-1"]
        assert missing[0].root_path == str((tmp_path / "library").resolve())
        assert missing[0].missing_at is not None
    finally:
        conn.close()


def test_prune_missing_tracks_deletes_missing_rows_and_related_data_only(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        root_id = _insert_root(conn, tmp_path / "library")
        _insert_track(conn, "missing-1", tmp_path / "library" / "gone.mp3", root_id, missing=True)
        _insert_track(conn, "active-1", tmp_path / "library" / "active.mp3", root_id, missing=False)
        _insert_related_rows(conn, "missing-1")
        _insert_related_rows(conn, "active-1")
        conn.commit()

        pruned = prune_missing_tracks(conn)

        assert pruned == 1
        assert _track_exists(conn, "missing-1") is False
        assert _track_exists(conn, "active-1") is True
        assert _related_count(conn, "tracks_fts", "missing-1") == 0
        assert _related_count(conn, "tracks_fts", "active-1") == 1
        assert _related_count(conn, "audio_features", "missing-1") == 0
        assert _related_count(conn, "track_tags", "missing-1") == 0
        assert _related_count(conn, "track_profiles", "missing-1") == 0
        assert _related_count(conn, "embeddings", "missing-1") == 0
    finally:
        conn.close()


def test_prune_missing_track_id_does_not_delete_active_track(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        root_id = _insert_root(conn, tmp_path / "library")
        _insert_track(conn, "active-1", tmp_path / "library" / "active.mp3", root_id, missing=False)
        conn.commit()

        assert prune_missing_tracks(conn, track_id="active-1") == 0
        assert _track_exists(conn, "active-1") is True
    finally:
        conn.close()


def test_missing_and_prune_missing_cli_json(tmp_path):
    db_path = tmp_path / "index.sqlite"
    conn = connect_db(db_path)
    try:
        init_db(conn)
        root_id = _insert_root(conn, tmp_path / "library")
        _insert_track(conn, "missing-1", tmp_path / "library" / "gone.mp3", root_id, missing=True)
        conn.commit()
    finally:
        conn.close()

    runner = CliRunner()
    missing_result = runner.invoke(app, ["missing", "--db", str(db_path), "--json"])
    assert missing_result.exit_code == 0, missing_result.output
    missing_payload = json.loads(missing_result.output)
    assert missing_payload["count"] == 1
    assert missing_payload["missing"][0]["id"] == "missing-1"

    prune_result = runner.invoke(app, ["prune-missing", "--db", str(db_path), "--all", "--json"])
    assert prune_result.exit_code == 0, prune_result.output
    prune_payload = json.loads(prune_result.output)
    assert prune_payload["pruned"] == 1


def _insert_root(conn, path) -> int:
    resolved = str(path.resolve())
    cursor = conn.execute(
        """
        INSERT INTO library_roots (path, created_at, updated_at)
        VALUES (?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
        """,
        (resolved,),
    )
    return int(cursor.lastrowid)


def _insert_track(conn, track_id: str, path, root_id: int, *, missing: bool) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, root_id, path, path_hash, extension, file_size, file_mtime_ns,
            indexed_at, missing_at, title, artist, album
        ) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            root_id,
            str(path.resolve()),
            f"hash-{track_id}",
            path.suffix,
            "2026-01-01T00:00:00+00:00",
            "2026-01-02T00:00:00+00:00" if missing else None,
            f"Title {track_id}",
            "Artist",
            "Album",
        ),
    )


def _insert_related_rows(conn, track_id: str) -> None:
    conn.execute(
        "INSERT INTO audio_features (track_id, updated_at) VALUES (?, ?)",
        (track_id, "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO track_tags (track_id, source, tag, score, updated_at) VALUES (?, ?, ?, ?, ?)",
        (track_id, "essentia:test", "tag", 0.8, "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO track_profiles (track_id, profile_text, profile_json, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (track_id, "profile", "{}", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO embeddings (track_id, kind, model, dim, vector, text, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (track_id, "profile", "model", 1, b"x", "profile", "2026-01-01T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO tracks_fts (track_id, title, artist, album, genre, profile_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (track_id, "Title", "Artist", "Album", None, "profile"),
    )


def _track_exists(conn, track_id: str) -> bool:
    return conn.execute("SELECT 1 FROM tracks WHERE id = ?", (track_id,)).fetchone() is not None


def _related_count(conn, table: str, track_id: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE track_id = ?", (track_id,)).fetchone()
    return int(row[0])
