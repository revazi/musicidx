from __future__ import annotations

from musicidx.db import connect_db, init_db
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


def _track_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
