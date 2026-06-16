from __future__ import annotations

from musicidx.db import connect_db, init_db
from musicidx.failures import list_failed_tracks, record_track_error, reset_failed_tracks
from musicidx.metadata import MetadataExtractionError, process_metadata


def test_record_track_error_quarantines_after_threshold(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", tmp_path / "bad.mp3")
        conn.commit()

        record_track_error(conn, "track-1", "first", threshold=2)
        record_track_error(conn, "track-1", "second", threshold=2)
        conn.commit()

        failed = list_failed_tracks(conn)
        assert len(failed) == 1
        assert failed[0].error_count == 2
        assert failed[0].last_error == "second"
        assert failed[0].quarantined_at is not None
        assert failed[0].quarantine_reason == "second"
    finally:
        conn.close()


def test_reset_failed_tracks_clears_quarantine(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", tmp_path / "bad.mp3")
        conn.commit()
        record_track_error(conn, "track-1", "bad", threshold=1)
        conn.commit()

        reset = reset_failed_tracks(conn, track_id="track-1")

        assert reset >= 1
        assert list_failed_tracks(conn) == []
    finally:
        conn.close()


def test_metadata_skips_quarantined_tracks(monkeypatch, tmp_path):
    track_path = tmp_path / "bad.mp3"
    track_path.write_bytes(b"bad")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        conn.commit()
        record_track_error(conn, "track-1", "bad", threshold=1)
        conn.commit()

        def fake_extract_metadata(path):
            raise MetadataExtractionError("should not be called")

        monkeypatch.setattr("musicidx.metadata.extract_metadata", fake_extract_metadata)

        summary = process_metadata(conn, missing_only=True)

        assert summary.processed == 0
        assert summary.errors == 0
        assert summary.skipped == 0
    finally:
        conn.close()


def _insert_track(conn, track_id: str, path) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix),
    )
