from __future__ import annotations

import array
from types import SimpleNamespace

import chromaprint

from musicidx.db import connect_db, init_db
from musicidx.fingerprint import (
    TrackFingerprint,
    find_duplicate_groups,
    fingerprint_path,
    process_fingerprints,
)


def test_fingerprint_path_parses_fpcalc_json(monkeypatch, tmp_path):
    track_path = tmp_path / "song.mp3"
    track_path.write_bytes(b"audio")

    def fake_run(command, capture_output, text, check):
        assert command == ["fpcalc", "-json", str(track_path)]
        assert capture_output is True
        assert text is True
        assert check is False
        return SimpleNamespace(
            returncode=0,
            stdout='{"duration": 123.4, "fingerprint": "abc123"}',
            stderr="",
        )

    monkeypatch.setattr("musicidx.fingerprint.subprocess.run", fake_run)
    monkeypatch.setattr("musicidx.fingerprint.resolve_executable", lambda name, env_var: name)

    fingerprint = fingerprint_path(track_path)

    assert fingerprint.chromaprint == "abc123"
    assert fingerprint.duration_sec == 123.4


def test_process_fingerprints_updates_track(monkeypatch, tmp_path):
    track_path = tmp_path / "song.flac"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        encoded = _encoded_test_fingerprint()

        def fake_fingerprint_path(path):
            assert path == track_path
            return TrackFingerprint(chromaprint=encoded, duration_sec=98.0)

        monkeypatch.setattr("musicidx.fingerprint.fingerprint_path", fake_fingerprint_path)
        summary = process_fingerprints(conn)

        assert summary.processed == 1
        assert summary.updated == 1
        assert summary.errors == 0
        row = conn.execute(
            """
            SELECT chromaprint, fingerprint_duration, chromaprint_algorithm,
                   chromaprint_frames, chromaprint_frame_count
            FROM tracks WHERE id = 'track-1'
            """
        ).fetchone()
        assert row["chromaprint"] == encoded
        assert row["fingerprint_duration"] == 98.0
        assert row["chromaprint_algorithm"] == 1
        assert row["chromaprint_frames"] is not None
        assert row["chromaprint_frame_count"] == 4
    finally:
        conn.close()


def test_find_duplicate_groups_reports_audio_duplicates_and_possible_moves(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-1",
            tmp_path / "a.mp3",
            artist="Air",
            title="Kelly Watch the Stars",
            content_hash="hash-a",
            chromaprint="fp-a",
            duration_sec=250.0,
            fingerprint_duration=250.0,
        )
        _insert_track(
            conn,
            "track-2",
            tmp_path / "copy.mp3",
            artist="Air",
            title="Kelly Watch the Stars",
            content_hash="hash-a",
            chromaprint="fp-a",
            duration_sec=251.0,
            fingerprint_duration=251.0,
        )
        _insert_track(
            conn,
            "track-3",
            tmp_path / "old-path.mp3",
            content_hash="hash-move",
            missing_at="2026-01-01T00:00:00+00:00",
        )
        _insert_track(conn, "track-4", tmp_path / "new-path.mp3", content_hash="hash-move")

        groups = find_duplicate_groups(conn)

        exact_groups = [group for group in groups if group.kind == "exact_duplicate"]
        move_groups = [group for group in groups if group.kind == "possible_move"]
        assert any(
            {track.track_id for track in group.tracks} == {"track-1", "track-2"}
            for group in exact_groups
        )
        assert any(
            {track.track_id for track in group.tracks} == {"track-3", "track-4"}
            for group in move_groups
        )
    finally:
        conn.close()


def _encoded_test_fingerprint() -> str:
    payload = array.array("I", [1, 2, 3, 4])
    return chromaprint.encode_fingerprint(payload, 1).decode("ascii")


def _insert_track(
    conn,
    track_id: str,
    path,
    *,
    artist: str | None = None,
    title: str | None = None,
    content_hash: str | None = None,
    chromaprint: str | None = None,
    duration_sec: float | None = None,
    fingerprint_duration: float | None = None,
    missing_at: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns, content_hash,
            chromaprint, title, artist, duration_sec, fingerprint_duration, indexed_at,
            missing_at
        ) VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?, '2026-01-01T00:00:00+00:00', ?)
        """,
        (
            track_id,
            str(path),
            f"hash-{track_id}",
            path.suffix.lower(),
            content_hash,
            chromaprint,
            title,
            artist,
            duration_sec,
            fingerprint_duration,
            missing_at,
        ),
    )
    conn.commit()
