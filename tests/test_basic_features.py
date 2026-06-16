from __future__ import annotations

import json
import sys
import warnings
from types import SimpleNamespace

import pytest

from musicidx.analyzer.basic_features import (
    ANALYSIS_VERSION,
    AudioAnalysisError,
    AudioFeatures,
    _chunk_windows,
    analyze_basic_features,
    estimate_key_mode,
    process_basic_analysis,
    save_audio_features,
)
from musicidx.db import connect_db, init_db
from musicidx.metadata import search_text


def test_chunk_windows_can_sample_long_files_evenly():
    assert _chunk_windows(600.0, chunk_sec=60.0, max_chunks=3) == [
        (0.0, 60.0),
        (240.0, 60.0),
        (540.0, 60.0),
    ]


def test_estimate_key_mode_returns_rough_key_for_chroma_profile():
    key_name, mode = estimate_key_mode([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert key_name == "C"
    assert mode in {"major", "minor"}


def test_save_audio_features_updates_features_profile_fts_and_track_version(tmp_path):
    track_path = tmp_path / "song.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-1",
            track_path,
            title="Warm Lights",
            artist="Night Bar",
            album="Late Set",
            genre="Downtempo",
        )
        features = _features()

        save_audio_features(conn, "track-1", features)
        conn.commit()

        feature_row = conn.execute(
            "SELECT bpm, energy, brightness, mfcc_mean_json FROM audio_features WHERE track_id = ?",
            ("track-1",),
        ).fetchone()
        assert feature_row["bpm"] == 92.0
        assert feature_row["energy"] == 0.25
        assert json.loads(feature_row["mfcc_mean_json"]) == [1.0, 2.0]

        track_row = conn.execute(
            "SELECT analysis_version, analyzed_at, last_error FROM tracks WHERE id = ?",
            ("track-1",),
        ).fetchone()
        assert track_row["analysis_version"] == ANALYSIS_VERSION
        assert track_row["analyzed_at"] is not None
        assert track_row["last_error"] is None

        profile_row = conn.execute(
            "SELECT profile_text, profile_json FROM track_profiles WHERE track_id = ?",
            ("track-1",),
        ).fetchone()
        assert "Audio:" in profile_row["profile_text"]
        assert "low energy" in profile_row["profile_text"]
        assert json.loads(profile_row["profile_json"])["analysis_version"] == ANALYSIS_VERSION

        results = search_text(conn, "downtempo energy")
        assert [result.track_id for result in results] == ["track-1"]
    finally:
        conn.close()


def test_process_basic_analysis_skips_current_version(monkeypatch, tmp_path):
    track_path = tmp_path / "song.flac"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        def fake_analyze_basic_features(path, *, quick=False, **kwargs):
            assert path == track_path
            assert quick is True
            return _features()

        monkeypatch.setattr(
            "musicidx.analyzer.basic_features.analyze_basic_features",
            fake_analyze_basic_features,
        )

        first = process_basic_analysis(conn, quick=True)
        second = process_basic_analysis(conn, quick=True)

        assert first.processed == 1
        assert first.updated == 1
        assert second.processed == 0
        assert second.skipped == 1
    finally:
        conn.close()


def test_analyze_basic_features_captures_decoder_noise(monkeypatch, tmp_path, capsys):
    track_path = tmp_path / "bad.mp3"
    track_path.write_bytes(b"bad")

    def fake_load(*args, **kwargs):
        warnings.warn("PySoundFile failed. Trying audioread instead.", UserWarning, stacklevel=2)
        print("[src/libmpg123] bad header", file=sys.stderr)
        raise RuntimeError("")

    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=fake_load))

    with pytest.raises(AudioAnalysisError) as exc_info:
        analyze_basic_features(track_path, quick=True)

    assert "bad header" in str(exc_info.value)
    assert "PySoundFile failed" in str(exc_info.value)
    assert capsys.readouterr().err == ""


def test_process_basic_analysis_records_errors_without_crashing(monkeypatch, tmp_path):
    track_path = tmp_path / "corrupt.wav"
    track_path.write_bytes(b"not audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        def fake_analyze_basic_features(path, *, quick=False, **kwargs):
            raise AudioAnalysisError("failed to decode audio")

        monkeypatch.setattr(
            "musicidx.analyzer.basic_features.analyze_basic_features",
            fake_analyze_basic_features,
        )

        summary = process_basic_analysis(conn)

        assert summary.processed == 1
        assert summary.updated == 0
        assert summary.errors == 1
        row = conn.execute("SELECT last_error FROM tracks WHERE id = ?", ("track-1",)).fetchone()
        assert row["last_error"] == "failed to decode audio"
    finally:
        conn.close()


def _features() -> AudioFeatures:
    return AudioFeatures(
        bpm=92.0,
        key_name="C",
        mode="minor",
        dynamic_range=0.10,
        energy=0.25,
        danceability=0.55,
        aggression=0.15,
        brightness=0.30,
        spectral_centroid_mean=1500.0,
        spectral_centroid_std=200.0,
        spectral_flatness_mean=0.05,
        spectral_rolloff_mean=3200.0,
        zero_crossing_rate_mean=0.04,
        mfcc_mean=[1.0, 2.0],
        mfcc_std=[0.5, 0.6],
        chroma_profile=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        raw_features={"sample_rate": 22050},
    )


def _insert_track(
    conn,
    track_id: str,
    path,
    *,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
    genre: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns, title, artist,
            album, genre, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?, '2026-01-01T00:00:00+00:00')
        """,
        (
            track_id,
            str(path),
            f"hash-{track_id}",
            path.suffix.lower(),
            title,
            artist,
            album,
            genre,
        ),
    )
    conn.commit()
