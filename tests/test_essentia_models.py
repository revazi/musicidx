from __future__ import annotations

import json

from musicidx.analyzer.essentia_models import (
    EssentiaModelSpec,
    TrackTag,
    available_model_specs,
    list_track_tags,
    model_manifest_status,
    process_tags,
    save_track_tags,
)
from musicidx.db import connect_db, init_db
from musicidx.metadata import search_text


def test_model_manifest_status_lists_available_local_models(tmp_path):
    models_path = tmp_path / "models"
    models_path.mkdir()
    (models_path / "classifier.pb").write_bytes(b"model")
    (models_path / "manifest.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "mood-basic",
                        "kind": "mood",
                        "profile": "direct_2d",
                        "model": "classifier.pb",
                        "labels": ["sad", "melancholic", "happy"],
                        "top_k": 2,
                    }
                ]
            }
        )
    )

    specs = available_model_specs(models_path)
    status = model_manifest_status(models_path)

    assert len(specs) == 1
    assert specs[0].name == "mood-basic"
    assert specs[0].labels == ["sad", "melancholic", "happy"]
    assert status.manifest_exists is True
    assert status.models[0]["available"] is True


def test_save_track_tags_updates_tags_profile_and_fts(tmp_path):
    track_path = tmp_path / "song.mp3"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(
            conn,
            "track-1",
            track_path,
            title="Blue Room",
            artist="Late Night",
            genre="Ambient",
        )
        tags = [
            TrackTag(source="essentia:mood", tag="melancholic", score=0.82),
            TrackTag(source="essentia:genre", tag="ambient", score=0.74),
        ]

        save_track_tags(conn, "track-1", tags)
        conn.commit()

        stored = list_track_tags(conn, track_id="track-1")
        assert [(tag.tag, tag.score) for tag in stored] == [
            ("melancholic", 0.82),
            ("ambient", 0.74),
        ]

        profile = conn.execute(
            "SELECT profile_text, profile_json FROM track_profiles WHERE track_id = ?",
            ("track-1",),
        ).fetchone()
        assert "Tags: melancholic 0.82, ambient 0.74." in profile["profile_text"]
        assert json.loads(profile["profile_json"])["tags"][0]["tag"] == "melancholic"

        results = search_text(conn, "melancholic ambient")
        assert [result.track_id for result in results] == ["track-1"]
    finally:
        conn.close()


def test_process_tags_uses_available_models_and_mocked_analyzer(monkeypatch, tmp_path):
    track_path = tmp_path / "song.flac"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)
        spec = EssentiaModelSpec(
            name="genre-basic",
            kind="genre",
            profile="direct_2d",
            labels=["electronic"],
            model=tmp_path / "model.pb",
        )

        monkeypatch.setattr(
            "musicidx.analyzer.essentia_models.available_model_specs",
            lambda models_path: [spec],
        )

        def fake_analyze_essentia_tags(path, specs, *, min_score):
            assert path == track_path
            assert specs == [spec]
            assert min_score == 0.25
            return [TrackTag(source="essentia:genre-basic", tag="electronic", score=0.91)]

        monkeypatch.setattr(
            "musicidx.analyzer.essentia_models.analyze_essentia_tags",
            fake_analyze_essentia_tags,
        )

        summary = process_tags(conn, models_path=tmp_path, min_score=0.25)

        assert summary.processed == 1
        assert summary.updated == 1
        assert summary.errors == 0
        assert summary.model_count == 1
        stored = list_track_tags(conn, track_id="track-1")
        assert [(tag.source, tag.tag, tag.score) for tag in stored] == [
            ("essentia:genre-basic", "electronic", 0.91)
        ]
    finally:
        conn.close()


def test_process_tags_records_error_when_no_models_are_available(tmp_path):
    track_path = tmp_path / "song.wav"
    track_path.write_bytes(b"audio")
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", track_path)

        summary = process_tags(conn, models_path=tmp_path)

        assert summary.processed == 0
        assert summary.updated == 0
        assert summary.errors == 1
        row = conn.execute("SELECT last_error FROM tracks WHERE id = ?", ("track-1",)).fetchone()
        assert row["last_error"] == "no available local Essentia model specs found"
    finally:
        conn.close()


def _insert_track(
    conn,
    track_id: str,
    path,
    *,
    title: str | None = None,
    artist: str | None = None,
    genre: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns,
            title, artist, genre, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, '2026-01-01T00:00:00+00:00')
        """,
        (
            track_id,
            str(path),
            f"hash-{track_id}",
            path.suffix.lower(),
            title,
            artist,
            genre,
        ),
    )
    conn.commit()
