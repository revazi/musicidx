from __future__ import annotations

import json

from musicidx.db import connect_db, init_db
from musicidx.derived import (
    DERIVED_TAG_SOURCE,
    derive_context_fit,
    derive_feature_tags,
    rebuild_derived_signals,
)


def test_derive_context_fit_scores_club_and_background():
    club = {
        "bpm": 124.0,
        "energy": 0.82,
        "danceability": 0.90,
        "aggression": 0.25,
        "brightness": 0.55,
        "vocalness": None,
        "instrumentalness": None,
    }
    ambient = {
        "bpm": 82.0,
        "energy": 0.28,
        "danceability": 0.30,
        "aggression": 0.10,
        "brightness": 0.25,
        "vocalness": None,
        "instrumentalness": None,
    }

    club_scores = {fit.context: fit.score for fit in derive_context_fit(club)}
    ambient_scores = {fit.context: fit.score for fit in derive_context_fit(ambient)}

    assert club_scores["club"] > 0.85
    assert club_scores["party"] > 0.80
    assert ambient_scores["dark_ambient"] > 0.80
    assert ambient_scores["background"] > 0.80


def test_double_time_raw_bpm_uses_perceived_tempo_for_feature_tags():
    tags = {
        tag.tag: tag.score
        for tag in derive_feature_tags(
            {
                "bpm": 172.0,
                "energy": 0.70,
                "danceability": 0.70,
                "aggression": 0.20,
                "brightness": 0.45,
                "vocalness": None,
                "instrumentalness": None,
            }
        )
    }

    assert "fast" not in tags
    assert tags["slow"] > 0.8


def test_no_vocals_tag_requires_vocal_or_instrumental_evidence():
    no_vocal_evidence = {
        "bpm": 90.0,
        "energy": 0.30,
        "danceability": 0.30,
        "aggression": 0.10,
        "brightness": 0.35,
        "vocalness": None,
        "instrumentalness": None,
    }
    instrumental_evidence = {**no_vocal_evidence, "instrumentalness": 0.95}

    no_evidence_tags = {tag.tag for tag in derive_feature_tags(no_vocal_evidence)}
    with_evidence_tags = {tag.tag for tag in derive_feature_tags(instrumental_evidence)}
    no_evidence_contexts = {fit.context: fit for fit in derive_context_fit(no_vocal_evidence)}
    with_evidence_contexts = {fit.context: fit for fit in derive_context_fit(instrumental_evidence)}

    assert "no_vocals_background" not in no_evidence_tags
    assert no_evidence_contexts["no_vocals_background"].score <= 0.45
    assert no_evidence_contexts["no_vocals_background"].confidence < 0.5
    assert "no_vocals_background" in with_evidence_tags
    assert with_evidence_contexts["no_vocals_background"].score > 0.75


def test_rebuild_derived_writes_tags_context_and_profile(tmp_path):
    conn = connect_db(tmp_path / "index.sqlite")
    try:
        init_db(conn)
        _insert_track(conn, "track-1", tmp_path / "song.mp3")
        _insert_features(
            conn,
            "track-1",
            bpm=122.0,
            energy=0.78,
            danceability=0.88,
            aggression=0.22,
            brightness=0.50,
            instrumentalness=0.9,
        )
        conn.commit()

        summary = rebuild_derived_signals(conn)

        assert summary.processed == 1
        assert summary.updated == 1
        tags = conn.execute(
            """
            SELECT tag, score FROM track_tags
            WHERE track_id = 'track-1' AND source = ?
            ORDER BY score DESC
            """,
            (DERIVED_TAG_SOURCE,),
        ).fetchall()
        assert any(row["tag"] == "danceable" for row in tags)
        assert any(row["tag"] == "club_friendly" for row in tags)

        context = conn.execute(
            """
            SELECT context, score FROM track_context_fit
            WHERE track_id = 'track-1'
            ORDER BY score DESC
            """
        ).fetchall()
        assert context[0]["context"] in {"club", "party", "driving"}

        profile = conn.execute(
            "SELECT profile_text, profile_json FROM track_profiles WHERE track_id = 'track-1'"
        ).fetchone()
        assert "Context fit:" not in profile["profile_text"]
        doc = json.loads(profile["profile_json"])
        assert doc["schema_version"] == 2
        assert doc["context_fit"]["club"] > 0.8
        assert "club" in doc["search_text"]["embedding_text"]
    finally:
        conn.close()


def _insert_track(conn, track_id: str, path) -> None:
    path.write_bytes(b"audio")
    conn.execute(
        """
        INSERT INTO tracks (
            id, path, path_hash, extension, file_size, file_mtime_ns,
            title, artist, title_norm, artist_norm, artist_title_norm,
            metadata_confidence, indexed_at
        ) VALUES (?, ?, ?, ?, 1, 1, 'Dance Song', 'Test Artist',
                  'dance song', 'test artist', 'test artist dance song', 0.8,
                  '2026-01-01T00:00:00+00:00')
        """,
        (track_id, str(path), f"hash-{track_id}", path.suffix.lower()),
    )


def _insert_features(
    conn,
    track_id: str,
    *,
    bpm: float,
    energy: float,
    danceability: float,
    aggression: float,
    brightness: float,
    instrumentalness: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audio_features (
            track_id, bpm, energy, danceability, aggression, brightness,
            instrumentalness, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, '2026-01-01T00:00:00+00:00')
        """,
        (track_id, bpm, energy, danceability, aggression, brightness, instrumentalness),
    )
